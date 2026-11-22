import json
import logging

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("comfy")

from mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerMechanistic
from test_comfy_integration import draw, model, traces


def capture(model, caplog, tmp_path, monkeypatch, *, draw_options=None, **settings):
    import folder_paths

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    patch, = ModeExplorer().apply(model, trajectory_label="discovery", **settings)
    result = draw(patch, **(draw_options or {}))
    entry = traces(caplog)[-1]["trajectory"]
    manifest = json.loads((tmp_path / "modeexplorer_trajectory" / entry["id"] / "manifest.json").read_text())
    return result, manifest


def test_actual_trajectory_captures_every_state_without_changing_sampler(model, caplog, tmp_path, monkeypatch):
    native = draw(model)
    rng = torch.random.get_rng_state().clone()
    observed, run = capture(model, caplog, tmp_path, monkeypatch, feature="native")
    assert torch.equal(native, observed)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert run["status"] == "completed", run
    assert run["native_dit_calls"] == len(run["states"]) == 5
    assert run["auxiliary_dit_calls"] == 0
    assert [state["solver_step"] for state in run["states"]] == list(range(5))
    for state in run["states"]:
        export_id = state["export_id"]
        path = tmp_path / "modeexplorer_trajectory" / f"{export_id}.safetensors"
        tensors = load_file(path)
        assert set(tensors) == {"frozen_native_input", "conditional_velocity", "baseline_clean"}
        latent, _ = ModeExplorerLoadResponse().load(export_id, "baseline_clean")
        assert torch.equal(latent["samples"], tensors["baseline_clean"])


def test_active_capture_records_rounding_without_changing_float32_edit(model, caplog, tmp_path, monkeypatch):
    cfg = dict(strength=.03, fidelity=.7, adaptive=False)
    expected, = ModeExplorer().apply(model, **cfg)
    clean = draw(expected)
    observed, run = capture(model, caplog, tmp_path, monkeypatch, **cfg)
    assert torch.equal(clean, observed)
    assert run["live_reference_dit_calls"] == 2
    assert len(run["states"]) == 5
    assert run["states"][0]["feature_applied"]
    assert run["states"][0]["realized_shift_rms"][0] > 0
    assert not run["states"][-1]["feature_applied"]


def test_factorial_reuses_exact_frozen_native_velocity_and_requires_int8(model, caplog, tmp_path, monkeypatch):
    native, run = capture(model, caplog, tmp_path, monkeypatch, feature="native")
    export = run["states"][1]["export_id"]
    patched, = ModeExplorerMechanistic().apply(
        model, feature_block=1, blocks="1", evaluation=1, frozen_state=export, export=True,
        projection_study="activation_factorial", projection_activation_check=True,
        analysis_device="cpu", feature_strength=.03, auxiliary_budget=6, all_actual=False,
    )
    assert torch.equal(draw(patched), native)
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["mechanistic"]["projection"]["reason"] == "requires_observed_tensorwise_int8_weight", point
    assert point["auxiliary_dit_calls"] <= 4


def test_frozen_factorial_replaces_generated_tokens_on_both_int8_arms(model, caplog, tmp_path, monkeypatch):
    import comfy_kitchen
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    block = model.model.diffusion_model.blocks[2]
    old = block.attn.wo
    layer = ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(64, 64, bias=False, device="cpu").eval()
    with torch.inference_mode(), comfy_kitchen.use_backend("eager"):
        layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
            old.weight.detach(), "TensorWiseINT8Layout", convrot=True, convrot_groupsize=4, per_channel=True,
        ), requires_grad=False)
        layer.quant_format = "int8_tensorwise"
        layer.layout_type = "TensorWiseINT8Layout"
        block.attn.wo = layer
        native, run = capture(model, caplog, tmp_path, monkeypatch, feature="native")
        export = run["states"][1]["export_id"]
        patched, = ModeExplorerMechanistic().apply(
            model, feature_block=1, projection_block=2, blocks="1,2", evaluation=1, frozen_state=export,
            export=True, projection_study="activation_factorial", projection_activation_check=True,
            analysis_device="cpu", feature_strength=.03, auxiliary_budget=6,
        )
        assert torch.equal(draw(patched), native)
    point, = traces(caplog)[-1]["response"]["points"]
    report = point["mechanistic"]["projection"]
    assert report["status"] == "qualified", report
    assert point["auxiliary_dit_calls"] == 6
    assert report["block"] == 2 and point["mechanistic"]["injection_block"] == 1
    assert report["activation_counterfactual"]["qualification"] == "bit_exact"
    assert report["factorial"]["comparisons"]["precision_native"]["displacement_l2"] > 0
    assert report["factorial"]["comparisons"]["interaction"]["displacement_l2"] > 0
    assert report["factorial"]["production_cuda"] is False
    for name in ("baseline_clean", "D01_clean", "continuous_native_clean", "continuous_feature_clean"):
        latent, _ = ModeExplorerLoadResponse().load(point["export_id"], name)
        assert latent["samples"].shape == native.shape
    assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_trajectory_keeps_er_sde_spectrum_forecasts_out_of_exports(model, caplog, tmp_path, monkeypatch):
    pytest.importorskip("comfyui_spectrum_krea2")
    from test_spectrum_integration import SIGMAS, spectrum
    import folder_paths

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    patched, = ModeExplorer().apply(model, feature="native", trajectory_label="spectrum")
    options = dict(sampler="er_sde", stage="planner", sigmas=SIGMAS)
    observed = draw(spectrum(patched), **options)
    assert torch.equal(observed, draw(spectrum(model), **options))
    run = traces(caplog)[-1]
    manifest = json.loads((tmp_path / "modeexplorer_trajectory" / run["trajectory"]["id"] / "manifest.json").read_text())
    assert run["spectrum_forecast_calls"] > 0
    assert len(manifest["states"]) == run["native_dit_calls"]
    assert {state["route"] for state in manifest["states"]} == {"actual"}
    assert len(manifest["routes"]) == len(SIGMAS) - 1


def test_factorial_rejects_later_unedited_state_from_guided_trajectory(model, caplog, tmp_path, monkeypatch):
    native = draw(model)
    _, run = capture(model, caplog, tmp_path, monkeypatch, strength=.03, fidelity=.7, adaptive=False)
    state = run["states"][2]
    assert not state["feature_applied"]
    patched, = ModeExplorerMechanistic().apply(
        model, feature_block=1, blocks="1", evaluation=2, frozen_state=state["export_id"],
        export=True, projection_study="activation_factorial", projection_activation_check=True,
        analysis_device="cpu", feature_strength=.03, auxiliary_budget=6,
    )
    assert torch.equal(draw(patched), native)
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["reason"] == "frozen_replay_requires_native_trajectory"
    assert point["auxiliary_dit_calls"] == 0


@pytest.mark.parametrize("selection", ["downstream", "b1.attn.wo,b2.mlp.down"])
def test_frozen_factorial_qualifies_each_current_packed_layer(model, caplog, tmp_path, monkeypatch, selection):
    import comfy_kitchen
    from mode_explorer.continuous_layers import ContinuousLayers
    from mode_explorer.projection import ProjectionUnavailable
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    dit = model.model.diffusion_model
    targets = (("b1.attn.wo", dit.blocks[1].attn, "wo"), ("b2.mlp.down", dit.blocks[2].mlp, "down"))
    original_replace = ContinuousLayers.replace

    def assert_native_nongenerated(self, name, module, source, output, bounds):
        replaced, report = original_replace(self, name, module, source, output, bounds)
        lo, hi = bounds
        assert 0 < lo < hi < output.shape[1]  # Text and reference rows are both present.
        assert torch.equal(replaced[:, :lo], output[:, :lo])
        assert torch.equal(replaced[:, hi:], output[:, hi:])
        return replaced, report

    monkeypatch.setattr(ContinuousLayers, "replace", assert_native_nongenerated)
    with torch.inference_mode(), comfy_kitchen.use_backend("eager"):
        for _, parent, key in targets:
            old = getattr(parent, key)
            layer = ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(
                old.in_features, old.out_features, bias=False, device="cpu").eval()
            layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
                old.weight.detach(), "TensorWiseINT8Layout", convrot=True, convrot_groupsize=4, per_channel=True,
            ), requires_grad=False)
            layer.quant_format = "int8_tensorwise"
            layer.layout_type = "TensorWiseINT8Layout"
            setattr(parent, key, layer)

        native, native_run = capture(model, caplog, tmp_path, monkeypatch, feature="native", draw_options={"reference": True})
        _, active_run = capture(model, caplog, tmp_path, monkeypatch, strength=.03, fidelity=.81, adaptive=False, draw_options={"reference": True})
        opts = dict(feature_block=1, projection_layers=selection, blocks="1,2", sites="clean", evaluation=0,
                    frozen_state=native_run["states"][0]["export_id"], export=True, projection_study="activation_factorial",
                    projection_activation_check=True, analysis_device="cpu", feature_strength=.03, auxiliary_budget=5,
                    factorial_feature_source=active_run["states"][0]["export_id"])
        patched, = ModeExplorerMechanistic().apply(model, **opts)
        assert torch.equal(draw(patched, reference=True), native)
        point, = traces(caplog)[-1]["response"]["points"]
        report = point["mechanistic"]["projection"]
        assert report["status"] == "qualified", report
        assert report["selected_layers"] == [name for name, _, _ in targets]
        assert report["local_projection_calls"] == 4 and point["auxiliary_dit_calls"] == 5
        assert point["mechanistic"]["factorial_feature_settings"]["fidelity"] == .81
        assert point["mechanistic"]["proposal"]["active_archive_same_noisy_input"]
        assert point["mechanistic"]["proposal"]["active_archive_same_proposal"]
        assert report["factorial"]["comparisons"]["precision_feature"]["displacement_l2"] > 0
        for arm in ("continuous_native", "continuous_feature"):
            assert all(layer["packed_native_replay_bit_exact"] and layer["activation_emulation"]["packed_bit_exact"]
                       for layer in report["layers"][arm].values())
        rejection, = ModeExplorerMechanistic().apply(model, **{**opts, "feature_strength": .04})
        assert torch.equal(draw(rejection, reference=True), native)
        rejected, = traces(caplog)[-1]["response"]["points"]
        assert rejected["reason"] == "factorial_active_feature_settings_mismatch"
        assert rejected["auxiliary_dit_calls"] == 0

        def reject_final_layer(self, name, *args):
            if name == targets[-1][0]:
                raise ProjectionUnavailable("simulated_layer_qualification_failure")
            return assert_native_nongenerated(self, name, *args)

        monkeypatch.setattr(ContinuousLayers, "replace", reject_final_layer)
        assert torch.equal(draw(patched, reference=True), native)
        rejected, = traces(caplog)[-1]["response"]["points"]
        assert rejected["mechanistic"]["projection"]["reason"] == "simulated_layer_qualification_failure"
        from mode_explorer.response_runtime import read_export
        from safetensors.torch import load_file

        _, path = read_export(rejected["export_id"])
        assert "continuous_native_clean" not in load_file(path)
        assert "continuous_feature_clean" not in load_file(path)
    assert all(not getattr(parent, key)._forward_hooks for _, parent, key in targets)


def test_trajectory_measures_unpooled_token_roi_without_edit(model, caplog, tmp_path, monkeypatch):
    native = draw(model)
    observed, captured = capture(model, caplog, tmp_path, monkeypatch, feature="native", trajectory_roi="1,3,1,5")
    assert torch.equal(observed, native)
    for state in captured["states"]:
        stats = state["token_roi"]
        assert stats["status"] == "measured" and stats["bounds_y0_y1_x0_x1"] == [1, 3, 1, 5]
        assert len(stats["covariance_eigenvalues"]) == 8
        assert 1 <= stats["effective_rank"] <= 8
        assert len(stats["vertical_neighbor_cosines"][0]) == 4
    with pytest.raises(ValueError, match="between 1 and 64"):
        ModeExplorer().apply(model, feature="native", trajectory_label="test", trajectory_roi="0,9,0,9")


def test_downstream_discovery_budget_admits_wide_projection_at_planner_resolution():
    from types import SimpleNamespace

    from mode_explorer.continuous_layers import workspace_bytes
    from mode_explorer.mechanism import MechanisticSettings
    from mode_explorer.mechanism_runtime import estimate_capture_workspace
    from tools.structural_workflows import factorial_controls

    spec = MechanisticSettings(
        blocks="1,5", sites="clean", projection_study="activation_factorial",
        projection_activation_check=True, projection_layers="downstream",
    )
    packed_rows = 102 * 154 + 512
    staged, reserve = estimate_capture_workspace(
        spec, feature_block=1, grid=(102, 154), latent_shape=(1, 16, 204, 308),
        attention_channels=6144, patch_size=2, baseline_reused=False, chunk=spec.cuda_chunk,
        packed_tokens=packed_rows, retain_packed_projection=False,
    )
    targets = {"b1.mlp.up": (None, SimpleNamespace(in_features=6144, out_features=16384))}
    required = staged + reserve + workspace_bytes(targets, packed_rows, spec.projection_token_chunk)
    mib = 1024**2
    assert 12288 * mib < required <= factorial_controls("factorial_downstream")["capture_vram_mib"] * mib
    assert required <= factorial_controls("factorial_downstream_first")["capture_vram_mib"] * mib
    assert factorial_controls("factorial_injection")["capture_vram_mib"] == 12288


def test_temporal_ablation_matches_guarded_spectrum_and_reference_schedule(model, caplog, tmp_path, monkeypatch):
    pytest.importorskip("comfyui_spectrum_krea2")
    from test_spectrum_integration import SIGMAS, spectrum
    import folder_paths

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    options = dict(sampler="er_sde", stage="planner", sigmas=SIGMAS)
    common = dict(strength=.03, fidelity=.7, adaptive=False, trajectory_label="ablation")

    def run(**widgets):
        patched, = ModeExplorer().apply(model, **common, **widgets)
        final = draw(spectrum(patched), **options)
        trace = traces(caplog)[-1]
        folder = tmp_path / "modeexplorer_trajectory" / trace["trajectory"]["id"]
        manifest = json.loads((folder / "manifest.json").read_text())
        records = [load_file(folder / f"{state['ordinal']:03}.safetensors") for state in manifest["states"]]
        assert torch.equal(final, records[-1]["solver_clean"])
        return final, manifest, records

    original_sham, old_sham, _ = run(sham=True)
    original_active, _, _ = run()
    route = "".join("A" if r["route"] == "actual" else "F" for r in old_sham["routes"])
    assert "F" in route
    rng = torch.random.get_rng_state().clone()
    arms = (("all", True, []), ("first_only", False, [0]), ("second_only", False, [1]), ("all", False, [0, 1]))
    results = [run(sham=sham, feature_application=app, ablation_route=route) for app, sham, _ in arms]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(results[0][0], original_sham)
    assert torch.equal(results[-1][0], original_active)
    for (final, m, _), (_, _, applied) in zip(results, arms):
        assert m["ablation"]["status"] == "qualified"
        assert m["ablation"]["applied_evaluations"] == applied
        assert m["ablation"]["reference_calls"] == [1, 1] + [0] * (len(route) - 2)
        assert m["solver_terminal"]["status"] == "bit_exact"
        assert [s["reference_history"] for s in m["states"][:2]] == [
            s["reference_history"] for s in results[0][1]["states"][:2]]
    assert torch.equal(results[0][2][0]["compact_feature_proposal"], results[-1][2][0]["compact_feature_proposal"])
    first, second = results[1][1]["states"][:2]
    assert first["feature_geometry"]["block"] == 1
    assert first["feature_geometry"]["proposal_shape"] == list(results[1][2][0]["compact_feature_proposal"].shape)
    assert 0 <= first["realization_error_rms"][0] < first["realized_shift_rms"][0]
    assert first["objective_after"][0] < first["objective"]
    assert second["realization_error_rms"] is None
    assert second["objective_after"][0] == second["objective"]
    assert torch.equal(results[0][2][1]["solver_input"], results[2][2][1]["solver_input"])
    assert not torch.equal(results[0][2][1]["solver_input"], results[1][2][1]["solver_input"])
    assert not torch.equal(results[0][0], results[1][0])
    assert not torch.equal(results[0][0], results[2][0])

    from mode_explorer.trajectory_review import ModeExplorerTrajectoryReview, TEMPORAL

    class FixtureVAE:
        calls = 0

        def decode(self, samples):
            self.calls += 1
            return torch.full((1, 8, 8, 3), .5)

    vae = FixtureVAE()
    plan = {"case": "fixture", "experiment": "temporal", "resolution": [8, 8],
            "crops": {"fixed_crop": [1, 2, 5, 7]},
            "arms": {arm: {"id": data["states"][0]["export_id"].split("/")[0]}
                     for arm, (_, data, _) in zip(TEMPORAL, results)}}
    ModeExplorerTrajectoryReview().review(vae, json.dumps(plan))
    review, = (tmp_path / "modeexplorer_phase_iv").glob("*/manifest.json")
    report = json.loads(review.read_text())
    assert report["decode_tensor"] == "solver_clean"
    assert report["auxiliary_dit_calls"] == 0
    assert vae.calls == report["vae_decodes"] == (len(route) - route.count("F")) * 4
    assert (review.parent / "step_00" / "fixed_crop.png").is_file()
    assert (review.parent / "step_00" / "full.png").is_file()
    assert not (review.parent / f"step_{route.index('F'):02d}").exists()

    wrong, = ModeExplorer().apply(model, **common, feature_application="first_only", ablation_route="A" * len(route))
    with pytest.raises(RuntimeError, match="ablation_route_mismatch"):
        draw(spectrum(wrong), **options)
    rejected = traces(caplog)[-1]
    assert rejected["ablation"]["status"] == "unqualified" and rejected["trajectory"]["status"] == "partial"
