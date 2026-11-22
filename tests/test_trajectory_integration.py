import json
import logging

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("comfy")

from mode_explorer.nodes import ModeExplorer, ModeExplorerLoadResponse, ModeExplorerMechanistic
from test_comfy_integration import draw, model, traces


def capture(model, caplog, tmp_path, monkeypatch, **settings):
    import folder_paths

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    patch, = ModeExplorer().apply(model, trajectory_label="discovery", **settings)
    result = draw(patch)
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
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor

    dit = model.model.diffusion_model
    targets = (("b1.attn.wo", dit.blocks[1].attn, "wo"), ("b2.mlp.down", dit.blocks[2].mlp, "down"))
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

        native, native_run = capture(model, caplog, tmp_path, monkeypatch, feature="native")
        _, active_run = capture(model, caplog, tmp_path, monkeypatch, strength=.03, fidelity=.81, adaptive=False)
        opts = dict(feature_block=1, projection_layers=selection, blocks="1,2", sites="clean", evaluation=0,
                    frozen_state=native_run["states"][0]["export_id"], export=True, projection_study="activation_factorial",
                    projection_activation_check=True, analysis_device="cpu", feature_strength=.03, auxiliary_budget=5,
                    factorial_feature_source=active_run["states"][0]["export_id"])
        patched, = ModeExplorerMechanistic().apply(model, **opts)
        assert torch.equal(draw(patched), native)
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
        assert torch.equal(draw(rejection), native)
        rejected, = traces(caplog)[-1]["response"]["points"]
        assert rejected["reason"] == "factorial_active_feature_settings_mismatch"
        assert rejected["auxiliary_dit_calls"] == 0
    assert all(not getattr(parent, key)._forward_hooks for _, parent, key in targets)
