import logging

import pytest
import torch

pytest.importorskip("comfy")
from test_comfy_integration import conditions, draw, model, traces

from mode_explorer.mechanism_runtime import Capture, Workspace
from mode_explorer.nodes import ModeExplorerMechanistic
from mode_explorer.mechanism import MechanisticSettings
from tools.analyze_mechanism import compare, component_pair, load, profile


def point(caplog):
    measurement, = traces(caplog)[-1]["response"]["points"]
    return measurement


def patched(model, **changes):
    return ModeExplorerMechanistic().apply(model, **{"feature_block": 1, "blocks": "0-2",
                                                     "evaluation": 1, "export": False, "analysis_device": "cpu", **changes})[0]


def test_signed_uses_exact_matched_image_sites_and_preserves_native_cfg(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, cfg=2, reference=True, split_cfg=True)
    rng = torch.random.get_rng_state().clone()
    proxied = patched(model, sites="attention_pre,attention_post,gate_logits,sigmoid_gate,projection_input,projection_output,"
                                 "attention_contribution,after_attention,mlp_input,mlp_output,mlp_contribution,"
                                 "after_mlp,final_projection,final_projection_unpatched,velocity,clean", amplitude_check=True, repeat_baseline=True)
    assert torch.equal(native, draw(proxied, cfg=2, reference=True, split_cfg=True))
    assert torch.equal(rng, torch.random.get_rng_state())
    m = point(caplog)
    assert m["status"] == "completed", m
    exp = m["mechanistic"]
    assert m["auxiliary_dit_calls"] == m["auxiliary_dit_rows"] == 8
    assert exp["repeat_clean_rms"] == 0
    assert exp["analysis_device"] == "cpu"
    assert exp["capture_transfer_bytes"] == 0
    assert exp["timings_ms"]["auxiliary_dit"] > 0
    assert exp["timings_ms"]["capture_copy"] > 0
    assert exp["timings_ms"]["spectral"] > 0
    assert exp["timings_ms"]["export"] == 0
    assert exp["sites"]["b1.attention_pre"]["comparisons"]["positive"]["displacement_rms"] == 0
    assert exp["sites"]["b1.attention_post"]["comparisons"]["positive"]["displacement_rms"] > 0
    assert exp["sites"]["b0.mlp_output"]["comparisons"]["positive"]["displacement_rms"] == 0
    assert exp["sites"]["b2.after_mlp"]["comparisons"]["positive"]["displacement_rms"] > 0
    assert exp["sites"]["b1.sigmoid_gate"]["status"] == "unsupported"
    assert exp["sites"]["b1.attention_contribution"]["status"] == "unsupported"
    assert exp["sites"]["b1.projection_input"]["comparisons"]["positive"]["shape_chw"] == [64, 4, 6]
    assert exp["sites"]["final_projection"]["comparisons"]["positive"]["shape_chw"] == [64, 4, 6]
    assert exp["sites"]["clean"]["comparisons"]["positive"]["shape_chw"] == [16, 7, 11]
    assert exp["sites"]["final_projection"]["latent_pixels_per_sample"] == 2
    assert exp["sites"]["final_projection_unpatched"]["latent_pixels_per_sample"] == 1
    assert exp["sites"]["final_projection_unpatched"]["comparisons"] == exp["sites"]["velocity"]["comparisons"]
    assert exp["sites"]["b1.projection_input"]["comparisons"]["positive"]["bands"]["sub_patch"]["frequency_bins"] == 0
    assert exp["sites"]["clean"]["comparisons"]["positive"]["bands"]["sub_patch"]["frequency_bins"] > 0
    assert exp["frequency_cycles_per_latent_pixel"] == [.0625, .125]
    assert exp["signed_realized_edit"]["status"] == "matched"
    assert exp["amplitude"]["status"] == "measured"


@pytest.mark.parametrize("mode,calls", [("conditioning_fixed", 6), ("joint_fixed", 7), ("joint_recomputed", 9)])
def test_conditioning_different_text_lengths_and_same_reference(model, caplog, mode, calls):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    positive, _ = conditions(reference=True)
    alt = [[positive[0][0][:, :3] * -.75, {**positive[0][1], "pooled_output": None,
                                           "attention_mask": torch.tensor([[1, 1, 0]])}]]
    m = patched(model, experiment=mode, alternate=alt, sites="attention_post,clean", auxiliary_budget=calls)
    assert torch.equal(draw(model, reference=True), draw(m, reference=True))
    p = point(caplog)
    assert p["status"] == "completed", p
    assert p["auxiliary_dit_calls"] == calls
    e = p["mechanistic"]
    if mode == "conditioning_fixed":
        assert set(e["arms"]) == {"D00", "D01", "D10", "D11"}
    assert [e["conditioning"][k]["tokens"] for k in ("C0", "C1")] == [5, 3]
    assert e["conditioning"]["C1"]["metadata_mask_sha256"]
    assert e["conditioning"]["C1"]["mask_sha256"] is None
    assert e["sites"]["clean"]["comparisons"]["conditioning_interaction"]["displacement_l2"] > 0
    if mode.endswith("fixed"):
        assert e["conditioning_realized_edit"]["status"] == "matched"
        assert "C1_sha256" not in e["proposal"]
    else:
        assert e["conditioning_realized_edit"]["status"] == "independent_proposals"
        assert e["proposal"]["C0_sha256"] != e["proposal"]["C1_sha256"]


def test_sham_budget_and_active_component(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model)
    no_edit = patched(model, feature_strength=0, sites="projection_input,clean")
    assert torch.equal(native, draw(no_edit))
    assert point(caplog)["mechanistic"]["sites"]["clean"]["comparisons"]["positive"]["displacement_rms"] == 0
    constrained = patched(model, auxiliary_budget=2)
    assert torch.equal(native, draw(constrained))
    p = point(caplog)
    assert p["reason"] == "auxiliary_forward_budget" and p["auxiliary_dit_calls"] == 0
    active = patched(model, treatment="minimum_change", component="odd", band="low", sites="clean", clean_limit=.02)
    assert not torch.equal(native, draw(active))
    p = point(caplog)
    assert p["applied"] and 0 < p["applied_clean_displacement_rms"] <= .02
    assert torch.equal(native, draw(active, stage="highres"))
    assert torch.equal(draw(model, cfg=2), draw(active, cfg=2))
    assert traces(caplog)[-1]["response"]["reason"] == "response_steering_requires_native_cfg_one"


def test_capture_hooks_release_on_failure(model):
    dit = model.get_model_object("diffusion_model")
    spec = MechanisticSettings(blocks="0", sites="gate_logits")
    capture = Capture(spec, (4, 6), Workspace(1), 0)
    gate = dit.blocks[0].attn.gate
    counts = (len(gate._forward_hooks), len(dit.blocks[0]._forward_pre_hooks))
    with pytest.raises(RuntimeError, match="interrupted"), capture.hooks(dit):
        raise RuntimeError("interrupted")
    assert counts == (len(gate._forward_hooks), len(dit.blocks[0]._forward_pre_hooks))


def test_export_frozen_match_and_shared_reference_start(model, caplog, tmp_path, monkeypatch):
    from json import loads

    import folder_paths
    from mode_explorer.nodes import ModeExplorerLoadResponse

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    cfg = patched(model, evaluation=-1, states=3, blocks="1", amplitude_check=True, auxiliary_budget=21, export=True,
                  sites="attention_pre,attention_post,projection_input,projection_output,after_attention,after_mlp,"
                        "final_projection_unpatched,clean")
    draw(cfg)
    first, second, third = traces(caplog)[-1]["response"]["points"]
    assert all(p["export_status"] == "completed" for p in (first, second, third))
    assert [p["auxiliary_dit_calls"] for p in (first, second, third)] == [6, 6, 6]
    assert [p["mechanistic"]["first_estimate_reused_for"] for p in (first, second, third)] == [[], ["C0"], ["C0"]]
    assert [p["mechanistic"]["feature_reference_calls"] for p in (first, second, third)] == [1, 1, 1]
    export_id = second["export_id"]
    manifest = loads((tmp_path / "modeexplorer_response" / export_id / "manifest.json").read_text())
    assert manifest["native_input_fp32_sha256"] == second["native_input_fp32_sha256"]
    assert 0 < manifest["mechanistic"]["timings_ms"]["export"] <= second["mechanistic"]["timings_ms"]["export"]
    export_path, exported, study = load(tmp_path / "modeexplorer_response" / export_id)
    assert compare((exported, study), (exported, study))["status"] == "matched_inputs"
    old = {**study, "schema": 1, "frequency_cycles_per_grid_cell": [.125, .25]}
    assert compare((exported, old), (exported, study))["status"] == "incompatible_inputs"
    assert profile(study, "conditioning_interaction")["clean"]["status"] == "comparison_missing"
    components = component_pair(export_path, "complete_native_delta", export_path, "complete_native_delta")
    assert components["displacement_cosine"] == pytest.approx(1)
    assert components["spatial_energy_overlap"] == pytest.approx(1)
    decoded, sigmas = ModeExplorerLoadResponse().load(export_id, "D01_clean")
    assert decoded["samples"].shape == (1, 16, 1, 7, 11)
    replay = patched(model, frozen_state=export_id, sites="clean")
    assert torch.equal(draw(model, sigmas=sigmas, seed=101), draw(replay, sigmas=sigmas, seed=101))
    frozen = point(caplog)
    assert frozen["status"] == "completed", frozen
    assert frozen["native_input_sha256"] == second["native_input_sha256"]
    assert frozen["mechanistic"]["sites"]["clean"] == second["mechanistic"]["sites"]["clean"]


def test_cross_dtype_replay_matches_condition_values_and_rejects_changes(model, caplog, tmp_path, monkeypatch):
    import folder_paths
    import test_comfy_integration as fixture

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    def rounded_conditions(*args, **kwargs):
        p, n = conditions(*args, **kwargs)
        p[0][0] = p[0][0].bfloat16().float()
        return p, n

    monkeypatch.setattr(fixture, "conditions", rounded_conditions)
    dit = model.model.diffusion_model
    dit.bfloat16()
    dit.dtype = torch.bfloat16
    draw(patched(model, evaluation=0, sites="clean", export=True))
    src = point(caplog)
    assert src["export_status"] == "completed" and src["input_dtype"] == "torch.bfloat16"
    dit.float()
    dit.dtype = torch.float32
    options = dict(evaluation=0, sites="clean", frozen_state=src["export_id"])
    draw(patched(model, **options))
    assert point(caplog)["reason"] == "frozen_sigma_or_conditioning_mismatch"
    cfg = patched(model, match_cross_dtype=True, **options)
    assert torch.equal(draw(model, seed=19), draw(cfg, seed=19))
    p = point(caplog)
    assert p["status"] == "completed", p
    assert p["native_input_fp32_sha256"] == src["native_input_fp32_sha256"]
    assert p["input_dtype"] == "torch.float32" and p["conditioning_sha256"] != src["conditioning_sha256"]
    assert p["native_condition_values"] == src["native_condition_values"]
    monkeypatch.setattr(fixture, "conditions", conditions)
    draw(cfg)
    assert point(caplog)["reason"] == "frozen_condition_values_mismatch"


def test_three_production_size_exports_keep_all_seven_tensors(tmp_path, monkeypatch):
    import folder_paths
    from safetensors import safe_open

    from mode_explorer.response import MAX_EXPORT_BYTES, ResponseSettings
    from mode_explorer.response_runtime import ResponseRun, _export, read_export

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    spec = MechanisticSettings()
    response = ResponseRun(ResponseSettings(states=3, export=True, mechanistic=spec), lambda tensor: tensor + 1)
    names = ("frozen_native_input", "baseline_clean", "D01_clean", "D0minus_clean",
             "complete_native_delta", "odd_native_delta", "even_native_delta")
    shape = (1, 16, 1, 250, 250)
    for state in range(3):
        tensor = torch.full(shape, state, dtype=torch.float32)
        point = {"schema": 1, "mechanistic": {"schema": 2}, "evaluation": state}
        _export(response, dict.fromkeys(names, tensor), point)
        assert point["export_status"] == "completed"
        meta, package = read_export(point["export_id"])
        assert meta["export_budget_scope"] == "per_state"
        assert meta["export_tensor_bytes"] == 28_000_000
        assert meta["evaluation"] == state
        with safe_open(package, framework="pt") as opened:
            assert set(opened.keys()) == set(names)
            assert opened.get_slice("even_native_delta")[0, 0, 0, 0, 0].item() == state
            assert opened.get_slice("baseline_clean")[0, 0, 0, 0, 0].item() == state + 1
    assert response.exports == 84_000_000 > MAX_EXPORT_BYTES

    oversized = torch.zeros(1).expand(MAX_EXPORT_BYTES // 4 + 1)
    _export(response, {"frozen_native_input": oversized}, point := {})
    assert point["export_status"] == "export_budget"


def test_legacy_exports_keep_sampler_budget(tmp_path, monkeypatch):
    import folder_paths

    from mode_explorer.response import ResponseSettings
    from mode_explorer import response_runtime

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(response_runtime, "MAX_EXPORT_BYTES", 32)
    response = response_runtime.ResponseRun(ResponseSettings(states=2, export=True), lambda tensor: tensor)
    tensors = {"baseline_clean": torch.zeros(6)}
    response_runtime._export(response, tensors, first := {})
    response_runtime._export(response, tensors, second := {})
    assert first["export_status"] == "completed"
    assert second["export_status"] == "export_budget"


def test_cuda_policy_does_not_fall_back_to_cpu(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    assert ModeExplorerMechanistic.INPUT_TYPES()["optional"]["analysis_device"][1]["default"] == "cuda"
    assert torch.equal(draw(model), draw(patched(model, analysis_device="cuda")))
    measured = point(caplog)
    assert measured["status"] == "skipped"
    assert measured["reason"] == "mechanistic_cuda_requires_cuda_inference"
    assert measured["auxiliary_dit_calls"] == 0
    assert measured["mechanistic"]["analysis_device"] is None



def test_capture_allocation_failure_keeps_prediction_and_releases_hooks(model, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model)
    spec = patched(model, sites="projection_input,clean")
    stage = Workspace.stage
    failed = False

    def fail_once(self, value, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise torch.OutOfMemoryError("capture allocation")
        return stage(self, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Workspace, "stage", fail_once)
        assert torch.equal(native, draw(spec))
    assert failed
    measured = point(caplog)
    assert measured["status"] == "skipped"
    assert measured["reason"] == "mechanistic_out_of_memory"
    assert measured["auxiliary_dit_calls"] == 1
    assert torch.equal(native, draw(spec))
    assert point(caplog)["status"] == "completed"


def test_packed_projection_replay_and_downstream_isolation(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, reference=True)
    rng = torch.random.get_rng_state().clone()
    configuration = dict(sites="", projection_study="observe", projection_cast_check=True, amplitude_check=True)
    assert torch.equal(native, draw(patched(model, **configuration), reference=True))
    assert torch.equal(rng, torch.random.get_rng_state())
    p = point(caplog)
    study = p["mechanistic"]["projection"]
    assert study["status"] == "qualified", study
    assert study["local_projection_calls"] == 5
    assert study["geometry"]["D00"]["input"]["shape"][1] > 4 * 6
    assert all(row["packed_bit_exact"] for row in study["native_replay"].values())
    assert study["effective_operator"]["weight_representation"] == "effective_dense_cast_operand"
    assert "half_even" in study["components"]
    assert study["components"]["even"]["linear"]["displacement_l2"] > 0
    assert "output_cast_only" in study["components"]["even"]
    experiment = patched(model, sites="", projection_study="executed", projection_effect="even",
                         component="complete", auxiliary_budget=6)
    assert torch.equal(native, draw(experiment, reference=True))
    p = point(caplog)
    study = p["mechanistic"]["projection"]
    assert study["status"] == "qualified", study
    assert p["auxiliary_dit_calls"] == 6
    assert study["downstream_clean_rms"] > 0
    assert not p["applied"]


def test_projection_factorial_keeps_distinct_packed_text_geometry(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    positive, _ = conditions(reference=True)
    alt = [[positive[0][0][:, :3] * -.75, positive[0][1]]]
    config = patched(model, experiment="conditioning_fixed", alternate=alt, sites="",
                     projection_study="observe", auxiliary_budget=6)
    assert torch.equal(draw(model, reference=True), draw(config, reference=True))
    p = point(caplog)
    study = p["mechanistic"]["projection"]
    assert study["status"] == "qualified", study
    assert study["local_projection_calls"] == 4
    assert study["geometry"]["D00"]["input"]["shape"][1] - study["geometry"]["D10"]["input"]["shape"][1] == 2
    metrics = study["components"]["conditioning_interaction"]
    assert metrics["executed"]["displacement_l2"] > 0
    assert metrics["residual"]["displacement_l2"] < metrics["linear"]["displacement_l2"]



def test_projection_diagnostic_oom_keeps_native_prediction(model, caplog, monkeypatch):
    import mode_explorer.mechanism_runtime as runtime

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model)
    candidate = patched(model, sites="", component="complete", treatment="minimum_change",
                        projection_study="linear", auxiliary_budget=6)

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("effective weight allocation")

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "study_projection", fail)
        assert torch.equal(native, draw(candidate))
    assert point(caplog)["reason"] == "mechanistic_out_of_memory"
    assert not point(caplog).get("applied", False)


@pytest.mark.parametrize("component", ["observe", "contrast", "contrast_reverse", "contrast_spatial"])
def test_conditioning_contrast_matched_native_state(model, caplog, monkeypatch, tmp_path, component):
    import json
    import folder_paths
    from safetensors.torch import load_file

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    positive, _ = conditions()
    alt = [[positive[0][0][:, :3] * -.75, {}]]
    cfg = patched(model, experiment="conditioning_contrast", alternate=alt, sites="clean",
                  treatment="export_only" if component == "observe" else "minimum_change",
                  component=component, candidate_gain=1., candidate_target_rms=.0001,
                  auxiliary_budget=1, export=True)
    native = draw(model)
    rng = torch.random.get_rng_state().clone()
    result = draw(cfg)
    assert torch.equal(rng, torch.random.get_rng_state())
    m = point(caplog)
    assert m["status"] == "completed", m
    assert m["auxiliary_dit_calls"] == 1
    assert m["mechanistic"]["feature_reference_calls"] == 0
    assert m["mechanistic"]["baseline_reused"]
    assert m["mechanistic"]["contrast_rms"] > 0
    assert m["export_status"] == "completed"
    assert m["applied"] == (component != "observe")
    assert torch.equal(result, native) == (component == "observe")
    data = load_file(tmp_path / "modeexplorer_response" / m["export_id"] / "predictions.safetensors")
    if component == "observe":
        assert "candidate_clean" not in data
    else:
        delta = data["baseline_clean"] - data["D10_clean"]
        if component == "contrast_reverse":
            delta = -delta
        if component == "contrast_spatial":
            delta = torch.roll(delta, shifts=(delta.shape[-2] // 2, delta.shape[-1] // 2), dims=(-2, -1))
        actual = data["applied_clean"] - data["baseline_clean"]
        assert torch.nn.functional.cosine_similarity(actual.flatten(), delta.flatten(), dim=0) > .999
        assert m["applied_clean_displacement_rms"] == pytest.approx(.0001, rel=.1)


def test_conditioning_contrast_fail_closed(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    positive, _ = conditions()
    options = dict(experiment="conditioning_contrast", alternate=positive, sites="clean",
                   component="contrast", treatment="minimum_change", auxiliary_budget=1)
    native = draw(model)
    assert torch.equal(draw(patched(model, **options)), native)
    assert not point(caplog)["applied"]
    assert torch.equal(draw(patched(model, **{**options, "auxiliary_budget": 0})), native)
    p = point(caplog)
    assert p["reason"] == "auxiliary_forward_budget" and p["auxiliary_dit_calls"] == 0
    cfg = patched(model, **options)
    assert torch.equal(draw(cfg, cfg=2), draw(model, cfg=2))
    assert torch.equal(draw(cfg, stage="highres"), native)
