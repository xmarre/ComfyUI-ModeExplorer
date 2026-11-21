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
                                                     "evaluation": 1, "export": False, **changes})[0]


def test_signed_uses_exact_matched_image_sites_and_preserves_native_cfg(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, cfg=2, reference=True, split_cfg=True)
    rng = torch.random.get_rng_state().clone()
    proxied = patched(model, sites="attention_pre,attention_post,gate_logits,sigmoid_gate,projection_input,projection_output,"
                                 "attention_contribution,after_attention,mlp_input,mlp_output,mlp_contribution,"
                                 "after_mlp,final_projection,velocity,clean", amplitude_check=True, repeat_baseline=True)
    assert torch.equal(native, draw(proxied, cfg=2, reference=True, split_cfg=True))
    assert torch.equal(rng, torch.random.get_rng_state())
    m = point(caplog)
    assert m["status"] == "completed", m
    exp = m["mechanistic"]
    assert m["auxiliary_dit_calls"] == m["auxiliary_dit_rows"] == 8
    assert exp["repeat_clean_rms"] == 0
    assert exp["sites"]["b1.attention_pre"]["comparisons"]["positive"]["displacement_rms"] == 0
    assert exp["sites"]["b1.attention_post"]["comparisons"]["positive"]["displacement_rms"] > 0
    assert exp["sites"]["b0.mlp_output"]["comparisons"]["positive"]["displacement_rms"] == 0
    assert exp["sites"]["b2.after_mlp"]["comparisons"]["positive"]["displacement_rms"] > 0
    assert exp["sites"]["b1.sigmoid_gate"]["status"] == "unsupported"
    assert exp["sites"]["b1.attention_contribution"]["status"] == "unsupported"
    assert exp["sites"]["b1.projection_input"]["comparisons"]["positive"]["shape_chw"] == [64, 4, 6]
    assert exp["sites"]["final_projection"]["comparisons"]["positive"]["shape_chw"] == [64, 4, 6]
    assert exp["sites"]["clean"]["comparisons"]["positive"]["shape_chw"] == [16, 7, 11]
    assert exp["signed_realized_edit"]["status"] == "matched"
    assert exp["amplitude"]["status"] == "measured"


@pytest.mark.parametrize("mode,calls", [("joint_fixed", 7), ("joint_recomputed", 9)])
def test_conditioning_different_text_lengths_and_same_reference(model, caplog, mode, calls):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    positive, _ = conditions(reference=True)
    alt = [[positive[0][0][:, :3] * -.75, {**positive[0][1], "pooled_output": None,
                                           "attention_mask": torch.tensor([[1, 1, 0]])}]]
    m = patched(model, experiment=mode, alternate=alt, sites="attention_post,clean")
    assert torch.equal(draw(model, reference=True), draw(m, reference=True))
    p = point(caplog)
    assert p["status"] == "completed", p
    assert p["auxiliary_dit_calls"] == calls
    e = p["mechanistic"]
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
    cfg = patched(model, evaluation=-1, states=3, sites="clean", export=True)
    draw(cfg)
    first, second, third = traces(caplog)[-1]["response"]["points"]
    assert all(p["export_status"] == "completed" for p in (first, second, third))
    assert [p["mechanistic"]["first_estimate_reused_for"] for p in (first, second, third)] == [[], ["C0"], ["C0"]]
    assert [p["mechanistic"]["feature_reference_calls"] for p in (first, second, third)] == [1, 1, 1]
    export_id = second["export_id"]
    manifest = loads((tmp_path / "modeexplorer_response" / export_id / "manifest.json").read_text())
    assert manifest["native_input_fp32_sha256"] == second["native_input_fp32_sha256"]
    export_path, exported, study = load(tmp_path / "modeexplorer_response" / export_id)
    assert compare((exported, study), (exported, study))["status"] == "matched_inputs"
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
