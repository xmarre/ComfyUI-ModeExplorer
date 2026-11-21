"""Tiny Krea 2 ModelPatcher and Spectrum interoperability; not an image quality test."""
import json
import logging
import os

import pytest
import torch

pytest.importorskip("comfyui_spectrum_krea2")

from test_comfy_integration import model, draw
from mode_explorer.nodes import ModeExplorer, ModeExplorerResponse
from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
from comfyui_spectrum_krea2.runtime import CURRENT


SIGMAS = torch.tensor([1.0, 0.96, 0.92, 0.85, 0.75, 0.65,
                       0.55, 0.45, 0.35, 0.25, 0.15, 0.0])


def spectrum(model, warmup=2):
    return SpectrumApplyKrea2().apply(
        model, enabled=True, preset="turbo", degree=1,
        warmup=warmup, tail=1, window=1, bootstrap=True, debug=True,
    )[0]


def logs(caplog, marker):
    return [json.loads(r.message.split(marker, 1)[1])
            for r in caplog.records if marker in r.message]


@pytest.mark.parametrize("feature,initialization", [
    ("feature_cosine", False), ("native", True), ("feature_volume", True),
])
def test_joint_sampling_isolated_refs_native_boundary_and_forecasts(caplog, model, feature, initialization):
    caplog.set_level(logging.INFO)
    baseline = spectrum(model)
    guided, = ModeExplorer().apply(
        model, feature=feature, initialization=initialization,
        adaptive=False, diagnostics=True)
    joint = spectrum(guided)
    base_image = draw(baseline, sigmas=SIGMAS)
    on_image = draw(joint, sigmas=SIGMAS)
    assert torch.isfinite(on_image).all()
    if feature != "native":
        assert not torch.equal(on_image, base_image)
    assert torch.equal(on_image, draw(joint, sigmas=SIGMAS))
    reports = logs(caplog, "Spectrum Krea 2 summary: ")
    assert reports
    report = reports[-1]
    assert report["fallback"] is None, report
    assert report["forecast"] > 0, report
    assert report["actual"] + report["forecast"] == len(SIGMAS) - 1, report
    assert "native final-layer capture unavailable" not in report["actual_reasons"], report
    assert "repeated branch evaluation at one outer step" not in report["actual_reasons"], report
    if feature != "native":
        assert report["mode_explorer_native_steps"] == 2, report
        assert "ModeExplorer early-feature regime transition" in report["actual_reasons"], report
    traces = logs(caplog, "MODEEXPLORER_TRACE ")
    assert traces
    trace = traces[-1]
    assert trace["completed"]
    assert trace["initialization_guider_calls"] == (4 if initialization else 0)
    assert trace["feature_reference_dit_calls"] == (2 if feature != "native" else 0)
    assert trace["spectrum_forecast_calls"] > 0
    assert CURRENT.get() is None


def test_off_path_matches_spectrum_without_modeexplorer(model):
    baseline = spectrum(model)
    with pytest.raises(RuntimeError, match="before Spectrum"):
        ModeExplorer().apply(baseline)
    bypass, = ModeExplorer().apply(model, enabled=False)
    assert bypass is model
    assert torch.equal(draw(baseline, sigmas=SIGMAS),
                       draw(spectrum(bypass), sigmas=SIGMAS))


def test_highres_stage_preserves_spectrum_native_model(model):
    baseline = spectrum(model)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True)
    joint = spectrum(guided)
    assert torch.equal(draw(baseline, sigmas=SIGMAS, stage="highres"),
                       draw(joint, sigmas=SIGMAS, stage="highres"))


def test_full_spectrum_keeps_native_forecast_plan_and_applies_guidance(caplog, model):
    """No intervention-driven reset; early native steps retain feature guidance."""
    caplog.set_level(logging.INFO)
    baseline = spectrum(model)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True,
                                   full_spectrum=True)
    full = spectrum(guided)
    baseline_image = draw(baseline, sigmas=SIGMAS)
    off = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    full_image = draw(full, sigmas=SIGMAS)
    on = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]

    assert off["mode_explorer_spectrum_policy"] == "spectrum_only"
    assert on["mode_explorer_spectrum_policy"] == "full"
    assert on["actual"] == off["actual"], (off, on)
    assert on["forecast"] == off["forecast"], (off, on)
    assert on["mode_explorer_native_steps"] == 0
    assert "ModeExplorer early-feature regime transition" not in on["actual_reasons"]
    assert on["fallback"] is None
    assert trace["spectrum_policy"] == "full"
    assert trace["feature_reference_dit_calls"] == 2
    assert trace["feature_attempts"] == 2
    points = [p for p in trace["points"] if "shift_rms" in p]
    assert len(points) == 2
    assert all(p["shift_rms"][0] > 0 for p in points)
    assert not torch.equal(full_image, baseline_image)
    assert torch.equal(full_image, draw(full, sigmas=SIGMAS))
    assert CURRENT.get() is None


def test_default_remains_guarded_on_same_spectrum_configuration(caplog, model):
    caplog.set_level(logging.INFO)
    guarded, = ModeExplorer().apply(model, adaptive=False, diagnostics=True)
    result = draw(spectrum(guarded), sigmas=SIGMAS)
    assert torch.isfinite(result).all()
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert report["mode_explorer_spectrum_policy"] == "guarded"
    assert report["mode_explorer_native_steps"] == 2
    assert "ModeExplorer early-feature regime transition" in report["actual_reasons"]
    assert trace["spectrum_policy"] == "guarded"


def test_full_spectrum_does_not_spend_references_on_bootstrap(caplog, model):
    caplog.set_level(logging.INFO)
    options = dict(adaptive=False, diagnostics=True, full_spectrum=True)
    baseline = spectrum(model, warmup=1)
    sham, = ModeExplorer().apply(model, sham=True, **options)
    guided, = ModeExplorer().apply(model, **options)
    baseline_result = draw(baseline, sigmas=SIGMAS)
    sham_result = draw(spectrum(sham, warmup=1), sigmas=SIGMAS)
    assert torch.equal(baseline_result, sham_result)
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    assert any(d["step"] == 1 and d["route"] == "bootstrap" for d in report["denoiser_decisions"])
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert any(d["route"] == "forecast" and d["reference_calls"] == 0 for d in trace["decisions"])
    changed = draw(spectrum(guided, warmup=1), sigmas=SIGMAS)
    assert not torch.equal(changed, sham_result)
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert trace["feature_attempts"] == trace["feature_reference_dit_calls"] == 2
    assert all(d["reference_calls"] == 0 for d in trace["decisions"] if d["route"] == "forecast")
    assert CURRENT.get() is None


@pytest.mark.parametrize("full", [True, False])
def test_er_sde_interop_stage_and_seed_reproducibility(caplog, model, full):
    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True, full_spectrum=full,
                                   initialization=True, reference_continuation="rolling_euler")
    joint = spectrum(guided)
    options = dict(sigmas=SIGMAS, sampler="er_sde", stage="planner")
    result = draw(joint, **options)
    assert torch.equal(result, draw(joint, **options))
    assert not torch.equal(result, draw(spectrum(model), **options))
    options["stage"] = "highres"
    assert torch.equal(draw(joint, **options), draw(spectrum(model), **options))
    assert CURRENT.get() is None


@pytest.mark.skipif(not os.environ.get("MODEEXPLORER_SCALE_LOCKED"), reason="Scale-Locked is optional")
def test_real_scale_locked_regenerate_stage_isolation(caplog, model):
    import asyncio

    import nodes
    from test_comfy_integration import conditions

    assert asyncio.run(nodes.load_custom_node(os.environ["MODEEXPLORER_SCALE_LOCKED"]))
    node = nodes.NODE_CLASS_MAPPINGS["Krea2TurboFlowAlignedRegenerate"]()
    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(model, diagnostics=True, adaptive=False)
    guided, = ModeExplorerResponse().apply(guided, min_sigma=0.9, max_sigma=1.0,
                                             directions=1, calibration="off")
    p, n = conditions()
    latent = {"samples": torch.zeros(1, 16, 1, 80, 80)}
    with torch.inference_mode():
        output, planner, _ = node.sample(spectrum(guided), p, n, latent, 902, 0.25, 0.0, 0.0,
                                        steps=6, composition_lock_sigma=0.60, sampler_name="er_sde")
    assert output["samples"].shape[-2:] == (80, 80)
    assert planner["samples"].shape[-1] < output["samples"].shape[-1]
    assert torch.isfinite(output["samples"]).all()
    traces = logs(caplog, "MODEEXPLORER_TRACE ")
    assert len(traces) == 1 and traces[0]["stage"] == "planner" and traces[0]["feature_attempts"] > 0
    point, = traces[0]["response"]["points"]
    assert point["stage"] == "planner" and point["status"] == "completed"
    reports = logs(caplog, "Spectrum Krea 2 summary: ")
    assert {item["stage"] for item in reports} == {"planner", "highres"}
    assert CURRENT.get() is None


def test_full_trace_indicates_cross_regime_feature_anchors(caplog, model):
    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True, min_sigma=0.97,
                                   full_spectrum=True)
    guarded, = ModeExplorer().apply(model, adaptive=False, diagnostics=True, min_sigma=0.97)
    full = draw(spectrum(guided, warmup=1), sigmas=SIGMAS)
    full_report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    changed = draw(spectrum(guarded, warmup=1), sigmas=SIGMAS)
    guarded_report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    assert not torch.equal(full, changed)
    pair = next(d for d in full_report["denoiser_decisions"] if d["step"] == 1)
    assert pair["route"] == "bootstrap" and pair["feature_anchor_steps"] == [0]
    assert not any(d.get("feature_anchor_steps") for d in guarded_report["denoiser_decisions"])


def test_sol_spectrum_and_modeexplorer_share_one_pipeline(caplog, model):
    sol = pytest.importorskip("sol_h3.krea2")
    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True, initialization=True)
    joint = spectrum(sol.install(guided, sol.Settings(), verbose=True))
    result = draw(joint, sigmas=SIGMAS)
    assert torch.isfinite(result).all()
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert report["fallback"] is None and report["forecast"] > 0
    assert trace["feature_reference_dit_calls"] == 2 and trace["initialization_dit_calls"] >= 4
    assert any(f"Sol-Krea2: model_evaluations={report['actual']} " in record.message for record in caplog.records)
    assert CURRENT.get() is None and sol._ACTIVE.get() is None


@pytest.mark.parametrize("full", [False, True], ids=["guarded", "full"])
def test_spectrum_rejected_outer_wrapper_keeps_native_guidance(caplog, model, full):
    from comfy.patcher_extension import WrappersMP

    def observe(executor, *args, **kwargs):
        return executor(*args, **kwargs)

    caplog.set_level(logging.INFO)
    wrapped = model.clone()
    wrapped.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "test_outer_observer", observe)
    guided, = ModeExplorer().apply(wrapped, adaptive=False, diagnostics=True, full_spectrum=full)
    native = draw(wrapped, sigmas=SIGMAS)
    expected = draw(guided, sigmas=SIGMAS)
    assert not torch.equal(native, expected)
    assert logs(caplog, "MODEEXPLORER_TRACE ")[-1]["feature_reference_dit_calls"] == 2

    actual = draw(spectrum(guided), sigmas=SIGMAS)
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert "external transformer patches or wrappers are active" in report["fallback"]
    assert report["forecast"] == 0 and report["actual"] == len(SIGMAS) - 1
    assert trace["feature_reference_dit_calls"] == trace["feature_attempts"] == 2
    assert "reference_prepare_unavailable" not in trace["reasons"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert CURRENT.get() is None


def test_live_spectrum_mismatch_keeps_native_guidance(caplog, model, monkeypatch):
    from comfyui_spectrum_krea2 import adapter

    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(model, adaptive=False, diagnostics=True, full_spectrum=True)
    expected = draw(guided, sigmas=SIGMAS)
    # Initial patch qualification succeeds, but the live adapter rejects dispatch.
    monkeypatch.setattr(adapter, "matches", lambda policy, options: False)
    actual = draw(spectrum(guided), sigmas=SIGMAS)
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert report["mode_explorer_spectrum_policy"] == "full"
    assert "external transformer modifications are active" in report["actual_reasons"]
    assert report["forecast"] == 0 and report["actual"] == len(SIGMAS) - 1
    assert trace["feature_reference_dit_calls"] == trace["feature_attempts"] == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kj_loader_sage_override_preserves_guidance_and_spectrum(caplog, model, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from types import ModuleType

    kj_root = os.environ.get("MODEEXPLORER_KJNODES")
    if not kj_root:
        pytest.skip("KJNodes checkout is optional")
    path = Path(kj_root) / "nodes/model_optimization_nodes.py"
    spec = importlib.util.spec_from_file_location("test_kj.nodes.model_optimization_nodes", path)
    kj = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, kj)
    spec.loader.exec_module(kj)

    calls = []

    def sageattn(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
        calls.append(tensor_layout)
        if tensor_layout == "NHD":
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, is_causal=is_causal)
        return out.transpose(1, 2) if tensor_layout == "NHD" else out

    backend = ModuleType("sageattention")
    backend.sageattn = sageattn
    monkeypatch.setitem(sys.modules, "sageattention", backend)
    monkeypatch.setattr(kj.folder_paths, "get_full_path_or_raise", lambda *args: str(path))
    monkeypatch.setattr(kj, "_load_diffusion_model_kj", lambda *a, **kw: model.clone())
    monkeypatch.setattr(kj, "args", type("Args", (), {"fast": set()})())
    loaded, = kj.DiffusionModelLoaderKJ().patch_and_load("tiny", "default", "default", False, "auto", False)
    override = loaded.model_options["transformer_options"]["optimized_attention_override"]

    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(loaded, adaptive=False, diagnostics=True)
    assert guided.model_options["transformer_options"]["optimized_attention_override"] is override
    assert not torch.equal(draw(loaded, sigmas=SIGMAS), draw(guided, sigmas=SIGMAS))
    assert logs(caplog, "MODEEXPLORER_TRACE ")[-1]["feature_reference_dit_calls"] == 2
    calls.clear()

    result = draw(spectrum(guided), sigmas=SIGMAS)
    assert torch.isfinite(result).all() and calls
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert report["fallback"] is None and report["forecast"] > 0
    assert trace["feature_reference_dit_calls"] == trace["feature_attempts"] == 2

    probe, = ModeExplorerResponse().apply(loaded, mode="feature_replay", feature_block=1)
    assert torch.equal(draw(loaded, sigmas=SIGMAS), draw(probe, sigmas=SIGMAS))
    assert logs(caplog, "MODEEXPLORER_TRACE ")[-1]["response"]["points"][0]["feature"][0]["status"] == "completed"

    foreign = loaded.clone()

    def unrelated(func, *args, **kwargs):
        return func(*args, **kwargs)

    unrelated.__name__ = override.__name__
    foreign.model_options["transformer_options"]["optimized_attention_override"] = unrelated
    with pytest.raises(RuntimeError, match="unrecognized optimized_attention_override"):
        ModeExplorer().apply(foreign)


@pytest.mark.parametrize("backend", ["pytorch attention", "comfy kitchen attention"])
def test_core_attention_selector_preserves_guidance_and_spectrum(caplog, model, monkeypatch, backend):
    from comfy.ldm.modules import attention
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend

    calls = []

    def prequantize(q, k, v, *, scale, attn_mask):
        return q, k, v, scale, attn_mask

    def from_prequantized(args):
        q, k, v, scale, mask = args
        calls.append(True)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    monkeypatch.setitem(attention.REGISTERED_ATTENTION_FUNCTIONS, "comfy_kitchen_int8",
                        attention.attention_comfy_kitchen_int8)
    monkeypatch.setattr(attention.comfy_kitchen, "prequantize_int8_attention", prequantize)
    monkeypatch.setattr(attention.comfy_kitchen, "int8_attention_from_prequantized", from_prequantized)

    patched, = ModelAttentionBackend.execute(model, backend).result
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(patched, adaptive=False, diagnostics=True)
    assert guided.model_options["transformer_options"]["optimized_attention_override"] is override
    native = draw(patched, sigmas=SIGMAS)
    assert not torch.equal(native, draw(guided, sigmas=SIGMAS))
    calls.clear()

    result = draw(spectrum(guided), sigmas=SIGMAS)
    assert torch.isfinite(result).all()
    assert bool(calls) == (backend == "comfy kitchen attention")
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    trace = logs(caplog, "MODEEXPLORER_TRACE ")[-1]
    assert report["fallback"] is None and report["forecast"] > 0
    assert trace["feature_reference_dit_calls"] == trace["feature_attempts"] == 2

    probe, = ModeExplorerResponse().apply(patched, mode="feature_replay", feature_block=1)
    assert torch.equal(native, draw(probe, sigmas=SIGMAS))
    arm = logs(caplog, "MODEEXPLORER_TRACE ")[-1]["response"]["points"][0]["feature"][0]
    assert arm["status"] == "completed" and arm["realized_edit_l2"] > 0


def test_mechanistic_preserves_passive_forecasts_and_forces_active_native(caplog, model):
    from mode_explorer.nodes import ModeExplorerMechanistic

    caplog.set_level(logging.INFO)
    base = draw(spectrum(model), sigmas=SIGMAS)
    node = ModeExplorerMechanistic()
    options = dict(feature_block=1, blocks="1", sites="projection_input,clean", evaluation=1, analysis_device="cpu")
    passive, = node.apply(model, **options)
    assert torch.equal(base, draw(spectrum(passive), sigmas=SIGMAS))
    passive_report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    passive_point, = logs(caplog, "MODEEXPLORER_TRACE ")[-1]["response"]["points"]
    assert passive_report["forecast"] > 0
    assert passive_point["status"] == "completed"
    active, = node.apply(model, treatment="minimum_change", component="complete", **options)
    draw(spectrum(active), sigmas=SIGMAS)
    active_report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    active_point, = logs(caplog, "MODEEXPLORER_TRACE ")[-1]["response"]["points"]
    assert active_report["forecast"] == 0
    assert active_point["applied"]
    assert CURRENT.get() is None


@pytest.mark.skipif(not os.environ.get("MODEEXPLORER_SCALE_LOCKED"), reason="Scale-Locked is optional")
def test_mechanistic_scale_locked_uses_original_planner_and_zero_target_evaluations(caplog, model):
    import asyncio

    import nodes
    from mode_explorer.nodes import ModeExplorerMechanistic
    from test_comfy_integration import conditions

    assert asyncio.run(nodes.load_custom_node(os.environ["MODEEXPLORER_SCALE_LOCKED"]))
    node = nodes.NODE_CLASS_MAPPINGS["Krea2TurboFlowAlignedRegenerate"]()
    caplog.set_level(logging.INFO)
    p, n = conditions()
    latent = {"samples": torch.zeros(1, 16, 1, 80, 80)}
    patched, = ModeExplorerMechanistic().apply(model, feature_block=1, blocks="1", sites="clean", evaluation=0,
                                                analysis_device="cpu")
    options = dict(steps=6, composition_lock_sigma=.60, sampler_name="er_sde")
    with torch.inference_mode():
        _, baseline_planner, _ = node.sample(spectrum(model), p, n, latent, 902, .25, 0., 0., **options)
        caplog.clear()
        _, planner, _ = node.sample(spectrum(patched), p, n, latent, 902, .25, 0., 0., **options,
                                   regeneration_mode="planner_final_upscale_only", handoff_diagnostics="off")
    assert torch.equal(planner["samples"], baseline_planner["samples"])
    assert "target denoiser evaluations=0" in caplog.text
    assert {report["stage"] for report in logs(caplog, "Spectrum Krea 2 summary: ")} == {"planner"}
    point, = logs(caplog, "MODEEXPLORER_TRACE ")[-1]["response"]["points"]
    assert point["stage"] == "planner" and point["status"] == "completed"
