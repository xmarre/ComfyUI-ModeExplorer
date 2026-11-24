"""ReDiverse with Spectrum and Scale-Locked on the tiny native transformer (routing, not quality)."""

import json
import logging
import os

import pytest
import torch

pytest.importorskip("comfyui_spectrum_krea2")

from test_comfy_integration import conditions, draw, model, traces  # noqa: F401
from test_rediverse_integration import rediverse, workspace  # noqa: F401
from test_spectrum_integration import SIGMAS, logs, spectrum
from comfyui_spectrum_krea2.runtime import CURRENT
from mode_explorer.nodes import ModeExplorer


def test_guarded_spectrum_keeps_adapted_step_actual_and_starts_fresh_history(caplog, model, workspace):
    caplog.set_level(logging.INFO)
    joint = spectrum(rediverse(model, diagnostics=True), warmup=1)
    result = draw(joint, sigmas=SIGMAS)
    assert torch.isfinite(result).all()
    report, = logs(caplog, "Spectrum Krea 2 summary: ")
    assert report["fallback"] is None, report
    routes = {item["step"]: item["route"] for item in report["denoiser_decisions"]
              if item["route"] != "forecast_rejected"}
    assert routes[0] == "actual" and routes[1] == "actual"
    assert report["actual_reasons"]["ModeExplorer early-feature regime transition"] == 1
    assert report["forecast"] > 0 and report["actual"] + report["forecast"] == len(SIGMAS) - 1
    summary = traces(caplog)[-1]["rediverse"]
    assert summary["adapted_evaluations"] == 1 and summary["forecast_consumed"] == 0
    assert "spectrum_all_actual" not in summary["reasons"]
    assert torch.equal(result, draw(joint, sigmas=SIGMAS))
    assert CURRENT.get() is None


def test_spectrum_without_deferred_contract_runs_all_actual(caplog, model, workspace, monkeypatch):
    import comfyui_spectrum_krea2.adapter as adapter

    monkeypatch.setattr(adapter, "MODE_EXPLORER_DEFERRED_PROBES", False)
    caplog.set_level(logging.INFO)
    draw(spectrum(rediverse(model, diagnostics=True), warmup=1), sigmas=SIGMAS)
    report, = logs(caplog, "Spectrum Krea 2 summary: ")
    assert report["forecast"] == 0
    summary = traces(caplog)[-1]["rediverse"]
    assert "spectrum_all_actual" in summary["reasons"] and summary["adapted_evaluations"] == 1
    assert CURRENT.get() is None


def test_combined_b1_and_ft_with_guarded_spectrum(caplog, model, workspace):
    caplog.set_level(logging.INFO)
    guided, = ModeExplorer().apply(rediverse(model), mode="Early Structural (B1)", strength=.03, fidelity=.7,
                                   diagnostics=True)
    draw(spectrum(guided, warmup=1), sigmas=SIGMAS)
    report, = logs(caplog, "Spectrum Krea 2 summary: ")
    trace = traces(caplog)[-1]
    assert report["fallback"] is None and report["forecast"] > 0
    assert trace["rediverse"]["adapted_evaluations"] == 1 and trace["feature_reference_dit_calls"] >= 1
    assert trace["spectrum_forecast_calls"] == report["forecast"]
    with pytest.raises(RuntimeError, match="before Spectrum"):
        rediverse(spectrum(model))


@pytest.mark.skipif(not os.environ.get("MODEEXPLORER_SCALE_LOCKED"), reason="Scale-Locked is optional")
def test_scale_locked_planner_only(caplog, model, workspace):
    import asyncio

    import nodes

    assert asyncio.run(nodes.load_custom_node(os.environ["MODEEXPLORER_SCALE_LOCKED"]))
    node = nodes.NODE_CLASS_MAPPINGS["Krea2TurboFlowAlignedRegenerate"]()
    caplog.set_level(logging.INFO)
    p, n = conditions()
    latent = {"samples": torch.zeros(1, 16, 1, 80, 80)}
    with torch.inference_mode():
        output, planner, _ = node.sample(spectrum(rediverse(model, diagnostics=True)), p, n, latent, 902, 0.25, 0.0, 0.0,
                                         steps=6, composition_lock_sigma=0.60, sampler_name="er_sde")
    assert torch.isfinite(output["samples"]).all()
    found = traces(caplog)
    assert len(found) == 1 and found[0]["stage"] == "planner"
    assert found[0]["rediverse"]["adapted_evaluations"] == 1
    reports = logs(caplog, "Spectrum Krea 2 summary: ")
    assert {item["stage"] for item in reports} == {"planner", "highres"}
    assert CURRENT.get() is None


def _routes(report):
    return {item["step"]: item["route"] for item in report["denoiser_decisions"] if item["route"] != "forecast_rejected"}


@pytest.mark.parametrize("profile", ["constant", "zero_inside", "sham"])
def test_delayed_window_routes_and_history_boundaries(caplog, model, workspace, profile):
    """Forecasts before the window, all-actual inside it, a forced actual exit and fresh native history after."""
    caplog.set_level(logging.INFO)
    kwargs = dict(adapted_sigmas=3, start_evaluation=4, diagnostics=True)
    if profile == "zero_inside":
        kwargs.update(strength_profile="custom", custom_profile="1, 0, 1")
    if profile == "sham":
        kwargs.update(strength=0.0)
    # Long enough for Spectrum to rebuild history and forecast again after the window.
    sigmas = torch.linspace(1.0, 0.0, 21)
    draw(spectrum(rediverse(model, **kwargs), warmup=1), sigmas=sigmas)
    report, = logs(caplog, "Spectrum Krea 2 summary: ")
    routes = _routes(report)
    assert report["fallback"] is None, report
    assert any(routes[step] != "actual" for step in (1, 2, 3)), routes  # ordinary forecasting before the window
    assert [routes[step] for step in (4, 5, 6, 7)] == ["actual"] * 4, routes  # window and its forced exit
    assert any(routes[step] != "actual" for step in range(8, 20)), routes  # resumed with fresh native history
    assert report["actual_reasons"]["ModeExplorer early-feature regime transition"] == 2  # entry and exit
    temporal = traces(caplog)[-1]["rediverse"]["temporal"]
    assert [s["evaluation"] for s in temporal["slots"]] == [4, 5, 6]
    assert all(s["live_forwards"] == 1 for s in temporal["slots"])  # no forecast consumed a slot
    assert temporal["slots"][0]["sigma"] == pytest.approx(float(sigmas[4]))  # forecasts did not shift the window
    expected = {"constant": [1.0, 1.0, 1.0], "zero_inside": [1.0, 0.0, 1.0], "sham": [0.0, 0.0, 0.0]}[profile]
    assert [s["effective_strength"] for s in temporal["slots"]] == expected
    assert CURRENT.get() is None


def test_delayed_sham_matches_treated_routes(caplog, model, workspace):
    caplog.set_level(logging.INFO)
    reports = []
    for strength in (1.0, 0.0):
        caplog.clear()
        draw(spectrum(rediverse(model, adapted_sigmas=2, start_evaluation=3, strength=strength), warmup=1), sigmas=SIGMAS)
        report, = logs(caplog, "Spectrum Krea 2 summary: ")
        reports.append(_routes(report))
    assert reports[0] == reports[1]


@pytest.mark.skipif(not os.environ.get("MODEEXPLORER_SCALE_LOCKED"), reason="Scale-Locked is optional")
@pytest.mark.parametrize("start,count", [(1, 3), (4, 8)])
def test_scale_locked_temporal_window_stays_in_the_planner(caplog, model, workspace, start, count):
    import asyncio

    import nodes

    assert asyncio.run(nodes.load_custom_node(os.environ["MODEEXPLORER_SCALE_LOCKED"]))
    node = nodes.NODE_CLASS_MAPPINGS["Krea2TurboFlowAlignedRegenerate"]()
    caplog.set_level(logging.INFO)
    p, n = conditions()
    patched = rediverse(model, diagnostics=True, adapted_sigmas=count, start_evaluation=start, strength=.5)
    with torch.inference_mode():
        output, _, _ = node.sample(spectrum(patched), p, n, {"samples": torch.zeros(1, 16, 1, 80, 80)}, 902, 0.25,
                                   0.0, 0.0, steps=6, composition_lock_sigma=0.60, sampler_name="er_sde")
    assert torch.isfinite(output["samples"]).all()
    found = traces(caplog)
    assert len(found) == 1 and found[0]["stage"] == "planner"  # the high-resolution stage never runs ReDiverse
    temporal = found[0]["rediverse"]["temporal"]
    assert temporal["slots"] and temporal["slots"][0]["evaluation"] == start
    if start + count > len(temporal["evaluation_sigmas"]):
        assert "temporal_window_truncated" in found[0]["rediverse"]["reasons"]
    assert CURRENT.get() is None
