"""Tiny Krea 2 ModelPatcher and Spectrum interoperability; not an image quality test."""
import json
import logging

import pytest
import torch

from test_comfy_integration import model, draw
from mode_explorer.nodes import ModeExplorer
from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
from comfyui_spectrum_krea2.runtime import CURRENT


SIGMAS = torch.tensor([1.0, 0.96, 0.92, 0.85, 0.75, 0.65,
                       0.55, 0.45, 0.35, 0.25, 0.15, 0.0])


def spectrum(model):
    return SpectrumApplyKrea2().apply(
        model, enabled=True, preset="turbo", degree=1,
        warmup=2, tail=1, window=1, bootstrap=True, debug=True,
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
