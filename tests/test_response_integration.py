import logging

import pytest
import torch

pytest.importorskip("comfy")

from test_comfy_integration import draw, model, traces
from mode_explorer.nodes import ModeExplorer, ModeExplorerResponse


SIGMAS = torch.tensor([1.0, 0.9743, 0.9456, 0.83, 0.62, 0.2, 0.0])


def configured(model, **overrides):
    return ModeExplorerResponse().apply(model, **{"feature_block": 1, **overrides})[0]


def test_probe_is_observational_and_runs_on_actual_positive_dispatch(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, sigmas=SIGMAS, cfg=2, reference=True, split_cfg=True)
    rng = torch.random.get_rng_state().clone()
    probe = configured(model)
    assert torch.equal(native, draw(probe, sigmas=SIGMAS, cfg=2, reference=True, split_cfg=True))
    assert torch.equal(rng, torch.random.get_rng_state())
    response = traces(caplog)[-1]["response"]
    point, = response["points"]
    assert point["sigma"] == pytest.approx(0.9743)
    assert point["evaluation"] == 1
    assert point["status"] == "completed"
    assert point["auxiliary_dit_calls"] == response["auxiliary_dit_calls"] == 7
    assert point["repeat_rms"] == 0
    assert len(point["projected_response"]["matrix"]) == 2
    assert not point["applied"]
    assert not model.wrappers


@pytest.mark.parametrize("direction,expected", [("sham", False), ("repel", True), ("tangent", True)])
def test_feature_replay_does_not_change_generation(model, caplog, direction, expected):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, sigmas=SIGMAS)
    patched = configured(model, mode="joint_probe", feature_direction=direction, localization="only_raw", head=0)
    assert torch.equal(native, draw(patched, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["auxiliary_dit_calls"] == 9
    arm, = point["feature"]
    assert arm["status"] == "completed"
    assert (arm["realized_edit_l2"] > 0) is expected
    assert (arm["clean_rms"] > 0) is expected
    assert "explained_energy_fraction" in arm


def test_steering_uses_bounded_candidates_and_restores_native_on_decline(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, sigmas=SIGMAS)
    patched = configured(model, mode="latent_steer")
    assert not torch.equal(native, draw(patched, sigmas=SIGMAS))
    trace, = traces(caplog)[-1]["response"]["points"]
    assert trace["applied"]
    assert 0 < trace["applied_clean_displacement_rms"] <= 0.06
    assert sum(c["reason"] == "selected" for c in trace["candidates"]) == 1
    missing = configured(model, mode="latent_steer", selection="external")
    assert torch.equal(native, draw(missing, sigmas=SIGMAS))
    assert traces(caplog)[-1]["response"]["reason"] == "score_unavailable"
    strict = configured(model, mode="latent_steer", auxiliary_budget=1)
    assert torch.equal(native, draw(strict, sigmas=SIGMAS))
    assert traces(caplog)[-1]["response"]["auxiliary_dit_calls"] == 1
    assert torch.equal(native, draw(patched, sigmas=SIGMAS, stage="highres"))


def test_probe_preserves_existing_feature_path_and_disabled_node(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    base, = ModeExplorer().apply(model, strength=0.025, fidelity=0.7, adaptive=False, diagnostics=True)
    research = configured(base, mode="feature_replay")
    disabled, = ModeExplorerResponse().apply(research, enabled=False)
    assert torch.equal(draw(research, sigmas=SIGMAS), draw(base, sigmas=SIGMAS))
    assert torch.equal(draw(disabled, sigmas=SIGMAS), draw(base, sigmas=SIGMAS))


def test_export_load_and_exact_frozen_replay(model, caplog, tmp_path, monkeypatch):
    import folder_paths
    from mode_explorer.nodes import ModeExplorerLoadResponse

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    first = configured(model, export=True)
    draw(first, sigmas=SIGMAS)
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["export_status"] == "completed"
    export_id = point["export_id"]
    latent, sigmas = ModeExplorerLoadResponse().load(export_id, "baseline_clean")
    assert latent["samples"].shape == (1, 16, 1, 7, 11)
    assert torch.equal(sigmas, SIGMAS[[0, 1, -1]])
    frozen = configured(model, frozen_state=export_id)
    assert torch.equal(draw(model, sigmas=sigmas, seed=89), draw(frozen, sigmas=sigmas, seed=89))
    new_point, = traces(caplog)[-1]["response"]["points"]
    assert new_point["native_input_sha256"] == point["native_input_sha256"]
    assert not new_point["baseline_reused"]
    draw(frozen, sigmas=sigmas, reference=True)
    assert traces(caplog)[-1]["response"]["points"][0]["reason"] == "frozen_sigma_or_conditioning_mismatch"


def test_feature_downstream_capture_and_disabled_cleanup(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    patched = configured(model, mode="feature_replay", feature_block=0, downstream=True)
    assert torch.equal(draw(patched, sigmas=SIGMAS), draw(model, sigmas=SIGMAS))
    feature, = traces(caplog)[-1]["response"]["points"][0]["feature"]
    assert feature["later_block_pooled_attention_rms"]["1"] > 0
    assert feature["later_block_pooled_attention_rms"]["2"] > 0
    disabled, = ModeExplorerResponse().apply(patched, enabled=False)
    assert not any(disabled.wrappers.values())


@pytest.mark.parametrize("all_actual", [False, True])
def test_spectrum_probe_state_and_route_parity(model, caplog, all_actual):
    adapter = pytest.importorskip("comfyui_spectrum_krea2.runtime")
    from test_spectrum_integration import logs, spectrum

    caplog.set_level(logging.INFO)
    baseline, probe = spectrum(model), spectrum(configured(model, all_actual=all_actual, min_sigma=0.9, max_sigma=1.0))
    if all_actual:
        token = adapter.FORCE_NATIVE_ALL_SAMPLERS.set(True)
        try:
            expected = draw(baseline, sigmas=SIGMAS)
        finally:
            adapter.FORCE_NATIVE_ALL_SAMPLERS.reset(token)
    else:
        expected = draw(baseline, sigmas=SIGMAS)
    off = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    assert torch.equal(expected, draw(probe, sigmas=SIGMAS))
    on = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    assert on["actual"] == off["actual"]
    assert on["forecast"] == off["forecast"]
    assert on["fallback"] is None
    if all_actual:
        assert on["forecast"] == 0
    assert traces(caplog)[-1]["response"]["auxiliary_dit_calls"] == 7
    assert adapter.CURRENT.get() is None
    assert not adapter.FORCE_NATIVE_ALL_SAMPLERS.get()


def test_feature_reference_budget_and_matching_live_anchor(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    bare = configured(model, mode="feature_replay", feature_references=2, calibration="off")
    assert torch.equal(draw(model, sigmas=SIGMAS), draw(bare, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["auxiliary_dit_calls"] == 3
    assert len(set(point["feature_reference_slots"])) == 2

    primary, = ModeExplorer().apply(model, strength=0.025, fidelity=0.70, adaptive=False)
    replay = configured(primary, mode="feature_replay", calibration="off")
    assert torch.equal(draw(primary, sigmas=SIGMAS), draw(replay, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["feature_reference_origin"] == "live_pooled_reference"
    assert point["auxiliary_dit_calls"] == 2


def test_active_gate_preserves_native_with_no_improvement_or_amplified_cfg(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    scored = configured(model, mode="latent_steer", selection="external", scorer=lambda clean, info: 4)
    assert torch.equal(draw(scored, sigmas=SIGMAS), draw(model, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["baseline_external_score"] == 4
    assert {item["reason"] for item in point["candidates"]} == {"no_valid_score_improvement"}
    assert torch.equal(draw(scored, sigmas=SIGMAS, cfg=2), draw(model, sigmas=SIGMAS, cfg=2))
    assert traces(caplog)[-1]["response"]["reason"] == "response_steering_requires_native_cfg_one"


def test_workspace_limit_precedes_forwards():
    from types import SimpleNamespace
    from mode_explorer.response import ResponseSettings
    from mode_explorer.response_runtime import ResponseRun, run_response

    def fail(*args, **kwargs):
        pytest.fail("A rejected workspace must not start a model forward")

    run = SimpleNamespace(response=ResponseRun(ResponseSettings(memory_mb=16), fail),
                          settings=SimpleNamespace(reference_source="bank"), seed=1, noise_scale=1.0, stage="planner")
    frame = SimpleNamespace(grid=(512, 512), offset=0, sigma=0.97)
    x = torch.zeros(1).expand(1, 16, 1, 1024, 1024)
    context = torch.zeros(1, 5, 64)
    args = (x, torch.tensor([0.97]), context, None, None, {})
    assert run_response(fail, run, frame, args, {}, x) is x
    assert run.response.forwards == 0
    assert run.response.points[0]["reason"] == "research_memory_budget"


def test_exception_restores_spectrum_scopes(model, monkeypatch):
    adapter = pytest.importorskip("comfyui_spectrum_krea2.runtime")
    from test_spectrum_integration import spectrum
    from mode_explorer import response_runtime

    def fail(*args, **kwargs):
        raise RuntimeError("controlled probe failure")

    patched = spectrum(configured(model, all_actual=True))
    with monkeypatch.context() as edits:
        edits.setattr(response_runtime, "direction_field", fail)
        with pytest.raises(RuntimeError, match="controlled probe failure"):
            draw(patched, sigmas=SIGMAS)
    assert adapter.CURRENT.get() is None
    assert not adapter.FORCE_NATIVE_ALL_SAMPLERS.get()
    assert torch.isfinite(draw(patched, sigmas=SIGMAS)).all()


def test_sol_response_forward_isolation(model, caplog):
    sol = pytest.importorskip("sol_h3.krea2")

    caplog.set_level(logging.INFO)
    baseline = sol.install(model, sol.Settings(), verbose=True)
    research = sol.install(configured(model, mode="feature_replay"), sol.Settings(), verbose=True)
    assert torch.equal(draw(baseline, sigmas=SIGMAS), draw(research, sigmas=SIGMAS))
    assert any("Sol-Krea2: model_evaluations=6 " in record.message for record in caplog.records)
    assert traces(caplog)[-1]["response"]["auxiliary_dit_calls"] == 3
    assert sol._ACTIVE.get() is None and sol._EXECUTION.get() is None


def test_active_spectrum_uses_all_actual_through_end_of_trajectory(model, caplog):
    adapter = pytest.importorskip("comfyui_spectrum_krea2.runtime")
    from test_spectrum_integration import logs, spectrum

    caplog.set_level(logging.INFO)
    patched = spectrum(configured(model, mode="latent_steer"))
    assert not torch.equal(draw(model, sigmas=SIGMAS), draw(patched, sigmas=SIGMAS))
    report = logs(caplog, "Spectrum Krea 2 summary: ")[-1]
    assert report["forecast"] == 0 and report["actual"] == len(SIGMAS) - 1
    assert traces(caplog)[-1]["response"]["points"][0]["applied"]
    assert not adapter.FORCE_NATIVE_ALL_SAMPLERS.get()
