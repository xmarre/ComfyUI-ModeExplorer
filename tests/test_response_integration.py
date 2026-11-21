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
    assert point["auxiliary_dit_calls"] == 10
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
    frozen = configured(model, frozen_state=export_id, export=True)
    assert torch.equal(draw(model, sigmas=sigmas, seed=89), draw(frozen, sigmas=sigmas, seed=89))
    new_point, = traces(caplog)[-1]["response"]["points"]
    assert new_point["native_input_sha256"] == point["native_input_sha256"]
    assert new_point["export_status"] == "completed"
    assert new_point["native_input_fp32_sha256"] == point["native_input_fp32_sha256"]
    assert new_point["native_condition_values"] == point["native_condition_values"]
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
    assert point["auxiliary_dit_calls"] == 5
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
    assert {item["reason"] for item in point["candidates"] if item["arm"] != "baseline"} == {"no_valid_score_improvement"}
    assert torch.equal(draw(scored, sigmas=SIGMAS, cfg=2), draw(model, sigmas=SIGMAS, cfg=2))
    assert traces(caplog)[-1]["response"]["reason"] == "response_steering_requires_native_cfg_one"


def test_workspace_widget_matches_backend_limit(model):
    from mode_explorer.response import MAX_WORKSPACE_MIB, ResponseSettings

    widget = ModeExplorerResponse.INPUT_TYPES()["required"]["memory_mb"][1]
    assert widget["default"] == ResponseSettings().memory_mb == 128
    assert widget["max"] == MAX_WORKSPACE_MIB == 1024
    assert configured(model, mode="joint_probe", memory_mb=1024) is not model
    with pytest.raises(ValueError, match="memory_mb"):
        configured(model, memory_mb=1025)


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
    response = traces(caplog)[-1]["response"]
    assert response["auxiliary_dit_calls"] == 4
    point, = response["points"]
    assert point["feature_reference_initial_sol_evaluations"] == 0
    assert point["sol_evaluations"] == 1
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


@pytest.mark.parametrize("evaluation", [1, 2])
def test_cold_reference_matches_live_after_first_evaluation(model, caplog, tmp_path, monkeypatch, evaluation):
    import folder_paths
    from mode_explorer.nodes import ModeExplorerLoadResponse

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    primary, = ModeExplorer().apply(model, strength=0.025, fidelity=0.70, adaptive=False, reference_budget=3)
    live = configured(primary, mode="feature_replay", export=True, calibration="off", evaluation=evaluation)
    draw(live, sigmas=SIGMAS)
    warm, = traces(caplog)[-1]["response"]["points"]
    assert warm["feature_reference_origin"] == "live_pooled_reference"
    assert warm["sigma"] == float(SIGMAS[evaluation])
    loader = ModeExplorerLoadResponse()
    expected, sigmas = loader.load(warm["export_id"], "feature_repel_clean")
    cold = configured(model, mode="feature_replay", export=True, calibration="off", frozen_state=warm["export_id"])
    draw(cold, sigmas=sigmas)
    reconstructed, = traces(caplog)[-1]["response"]["points"]
    assert reconstructed["feature_reference_origin"] == "reconstructed_first_estimate"
    assert reconstructed["feature_reference_initial_sigma"] == 1.0
    assert reconstructed["auxiliary_dit_calls"] == reconstructed["auxiliary_dit_rows"] == 4
    actual, _ = loader.load(reconstructed["export_id"], "feature_repel_clean")
    assert torch.equal(actual["samples"], expected["samples"])


def test_reapplying_primary_node_preserves_response_config(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    research = configured(model)
    primary, = ModeExplorer().apply(research, strength=0.025, feature_block=1)
    draw(primary, sigmas=SIGMAS)
    record = traces(caplog)[-1]
    assert record["feature_attempts"] == 2 and record["response"]["status"] == "completed"
    disabled, = ModeExplorer().apply(primary, enabled=False)
    assert torch.equal(draw(disabled, sigmas=SIGMAS), draw(model, sigmas=SIGMAS))
    record = traces(caplog)[-1]
    assert record["feature_attempts"] == 0 and record["response"]["status"] == "completed"


def test_external_scorer_sees_exact_exported_vae_baseline(model, caplog, tmp_path, monkeypatch):
    import numpy as np
    import folder_paths
    from mode_explorer.nodes import ModeExplorerLoadResponse

    seen = []
    def scorer(clean, provenance):
        assert provenance["scorer_latent_format"] == "comfy_vae_latent"
        seen.append(clean.detach().clone())
        return np.float32(1)

    caplog.set_level(logging.INFO, logger="ModeExplorer")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    patched = configured(model, mode="latent_steer", selection="external", scorer=scorer, export=True)
    assert torch.equal(draw(patched, sigmas=SIGMAS), draw(model, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    vae, _ = ModeExplorerLoadResponse().load(point["export_id"], "baseline_clean")
    assert len(seen) == 3 and torch.equal(seen[0], vae["samples"])
    assert point["candidates"][0]["arm"] == point["selected_candidate"] == "baseline"
    assert point["candidates"][0]["external_scores"] == {"quality": 1.0}


def test_packed_cfg_reports_both_rows_per_auxiliary_call(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, cfg=2, sigmas=SIGMAS)
    assert torch.equal(native, draw(configured(model), cfg=2, sigmas=SIGMAS))
    response = traces(caplog)[-1]["response"]
    assert response["auxiliary_dit_calls"] == 7 and response["auxiliary_dit_rows"] == 14
    point, = response["points"]
    assert set(point["branches"]) == {0, 1}
    assert point["auxiliary_dit_rows"] == 14


def test_native_bf16_probe_reports_half_amplitude_rounding(model, caplog):
    model.model.diffusion_model.bfloat16()
    model.model.diffusion_model.dtype = torch.bfloat16
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, sigmas=SIGMAS)
    patched = configured(model, directions=1, calibration_max_change=0.01, probe_precision="fixed")
    assert torch.equal(native, draw(patched, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["input_dtype"] == "torch.bfloat16"
    diff, = point["differences"]
    assert 0.02 < diff["off_axis_fraction"] < 0.25
    half = diff["half_calibration"]
    assert half["off_axis_fraction"] > diff["off_axis_fraction"]
    assert half["rounding_warning"] and half["status"] == "rounding_limited"
    assert "relative_first_order_change" not in half


def test_compares_feature_sites_using_projected_candidates(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    opts = dict(mode="feature_steer", feature_sites="0:0,1:1", feature_pair=False,
                feature_projection="response_subspace", selection="response_alignment",
                calibration="off", auxiliary_budget=14, latent_cell=1)
    draw(configured(model, **opts), sigmas=SIGMAS)
    point, = traces(caplog)[-1]["response"]["points"]
    candidates = [entry for entry in point["candidates"] if entry["arm"] != "baseline"]
    assert {(entry["block"], entry["head"]) for entry in candidates} == {(0, 0), (1, 1)}
    assert all(0 < info["projected_clean_rms"] <= info["clean_rms"] for info in point["feature"])
    assert point["latent_region"]["cell"] == 1
    best = max(candidates, key=lambda entry: entry["response_alignment"])
    assert point["selected_candidate"] == best["arm"]
    assert point["auxiliary_dit_calls"] == 8
    assert point["feature_reference_calls"] == 2
    assert point["feature_reference_capture_blocks"] == [0, 1]


def test_steering_accepts_solver_start_tolerance_above_one(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    sigmas = torch.tensor([1.000005, 0.93, 0.0])
    steering = configured(model, mode="latent_steer", max_sigma=1.0, min_sigma=0.99)
    assert torch.isfinite(draw(steering, sigmas=sigmas)).all()
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["sigma"] > 1 and point["applied"]



def test_reconstruction_preflights_initial_and_current_reference_calls(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    limited = configured(model, mode="feature_replay", auxiliary_budget=2, calibration="off")
    assert torch.equal(draw(limited, sigmas=SIGMAS), draw(model, sigmas=SIGMAS))
    trace = traces(caplog)[-1]["response"]
    assert trace["auxiliary_dit_calls"] == trace["auxiliary_dit_rows"] == 0
    assert trace["points"][0]["feature"][0]["reason"] == "auxiliary_forward_budget"


def test_explicit_reference_start_survives_frozen_replay_without_that_solver_step(model, caplog, tmp_path, monkeypatch):
    import folder_paths
    from mode_explorer.nodes import ModeExplorerLoadResponse

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    anchor, target = map(float, SIGMAS[1:3])
    settings = {"mode": "feature_replay", "min_sigma": target - .001, "max_sigma": target + .001,
                "calibration": "off", "export": True}
    explicit = configured(model, **settings, reference_start_sigma=0.9743)
    draw(explicit, sigmas=SIGMAS)
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["feature_reference_initial_sigma"] == anchor
    assert point["feature_reference_sigma_source"] == "configured_sigma"
    assert point["feature_reference_calls"] == 2
    loader = ModeExplorerLoadResponse()
    expected, schedule = loader.load(point["export_id"], "feature_repel_clean")
    assert not any(float(t) == anchor for t in schedule)

    for reference_options in ({}, {"reference_start_sigma": 0.9743}):
        draw(configured(model, **settings, **reference_options, frozen_state=point["export_id"]), sigmas=schedule)
        replay, = traces(caplog)[-1]["response"]["points"]
        assert replay["feature_reference_sigma_source"] == "export"
        assert replay["feature_reference_initial_sigma"] == anchor
        actual, _ = loader.load(replay["export_id"], "feature_repel_clean")
        assert torch.equal(expected["samples"], actual["samples"])

    draw(configured(model, **settings, reference_start_sigma=0.9743), sigmas=schedule)
    missing, = traces(caplog)[-1]["response"]["points"]
    assert missing["feature"][0]["reason"] == "reference_start_not_observed"
    assert missing["feature_reference_calls"] == 0


def test_explicit_floor_never_falls_back_to_an_ineligible_reference(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    sigmas = SIGMAS.clone()
    sigmas[0] = 0.999995
    draw(configured(model, mode="feature_replay", reference_min_sigma=1.0, calibration="off"), sigmas=sigmas)
    point, = traces(caplog)[-1]["response"]["points"]
    assert point["feature_reference_sigma_source"] == "configured_min_sigma"
    assert point["feature_reference_initial_sigma"] is None
    assert point["feature"][0]["reason"] == "reference_start_not_observed"
    assert point["auxiliary_dit_calls"] == 0


def test_native_bf16_calibration_improves_geometry_without_extra_model_calls(model, caplog):
    model.model.diffusion_model.bfloat16()
    model.model.diffusion_model.dtype = torch.bfloat16
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    native = draw(model, sigmas=SIGMAS)
    assert torch.equal(native, draw(configured(model, directions=1), sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    diff, = point["differences"]
    assert diff["precision_adjusted"] and 0.02 < diff["probe_displacement"] <= 0.08
    half = diff["half_calibration"]
    assert half["status"] in ("stable", "unstable")
    assert not half["rounding_warning"]
    assert half["off_axis_fraction"] <= 0.10
    assert half["relative_input_direction_change"] > 0 and point["auxiliary_dit_calls"] == 5

    constrained = configured(model, mode="latent_steer", directions=1, input_limit=0.024)
    assert torch.equal(native, draw(constrained, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    diff, = point["differences"]
    assert diff["half_calibration"]["status"] == "rounding_limited"
    assert diff["reason"] == "failed_half_amplitude_calibration"
    assert point["auxiliary_dit_calls"] == 1 and point["selected_candidate"] == "baseline"


def test_four_sites_share_reference_forwards_and_match_single_site_responses(model, caplog):
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    sites = ("0:0", "0:1", "1:0", "1:1")
    settings = dict(mode="feature_steer", selection="export_only", feature_pair=False,
                    calibration="off", feature_references=2)
    native = draw(model, sigmas=SIGMAS)
    combined = configured(model, **settings, feature_sites=",".join(sites), auxiliary_budget=8)
    assert torch.equal(native, draw(combined, sigmas=SIGMAS))
    together, = traces(caplog)[-1]["response"]["points"]
    assert together["feature_reference_calls"] == 4
    assert together["feature_reference_capture_blocks"] == [0, 1]
    assert together["auxiliary_dit_calls"] == together["auxiliary_dit_rows"] == 8

    for site, expected in zip(sites, together["feature"]):
        individual = configured(model, **settings, feature_sites=site, auxiliary_budget=5)
        draw(individual, sigmas=SIGMAS)
        point, = traces(caplog)[-1]["response"]["points"]
        assert point["feature"] == [expected]

    limited = configured(model, **settings, feature_sites=",".join(sites), auxiliary_budget=7)
    draw(limited, sigmas=SIGMAS)
    response = traces(caplog)[-1]["response"]
    assert response["auxiliary_dit_calls"] == 0
    assert response["points"][0]["feature"][0]["reason"] == "auxiliary_forward_budget"


def test_native_bf16_projected_matrix_uses_one_amplitude_per_state(model, caplog):
    model.model.diffusion_model.bfloat16()
    model.model.diffusion_model.dtype = torch.bfloat16
    caplog.set_level(logging.INFO, logger="ModeExplorer")
    options = dict(directions=4, probe_grid=1, min_sigma=1.0, max_sigma=1.0)
    patched = configured(model, **options)
    assert torch.equal(draw(model, seed=2, sigmas=SIGMAS), draw(patched, seed=2, sigmas=SIGMAS))
    point, = traces(caplog)[-1]["response"]["points"]
    indices = point["projected_response"]["direction_indices"]
    assert indices == point["precision_eligible_directions"] == [0, 1, 2, 3]
    assert point["probe_displacement"] == 0.04 and point["precision_adjusted"]
    assert {point["differences"][i]["probe_displacement"] for i in indices} == {0.04}
    assert point["differences"][0]["half_calibration"]["requested_rms"] == 0.02
    assert point["auxiliary_dit_calls"] == 11
