import json
import shutil
from zipfile import ZipFile

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("comfy")

from test_comfy_integration import model, draw
from comfy.latent_formats import Wan21
from mode_explorer.nodes import ModeExplorer
from mode_explorer.report_inputs import load_state, resolve_arm
from mode_explorer.report_metrics import odd_even, region_metrics, response_geometry, spectral
from mode_explorer.research_report import ModeExplorerResearchReport, SCHEMA, analyze_case
from mode_explorer.response import flow_clean
from mode_explorer.response_runtime import _tensor_hash
from tools.structural_workflows import validate_widget_serialization


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    root = tmp_path / "modeexplorer_trajectory"
    spec = {"seed": 836, "slot": 3, "route": "AAFA", "resolution": [32, 32], "crops": {"left": [0, 0, 16, 32]}, "arms": {}}
    for arm, archive, block, legacy in (("sham", "trajectory-aaaaaa01", 1, True), ("edit", "trajectory-bbbbbb02", 2, False)):
        path = root / archive
        path.mkdir(parents=True)
        settings = {"feature_block": block, "feature_grid": 2, "reference_continuation": "first_estimate",
                    "strength": 0.03, "feature_direction": "repel", "direction_trial": 0}
        schedule = [0.99996, .9, .8, .6, 0]
        routes = [{"solver_step": i, "opportunity": i, "route": route, "sigmas": [schedule[i]],
                   "reference_calls": int(i < 2), "branches": [0]}
                  for i, route in enumerate(["actual", "actual", "forecast", "actual"])]
        states = []
        for i, route in enumerate(x for x in routes if x["route"] == "actual"):
            sigma = route["sigmas"][0]
            solver = torch.full((1, 16, 1, 4, 4), .44449 + i)
            native = solver.half().float()
            velocity = torch.full_like(solver, .5 if legacy else .375)
            tensors = {"frozen_native_input": native, "conditional_velocity": velocity,
                       "baseline_clean": Wan21().process_out(flow_clean(native, velocity, sigma))}
            applied = not legacy and i == 0
            state = {**route, "ordinal": i, "export_id": f"{archive}/{i:03}", "sigma": sigma,
                     "feature_applied": applied, "reference_slots": [3] if i < 2 else [],
                     "feature_reference_initial_sigma": schedule[0], "conditioning_sha256": "same",
                     "native_input_sha256": _tensor_hash(native.half()), "input_dtype": "torch.float16",
                     "velocity_dtype": "torch.float16"}
            if not legacy:
                clean = Wan21().process_out(flow_clean(solver, velocity, sigma))
                tensors.update(solver_input=solver, solver_clean=clean)
                state.update(solver_input_sha256=_tensor_hash(solver), solver_clean_sha256=_tensor_hash(clean),
                             reference_history=[{"slot": 3, "sigma": sigma, "continuation": "first_estimate",
                                                 "first_sigma": schedule[0], "previous_sigma": schedule[0] if i else None,
                                                 "previous_clean_sha256": "known" if i else None}] if i < 2 else [])
            if i < 2:
                proposal = torch.full((1, 4, 2, 2), .03125)
                tensors["compact_feature_proposal"] = proposal
                state.update(compact_proposal_sha256=_tensor_hash(proposal), proposed_shift_rms=[.03125],
                             realized_shift_rms=[.03 if applied else 0.])
            save_file(tensors, path / f"{i:03}.safetensors")
            states.append(state)
        terminal = None if legacy else {"status": "bit_exact", "solver_step": 3, "solver_clean_sha256": states[-1]["solver_clean_sha256"]}
        data = {"schema": 1, "label": arm, "seed": spec["seed"], "cfg": 1, "stage": "planner", "status": "completed",
                "settings": settings, "solver": "comfy.k_diffusion.sampling.sample_er_sde",
                "schedule": schedule, "schedule_length": 5, "routes": routes, "states": states,
                "solver_terminal": terminal, "single_stream_blocks": 5,
                "ablation": None if legacy else {"status": "qualified"}, "initial_noise_sha256": "same",
                "native_dit_calls": 3, "spectrum_forecast_calls": 1, "live_reference_dit_calls": 2,
                "auxiliary_dit_calls": 0}
        (path / "manifest.json").write_text(json.dumps(data))
        spec["arms"][arm] = {"label": arm, "block": 1 if legacy else "middle", "applied": [] if legacy else [0],
                             "settings": {"strength": .03}, "require_exact": not legacy}
        if legacy:
            spec["arms"][arm]["id"] = archive
    return root, spec


def modify(root, update, ordinal=0):
    folder = root / "trajectory-bbbbbb02"
    manifest = folder / "manifest.json"
    data = json.loads(manifest.read_text())
    update(data, data["states"][ordinal])
    manifest.write_text(json.dumps(data))


def test_report_accepts_legacy_and_exact_without_mixing_coordinates(campaign):
    root, spec = campaign
    result, _ = analyze_case("mixed", spec, root, lambda: None)
    assert result["qualification"] == "qualified"
    assert result["clean_coordinate"] == "baseline_clean"
    assert [e["solver_step"] for e in result["evaluations"]] == [0, 1, 3]
    assert result["arms"]["edit"]["forecast_only"][0]["solver_step"] == 2
    assert result["arms"]["sham"]["checks"]["terminal_solver"] == "unavailable_legacy"
    checks = result["arms"]["edit"]["states"][0]["checks"]
    assert checks["native_input_sha256"] == "verified_in_original_dtype"
    assert checks["solver_clean_sha256"] == "verified_and_reconstructed_exact"
    initial = result["evaluations"][0]["arms"]["edit"]
    assert initial["solver_input_delta"] is None
    assert initial["clean_delta"]["l2"] / initial["clean_delta"]["rms"] == pytest.approx(16)
    assert initial["immediate_gain_rms"] == pytest.approx(initial["clean_delta"]["rms"] / .03)
    assert initial["immediate_gain_l2"] == pytest.approx(initial["clean_delta"]["l2"] / (.03 * 4))


@pytest.mark.parametrize(("mutation", "reason"), [
    (lambda data, state: data["settings"].update(feature_block=1), "feature_block_mismatch"),
    (lambda data, state: state.update(feature_applied=False), "intervention_schedule_mismatch"),
    (lambda data, state: data["routes"][2].update(route="actual"), "forecast_route_mismatch"),
    (lambda data, state: data.update(initial_noise_sha256="different"), "provenance_mismatch:initial_noise_sha256"),
    (lambda data, state: state.update(conditioning_sha256="different"), "state_provenance_mismatch:conditioning_sha256"),
    (lambda data, state: state.update(native_input_sha256="different"), "initial_native_input_mismatch"),
    (lambda data, state: state["reference_history"][0].update(continuation="different"), "reference_continuation_mismatch"),
    (lambda data, state: data["solver_terminal"].update(status="solver_output_mismatch"), "unqualified_solver_terminal"),
])
def test_report_rejects_wrong_provenance_and_execution(campaign, mutation, reason):
    root, spec = campaign
    modify(root, mutation)
    result, _ = analyze_case("rejected", spec, root, lambda: None)
    assert result["qualification"] == "unqualified"
    assert result["evaluations"] == []
    assert reason in result["arms"]["edit"]["reason"]


@pytest.mark.parametrize(("transform", "reason"), [
    (lambda tensors: tensors.pop("conditional_velocity"), "missing_or_unknown_tensor"),
    (lambda tensors: tensors.update(conditional_velocity=tensors["conditional_velocity"].half()), "invalid_tensor_values"),
    (lambda tensors: tensors.update(conditional_velocity=tensors["conditional_velocity"][..., :2].contiguous()), "latent_shape_mismatch"),
    (lambda tensors: tensors["conditional_velocity"].fill_(float("nan")), "invalid_tensor_values"),
    (lambda tensors: tensors["compact_feature_proposal"].mul_(2), "compact_proposal_fingerprint_mismatch"),
    (lambda tensors: tensors["solver_clean"].add_(1), "solver clean fingerprint mismatch"),
])
def test_report_rejects_corrupted_tensor(campaign, transform, reason):
    root, spec = campaign
    path = root / "trajectory-bbbbbb02/000.safetensors"
    tensors = load_file(path)
    transform(tensors)
    save_file(tensors, path)
    result, _ = analyze_case("bad", spec, root, lambda: None)
    assert result["qualification"] == "unqualified"
    assert reason in result["arms"]["edit"]["reason"]


def test_report_duplicate_resolution_requires_full_byte_equivalence(campaign):
    root, spec = campaign
    left, right = root / "trajectory-bbbbbb02", root / "trajectory-cccccc03"
    shutil.copytree(left, right)
    path = right / "manifest.json"
    path.write_text(path.read_text().replace(left.name, right.name))
    _, data, info = resolve_arm(root, spec["arms"]["sham"], spec, None, lambda: None)
    candidate, _, info = resolve_arm(root, spec["arms"]["edit"], spec, data, lambda: None)
    assert candidate == left
    assert info["rule"] == "lexical_first_of_byte_equivalent"
    point = right / "001.safetensors"
    tensors = load_file(point)
    tensors["conditional_velocity"].mul_(2)
    save_file(tensors, point)
    with pytest.raises(ValueError, match="ambiguous_non_equivalent_archives"):
        resolve_arm(root, spec["arms"]["edit"], spec, data, lambda: None)


def test_native_fingerprint_uses_original_dtype_in_later_state(campaign):
    root, spec = campaign
    modify(root, lambda data, state: state.update(native_input_sha256="invalid"), ordinal=1)
    folder = root / "trajectory-bbbbbb02"
    state = json.loads((folder / "manifest.json").read_text())["states"][1]
    with pytest.raises(ValueError, match="native_input_fingerprint_mismatch"):
        load_state(folder, state, spec, True)


def test_known_response_matrix_and_norms():
    positive = torch.tensor([2., 0.])
    negative = torch.tensor([0., -2.])
    report = response_geometry({"positive": positive, "negative": negative})
    assert report["gram"] == [[4, 0], [0, 4]]
    assert report["cosine"] == [[1, 0], [0, 1]]
    assert report["singular_values_sampled_responses"] == [2, 2]
    assert report["entropy_effective_rank_sampled_responses"] == pytest.approx(2)
    assert report["pairwise_l2"][0][1] == pytest.approx(8**.5)
    for value in (odd_even(positive, negative)["odd"], odd_even(positive, negative)["even"]):
        assert value["rms"] == pytest.approx(1)
        assert value["l2"] == pytest.approx(2**.5)


def test_registered_region_and_spectral_energy():
    delta = torch.zeros(1, 4, 4)
    delta[:, :, :2] = 1
    report = region_metrics(delta, {"left": [0, 0, 16, 32]}, [32, 32])
    left = report["regions"]["left"]
    assert left["latent_xyxy"] == [0, 0, 2, 4]
    assert left["energy"] == 8 and left["density_concentration"] == 2 and left["complement_rms"] == 0
    bands = spectral(torch.zeros_like(delta), {"edit": delta})["edit"]
    assert sum(v["displacement"] for v in bands.values()) == pytest.approx(8)
    assert sum(v["net_change"] for v in bands.values()) == pytest.approx(8)


def test_zip_contents_precise_size_and_no_dit(campaign):
    root, spec = campaign
    plan = {"name": "fixture", "budget_mib": 2, "cases": {"paired": spec}}
    payload = json.dumps(plan)
    graph = {"nodes": [{"id": 1, "type": "ModeExplorerResearchReportKrea2", "widgets_values": [payload],
                        "widgets_values_named": {"campaign": payload}}]}
    validate_widget_serialization(graph)
    assert ModeExplorerResearchReport().report(payload) == ()
    final, = (root.parent / "modeexplorer_research_reports").glob("*.zip")
    with ZipFile(final) as archive:
        report = json.loads(archive.read("report.json"))
        assert report["schema"] == SCHEMA
        assert report["qualification"] == "qualified"
        assert report["report_dit_calls"] == 0
        assert report["cases"]["paired"]["vae_decode_seconds"] == 0
        assert report["inventory"] == {info.filename: info.file_size for info in archive.infolist()}
        assert report["archive_size_bytes"] == final.stat().st_size < 2**21
        assert report["cases"]["paired"]["evaluations"][0]["sigma"] == .99996
        assert len(archive.read("evaluations.csv").splitlines()) == 7
        assert archive.testzip() is None
        assert not any(n.endswith(".safetensors") for n in archive.namelist())
    assert (root / "trajectory-bbbbbb02/000.safetensors").is_file()


def test_report_interruption_removes_staging(campaign, monkeypatch):
    from mode_explorer import research_report
    from comfy.model_management import InterruptProcessingException

    root, spec = campaign

    def interrupt_after_staging(path, *args):
        assert path.is_file()
        raise InterruptProcessingException()

    monkeypatch.setattr(research_report, "write_report", interrupt_after_staging)
    with pytest.raises(InterruptProcessingException):
        ModeExplorerResearchReport().report(json.dumps({"name": "stop", "cases": {"paired": spec}}))
    assert not list((root.parent / "modeexplorer_research_reports").iterdir())


def test_middle_resolves_loaded_blocks_and_preserves_explicit_behavior(model):
    node = ModeExplorer()
    assert node.verify_model(model, -1) == 1
    assert torch.equal(draw(node.apply(model, feature_block=-1)[0]), draw(node.apply(model, feature_block=1)[0]))
    with pytest.raises(ValueError, match="outside"):
        node.apply(model, feature_block=3)


def test_final_png_requires_executable_prompt_and_unambiguous_pixels(tmp_path):
    from copy import deepcopy
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from mode_explorer.report_visuals import open_final, prompt_signature

    archived = {"1": {"class_type": "ModeExplorerKrea2", "_meta": {"title": "First only"},
                      "inputs": {"feature_block": 1, "trajectory_label": "old", "strength": .03}},
                "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old"}},
                "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "matched"}}}
    current = deepcopy(archived)
    current["1"]["inputs"].update(feature_block=-1, trajectory_label="new")
    current["1"]["_meta"].update(title="Middle block", ver="updated registry metadata")
    current["2"]["inputs"]["filename_prefix"] = "new"
    pin = {"label": "new", "block": "middle"}
    final = {"prefix": "new", "prompt_sha256": prompt_signature(archived), "verify_feature_widget": True}

    def write(counter, color):
        info = PngInfo()
        info.add_text("prompt", json.dumps(current))
        Image.new("RGB", (32, 32), color).save(tmp_path / f"new_{counter:05}_.png", pnginfo=info)

    write(1, "black")
    current["3"]["inputs"]["unet_name"] = "different"
    write(2, "white")
    _, provenance = open_final(tmp_path, pin, final, [32, 32])
    assert provenance["path"] == "new_00001_.png"
    current["3"]["inputs"]["unet_name"] = "matched"
    write(3, "white")
    with pytest.raises(ValueError, match="ambiguous_non_equivalent_final_images"):
        open_final(tmp_path, pin, final, [32, 32])
