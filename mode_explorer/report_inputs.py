"""Bounded archive qualification shared by local campaign reports."""

import hashlib
import math

import torch

from .response import flow_clean
from .response_runtime import _tensor_hash
from .trajectory import IDENTIFIER, MAX_BYTES, MAX_STATES
from .trajectory_review import CONTROL_KEYS, STATE_KEYS, _manifest, checked_tensors

PROVENANCE = (*CONTROL_KEYS, "noise_scale", "schedule_length")
SETTINGS_EXCEPTIONS = {"trajectory_label", "feature_application", "ablation_route", "sham",
                       "feature_direction", "direction_trial", "feature_block", "strength"}
D_TYPES = {str(t): t for t in (torch.float16, torch.bfloat16, torch.float32)}
STATE_TENSORS = {"baseline_clean", "frozen_native_input", "conditional_velocity"}
SOLVER_TENSORS = {"solver_input", "solver_clean"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def key(state):
    return state["solver_step"], state["opportunity"], state["sigma"]


def qualified_metadata(data, pin, case, control=None):
    settings = data["settings"]
    expected = pin["block"]
    if expected == "middle":
        blocks = data.get("single_stream_blocks")
        require(type(blocks) is int and blocks > 0, "missing_loaded_block_count")
        expected = blocks // 2
    require(settings["feature_block"] == expected, "feature_block_mismatch")
    for name, value in pin["settings"].items():
        require(settings.get(name) == value, f"setting_mismatch:{name}")
    require(data["label"] == pin["label"] and data["seed"] == case["seed"], "label_or_seed_mismatch")
    require(data["status"] == "completed" and data["cfg"] == 1 and data["stage"] == "planner",
            "incomplete_or_wrong_planner_contract")
    schedule, routes = data["schedule"], data["routes"]
    states = data["states"]
    require(0 < len(states) <= MAX_STATES and len(routes) + 1 == len(schedule) == data["schedule_length"]
            and len(routes) <= MAX_STATES and schedule[-1] == 0
            and data["solver"].endswith(".sample_er_sde"), "invalid_solver_or_schedule")
    require(all(math.isfinite(v) for v in schedule) and all(a > b for a, b in zip(schedule, schedule[1:])),
            "invalid_sigma_schedule")
    require(all(r["route"] in ("actual", "forecast") for r in routes), "unknown_route")
    require("".join("A" if r["route"] == "actual" else "F" for r in routes) == case["route"], "forecast_route_mismatch")
    require(all(r["solver_step"] == i and r["opportunity"] == i and r["branches"] == [0]
                and len(r["sigmas"]) == 1 and 0 < r["sigmas"][0] <= 1.00001
                for i, r in enumerate(routes)), "opportunity_or_sigma_mismatch")
    ref_calls = [r["reference_calls"] for r in routes]
    require(ref_calls == [1, 1] + [0] * (len(routes) - 2) and data["live_reference_dit_calls"] == sum(ref_calls),
            "reference_forward_schedule_mismatch")
    actuals = [r for r in routes if r["route"] == "actual"]
    require(len(actuals) == len(states) == data["native_dit_calls"]
            and len(routes) - len(actuals) == data["spectrum_forecast_calls"], "route_count_mismatch")
    require([s["ordinal"] for s in states if s["feature_applied"]] == pin["applied"], "intervention_schedule_mismatch")
    require(not data.get("auxiliary_dit_calls") and not data.get("initialization_dit_calls"), "unexpected_auxiliary_forward")
    terminal = data.get("solver_terminal")
    if terminal is not None or (data.get("ablation") or {}).get("status") == "qualified":
        require(terminal is not None and terminal.get("status") == "bit_exact"
                and terminal.get("solver_step") == actuals[-1]["solver_step"] == len(routes) - 1
                and terminal.get("solver_clean_sha256") == states[-1].get("solver_clean_sha256"),
                "unqualified_solver_terminal")
    if pin.get("require_exact"):
        require(terminal is not None and (data.get("ablation") or {}).get("status") == "qualified",
                "missing_exact_ablation_record")
    for i, (state, route) in enumerate(zip(states, actuals, strict=True)):
        require(all(state[k] == v for k, v in route.items()), "state_route_mismatch")
        require(state["sigma"] == route["sigmas"][0], "state_sigma_mismatch")
        require(state["reference_slots"] == ([case["slot"]] if i < 2 else []), "reference_slot_mismatch")
        require(state["feature_reference_initial_sigma"] == states[0]["sigma"], "reference_start_mismatch")
        if "reference_history" in state:
            history = state["reference_history"]
            require(len(history) == (1 if i < 2 else 0), "reference_history_count_mismatch")
            if history:
                h = history[0]
                require(h["slot"] == case["slot"] and h["sigma"] == state["sigma"]
                        and h["continuation"] == settings["reference_continuation"]
                        and h["first_sigma"] == states[0]["sigma"]
                        and h["previous_sigma"] == (states[i-1]["sigma"] if i else None)
                        and (bool(h["previous_clean_sha256"]) == bool(i)), "reference_continuation_mismatch")
    if control is not None:
        for name in PROVENANCE:
            require(data.get(name) == control.get(name), f"provenance_mismatch:{name}")
        require(routes == control["routes"], "actual_or_forecast_route_provenance_mismatch")
        recipe = {k: v for k, v in settings.items() if k not in SETTINGS_EXCEPTIONS}
        require(recipe == {k: v for k, v in control["settings"].items() if k not in SETTINGS_EXCEPTIONS},
                "feature_reference_recipe_mismatch")
        for state, base in zip(states, control["states"], strict=True):
            require(key(state) == key(base), "actual_state_alignment_mismatch")
            for name in STATE_KEYS:
                require(state.get(name) == base.get(name), f"state_provenance_mismatch:{name}")
            if state["ordinal"] == 0:
                require(state["native_input_sha256"] == base["native_input_sha256"], "initial_native_input_mismatch")


def _file(folder, state):
    require(state["export_id"] == f"{folder.name}/{state['ordinal']:03d}", "foreign_tensor_file")
    path = folder / f"{state['ordinal']:03d}.safetensors"
    require(path.resolve().parent == folder.resolve() and path.stat().st_size <= MAX_BYTES + 65536,
            "unsafe_or_oversized_tensor_file")
    return path


def equivalent_signature(folder, data, interrupt):
    # Only used on duplicate eligible runs. Identical clean hashes alone
    # are insufficient to establish identical recorded velocity and proposals.
    ordered = []
    for state in data["states"]:
        interrupt()
        digest = hashlib.sha256()
        with _file(folder, state).open("rb") as f:
            for part in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(part)
        ordered.append((tuple((k, v) for k, v in state.items() if k != "export_id"), digest.hexdigest()))
    return ordered, tuple(sorted(data["settings"].items())), data.get("single_stream_blocks"), data.get("solver_terminal")


def resolve_arm(root, pin, case, control, interrupt):
    if "id" in pin:
        name = pin["id"]
        require(isinstance(name, str) and IDENTIFIER.fullmatch(f"{name}/000"), "invalid_archive_id")
        folder = root / name
        data = _manifest(folder)
        qualified_metadata(data, pin, case, control)
        return folder, data, {"rule": "pinned", "rejected_candidates": []}
    eligible, rejected = [], []
    for folder in sorted(root.glob("trajectory-*")):
        interrupt()
        path = folder / "manifest.json"
        if not IDENTIFIER.fullmatch(f"{folder.name}/000") or not path.is_file():
            continue
        try:
            data = _manifest(folder)
        except (ValueError, OSError, AttributeError, TypeError, KeyError):
            continue  # The label of an unreadable manifest cannot be established.
        if data.get("label") != pin["label"]:
            continue
        try:
            qualified_metadata(data, pin, case, control)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            rejected.append({"id": folder.name, "reason": str(exc)})
        else:
            eligible.append((folder, data))
    if not eligible:
        raise ValueError(f"no_compatible_archive:{pin['label']}:{rejected}")
    if len(eligible) > 1:
        signatures = [equivalent_signature(folder, data, interrupt) for folder, data in eligible]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("ambiguous_non_equivalent_archives:" + ",".join(folder.name for folder, _ in eligible))
    folder, data = eligible[0]
    return folder, data, {"rule": "lexical_first_of_byte_equivalent" if len(eligible) > 1 else "unique_compatible",
                          "eligible_ids": [p.name for p, _ in eligible], "rejected_candidates": rejected}


def load_state(folder, state, case, exact_expected):
    from comfy.latent_formats import Wan21
    from safetensors.torch import load_file

    path = _file(folder, state)
    tensors = load_file(path)
    names = set(tensors)
    allowed = STATE_TENSORS | SOLVER_TENSORS | {"compact_feature_proposal"}
    require(STATE_TENSORS <= names <= allowed, "missing_or_unknown_tensor")
    require(("compact_feature_proposal" in names) == (state["ordinal"] < 2), "compact_proposal_schedule_mismatch")
    require(not (names & SOLVER_TENSORS) or SOLVER_TENSORS <= names, "incomplete_solver_capture")
    require(not exact_expected or SOLVER_TENSORS <= names, "missing_exact_solver_tensors")
    width, height = case["resolution"]
    shape = (1, 16, 1, height // 8, width // 8)
    for name, tensor in tensors.items():
        if name == "compact_feature_proposal":
            require(tensor.ndim == 4 and tensor.shape[0] == 1 and 0 < tensor.shape[1] <= 32768
                    and all(0 < v <= 8 for v in tensor.shape[-2:]), "invalid_compact_proposal_shape")
        else:
            require(tuple(tensor.shape) == shape, f"latent_shape_mismatch:{name}")
        require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), f"invalid_tensor_values:{name}")
    checked_tensors(state, tensors)
    native_dtype = D_TYPES.get(state["input_dtype"])
    velocity_dtype = D_TYPES.get(state["velocity_dtype"])
    require(native_dtype is not None and velocity_dtype is not None, "unrecognized_native_dtype")
    native = tensors["frozen_native_input"].to(native_dtype)
    require(torch.equal(native.float(), tensors["frozen_native_input"])
            and torch.equal(tensors["conditional_velocity"].to(velocity_dtype).float(), tensors["conditional_velocity"]),
            "saved_value_not_representable_in_native_dtype")
    require(_tensor_hash(native) == state["native_input_sha256"], "native_input_fingerprint_mismatch")
    fingerprints = {name: _tensor_hash(tensor) for name, tensor in tensors.items()}
    proposal = "compact_feature_proposal"
    hash_verification = {"native_input_sha256": "verified_in_original_dtype", "baseline_clean": "reconstructed_exact"}
    if proposal in names and "compact_proposal_sha256" in state:
        require(fingerprints[proposal] == state["compact_proposal_sha256"], "compact_proposal_fingerprint_mismatch")
        hash_verification["compact_proposal_sha256"] = "verified"
    if SOLVER_TENSORS <= names:
        require(fingerprints["solver_input"] == state.get("solver_input_sha256"), "solver_input_fingerprint_mismatch")
        require(torch.equal(tensors["solver_input"].to(native_dtype), native), "solver_to_transformer_cast_mismatch")
        estimated = Wan21().process_out(flow_clean(tensors["solver_input"], tensors["conditional_velocity"], state["sigma"]))
        require(torch.equal(estimated, tensors["solver_clean"]), "solver_clean_reconstruction_mismatch")
        hash_verification.update(solver_input_sha256="verified", solver_clean_sha256="verified_and_reconstructed_exact")
    return tensors, fingerprints, hash_verification
