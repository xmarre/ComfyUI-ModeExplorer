"""Generate one compact research archive from local Krea 2 trajectories."""

import csv
import json
import logging
import math
import os
import tempfile
from secrets import token_hex
from io import StringIO
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import torch
from safetensors import SafetensorError

from .objectives import direction_control
from .report_inputs import PROVENANCE, STATE_KEYS, key, load_state, require, resolve_arm
from .report_metrics import cosine, norms, odd_even, proposal_metrics, ratio, region_metrics, response_geometry, restrict_geometry, spectral
from .report_visuals import blind_labels, encode, jpeg_evidence, make_visuals, panel, preview
from .trajectory import LABEL, _root

LOGGER = logging.getLogger("ModeExplorer")
SCHEMA = "modeexplorer.research_report/1"
INPUT_ERRORS = (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, SafetensorError)
MIB = 1024**2


def _validate_plan(plan):
    name, cases = plan["name"], plan["cases"]
    budget = plan.get("budget_mib", 15)
    seed = plan.get("blind_seed")
    require(seed is None or type(seed) is int or isinstance(seed, str) and 0 < len(seed) <= 256,
            "invalid_blinding_seed")
    require(isinstance(name, str) and LABEL.fullmatch(name) and isinstance(cases, dict) and 0 < len(cases) <= 4
            and type(budget) is int and 2 <= budget <= 30, "invalid_campaign_name_case_count_or_zip_budget")
    for case, spec in cases.items():
        arms, crops = spec["arms"], spec["crops"]
        require(LABEL.fullmatch(case) and 2 <= len(arms) <= 16 and "sham" in arms and all(LABEL.fullmatch(a) and a != "_sham" for a in arms)
                and len(crops) <= 8 and all(LABEL.fullmatch(r) for r in crops), "invalid_campaign_arms_or_crops")
        w, h = spec["resolution"]
        require(type(w) is int and type(h) is int and w > 0 and h > 0 and w % 8 == 0 and h % 8 == 0 and w * h <= 8_500_000,
                "invalid_planner_resolution")
        require(all(len(box) == 4 and all(type(v) is int for v in box)
                    and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h for box in crops.values()),
                "invalid_fixed_crop_bounds")
        visual = spec.get("visual_crops", crops)
        require(visual.keys() == crops.keys() and all(len(box) == 4 and all(type(v) is int for v in box)
                and crops[region][0] <= box[0] < box[2] <= crops[region][2]
                and crops[region][1] <= box[1] < box[3] <= crops[region][3] for region, box in visual.items()),
                "visual_crop_must_be_inside_registered_region")
        require(len(set(spec.get("intermediate_steps", []))) <= 2, "select_at_most_two_intermediate_steps")
        require(len(set(pin["label"] for pin in arms.values())) == len(arms), "duplicate_arm_labels")
        require(all(type(pin.get("require_runtime_telemetry", False)) is bool for pin in arms.values()),
                "invalid_runtime_telemetry_policy")
        require(not arms["sham"]["applied"] and "id" in arms["sham"], "sham_must_be_a_pinned_unedited_control")
        require(spec.get("direction_reference") is None or spec["direction_reference"] in arms,
                "unknown_direction_reference_arm")
        require(type(spec.get("allow_partial", False)) is bool, "invalid_partial_comparison_policy")
        require(not spec.get("allow_partial") or not (spec.get("direction_reference") or plan.get("regressions")),
                "partial_comparisons_require_independent_arms_without_positional_regressions")


def _record_failure(report, arm, check, reason):
    report["arms"][arm]["checks"][check] = "failed"
    report["arms"][arm]["reason"] = str(reason)
    report["arms"][arm]["qualification"] = "unqualified"
    report["qualification"] = "unqualified"


def _performance(data):
    return {"actual": data["native_dit_calls"], "forecast": data["spectrum_forecast_calls"],
            "reference_dit_calls": data["live_reference_dit_calls"],
            "auxiliary_dit_calls": data.get("auxiliary_dit_calls"), "sampling_wall_seconds": data.get("sampler_seconds"),
            "capture_wall_seconds": data.get("capture_seconds"), "serialization_wall_seconds": data.get("serialization_seconds"),
            "peak_device_memory_bytes": data.get("peak_device_memory_bytes"),
            "peak_device_memory_scope": data.get("peak_device_memory_scope"), "source": "runtime_telemetry"}


def _measure_evaluation(records, base, chosen, first, spec, step):
    delta = {arm: (row[0][chosen] - base[chosen])[0].reshape(-1, *base[chosen].shape[-2:])
             for arm, row in records.items() if arm != "sham"}
    if not first:
        first.update(delta)
    calculations = spectral(base[chosen][0].reshape(-1, *base[chosen].shape[-2:]), delta)
    values = {}
    for arm, row in records.items():
        tensors, state, metadata, intervention = row
        entry = {"actual_ordinal": state["ordinal"], "applied": state["feature_applied"],
                 "proposal": intervention, "velocity_delta": norms(tensors["conditional_velocity"] - base["conditional_velocity"]),
                 "model_input_delta": norms(tensors["frozen_native_input"] - base["frozen_native_input"]),
                 "solver_input_delta": norms(tensors["solver_input"] - base["solver_input"])
                 if "solver_input" in tensors and "solver_input" in base else None}
        if arm != "sham":
            clean = norms(delta[arm])
            initial = norms(first[arm])
            entry.update(clean_delta=clean, spectral=calculations[arm],
                         **region_metrics(delta[arm], spec["crops"], spec["resolution"]),
                         norm_amplification_from_initial=ratio(clean["l2"], initial["l2"]),
                         cosine_to_initial=cosine(delta[arm], first[arm]),
                         immediate_gain_rms=ratio(clean["rms"], intervention["native_rounded_realized_rms"])
                         if step == 0 and intervention else None,
                         immediate_gain_l2=ratio(clean["l2"], intervention["native_rounded_realized_l2"])
                         if step == 0 and intervention else None)
        values[arm] = entry
    return delta, {"arms": values, "response_geometry": response_geometry(delta)}


def _check_direction(records, spec, report):
    first = spec.get("direction_reference")
    groups = {}
    for arm, (tensors, state, metadata, measurements) in records.items():
        proposal = tensors["compact_feature_proposal"]
        groups.setdefault(metadata["settings"]["feature_block"], {})[arm] = proposal
        if first is None:
            continue
        source, _, source_meta, _ = records[first]
        params = metadata["settings"]
        if params["feature_block"] != source_meta["settings"]["feature_block"]:
            _record_failure(report, arm, "initial_direction_transform", "direction_transform_requires_same_feature_block")
            continue
        mode, trial = params["feature_direction"], params["direction_trial"]
        want = direction_control(source["compact_feature_proposal"], mode, metadata["seed"], trial)
        if not torch.equal(proposal.view(torch.int32), want.view(torch.int32)):
            # Native signed zeros can differ for independently generated
            # sparsity, so classify exact magnitudes and recorded zero signs separately.
            if not torch.equal(proposal, want):
                _record_failure(report, arm, "initial_direction_transform", "compact_direction_transform_mismatch")
                continue
            report["arms"][arm]["checks"]["initial_direction_transform"] = "value_exact_signed_zero_differs"
        else:
            report["arms"][arm]["checks"]["initial_direction_transform"] = "byte_exact"
    return {str(block): response_geometry(group) for block, group in groups.items()}


def _check_initial(records, report):
    baseline = records["sham"][0]
    solvers = {arm: row[0]["solver_input"] for arm, row in records.items() if "solver_input" in row[0]}
    # Without an exact control, disagreement makes the exact input origin
    # ambiguous for every provider. Legacy comparisons retain their own basis.
    exact = solvers.get("sham") if "sham" in solvers else next(iter(solvers.values()), None)
    ambiguous = "sham" not in solvers and any(not torch.equal(exact, field) for field in solvers.values())
    for arm, (tensors, *_) in records.items():
        if not torch.equal(baseline["frozen_native_input"], tensors["frozen_native_input"]) or (
                arm in solvers and (ambiguous or not torch.equal(exact, solvers[arm]))):
            _record_failure(report, arm, "initial_noisy_input", "initial_noisy_input_mismatch")
        else:
            report["arms"][arm]["checks"]["initial_noisy_input"] = (
                "native_and_solver_exact_among_available" if arm in solvers else "native_exact_solver_unavailable")


def _check_histories(records, report):
    histories = {arm: row[1]["reference_history"] for arm, row in records.items() if "reference_history" in row[1]}
    if not histories:
        return
    baseline = histories.get("sham", next(iter(histories.values())))
    divergent = [arm for arm, history in histories.items() if history != baseline]
    if divergent:
        for arm in divergent if "sham" in histories else histories:
            _record_failure(report, arm, "matched_reference_history", "non_equivalent_reference_history")


def _restrict_case(report, surviving):
    for row in report["evaluations"]:
        row["arms"] = {arm: value for arm, value in row["arms"].items() if arm in surviving}
        row["response_geometry"] = restrict_geometry(row["response_geometry"], surviving)
    groups = report["initial_proposal_groups_by_block"]
    report["initial_proposal_groups_by_block"] = {
        block: restrict_geometry(group, surviving)
        for block, group in groups.items() if surviving.intersection(group["arms"])
    }


def _check_runtime_telemetry(state, intervention, required):
    if required and intervention is not None:
        require(state.get("feature_geometry") is not None and intervention["objective_after"] is not None,
                "missing_runtime_feature_geometry_or_post_edit_objective")
        if state["feature_applied"]:
            error = intervention["realization_error_rms"]
            require(error is not None and math.isfinite(error) and error >= 0, "missing_runtime_realization_error")


def analyze_case(name, spec, root, interrupt, vae=None, presentation=None):
    report = {"qualification": "qualified", "comparison_status": "unavailable", "arms": {}, "evaluations": [], "sham": "sham"}
    allow_partial = spec.get("allow_partial", False)
    selected, base_data = {}, None
    for arm in ("sham", *(a for a in spec["arms"] if a != "sham")):
        interrupt()
        pin = spec["arms"][arm]
        report["arms"][arm] = {"pins": pin, "qualification": "qualified", "checks": {}, "states": []}
        arm_report = report["arms"][arm]
        try:
            if arm != "sham" and base_data is None:
                raise ValueError("sham_archive_unqualified")
            folder, data, resolution = resolve_arm(root, pin, spec, base_data, interrupt)
        except INPUT_ERRORS as exc:
            _record_failure(report, arm, "archive_metadata_and_resolution", exc)
            continue
        if arm == "sham":
            base_data = data
        selected[arm] = folder, data
        arm_report.update(archive_id=folder.name, resolution=resolution, settings=data["settings"],
                          forecast_only=[r for r in data["routes"] if r["route"] == "forecast"],
                          routes=data["routes"], solver_terminal=data.get("solver_terminal"),
                          performance=_performance(data), single_stream_blocks=data.get("single_stream_blocks"),
                          provenance={k: data.get(k) for k in PROVENANCE})
        arm_report["checks"].update(
            archive_metadata_and_resolution="verified", feature_block_and_settings="verified",
            intervention_schedule="verified", actual_and_forecast_routes="verified",
            reference_slots_and_forward_schedule="verified",
            reference_continuation=("recorded_and_checked" if "reference_history" in data["states"][0]
                                    else "unavailable_legacy"),
            planner_and_state_provenance="matched_to_sham" if arm != "sham" else "pinned_control",
            terminal_solver="runtime_certified_and_manifest_bound" if data.get("solver_terminal") else "unavailable_legacy",
            initial_noisy_input="pending_tensor_verification",
        )

    if not selected:
        return report, {}
    chosen = "solver_clean" if all(d.get("solver_terminal") for _, d in selected.values()) else "baseline_clean"
    report["clean_coordinate"] = chosen
    report["coordinate_limit"] = ("exact_solver_predicted_clean_in_vae_coordinates" if chosen == "solver_clean"
                                  else "one_vae_conversion_of_cast_transformer_input_estimate_for_every_arm")
    first, intermediate, decode_time = {}, {}, 0.
    # Every eligible arm shares the pinned route and real sigma values. Iterate
    # one opportunity at a time and keep only one previous response reference.
    for i in range(len(next(iter(selected.values()))[1]["states"])):
        records = {}
        for arm, (folder, data) in selected.items():
            if report["arms"][arm]["qualification"] != "qualified":
                continue
            interrupt()
            state = data["states"][i]
            try:
                tensors, fingerprints, checks = load_state(folder, state, spec, bool(data.get("solver_terminal")))
                intervention = (proposal_metrics(tensors["compact_feature_proposal"], state, data["settings"],
                                                tensors["baseline_clean"].shape)
                                if "compact_feature_proposal" in tensors else None)
                _check_runtime_telemetry(state, intervention, spec["arms"][arm].get("require_runtime_telemetry"))
            except INPUT_ERRORS as exc:
                _record_failure(report, arm, f"tensor_state_{i:03d}", exc)
                continue
            report["arms"][arm]["states"].append({
                "key": key(state), "fingerprints": fingerprints, "checks": checks,
                "provenance": {k: state.get(k) for k in (*STATE_KEYS, "native_input_sha256", "input_dtype", "velocity_dtype")},
            })
            records[arm] = tensors, state, data, intervention
        if not records:
            continue
        step, opportunity, sigma = key(next(iter(records.values()))[1])
        LOGGER.info("ModeExplorer report %s: checked actual step %d (%d/%d)", name, step, i+1,
                    len(next(iter(selected.values()))[1]["states"]))
        if i == 0 and "sham" in records:
            _check_initial(records, report)
        if i < 2:
            _check_histories(records, report)
        records = {arm: row for arm, row in records.items() if report["arms"][arm]["qualification"] == "qualified"}
        if "sham" not in records or len(records) < 2 or (not allow_partial and report["qualification"] != "qualified"):
            continue
        if i == 0:
            report["initial_proposal_groups_by_block"] = _check_direction(records, spec, report)
            if report["qualification"] != "qualified" and not allow_partial:
                continue
            if spec.get("direction_reference") and all(v[3] is not None for v in records.values()):
                rounded = [v[3]["native_rounded_realized_rms"] for arm, v in records.items() if arm != "sham"]
                report["initial_realized_rms_relative_spread"] = (max(rounded) / min(rounded) - 1
                                                                if rounded and all(rounded) else None)
        interrupt()
        deltas, measurements = _measure_evaluation(records, records["sham"][0], chosen, first, spec, step)
        measurements.update(solver_step=step, opportunity=opportunity, sigma=sigma)
        if (i == 0 and spec.get("direction_reference") and "reverse" in deltas
                and records[spec["direction_reference"]][2]["settings"]["feature_direction"] == "repel"
                and records["reverse"][2]["settings"]["feature_direction"] == "reverse"
                and records["reverse"][1]["feature_applied"] and records[spec["direction_reference"]][1]["feature_applied"]):
            measurements["signed_decomposition"] = odd_even(deltas[spec["direction_reference"]], deltas["reverse"])
        report["evaluations"].append(measurements)
        if step in spec.get("intermediate_steps", []):
            from .trajectory_review import decode_rgb

            begin = perf_counter()
            LOGGER.info("ModeExplorer report %s: decoding selected actual step %d", name, step)
            imgs = {}
            for arm, (tensors, *_) in records.items():
                interrupt()
                imgs[arm] = preview(decode_rgb(vae, tensors[chosen]), 480)
            intermediate[step] = imgs
            decode_time += perf_counter() - begin
    report["vae_decode_seconds"] = decode_time
    for arm, value in report["arms"].items():
        if arm in selected and len(value["states"]) == len(selected[arm][1]["states"]):
            value["checks"]["tensor_states"] = "verified"
        if value["checks"].get("initial_noisy_input") == "pending_tensor_verification":
            value["checks"]["initial_noisy_input"] = "unavailable_incomplete_case"
    surviving = {arm for arm, value in report["arms"].items() if value["qualification"] == "qualified"}
    if ("sham" not in surviving or len(surviving) < 2 or not report["evaluations"]
            or (not allow_partial and report["qualification"] != "qualified")):
        for field in ("initial_proposal_groups_by_block", "initial_realized_rms_relative_spread"):
            report.pop(field, None)
        report["evaluations"].clear()
        return report, {}
    if report["qualification"] != "qualified":
        _restrict_case(report, surviving)
        report["comparison_status"] = "qualified_subset"
    else:
        report["comparison_status"] = "qualified"
    report["terminal_summary"] = {
        arm: {field: data[field] for field in ("clean_delta", "norm_amplification_from_initial", "cosine_to_initial")}
        for arm, data in report["evaluations"][-1]["arms"].items() if arm != "sham"
    }
    labels = presentation["arms"] if presentation else {arm: arm for arm in spec["arms"]}
    case_label = presentation["name"] if presentation else name
    return report, {
        f"visual/{case_label}/intermediate_step_{step:02d}.jpg":
        encode(panel([(label, images[arm]) for arm, label in labels.items() if arm in surviving]), "JPEG", quality=86)
        for step, images in intermediate.items()
    }


def regression_results(cases, expectations):
    rows = []
    for item in expectations:
        value = cases
        for token in item["path"]:
            if isinstance(value, dict):
                value = value.get(token)
            elif isinstance(value, list) and type(token) is int and 0 <= token < len(value):
                value = value[token]
            else:
                value = None
        expected, tolerance = item["value"], item["abs_tol"]
        rows.append({"name": item["name"], "expected": expected, "abs_tol": tolerance, "measured": value,
                     "status": ("unavailable" if type(value) not in (int, float)
                                else "passed" if abs(value - expected) <= tolerance else "mismatch")})
    return rows


def _summary(report):
    lines = [f"# {report['name']}", "", f"Numerical qualification: {report['qualification']}.",
             "No DiT forwards were executed by this report.", ""]
    for name, case in report["cases"].items():
        lines.extend([f"## {name}", "", f"Numerical qualification: {case['qualification']}."])
        if case["comparison_status"] == "qualified_subset":
            lines.append("Evidence is included for the qualified subset. The complete case gate failed.")
        if case.get("clean_coordinate"):
            lines.append(f"Clean coordinate: `{case['clean_coordinate']}`.")
        for arm, entry in case["arms"].items():
            lines.append(f"- {arm}: {entry['qualification']} ({entry.get('archive_id', 'no selected archive')})"
                         + (f" — {entry['reason']}" if entry.get("reason") else ""))
            if (image := entry.get("final_image")) and image["status"] != "qualified":
                lines.append(f"  Final image: {image['reason']}.")
        lines.append("")
    if report["regression"]:
        lines.extend(["## Independent numerical references", ""])
        for check in report["regression"]:
            lines.append(f"- {check['name']}: {check['status']} (measured {check['measured']}, expected {check['expected']}).")
    lines.extend(["", "Response norms, energy and effective ranks describe this finite sampled set. "
                  "Use the registered image evidence to judge anatomical coherence and meaningful detail.", ""])
    return "\n".join(lines).encode()


def _csv(cases):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("case", "arm", "solver_step", "opportunity", "actual_ordinal", "sigma", "applied",
                     "clean_rms", "clean_l2", "velocity_rms", "solver_input_rms", "amplification", "initial_cosine"))
    for name, data in cases.items():
        for row in data["evaluations"]:
            for arm, m in row["arms"].items():
                writer.writerow((name, arm, row["solver_step"], row["opportunity"], m["actual_ordinal"], row["sigma"], m["applied"],
                                 m.get("clean_delta", {}).get("rms"), m.get("clean_delta", {}).get("l2"),
                                 m["velocity_delta"]["rms"], (m["solver_input_delta"] or {}).get("rms"),
                                 m.get("norm_amplification_from_initial"), m.get("cosine_to_initial")))
    return stream.getvalue().encode()


def write_report(path, report, files, budget, interrupt):
    # report.json is stored at a fixed byte count. Updating its integer size
    # field does not shift the ZIP's layout or depend on compression feedback.
    name = "report.json"
    sizes = {key: len(value) for key, value in sorted(files.items())}
    sizes[name] = 0
    report["inventory"] = sizes
    report["archive_size_bytes"] = 0

    def body():
        return json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()

    reserve = len(body()) + 128
    require(reserve <= MIB, "numerical_json_exceeds_1_mib")
    sizes[name] = reserve

    def pack():
        contents = {**files, name: body().ljust(reserve, b" ")}
        with ZipFile(path, "w", allowZip64=False) as archive:
            for filename, content in sorted(contents.items()):
                interrupt()
                require(not filename.endswith(".safetensors"), "raw_tensors_forbidden")
                info = ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_STORED if filename == name else ZIP_DEFLATED
                archive.writestr(info, content)

    begin = perf_counter()
    pack()
    report["first_zip_write_seconds"] = perf_counter() - begin
    size = path.stat().st_size
    if size > budget:
        return False
    report["archive_size_bytes"] = size
    pack()
    require(path.stat().st_size == size, "archive_size_accounting_mismatch")
    return True


def _blind_review(report, presentation, review_id):
    cases = {}
    for name, labels in presentation.items():
        case = report["cases"][name]
        cases[labels["name"]] = {
            "comparison_status": case["comparison_status"],
            "arms": {label: {"final_image": "available" if case["arms"][arm].get("final_image", {}).get("status")
                            == "qualified" else "unavailable"} for arm, label in labels["arms"].items()},
        }
    return {"schema": "modeexplorer.blind_review/1", "review_id": review_id, "cases": cases,
            "qualification": report["qualification"], "visual_evidence": report["visual_evidence"]}


BLIND_SUMMARY = (
    b"# Blinded visual review\n\nUse only the anonymous case and arm IDs to record ratings. "
    b"The same IDs are used in previews, registered crops and selected intermediate panels. "
    b"Missing images have no score. Lock ratings before opening the local analysis and mapping.\n"
)


def _publish_blind_report(staging, final, folder, report, presentation, review, interrupt):
    private = folder / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    sidecar = private / f"{review['review_id']}.json"
    fd, name = tempfile.mkstemp(prefix="analysis_", suffix=".part", dir=private)
    pending = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"schema": "modeexplorer.blind_review_private/1", "review_id": review["review_id"],
                       "shareable_zip": final.name, "presentation": presentation, "report": report, "review": review},
                      f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        interrupt()
        pending.replace(sidecar)
        try:
            staging.replace(final)
        except BaseException:
            sidecar.unlink(missing_ok=True)
            raise
    finally:
        pending.unlink(missing_ok=True)
    return sidecar


class ModeExplorerResearchReport:
    CATEGORY = "model_patches/ModeExplorer"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "report"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"campaign": ("STRING", {"default": "", "multiline": True})}, "optional": {"vae": ("VAE",)}}

    @classmethod
    def IS_CHANGED(cls, campaign, vae=None):
        return float("nan")

    def report(self, campaign, vae=None):
        import folder_paths
        from comfy.model_management import throw_exception_if_processing_interrupted as interrupt

        started = perf_counter()
        require(len(campaign) <= 128 * 1024, "campaign_configuration_too_large")
        plan = json.loads(campaign)
        _validate_plan(plan)
        presentation = blind_labels(plan)
        require(vae is not None or not any(s.get("intermediate_steps") for s in plan["cases"].values()),
                "selected_intermediates_require_connected_production_vae")
        report = {"schema": SCHEMA, "name": plan["name"], "qualification": "qualified", "cases": {}, "analysis_device": "cpu",
                  "report_dit_calls": 0, "configuration": plan}
        intermediate = {}
        LOGGER.info("ModeExplorer report: qualifying %s on CPU", plan["name"])
        for name, spec in plan["cases"].items():
            case, previews = analyze_case(name, spec, _root(), interrupt, vae,
                                          presentation[name] if presentation else None)
            report["cases"][name] = case
            intermediate.update(previews)
        report["regression"] = regression_results(report["cases"], plan.get("regressions", []))
        if any(c["qualification"] != "qualified" for c in report["cases"].values()) or any(
                r["status"] != "passed" for r in report["regression"]):
            report["qualification"] = "unqualified"
        output = Path(folder_paths.get_output_directory()).resolve()
        LOGGER.info("ModeExplorer report: assembling final-image evidence")
        visuals, panels = make_visuals(report["cases"], plan, output, interrupt, presentation)
        report["visual_evidence"] = "complete" if all(
            arm.get("final_image", {}).get("status") == "qualified" for case in report["cases"].values()
            for arm in case["arms"].values()) else "partial_or_unavailable"
        base = {**visuals, **intermediate}
        if presentation:
            review_id = f"blind_review_{token_hex(12)}"
            upload = _blind_review(report, presentation, review_id)
            base["SUMMARY.md"] = BLIND_SUMMARY
        else:
            review_id = plan["name"]
            upload = report
            base.update({"SUMMARY.md": _summary(report), "evaluations.csv": _csv(report["cases"])})
        folder = output / "modeexplorer_research_reports"
        folder.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f"{review_id}_", suffix=".part", dir=folder)
        os.close(fd)
        staging = Path(temporary)
        budget = plan.get("budget_mib", 15) * MIB
        try:
            LOGGER.info("ModeExplorer report: writing ZIP with a %d MiB budget", budget // MIB)
            for width, quality in ((560, 90), (480, 86), (384, 82)):
                upload["preview"] = {"width_px": width, "jpeg_quality": quality, "crops": "lossless_native_pixels"}
                upload["archive_budget_bytes"] = budget
                report["report_generation_seconds_before_zip"] = perf_counter() - started
                if write_report(staging, upload, {**base, **jpeg_evidence(panels, width, quality)}, budget, interrupt):
                    final = staging.with_suffix(".zip")
                    if presentation:
                        local = _publish_blind_report(staging, final, folder, report, presentation, upload, interrupt)
                        LOGGER.info("ModeExplorer private analysis and arm mapping: %s", local)
                    else:
                        staging.replace(final)
                    LOGGER.info("ModeExplorer report complete: %s, %d bytes, %s, %.2fs",
                                final, upload["archive_size_bytes"], report["qualification"], perf_counter() - started)
                    return ()
            raise ValueError("Lossless evidence exceeds the report budget. Narrow the report's fixed crop policy or increase its budget.")
        finally:
            staging.unlink(missing_ok=True)
