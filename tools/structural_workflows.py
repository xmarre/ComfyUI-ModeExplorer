"""Validate the discovery contract shared by public examples and the private pack."""

import argparse
import hashlib
import json
from zipfile import ZipFile

from mode_explorer.nodes import ModeExplorer, ModeExplorerMechanistic

CASES = {"detail": (896275366471460, .03), "anatomy": (354937070410882, .05)}
FACTORIAL_TARGETS = {"factorial_injection": (-1, ""), "factorial_block5": (5, ""),
                     "factorial_downstream": (-1, "downstream")}
ARMS = ("native", "native_repeat", "sham", "active", *FACTORIAL_TARGETS,
        *(f"{arm}_first" for arm in FACTORIAL_TARGETS))
PLANNER = "Krea2TurboFlowAlignedRegenerate"



def factorial_controls(arm):
    evaluation = 0 if arm.endswith("_first") else 1
    block, layers = FACTORIAL_TARGETS[arm.removesuffix("_first")]
    return {
        "projection_block": block, "projection_layers": layers, "evaluation": evaluation,
        "auxiliary_budget": 5 + evaluation, "capture_vram_mib": 16384 if layers else 12288,
    }

def widget_fields(cls):
    return [name for inputs in cls.INPUT_TYPES().values() for name, spec in inputs.items()
            if len(spec) > 1 and "default" in spec[1]]


# Linked widgets still occupy serialized positions. Seed control and the
# wildcard chooser are frontend-only widgets, not server input names.
SERIALIZED_FIELDS = {
    PLANNER: (
        "seed control_after_generate planner_megapixels coarse_lock_strength mid_lock_strength "
        "steps composition_lock_sigma max_stage_upscale stage_megapixels cfg sampler_name scheduler "
        "er_sde_solver_type er_sde_max_stage er_sde_eta er_sde_s_noise sa_eta sa_sde_start_percent "
        "sa_sde_end_percent sa_s_noise sa_predictor_order sa_corrector_order sa_simple_order_2 "
        "sa_spectrum_forecasts sa_planner_spectrum_forecasts regeneration_mode img2img_start_sigma "
        "handoff_diagnostics planner_fidelity_strength planner_detail_strength latent_upscale_method "
        "progressive_transition_sigmas spectral_tolerance stage_planning highres_min_step_fraction "
        "patch_lattice_correction"
    ).split(),
    "ImpactWildcardProcessor": ["wildcard_text", "populated_text", "mode", "seed",
                               "control_after_generate", "Select to add Wildcard"],
    "SpectrumApplyKrea2": (
        "enabled preset degree ridge warmup tail window growth max_history storage budget_mb debug "
        "bootstrap extrapolation_guard_enabled max_extrapolation_span_ratio sa_pece_forecast_policy "
        "extrapolation_guard_scope"
    ).split(),
    "PrimitiveInt": ["value", "control_after_generate"],
    "PrimitiveBoolean": ["value"],
    "ComfySwitchNode": ["switch"],
    "SaveImage": ["filename_prefix"],
    "VAELoader": ["vae_name"],
    "ModeExplorerTrajectoryReviewKrea2": ["review_plan"],
}


def validate_widget_serialization(graph):
    """Reject conflicting widget encodings and redundant embedded workflow snapshots."""
    require("workflow" not in graph.get("extensions", {}), "stale embedded workflow snapshot")
    nodes = {str(node["id"]) for node in graph["nodes"]}
    for scope in (graph, graph.get("extensions", {})):
        for key in ("seed_widgets", "widget_idx_map"):
            require(set(scope.get(key, {})) <= nodes, f"stale {key} node identifiers")
    for node in graph["nodes"]:
        if "widgets_values_named" not in node:
            continue
        named = node["widgets_values_named"]
        values = node.get("widgets_values")
        kind = node["type"]
        require(isinstance(named, dict) and isinstance(values, list),
                f"{kind}: invalid named widget representation")
        if kind == "ModeExplorerKrea2":
            names = _feature_widget_fields(values)
        elif kind == "ModeExplorerMechanisticKrea2":
            names = widget_fields(ModeExplorerMechanistic)
        else:
            # Archived third-party nodes without a pinned schema use the
            # original exporter insertion order as a strict pack contract.
            names = SERIALIZED_FIELDS.get(kind, list(named))
        require(len(names) == len(values) and dict(zip(names, values)) == named,
                f"{kind}: conflicting or incomplete named widgets")



def _feature_widget_fields(values):
    names = widget_fields(ModeExplorer)
    # Historical Phase I–III archives predate the two trailing ablation controls.
    if isinstance(values, list) and len(values) == len(names) - 2:
        return names[:-2]
    return names

def widgets(node, cls):
    entries = node["widgets_values"]
    names = _feature_widget_fields(entries) if cls is ModeExplorer else widget_fields(cls)
    return dict(zip(names, entries, strict=True))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def validate(graph, case, arm, *, private=False):
    """Check an imported LiteGraph workflow without reading private input values into logs."""
    validate_widget_serialization(graph)
    seed, strength = CASES[case]
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {item[0]: item for item in graph["links"]}
    require(len(nodes) == len(graph["nodes"]) and len(links) == len(graph["links"]), "duplicate graph identifiers")
    for lid, src, source_slot, dst, dest_slot, kind in links.values():
        output = nodes[src]["outputs"][source_slot]
        port = nodes[dst]["inputs"][dest_slot]
        compatible = all(value == kind or (private and value == "*") for value in (output["type"], port["type"]))
        require(lid in output["links"] and port["link"] == lid and compatible,
                "invalid graph wiring")
    for node in nodes.values():
        for slot, port in enumerate(node["inputs"]):
            lid = port.get("link")
            if lid is not None and lid != -1:
                require(links[lid][3:5] == [node["id"], slot], "dangling graph input")
        for slot, output in enumerate(node["outputs"]):
            for lid in output.get("links") or []:
                require(links[lid][1:3] == [node["id"], slot], "dangling graph output")

    def find(kind):
        node, = (n for n in nodes.values() if n["type"] == kind)
        return node

    def linked(node, name):
        port = next((p for p in node["inputs"] if p["name"] == name), {})
        link = links.get(port.get("link"))
        return (nodes[link[1]], link[2]) if link else (None, None)

    def fixed_integer(planner, name, position):
        source, _ = linked(planner, name)
        if source is None:
            return planner["widgets_values"][position]
        require(source["type"] == "PrimitiveInt" and source["widgets_values"][1] == "fixed",
                f"{name} must come from a fixed PrimitiveInt")
        return source["widgets_values"][0]

    planner = find(PLANNER)
    values = planner["widgets_values"]
    require(fixed_integer(planner, "seed", 0) == seed and values[1] == "fixed", "wrong case seed or seed control")
    require(fixed_integer(planner, "steps", 5) == 13, "wrong sampler step count")
    require([values[i] for i in (9, 10, 11, 12, 13, 14, 15, 25)] ==
            [1.0, "er_sde", "simple", "ER-SDE", 1, 1.0, 1.0, "planner_final_upscale_only"], "wrong planner route")
    decoder, _ = linked(find("SaveImage"), "images")
    require(decoder["type"] == "VAEDecode" and linked(decoder, "samples") == (planner, 1),
            "output must decode the native planner latent")

    spectrum = find("SpectrumApplyKrea2")
    active_model, _ = linked(planner, "model")
    if private:
        require(active_model["type"] == "ModelPreviewOverrideKJ", "planner model wrapper changed")
        active_model, _ = linked(active_model, "model")
    require(active_model == spectrum, "planner must receive the guarded Spectrum model")
    sx = spectrum["widgets_values"]
    require(sx[0] is True and sx[1:7] == ["turbo", 1, .01, 2, 1, 1] and sx[13] is True
            and sx[14] == .5 and sx[16] == "planner_only", "wrong guarded Spectrum recipe")
    factorial = arm.startswith("factorial_")
    if factorial:
        study = find("ModeExplorerMechanisticKrea2")
        spec = widgets(study, ModeExplorerMechanistic)
        require(all(spec[key] == value for key, value in factorial_controls(arm).items())
                and spec["projection_study"] == "activation_factorial" and spec["projection_activation_check"]
                and spec["treatment"] == "export_only" and spec["feature_block"] == 1
                and spec["feature_strength"] == strength and spec["export"] and not spec["all_actual"]
                and spec["reference_start_sigma"] == .9999683499336243
                and spec["analysis_device"] == "cuda", "wrong factorial controls")
        require(linked(spectrum, "model")[0] == study, "factorial must precede Spectrum")

    if not factorial or private:
        feature = find("ModeExplorerKrea2")
        cfg = widgets(feature, ModeExplorer)
        scheduled = arm in ("first_only", "second_only")
        expected_mode = "feature_cosine" if arm in ("sham", "active") or scheduled else "native"
        if scheduled:
            require(cfg.get("feature_application") == arm and cfg.get("ablation_route", "").startswith("AA"),
                    "temporal arm has no guarded application schedule")
        require(cfg["enabled"] and cfg["feature"] == expected_mode and cfg["sham"] == (arm == "sham")
                and cfg["strength"] == strength and cfg["fidelity"] == .70 and cfg["feature_block"] == 1
                and cfg["feature_grid"] == "4" and not cfg["adaptive"] and not cfg["full_spectrum"]
                and cfg["reference_budget"] == 2 and cfg["min_sigma"] == .82 and not cfg["initialization"]
                and cfg["reference_source"] == "bank" and cfg["reference_bank_size"] == 8
                and cfg["reference_trajectories"] == 1 and cfg["reference_selection"] == "all"
                and cfg["reference_continuation"] == "first_estimate" and cfg["feature_update"] == "bounded"
                and cfg["feature_direction"] == "repel"
                and cfg["trajectory_label"] == ("" if factorial else f"{case}_iv_{arm}" if scheduled else f"{case}_{arm}"), "wrong feature or sham recipe")

        if factorial:
            require(linked(study, "model")[0] == feature, "factorial feature base is miswired")
        elif private:
            switch, _ = linked(spectrum, "model")
            enabled, _ = linked(switch, "switch")
            require(linked(switch, "on_true")[0] == feature and enabled["widgets_values"] == [True],
                    "private model switch does not select the configured feature branch")
        else:
            require(linked(spectrum, "model")[0] == feature, "feature must precede Spectrum")

    output = None
    if private:
        wildcard = find("ImpactWildcardProcessor")
        require(linked(wildcard, "seed")[0] == linked(planner, "seed")[0] and linked(wildcard, "wildcard_text")[0] is None,
                "wildcard seed or conditioning source changed")
        raw, resolved, mode = wildcard["widgets_values"][:3]
        require(mode == ("reproduce" if case == "detail" else "populate"), "wrong wildcard replay mode")
        output = {name: hashlib.sha256(val.encode()).hexdigest()
                  for name, val in (("raw_sha256", raw), ("resolved_sha256", resolved))}
    return output


def validate_pack(path):
    with ZipFile(path) as pack:
        provenance = json.loads(pack.read("workflow-provenance.json"))
        graphs = [name for name in pack.namelist() if name.endswith(".json") and "/" in name]
        seen, ids = set(), set()
        for file in graphs:
            _, case, name = file.split("/")
            arm = name.removesuffix(".json").split("_", 1)[1]
            key = (case, arm)
            require(key not in seen, "duplicate case arm")
            seen.add(key)
            graph = json.loads(pack.read(file))
            require(graph["id"] not in ids, "duplicate ComfyUI workflow identity")
            ids.add(graph["id"])
            require(validate(graph, case, arm, private=True) == provenance[case], "archive conditioning fingerprint mismatch")
        require(seen == {(case, arm) for case in CASES for arm in ARMS}, "missing discovery workflow variants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack")
    args = parser.parse_args()
    validate_pack(args.pack)
    print(f"Validated all {len(CASES) * len(ARMS)} private discovery workflows.")
