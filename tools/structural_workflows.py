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

def widgets(node, cls):
    fields = [name for inputs in cls.INPUT_TYPES().values() for name, spec in inputs.items()
              if len(spec) > 1 and "default" in spec[1]]
    return dict(zip(fields, node["widgets_values"], strict=True))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def validate(graph, case, arm, *, private=False):
    """Check an imported LiteGraph workflow without reading private input values into logs."""
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
        expected_mode = "feature_cosine" if arm in ("sham", "active") else "native"
        require(cfg["enabled"] and cfg["feature"] == expected_mode and cfg["sham"] == (arm == "sham")
                and cfg["strength"] == strength and cfg["fidelity"] == .70 and cfg["feature_block"] == 1
                and cfg["feature_grid"] == "4" and not cfg["adaptive"] and not cfg["full_spectrum"]
                and cfg["reference_budget"] == 2 and cfg["min_sigma"] == .82 and not cfg["initialization"]
                and cfg["reference_source"] == "bank" and cfg["reference_bank_size"] == 8
                and cfg["reference_trajectories"] == 1 and cfg["reference_selection"] == "all"
                and cfg["reference_continuation"] == "first_estimate" and cfg["feature_update"] == "bounded"
                and cfg["feature_direction"] == "repel"
                and cfg["trajectory_label"] == ("" if factorial else f"{case}_{arm}"), "wrong feature or sham recipe")

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
