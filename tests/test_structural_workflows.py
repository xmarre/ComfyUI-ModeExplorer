import copy
import json
from pathlib import Path

import pytest

from mode_explorer.nodes import ModeExplorer
from tools.structural_workflows import (
    PLANNER, SERIALIZED_FIELDS, validate, validate_widget_serialization, widget_fields, widgets,
)


def feature_graph():
    fields = widget_fields(ModeExplorer)
    values = {name: spec[1]["default"] for inputs in ModeExplorer.INPUT_TYPES().values()
              for name, spec in inputs.items() if len(spec) > 1 and "default" in spec[1]}
    values.update(strength=.05, diagnostics=False, trajectory_label="anatomy_active")
    return {"nodes": [{"id": 1, "type": "ModeExplorerKrea2",
                       "widgets_values": [values[name] for name in fields],
                       "widgets_values_named": values}]}


@pytest.mark.parametrize("field, stale", [("strength", .025), ("diagnostics", True),
                                             ("trajectory_label", "")])
def test_named_feature_conflict_rejected(field, stale):
    graph = feature_graph()
    validate_widget_serialization(graph)
    graph["nodes"][0]["widgets_values_named"][field] = stale
    with pytest.raises(ValueError, match="conflicting or incomplete named widgets"):
        validate_widget_serialization(graph)


def test_absent_trajectory_label_and_named_order():
    graph = feature_graph()
    named = graph["nodes"][0]["widgets_values_named"]
    graph["nodes"][0]["widgets_values_named"] = dict(reversed(list(named.items())))
    validate_widget_serialization(graph)
    del graph["nodes"][0]["widgets_values_named"]["trajectory_label"]
    with pytest.raises(ValueError, match="conflicting or incomplete named widgets"):
        validate_widget_serialization(graph)


@pytest.mark.parametrize("kind, field, expected, stale", [
    (PLANNER, "regeneration_mode", "planner_final_upscale_only", "trajectory_auto_native"),
    ("ImpactWildcardProcessor", "mode", "reproduce", "populate"),
    ("PrimitiveInt", "control_after_generate", "fixed", "randomize"),
])
def test_named_companion_conflict(kind, field, expected, stale):
    fields = SERIALIZED_FIELDS[kind]
    named = dict.fromkeys(fields, 0)
    named[field] = expected
    graph = {"nodes": [{"id": 1, "type": kind,
                        "widgets_values": list(named.values()), "widgets_values_named": copy.copy(named)}]}
    validate_widget_serialization(graph)
    graph["nodes"][0]["widgets_values_named"][field] = stale
    with pytest.raises(ValueError, match="conflicting or incomplete named widgets"):
        validate_widget_serialization(graph)


def test_obsolete_workflow_and_seed_metadata_rejected():
    graph = feature_graph()
    graph["extensions"] = {"workflow": copy.deepcopy(graph)}
    with pytest.raises(ValueError, match="stale embedded workflow"):
        validate_widget_serialization(graph)
    graph["extensions"] = {"widget_idx_map": {"2": {"seed": 0}}}
    with pytest.raises(ValueError, match="stale widget_idx_map"):
        validate_widget_serialization(graph)
    graph["extensions"] = {}
    graph["seed_widgets"] = {"2": 0}
    with pytest.raises(ValueError, match="stale seed_widgets"):
        validate_widget_serialization(graph)


def test_mode_is_appended_and_legacy_widget_encodings_round_trip():
    assert widget_fields(ModeExplorer)[-5:] == [
        "trajectory_label", "trajectory_roi", "feature_application", "ablation_route", "mode",
    ]
    for count in (27, 29, 31, 32):
        graph = feature_graph()
        node = graph["nodes"][0]
        del node["widgets_values"][count:]
        node["widgets_values_named"] = dict(zip(widget_fields(ModeExplorer), node["widgets_values"]))
        round_trip = json.loads(json.dumps(graph))
        validate_widget_serialization(round_trip)
        settings = widgets(round_trip["nodes"][0], ModeExplorer)
        assert settings["strength"] == .05
        assert ("mode" in settings) == (count == 32)
        if count == 32:
            assert settings["mode"] == "Original"
            round_trip["nodes"][0]["widgets_values_named"]["mode"] = "Custom"
            with pytest.raises(ValueError, match="conflicting or incomplete named widgets"):
                validate_widget_serialization(round_trip)


def test_custom_mode_controls_are_visible_in_the_standard_widget_group():
    optional = ModeExplorer.INPUT_TYPES()["optional"]
    for name in ("mode", "feature_block", "feature_application"):
        assert not optional[name][1].get("advanced")
    assert optional["trajectory_label"][1]["advanced"]


@pytest.mark.parametrize("filename,mode", [
    ("turbo.json", "Early Structural (B1)"),
    ("scale_locked_spectrum.json", "Early Structural (B2)"),
])
def test_production_example_widget_configuration(filename, mode):
    path = Path(__file__).resolve().parents[1] / "examples" / filename
    graph = json.loads(path.read_text())
    validate_widget_serialization(graph)
    feature = next(node for node in graph["nodes"] if node["type"] == "ModeExplorerKrea2")
    settings = widgets(feature, ModeExplorer)
    assert settings["mode"] == mode
    assert settings["strength"] == .03 and settings["fidelity"] == .70
    assert not (settings["diagnostics"] or settings["trajectory_label"] or settings["ablation_route"])
    assert any(link[1] == feature["id"] and link[-1] == "MODEL" for link in graph["links"])


def test_structural_campaign_requires_historical_mode_resolution():
    path = Path(__file__).resolve().parents[1] / "examples" / "structural_discovery_native.json"
    graph = json.loads(path.read_text())
    validate(graph, "detail", "native")
    feature = next(node for node in graph["nodes"] if node["type"] == "ModeExplorerKrea2")
    feature["widgets_values"].extend(["all", "", "Early Structural (B1)"])
    with pytest.raises(ValueError, match="structural campaign requires Original mode"):
        validate(graph, "detail", "native")
