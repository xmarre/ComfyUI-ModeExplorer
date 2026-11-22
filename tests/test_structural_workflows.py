import copy

import pytest

from mode_explorer.nodes import ModeExplorer
from tools.structural_workflows import PLANNER, SERIALIZED_FIELDS, validate_widget_serialization, widget_fields


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
