# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Selected declaration-v2, projection, binding, and origin regressions."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow._engine import Finding, FindingKind, QualifiedPath
from finn.dataflow.model import (
    BeatSequence,
    BoundaryContract,
    DataflowNetwork,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NetworkNode,
    Operand,
    OutputInterface,
    Port,
    RectangularDomain,
    RegionEndpoint,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.ops.selected import (
    SELECTED_DECLARATION_ID,
    SELECTED_DECLARATION_VERSION,
    ComputationOwner,
    ConstructionIdentity,
    ConstructionRegistry,
    EncodedSourceSemantics,
    GraphNodeBinding,
    GraphSlotKind,
    GraphSlotRef,
    InterfaceBinding,
    InterfaceDirection,
    OwnerKind,
    PositionRelation,
    QualifiedInterfaceRef,
    RecordedChoice,
    SelectedConstruction,
    SelectedGraphDeclaration,
    SelectedGraphError,
    SelectionFacts,
    SourceDirection,
    SourceOperandKey,
    SourceProvenance,
    SourceValueBinding,
    SourceValueRef,
    build_selected_snapshot,
    decode_selected_declaration,
    decode_selected_graph,
    encode_selected_declaration,
)


def _tensor(name: str, shape: tuple[int, ...]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _sequence() -> BeatSequence:
    return BeatSequence(1, (((0,),), ((1,),)))


def _requirements() -> ScheduledInputRequirements:
    return ScheduledInputRequirements(
        {((0,), (0,)): 1, ((1,), (1,)): 1},
        schedule_domain=RectangularDomain((2,)),
        position_domain=RectangularDomain((2,)),
    )


def _availability() -> ScheduledOutputAvailability:
    return ScheduledOutputAvailability(
        {(0,): (0,), (1,): (1,)},
        position_domain=RectangularDomain((2,)),
        schedule_domain=RectangularDomain((2,)),
    )


def _region() -> DataflowRegion:
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("i", 2),)),
        (
            InputInterface(
                Port("in", Operand("X", DataType["INT8"], (2,)), _sequence()),
                _requirements(),
            ),
        ),
        (
            OutputInterface(
                Port("out", Operand("Y", DataType["INT8"], (2,)), _sequence()),
                _availability(),
            ),
        ),
    )


def _network(node_id: str = "compute") -> DataflowNetwork:
    region = _region()
    return DataflowNetwork(
        (NetworkNode(node_id, region),),
        (),
        (
            BoundaryContract("input", RegionEndpoint(node_id, "in"), _sequence()),
            BoundaryContract("output", RegionEndpoint(node_id, "out"), _sequence()),
        ),
    )


def _source(*, problem: str = "problem", scope: str | None = "scope", schema: int = 1):
    return SourceProvenance.create(
        family="test.selected.identity",
        family_version="1",
        schema_version=schema,
        problem_fingerprint=problem,
        scope_id=scope,
        operands=(
            SourceValueRef(
                SourceOperandKey("activation", SourceDirection.INPUT, 0),
                (2,),
                TensorProto.FLOAT,
                "INT8",
                None,
            ),
            SourceValueRef(
                SourceOperandKey("result", SourceDirection.OUTPUT, 0),
                (2,),
                TensorProto.FLOAT,
                "INT8",
                None,
            ),
        ),
        semantics=EncodedSourceSemantics("test.selected.identity", 1, {"operation": "identity"}),
    )


def _decode_semantics(value: EncodedSourceSemantics) -> str:
    if value.identity != "test.selected.identity" or value.version != 1:
        raise ValueError("unsupported identity semantics")
    if dict(value.payload) != {"operation": "identity"}:  # type: ignore[arg-type]
        raise ValueError("invalid identity semantics")
    return "identity"


def _derive(
    identity: ConstructionIdentity,
    source: SourceProvenance,
    semantics: str,
    choices: tuple[RecordedChoice, ...],
) -> SelectionFacts[str, None]:
    if choices:
        raise ValueError("identity construction has no choices")
    return SelectionFacts(identity, source, semantics, (), None)


def _project(facts: SelectionFacts[str, None]) -> DataflowNetwork:
    if facts.construction.form == "canonical":
        if facts.construction.form_version != 1 or facts.construction.form_arguments:
            raise ValueError("canonical identity form takes no arguments")
        return _network()
    if facts.construction.form != "identity_elided" or facts.construction.form_version != 1:
        raise ValueError("unsupported identity form")
    arguments = dict(facts.construction.form_arguments)
    if set(arguments) != {"region_id"} or arguments["region_id"] not in {"left", "right"}:
        raise ValueError("identity_elided requires one stable region_id")
    surviving = "right" if arguments["region_id"] == "left" else "left"
    return _network(surviving)


def _verify(snapshot, facts):
    model = snapshot.model_copy()
    expected_name = _project(facts).nodes[0].id
    if (
        len(model.graph.node) != 1
        or model.graph.node[0].op_type != "Identity"
        or tuple(model.graph.node[0].input) != ("X",)
        or tuple(model.graph.node[0].output) != ("Y",)
        or model.graph.node[0].name != expected_name
    ):
        return (
            Finding(
                FindingKind.REJECTION,
                "selected-test-computation",
                QualifiedPath("selected.test"),
                "actual graph is not the projected identity computation",
            ),
        )
    return ()


CONSTRUCTION = SelectedConstruction(
    family="test.selected",
    version="1",
    source_semantics_identity="test.selected.identity",
    source_semantics_version=1,
    admitted_forms=("canonical", "identity_elided"),
    choice_paths=(),
    initializer_inputs=(),
    decode_source_semantics=_decode_semantics,
    derive_facts=_derive,
    project=_project,
    construct=lambda _facts, _inputs: pytest.fail("fixture construction is explicit"),
    verify=_verify,
)
REGISTRY = ConstructionRegistry({("test.selected", "1"): CONSTRUCTION})


def _fixture(
    *,
    identity: ConstructionIdentity | None = None,
    source: SourceProvenance | None = None,
):
    construction = identity or ConstructionIdentity("test.selected", "1")
    source = source or _source()
    facts = _derive(construction, source, "identity", ())
    network = _project(facts)
    node_id = network.nodes[0].id
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["X"], ["Y"], name=node_id)],
                "selected",
                [_tensor("X", (2,))],
                [_tensor("Y", (2,))],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_tensor_datatype("X", DataType["INT8"])
    model.set_tensor_datatype("Y", DataType["INT8"])
    domain = RectangularDomain((2,))
    declaration = SelectedGraphDeclaration(
        SELECTED_DECLARATION_ID,
        SELECTED_DECLARATION_VERSION,
        "",
        construction,
        source,
        (),
        (GraphNodeBinding("identity", 0),),
        (
            InterfaceBinding(
                QualifiedInterfaceRef(node_id, InterfaceDirection.INPUT, "X", "in"),
                "X",
                PositionRelation.direct(domain, domain),
                (
                    GraphSlotRef(GraphSlotKind.GRAPH_INPUT, "selected", 0),
                    GraphSlotRef(GraphSlotKind.NODE_INPUT, "identity", 0),
                ),
            ),
            InterfaceBinding(
                QualifiedInterfaceRef(node_id, InterfaceDirection.OUTPUT, "Y", "out"),
                "Y",
                PositionRelation.direct(domain, domain),
                (
                    GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "identity", 0),
                    GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),
                ),
            ),
        ),
        (
            SourceValueBinding(
                source.operands[0].key,
                "X",
                PositionRelation.direct(domain, domain),
                (GraphSlotRef(GraphSlotKind.GRAPH_INPUT, "selected", 0),),
            ),
            SourceValueBinding(
                source.operands[1].key,
                "Y",
                PositionRelation.direct(domain, domain),
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
            ),
        ),
        (),
        (ComputationOwner(OwnerKind.REGION, node_id, ("identity",)),),
    )
    return network, declaration, model


def test_v2_round_trip_derives_network_and_omits_redundant_authorities() -> None:
    network, declaration, model = _fixture()
    snapshot = build_selected_snapshot(model, declaration)
    decoded = decode_selected_graph(snapshot, constructions=REGISTRY)
    raw = json.loads(encode_selected_declaration(snapshot.declaration))

    assert decoded.network == network
    assert set(raw) == {
        "identity",
        "version",
        "graph_digest",
        "construction",
        "source",
        "choices",
        "graph_nodes",
        "interface_bindings",
        "source_bindings",
        "supplies",
        "ownership",
    }
    assert (
        not {
            "decoder_version",
            "selection_fingerprint",
            "design_id",
            "design_version",
            "opsets",
            "regions",
            "network",
            "constants",
        }
        & raw.keys()
    )
    assert set(raw["graph_nodes"][0]) == {"node_id", "index"}
    assert all(
        "value" not in anchor for item in raw["interface_bindings"] for anchor in item["anchors"]
    )
    assert set(raw["ownership"][0]) == {"kind", "owner_id", "node_ids"}
    assert "semantic_fingerprint" not in raw["source"]
    assert "initializer_present" not in raw["source"]["operands"][0]


def test_v1_unknown_fields_duplicate_keys_and_nonfinite_values_refuse() -> None:
    with pytest.raises(SelectedGraphError) as old:
        decode_selected_declaration('{"identity":"finn.dataflow.selected_graph","version":1}')
    assert old.value.code == "selected.declaration.v1_unsupported"

    _network_value, declaration, _model = _fixture()
    raw = json.loads(encode_selected_declaration(declaration))
    raw["extra"] = 1
    with pytest.raises(SelectedGraphError) as unknown:
        decode_selected_declaration(json.dumps(raw))
    assert unknown.value.code == "selected.declaration.fields"

    encoded = encode_selected_declaration(declaration)
    duplicate = encoded[:-1] + ',"version":2}'
    with pytest.raises(SelectedGraphError) as repeated:
        decode_selected_declaration(duplicate)
    assert repeated.value.code == "selected.declaration.invalid"

    with pytest.raises(SelectedGraphError) as nonfinite:
        decode_selected_declaration(encoded.replace('"version":2', '"version":NaN', 1))
    assert nonfinite.value.code == "selected.declaration.invalid"

    with pytest.raises(SelectedGraphError) as nominal:
        decode_selected_declaration(encoded.replace('"version":2', '"version":true', 1))
    assert nominal.value.code == "selected.declaration.type"


def test_actual_graph_and_binding_slots_are_checked_after_projection() -> None:
    _network_value, declaration, model = _fixture()
    model.graph.node[0].op_type = "Neg"
    wrong_graph = build_selected_snapshot(model, declaration)
    with pytest.raises(SelectedGraphError) as computation:
        decode_selected_graph(wrong_graph, constructions=REGISTRY)
    assert computation.value.code == "selected.construction.verification"

    _network_value, declaration, model = _fixture()
    broken = replace(
        declaration.interface_bindings[0],
        anchors=(GraphSlotRef(GraphSlotKind.NODE_INPUT, "identity", 9),),
    )
    snapshot = build_selected_snapshot(
        model,
        replace(
            declaration,
            interface_bindings=(broken, declaration.interface_bindings[1]),
        ),
    )
    with pytest.raises(SelectedGraphError) as binding:
        decode_selected_graph(snapshot, constructions=REGISTRY)
    assert binding.value.code == "selected.binding.slot"


def test_invalid_registered_projection_is_rejected_before_binding_validation() -> None:
    _network_value, declaration, model = _fixture()
    invalid = replace(
        CONSTRUCTION,
        project=lambda _facts: DataflowNetwork(
            (NetworkNode("compute", _region()), NetworkNode("compute", _region())),
            (),
            (),
        ),
    )
    registry = ConstructionRegistry({("test.selected", "1"): invalid})
    with pytest.raises(SelectedGraphError) as error:
        decode_selected_graph(build_selected_snapshot(model, declaration), constructions=registry)
    assert error.value.code.startswith("selected.projection.network.")


def test_origin_survives_but_is_outside_semantic_and_selection_identity() -> None:
    network, declaration, model = _fixture()
    original = build_selected_snapshot(model, declaration)
    changed_source = _source(problem="other-problem", scope="other-scope", schema=99)
    changed = build_selected_snapshot(model, replace(declaration, source=changed_source))

    original_decoded = decode_selected_graph(original, constructions=REGISTRY)
    changed_decoded = decode_selected_graph(changed, constructions=REGISTRY)
    assert changed_decoded.network == network
    assert changed_decoded.declaration.source.problem_fingerprint == "other-problem"
    assert changed_decoded.declaration.source.scope_id == "other-scope"
    assert changed_decoded.declaration.source.schema_version == 99
    assert (
        original_decoded.selection_facts.source.semantic_fingerprint
        == changed_decoded.selection_facts.source.semantic_fingerprint
    )
    assert (
        original_decoded.selection_facts.selection_fingerprint
        == changed_decoded.selection_facts.selection_fingerprint
    )
    assert original.content_digest != changed.content_digest

    raw = json.loads(encode_selected_declaration(declaration))
    for field, invalid in (("native_schema_version", 0), ("scope_id", "")):
        malformed = json.loads(json.dumps(raw))
        malformed["source"]["origin"][field] = invalid
        with pytest.raises(SelectedGraphError) as error:
            decode_selected_declaration(json.dumps(malformed))
        assert error.value.code in {"selected.declaration.invalid", "selected.declaration.type"}


def test_bounded_identity_form_arguments_select_unique_projection_and_identity() -> None:
    left_identity = ConstructionIdentity(
        "test.selected",
        "1",
        "identity_elided",
        1,
        (("region_id", "left"),),
    )
    right_identity = replace(
        left_identity,
        form_arguments=(("region_id", "right"),),
    )
    left_network, left_declaration, left_model = _fixture(identity=left_identity)
    right_network, right_declaration, right_model = _fixture(identity=right_identity)
    left = decode_selected_graph(
        build_selected_snapshot(left_model, left_declaration), constructions=REGISTRY
    )
    right = decode_selected_graph(
        build_selected_snapshot(right_model, right_declaration), constructions=REGISTRY
    )

    assert left_network.nodes[0].id == "right"
    assert right_network.nodes[0].id == "left"
    assert left.network == left_network
    assert right.network == right_network
    assert left.selection_facts.selection_fingerprint != right.selection_facts.selection_fingerprint

    missing = replace(left_identity, form_arguments=())
    _network_value, declaration, model = _fixture(identity=left_identity)
    declaration = replace(declaration, construction=missing)
    with pytest.raises(SelectedGraphError) as error:
        decode_selected_graph(build_selected_snapshot(model, declaration), constructions=REGISTRY)
    assert error.value.code == "selected.projection.invalid"

    wrong_version = replace(left_identity, form_version=99)
    declaration = replace(declaration, construction=wrong_version)
    with pytest.raises(SelectedGraphError) as version:
        decode_selected_graph(build_selected_snapshot(model, declaration), constructions=REGISTRY)
    assert version.value.code == "selected.projection.invalid"


def test_form_argument_inputs_are_frozen_from_caller_owned_containers() -> None:
    argument = {"candidate": ["left"]}
    identity = ConstructionIdentity(
        "test.selected",
        "1",
        "identity_elided",
        1,
        (("region_id", argument),),
    )
    argument["candidate"].append("right")
    stored = identity.form_arguments[0][1]
    assert stored["candidate"] == ("left",)  # type: ignore[index]
