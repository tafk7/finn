from __future__ import annotations

from dataclasses import replace
import runpy
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.analysis.tensor_value_summary import initializer_value_summaries
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.dataflow.model.maps import CoordinateSet, RectangularDomain
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops.mvau.computation import AccumulationMode
from finn.dataflow.ops.mvau.designs.dot_product import WeightSupply
from finn.dataflow.ops.mvau.selected import (
    WEIGHT_KEY,
    construct_mvau_snapshot,
)
from finn.dataflow.ops.replay.selected import construct_replay_snapshot
from finn.dataflow.ops.selected import (
    SELECTED_DECLARATION_ID,
    SELECTED_DECLARATION_VERSION,
    ComputationOwner,
    ConstructionIdentity,
    ConstructionInputs,
    EncodedSourceSemantics,
    GraphNodeBinding,
    GraphSlotKind,
    GraphSlotRef,
    OwnerKind,
    PositionRelation,
    SelectedGraphDeclaration,
    SourceDirection,
    SourceOperandKey,
    SourceProvenance,
    SourceValueBinding,
    SourceValueRef,
    build_selected_snapshot,
)
from finn.dataflow.ops.selected_verification import (
    frozen_initializer_for_source,
    verify_normalized_selected_snapshot,
)


ROOT = Path(__file__).resolve().parents[3]
REPLAY_CASES = runpy.run_path(str(ROOT / "tests/dataflow/ops/replay/test_selected_replay.py"))
MVAU_CASES = runpy.run_path(str(ROOT / "tests/dataflow/ops/mvau/test_selected_mvau.py"))


def _all_tensor_names(model: ModelWrapper) -> set[str]:
    names = {value for node in model.graph.node for value in (*node.input, *node.output) if value}
    names.update(item.name for item in model.graph.input)
    names.update(item.name for item in model.graph.output)
    names.update(item.name for item in model.graph.value_info)
    names.update(item.name for item in model.graph.initializer)
    names.update(item.tensor_name for item in model.graph.quantization_annotation)
    return names


def _renamed(snapshot, *, reorder: bool = True):
    model = snapshot.model_copy()
    declaration = snapshot.declaration
    original_graph_name = model.graph.name

    records = tuple(declaration.graph_nodes)
    if reorder:
        ordered = tuple(reversed(records))
        nodes = [model.graph.node[item.index] for item in ordered]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        records = tuple(GraphNodeBinding(item.node_id, index) for index, item in enumerate(ordered))

    for index, node in enumerate(model.graph.node):
        node.name = f"display_node_{index}"
    value_names = {
        name: f"renamed_value_{index}"
        for index, name in enumerate(sorted(_all_tensor_names(model)))
    }
    for old, new in value_names.items():
        model.rename_tensor(old, new)
    model.graph.name = "renamed_selected_graph"

    def repair_slot(slot: GraphSlotRef) -> GraphSlotRef:
        if slot.kind in (GraphSlotKind.GRAPH_INPUT, GraphSlotKind.GRAPH_OUTPUT):
            return replace(slot, owner=model.graph.name)
        if slot.kind is GraphSlotKind.INITIALIZER:
            return replace(slot, owner=value_names[slot.owner])
        return slot

    declaration = replace(
        declaration,
        graph_nodes=records,
        interface_bindings=tuple(
            replace(
                item,
                graph_value=value_names[item.graph_value],
                anchors=tuple(repair_slot(slot) for slot in item.anchors),
            )
            for item in declaration.interface_bindings
        ),
        source_bindings=tuple(
            replace(
                item,
                graph_value=value_names[item.graph_value],
                anchors=tuple(repair_slot(slot) for slot in item.anchors),
            )
            for item in declaration.source_bindings
        ),
        supplies=tuple(
            replace(item, root_graph_value=value_names[item.root_graph_value])
            for item in declaration.supplies
        ),
    )
    assert original_graph_name != model.graph.name
    return build_selected_snapshot(model, declaration)


def _assert_mismatch(actual, expected) -> None:
    findings = verify_normalized_selected_snapshot(
        actual,
        expected,
        finding_code="selected-test-mismatch",
        path="selected.test",
        message="selected graph differs",
    )
    assert len(findings) == 1
    assert findings[0].code == "selected-test-mismatch"


def test_real_replay_accepts_total_node_value_renaming_and_reordering() -> None:
    expected = construct_replay_snapshot(REPLAY_CASES["_facts"](), ConstructionInputs())
    actual = _renamed(expected)
    assert not verify_normalized_selected_snapshot(
        actual,
        expected,
        finding_code="selected-replay-mismatch",
        path="selected.replay",
        message="Replay graph differs",
    )


@pytest.mark.parametrize("mode", tuple(AccumulationMode))
@pytest.mark.parametrize("supply", tuple(WeightSupply))
def test_real_mvau_profiles_and_supplies_accept_total_renaming(mode, supply) -> None:
    facts, inputs, _activation, _weight, _expected = MVAU_CASES["_facts"](mode, supply)
    expected = construct_mvau_snapshot(facts, inputs)
    actual = _renamed(expected)
    assert not verify_normalized_selected_snapshot(
        actual,
        expected,
        finding_code="selected-mvau-mismatch",
        path="selected.mvau",
        message="MVAU graph differs",
    )
    frozen = frozen_initializer_for_source(actual, WEIGHT_KEY)
    assert (frozen is None) is (supply is WeightSupply.EXTERNAL)
    if frozen is not None:
        source = next(item for item in facts.source.operands if item.key == WEIGHT_KEY)
        assert frozen.summary.content_digest == source.initializer_content_digest


def test_real_graph_operator_attribute_constant_and_order_corruptions_refuse() -> None:
    facts, inputs, _activation, _weight, _expected = MVAU_CASES["_facts"](
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    expected = construct_mvau_snapshot(facts, inputs)

    operator = expected.model_copy()
    next(item for item in operator.graph.node if item.name == "compute.matmul").op_type = "Add"
    _assert_mismatch(build_selected_snapshot(operator, expected.declaration), expected)

    attribute = expected.model_copy()
    transpose = next(
        item for item in attribute.graph.node if item.name == "compute.weight.transpose"
    )
    del transpose.attribute[:]
    transpose.attribute.extend((helper.make_attribute("perm", (1, 0, 2)),))
    _assert_mismatch(build_selected_snapshot(attribute, expected.declaration), expected)

    constant = expected.model_copy()
    constant.set_initializer("shape_Y", np.asarray((1, 8), dtype=np.int64))
    _assert_mismatch(build_selected_snapshot(constant, expected.declaration), expected)

    order = expected.model_copy()
    matmul = next(item for item in order.graph.node if item.name == "compute.matmul")
    matmul.input[0], matmul.input[1] = matmul.input[1], matmul.input[0]
    _assert_mismatch(build_selected_snapshot(order, expected.declaration), expected)

    annotation = expected.model_copy()
    activation_annotation = next(
        item for item in annotation.graph.quantization_annotation if item.tensor_name == "X"
    )
    activation_annotation.quant_parameter_tensor_names.add(key="review_extra", value="literal")
    _assert_mismatch(build_selected_snapshot(annotation, expected.declaration), expected)

    opset = expected.model_copy()
    opset.model.opset_import[0].version -= 1
    _assert_mismatch(build_selected_snapshot(opset, expected.declaration), expected)

    binding = expected.declaration.source_bindings[0]
    malformed = replace(
        expected.declaration,
        source_bindings=(
            replace(binding, graph_value="XR"),
            *expected.declaration.source_bindings[1:],
        ),
    )
    _assert_mismatch(build_selected_snapshot(expected.model_copy(), malformed), expected)


def _minimal_source(*, two_inputs: bool = False, digest: str | None = None):
    operands = [
        SourceValueRef(
            SourceOperandKey("left", SourceDirection.INPUT, 0),
            (2,),
            TensorProto.FLOAT,
            "INT8",
            digest,
        )
    ]
    if two_inputs:
        operands.append(
            SourceValueRef(
                SourceOperandKey("right", SourceDirection.INPUT, 1),
                (2,),
                TensorProto.FLOAT,
                "INT8",
                digest,
            )
        )
    operands.append(
        SourceValueRef(
            SourceOperandKey("result", SourceDirection.OUTPUT, 0),
            (2,),
            TensorProto.FLOAT,
            "INT8",
            None,
        )
    )
    return SourceProvenance.create(
        family="test.verification",
        family_version="1",
        schema_version=1,
        problem_fingerprint="problem",
        scope_id="scope",
        operands=operands,
        semantics=EncodedSourceSemantics("test.verification", 1, {"operation": "sum"}),
    )


def _sum_snapshot(
    inputs: tuple[str, ...],
    constants: dict[str, np.ndarray],
    *,
    source: SourceProvenance | None = None,
    source_bindings: tuple[SourceValueBinding, ...] | None = None,
):
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Sum", list(inputs), ["Y"], name="display")],
                "selected",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])]
                if "X" in inputs
                else [],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    for name, value in constants.items():
        model.set_initializer(name, value)
    for name in {*inputs, "Y"}:
        model.set_tensor_datatype(name, DataType["INT8"])
    domain = RectangularDomain((2,))
    source = source or _minimal_source()
    source_bindings = source_bindings or (
        SourceValueBinding(
            source.operands[0].key,
            "X",
            PositionRelation.direct(domain, domain),
            (GraphSlotRef(GraphSlotKind.GRAPH_INPUT, "selected", 0),),
        ),
        SourceValueBinding(
            source.operands[-1].key,
            "Y",
            PositionRelation.direct(domain, domain),
            (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
        ),
    )
    declaration = SelectedGraphDeclaration(
        SELECTED_DECLARATION_ID,
        SELECTED_DECLARATION_VERSION,
        "",
        ConstructionIdentity("test.verification", "1"),
        source,
        (),
        (GraphNodeBinding("sum", 0),),
        (),
        source_bindings,
        (),
        (ComputationOwner(OwnerKind.REGION, "compute", ("sum",)),),
    )
    return build_selected_snapshot(model, declaration)


def test_parameter_copies_and_commutative_initializer_last_are_normalized() -> None:
    first = np.asarray((1.0, 2.0), dtype=np.float32)
    second = np.asarray((3.0, 4.0), dtype=np.float32)
    expected = _sum_snapshot(("C1", "X", "C2"), {"C1": first, "C2": second})

    reordered = _sum_snapshot(("X", "C1", "C2"), {"C1": first, "C2": second})
    assert not verify_normalized_selected_snapshot(
        reordered,
        expected,
        finding_code="selected-test-mismatch",
        path="selected.test",
        message="selected graph differs",
    )

    copied = _sum_snapshot(
        ("X", "C1", "C2_copy"),
        {"C1": first, "C2_copy": second},
    )
    assert not verify_normalized_selected_snapshot(
        copied,
        expected,
        finding_code="selected-test-mismatch",
        path="selected.test",
        message="selected graph differs",
    )

    swapped = _sum_snapshot(("X", "C2", "C1"), {"C1": first, "C2": second})
    _assert_mismatch(swapped, expected)

    extra = copied.model_copy()
    extra.set_initializer("unused", second)
    _assert_mismatch(build_selected_snapshot(extra, copied.declaration), expected)


def test_bounded_cleanup_may_remove_only_preexisting_unused_and_static_facts() -> None:
    value = np.asarray((1.0, 2.0), dtype=np.float32)
    clean = _sum_snapshot(("X", "C"), {"C": value})
    canonical = clean.model_copy()
    canonical.set_initializer("unused", value)
    canonical.graph.value_info.append(
        helper.make_tensor_value_info("unused_value", TensorProto.FLOAT, [2])
    )
    static_input = next(item for item in canonical.graph.value_info if item.name == "C")
    canonical.graph.input.append(static_input)
    kept = [item for item in canonical.graph.value_info if item.name != "C"]
    del canonical.graph.value_info[:]
    canonical.graph.value_info.extend(kept)
    with_unused = build_selected_snapshot(canonical, clean.declaration)
    assert not verify_normalized_selected_snapshot(
        clean,
        with_unused,
        finding_code="selected-test-mismatch",
        path="selected.test",
        message="selected graph differs",
    )

    added = clean.model_copy()
    added.set_initializer("new_unused", value)
    _assert_mismatch(build_selected_snapshot(added, clean.declaration), clean)


def test_equal_initializer_content_does_not_merge_distinct_source_identities() -> None:
    values = np.asarray((1.0, 2.0), dtype=np.float32)
    probe = ModelWrapper(helper.make_model(helper.make_graph([], "digest", [], [])))
    probe.set_initializer("value", values)
    digest = initializer_value_summaries(probe)["value"].content_digest
    source = _minimal_source(two_inputs=True, digest=digest)
    domain = RectangularDomain((2,))
    left, right, output = (item.key for item in source.operands)
    bindings = (
        SourceValueBinding(
            left,
            "A",
            PositionRelation.direct(domain, domain),
            (GraphSlotRef(GraphSlotKind.INITIALIZER, "A", None),),
        ),
        SourceValueBinding(
            right,
            "B",
            PositionRelation.direct(domain, domain),
            (GraphSlotRef(GraphSlotKind.INITIALIZER, "B", None),),
        ),
        SourceValueBinding(
            output,
            "Y",
            PositionRelation.direct(domain, domain),
            (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
        ),
    )
    expected = _sum_snapshot(
        ("A", "B"), {"A": values, "B": values}, source=source, source_bindings=bindings
    )
    swapped_bindings = (
        replace(bindings[0], source=right),
        replace(bindings[1], source=left),
        bindings[2],
    )
    actual = _sum_snapshot(
        ("A", "B"),
        {"A": values, "B": values},
        source=source,
        source_bindings=swapped_bindings,
    )
    _assert_mismatch(actual, expected)
    assert frozen_initializer_for_source(expected, left) is not None
    assert frozen_initializer_for_source(expected, right) is not None

    changed = expected.model_copy()
    changed.set_initializer("A", np.asarray((9.0, 9.0), dtype=np.float32))
    changed_snapshot = build_selected_snapshot(changed, expected.declaration)
    assert frozen_initializer_for_source(changed_snapshot, left) is None


def test_million_fold_replay_verification_never_enters_map_expansion(monkeypatch) -> None:
    facts = REPLAY_CASES["_facts"](source_shape=(2, 3), folds=1_048_576, simd=3)
    expected = construct_replay_snapshot(facts, ConstructionInputs())
    actual = _renamed(expected)

    def refuse(*_args, **_kwargs):
        raise AssertionError("normalized selected verification entered map expansion")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)
    assert not verify_normalized_selected_snapshot(
        actual,
        expected,
        finding_code="selected-replay-mismatch",
        path="selected.replay",
        message="Replay graph differs",
    )
