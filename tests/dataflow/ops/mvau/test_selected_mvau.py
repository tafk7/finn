from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from qonnx.core.datatype import DataType

from dataflow.ops.mvau.test_batch_interleaved import _interleaved_operation, _mvau_model
from dataflow.ops.test_dataflow_op import Build, _configured_mvau, _unbound
from finn.dataflow._engine import Absent, Decided
from finn.dataflow.analysis.integer_dot import (
    DotProductPremise,
    FixedWeightPremise,
    IntegerRange,
    InvocationScope,
    RuntimeWeightPromise,
    check_integer_dot_product_support,
    decode_dot_product_premise,
    encode_dot_product_premise,
)
from finn.dataflow.designs.design import SelectedGraph
from finn.dataflow.kernels.dotp_axi import BatchInterleavedDotpAxiKernel, DotpAxiKernel
from finn.dataflow.kernels.memstream import MemstreamKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.model.maps import (
    CoordinateSet,
    IdentityCoordinateMap,
    RectangularDomain,
)
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops.mvau.computation import AccumulationMode, ActivationMode
from finn.dataflow.ops.mvau.designs.batch_interleaved import BatchInterleavedDesign
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.dot_product import DotProductDesign, WeightSupply
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.mvau.numerics import (
    ACTIVATION_IDENTITY,
    WEIGHT_IDENTITY,
    integer_graph_profile_fingerprint,
    integer_type,
)
from finn.dataflow.ops.mvau.selected import (
    ACTIVATION_KEY,
    MVAU_CONSTRUCTION_FAMILY,
    MVAU_CONSTRUCTION_VERSION,
    MVAU_SELECTED_CONSTRUCTION,
    OUTPUT_KEY,
    WEIGHT_KEY,
    MvauSourceSemantics,
    construct_mvau_snapshot,
    derive_mvau_facts,
    encode_mvau_source_semantics,
)
from finn.dataflow.ops.selected_registry import DEFAULT_SELECTED_CONSTRUCTIONS
from finn.dataflow.ops.selected import (
    ConstructionIdentity,
    ConstructionInputs,
    ConstructionRegistry,
    GraphNodeBinding,
    RecordedChoice,
    RelationKind,
    SelectedGraphError,
    SourceProvenance,
    SourceValueRef,
    build_selected_snapshot,
    decode_selected_graph,
)
from finn.dataflow.ops.tensor_summary import FrozenInitializer, encode_frozen_initializer


def _case(mode: AccumulationMode):
    if mode is AccumulationMode.INTEGER:
        activation = np.asarray([[1, -2, 3, 0, 2, -1], [2, 1, -1, 3, 0, 1]], dtype=np.float32)
        weight = np.asarray(
            [
                [1, 2, -1, 0],
                [0, -1, 2, 1],
                [2, 0, 1, -2],
                [-1, 1, 0, 2],
                [3, -2, 1, 1],
                [1, 0, -3, 2],
            ],
            dtype=np.float32,
        )
        return activation, weight, activation @ weight, "INT8", "INT8"
    if mode is AccumulationMode.XNOR_POPCOUNT:
        activation = np.asarray([[0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0]], dtype=np.float32)
        weight = np.asarray(
            [
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        expected = np.sum(activation[:, :, None] == weight[None, :, :], axis=1).astype(np.float32)
        return activation, weight, expected, "BINARY", "BINARY"
    activation = np.asarray([[-1, 1, -1, 1, 1, -1], [1, 1, -1, -1, 1, -1]], dtype=np.float32)
    weight = np.asarray(
        [
            [-1, 1, 1, -1],
            [1, 1, -1, -1],
            [-1, 1, -1, 1],
            [1, -1, -1, 1],
            [1, -1, 1, -1],
            [-1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    expected = np.sum(activation[:, :, None] == weight[None, :, :], axis=1).astype(np.float32)
    return activation, weight, expected, "BIPOLAR", "BIPOLAR"


def _facts(
    mode: AccumulationMode,
    supply: WeightSupply,
    *,
    activation_override=None,
    weight_override=None,
    pe=2,
    simd=3,
):
    activation, weight, expected, activation_type, weight_type = _case(mode)
    if activation_override is not None:
        activation = np.asarray(activation_override, dtype=np.float32)
        if activation.min() < -128 or activation.max() > 127:
            activation_type = "INT16"
    if weight_override is not None:
        weight = np.asarray(weight_override, dtype=np.float32)
        if weight.min() < -128 or weight.max() > 127:
            weight_type = "INT16"
    if activation_override is not None or weight_override is not None:
        assert mode is AccumulationMode.INTEGER
        expected = activation @ weight
    frozen = FrozenInitializer.from_tensor_proto(numpy_helper.from_array(weight, name="weight"))
    integer_evidence: tuple[object | None, object | None, object | None, object | None] = (
        None,
        None,
        None,
        None,
    )
    if mode is AccumulationMode.INTEGER:
        activation_dtype = integer_type(DataType[activation_type])
        weight_dtype = integer_type(DataType[weight_type])
        result_dtype = integer_type(DataType["INT32"])
        premise = DotProductPremise(
            activation_dtype.value_range,
            activation_dtype,
            "FLOAT32",
            tuple(activation.shape),
            ACTIVATION_IDENTITY,
            FixedWeightPremise(
                tuple(int(item) for item in weight.reshape(-1)),
                frozen.summary.content_digest,
                "FLOAT32",
            ),
            weight_dtype,
            "FLOAT32",
            tuple(weight.shape),
            WEIGHT_IDENTITY,
            result_dtype,
            result_dtype,
            tuple(expected.shape),
            InvocationScope("scope"),
        )
        support = check_integer_dot_product_support(
            premise=premise,
            selected_internal_bits=32,
            selected_output_carrier="INT32",
            target_max_accumulator_bits=64,
        )
        assert support.supported and support.support is not None
        integer_evidence = (
            integer_graph_profile_fingerprint(support.support),
            "INT32",
            encode_dot_product_premise(premise),
            encode_frozen_initializer(frozen),
        )
    semantics = MvauSourceSemantics(
        mode,
        ActivationMode.NONE,
        "INT32",
        "INT32",
        None,
        *integer_evidence,
    )
    source = SourceProvenance.create(
        family="finn.dataflow.mvau",
        family_version="1",
        schema_version=5,
        problem_fingerprint="problem",
        scope_id="scope",
        operands=(
            SourceValueRef(
                ACTIVATION_KEY,
                activation.shape,
                TensorProto.FLOAT,
                activation_type,
                None,
            ),
            SourceValueRef(
                WEIGHT_KEY,
                weight.shape,
                TensorProto.FLOAT,
                weight_type,
                frozen.summary.content_digest,
            ),
            SourceValueRef(
                OUTPUT_KEY,
                expected.shape,
                TensorProto.FLOAT,
                "INT32",
                None,
            ),
        ),
        semantics=encode_mvau_source_semantics(semantics),
    )
    compute = "dotp_axi_embedded" if supply is WeightSupply.EMBEDDED else "dotp_axi"
    choices = (
        RecordedChoice("design.case", "dot_product"),
        RecordedChoice("design.dot_product.pe", pe),
        RecordedChoice("design.dot_product.simd", simd),
        RecordedChoice("design.dot_product.weight_supply", supply, supply.value),
        RecordedChoice("design.dot_product.compute.kernel", compute),
    )
    facts = derive_mvau_facts(
        ConstructionIdentity(
            MVAU_CONSTRUCTION_FAMILY,
            MVAU_CONSTRUCTION_VERSION,
            "canonical",
        ),
        source,
        semantics,
        choices,
    )
    inputs = (
        ConstructionInputs(((WEIGHT_KEY, frozen),))
        if supply is not WeightSupply.EXTERNAL or mode is AccumulationMode.INTEGER
        else ConstructionInputs()
    )
    return facts, inputs, activation, weight, expected


@pytest.mark.parametrize("mode", tuple(AccumulationMode))
@pytest.mark.parametrize("supply", tuple(WeightSupply))
def test_selected_mvau_profiles_and_weight_modes(mode, supply) -> None:
    facts, inputs, activation, weight, expected = _facts(mode, supply)
    snapshot = construct_mvau_snapshot(facts, inputs)
    decoded = decode_selected_graph(snapshot, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    feeds = {"X": activation}
    if supply is WeightSupply.EXTERNAL:
        feeds["W_source"] = weight
    actual_xr, actual = ReferenceEvaluator(snapshot.model_copy().model).run(["XR", "Y"], feeds)
    assert np.array_equal(actual_xr, np.repeat(activation, 2, axis=0))
    assert np.array_equal(actual, expected)

    replay = decoded.network.node("replay").region
    compute = decoded.network.node("compute").region
    assert replay.output_interface("activation_out").port.operand.shape == (4, 6)
    assert compute.input("XR").operand.shape == (4, 6)
    edge = next(item for item in decoded.network.edges if item.id == "activation_replay")
    assert isinstance(edge.sinks[0].position_map.coordinate_map, IdentityCoordinateMap)
    if supply is WeightSupply.EXTERNAL:
        assert snapshot.model_copy().get_initializer("W_source") is None
        weight_binding = next(
            item for item in decoded.declaration.source_bindings if item.source == WEIGHT_KEY
        )
        assert weight_binding.graph_value == "W"
        assert weight_binding.relation.kind is RelationKind.TRANSPOSE_2D
        assert {item.id for item in decoded.network.boundaries} == {
            "activation",
            "weight",
            "output",
        }
    elif supply is WeightSupply.EMBEDDED:
        assert {node.id for node in decoded.network.nodes} == {"replay", "compute"}
        assert {item.id for item in decoded.network.boundaries} == {
            "activation",
            "output",
        }
    else:
        assert {node.id for node in decoded.network.nodes} == {
            "replay",
            "compute",
            "memory",
        }
        weight_edge = next(
            item for item in decoded.network.edges if item.id == "weight_supply_edge"
        )
        assert isinstance(weight_edge.sinks[0].position_map.coordinate_map, IdentityCoordinateMap)


@pytest.mark.parametrize("supply", tuple(WeightSupply))
def test_bound_mvau_routes_through_the_design_construction_hook(supply) -> None:
    _model, operation = _configured_mvau(supply=supply, pe=2, simd=2)
    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    decoded = decode_selected_graph(answer.value, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    assert decoded.network == operation.network.value


def test_generic_lowering_interprets_declared_initializer_inputs() -> None:
    _model, operation = _configured_mvau(supply=WeightSupply.EMBEDDED, pe=2, simd=2)
    declaration = DotProductDesign.selected_graph
    assert declaration is not None
    with patch.object(
        DotProductDesign,
        "selected_graph",
        SelectedGraph(replace(declaration.construction, initializer_inputs=())),
    ):
        answer = operation.selected_snapshot
    assert isinstance(answer, Absent)
    assert "initializer_keys" in answer.findings[0].message


def test_physical_compute_choice_does_not_change_the_selected_graph() -> None:
    _plain_model, plain = _configured_mvau(pumped=False)
    _pumped_model, pumped = _configured_mvau(pumped=True)
    plain_graph = plain.selected_snapshot
    pumped_graph = pumped.selected_snapshot
    assert isinstance(plain_graph, Decided)
    assert isinstance(pumped_graph, Decided)
    assert plain_graph.value.graph_digest == pumped_graph.value.graph_digest
    assert (
        decode_selected_graph(plain_graph.value).selection_facts.selection_fingerprint
        == decode_selected_graph(pumped_graph.value).selection_facts.selection_fingerprint
    )


def test_unresolved_physical_choice_does_not_block_selected_publication() -> None:
    model = _mvau_model()
    chosen = _unbound(model, "mvau0").bind(model, Build()).design.select("dot_product").root
    design = chosen.design.alternative("dot_product")
    chosen = design.assign(DotProductDesign.weight_supply, WeightSupply.EXTERNAL).root
    chosen = chosen.design.alternative("dot_product").compute.select("dotp_axi").root
    for declaration, value in (
        (WeightedDotProductDesign.pe, 2),
        (WeightedDotProductDesign.simd, 2),
    ):
        chosen = chosen.design.alternative("dot_product").assign(declaration, value).root

    assert isinstance(chosen.network, Decided)
    assert isinstance(chosen.selected_snapshot, Decided)
    assert "design.dot_product.compute.dotp_axi.compute_pumping" not in {
        item.path for item in chosen.selected_snapshot.value.declaration.choices
    }


@pytest.mark.parametrize("supply", (WeightSupply.EMBEDDED, WeightSupply.DECOUPLED))
def test_bound_local_mvau_uses_the_frozen_weight_after_source_mutation(supply, monkeypatch) -> None:
    model, operation = _configured_mvau(supply=supply, pe=2, simd=2)
    original = operation.source.operand("weight").initializer_value
    assert original is not None
    model.set_initializer("weight", np.full((8, 4), 99, dtype=np.float32))
    monkeypatch.setattr(
        model,
        "get_initializer",
        lambda *_args, **_kwargs: pytest.fail("selected construction reread the live model"),
    )

    answer = operation.selected_snapshot
    assert isinstance(answer, Decided)
    assert np.array_equal(
        answer.value.model_copy().get_initializer("W_source"), original.array_copy()
    )


def test_source_boundary_validation_rejects_a_missing_weight_transpose() -> None:
    facts, inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    model = snapshot.model_copy()
    node_index = next(
        item.index
        for item in snapshot.declaration.graph_nodes
        if item.node_id == "weight.to_region"
    )
    node = model.graph.node[node_index]
    node.CopyFrom(helper.make_node("Identity", ["W_integer"], ["W"], name=node.name))
    graph_nodes = tuple(
        GraphNodeBinding(item.node_id, item.index) if item.node_id == "weight.to_region" else item
        for item in snapshot.declaration.graph_nodes
    )
    broken = build_selected_snapshot(model, replace(snapshot.declaration, graph_nodes=graph_nodes))

    with pytest.raises(SelectedGraphError) as error:
        decode_selected_graph(broken, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    assert error.value.code == "selected.source.boundary_shape"


def test_mvau_verifier_rejects_a_wrong_weight_fold_permutation() -> None:
    facts, inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    model = snapshot.model_copy()
    node = next(item for item in model.graph.node if item.name == "compute.weight.transpose")
    del node.attribute[:]
    node.attribute.extend((helper.make_attribute("perm", (1, 0, 2)),))
    broken = build_selected_snapshot(model, snapshot.declaration)

    neutral = replace(MVAU_SELECTED_CONSTRUCTION, verify=lambda _snapshot, _facts: ())
    decode_selected_graph(
        broken,
        constructions=ConstructionRegistry(
            {(neutral.family, neutral.version): neutral},
            {
                (
                    neutral.family,
                    neutral.version,
                ): DEFAULT_SELECTED_CONSTRUCTIONS.resolve_choice_schema(facts.construction)
            },
        ),
    )
    with pytest.raises(SelectedGraphError) as error:
        decode_selected_graph(broken, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
    assert error.value.code == "selected.construction.verification"


def test_weight_modes_enforce_their_exact_construction_input_sets() -> None:
    external_facts, external_input, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    embedded_facts, local, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EMBEDDED
    )
    construct_mvau_snapshot(external_facts, external_input)
    with pytest.raises(SelectedGraphError) as external_missing:
        construct_mvau_snapshot(external_facts, ConstructionInputs())
    assert external_missing.value.code == "selected.construction.initializer_keys"
    construct_mvau_snapshot(embedded_facts, local)
    with pytest.raises(SelectedGraphError) as missing:
        construct_mvau_snapshot(embedded_facts, ConstructionInputs())
    assert missing.value.code == "selected.construction.initializer_keys"


def test_selected_dot_product_refuses_fused_or_mislabeled_profiles() -> None:
    facts, _inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    fused = MvauSourceSemantics(
        AccumulationMode.INTEGER,
        ActivationMode.MULTITHRESHOLD,
        "INT32",
        "INT8",
        0,
    )
    with pytest.raises(ValueError, match="no fused activation"):
        derive_mvau_facts(
            ConstructionIdentity(
                MVAU_CONSTRUCTION_FAMILY,
                MVAU_CONSTRUCTION_VERSION,
                "canonical",
            ),
            facts.source,
            fused,
            facts.choices,
        )
    with pytest.raises(ValueError, match="identity"):
        derive_mvau_facts(
            replace(facts.construction, form="unknown"),
            facts.source,
            facts.source_semantics,
            facts.choices,
        )


def test_selected_mvau_requires_semantically_consistent_choice_records() -> None:
    facts, _inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL
    )
    with pytest.raises(ValueError, match="requires compute candidate"):
        derive_mvau_facts(
            facts.construction,
            facts.source,
            facts.source_semantics,
            tuple(
                replace(item, value="dotp_axi_embedded")
                if item.path == "design.dot_product.compute.kernel"
                else item
                for item in facts.choices
            ),
        )


@pytest.mark.parametrize("change", ("result_type", "carrier", "activation_range"))
def test_detached_selected_integer_admission_revalidates_complete_premise(change: str) -> None:
    facts, inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER,
        WeightSupply.EMBEDDED,
        activation_override=np.full((1, 8), 127, dtype=np.float32),
        weight_override=np.ones((8, 4), dtype=np.float32),
        pe=2,
        simd=2,
    )
    semantics = facts.source_semantics
    source = facts.source
    if change == "result_type":
        semantics = replace(semantics, accumulator_datatype="INT8", output_datatype="INT8")
        source = replace(
            source,
            operands=tuple(
                replace(value, logical_datatype="INT8") if value.key == OUTPUT_KEY else value
                for value in source.operands
            ),
        )
    elif change == "carrier":
        source = replace(
            source,
            operands=tuple(
                replace(value, carrier_dtype=TensorProto.INT32)
                if value.key == ACTIVATION_KEY
                else value
                for value in source.operands
            ),
        )
    else:
        premise = decode_dot_product_premise(semantics.integer_premise)
        semantics = replace(
            semantics,
            integer_premise=encode_dot_product_premise(
                replace(premise, activation_range=IntegerRange(0, 1))
            ),
        )
    source = SourceProvenance.create(
        family=source.family,
        family_version=source.family_version,
        schema_version=source.schema_version,
        problem_fingerprint=source.problem_fingerprint,
        scope_id=source.scope_id,
        operands=source.operands,
        semantics=encode_mvau_source_semantics(semantics),
    )
    with pytest.raises(ValueError, match="selected integer MVAU"):
        derive_mvau_facts(facts.construction, source, semantics, facts.choices)

    if change == "result_type":
        snapshot = construct_mvau_snapshot(facts, inputs)
        broken = build_selected_snapshot(
            snapshot.model_copy(),
            replace(snapshot.declaration, source=source),
        )
        with pytest.raises(SelectedGraphError):
            decode_selected_graph(broken)


def test_fixed_value_facts_are_authenticated_against_initializer_payload() -> None:
    facts, inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER,
        WeightSupply.EMBEDDED,
        activation_override=np.full((1, 8), 127, dtype=np.float32),
        weight_override=np.ones((8, 4), dtype=np.float32),
        pe=2,
        simd=2,
    )
    premise = decode_dot_product_premise(facts.source_semantics.integer_premise)
    assert isinstance(premise.weights, FixedWeightPremise)
    narrow = integer_type(DataType["INT8"])
    false_premise = replace(
        premise,
        weights=replace(premise.weights, values=(0,) * len(premise.weights.values)),
        accumulator_type=narrow,
        result_type=narrow,
    )
    report = check_integer_dot_product_support(
        premise=false_premise,
        selected_internal_bits=8,
        selected_output_carrier="INT32",
        target_max_accumulator_bits=64,
    )
    assert report.supported and report.support is not None
    semantics = replace(
        facts.source_semantics,
        accumulator_datatype="INT8",
        output_datatype="INT8",
        integer_premise=encode_dot_product_premise(false_premise),
        integer_support_fingerprint=integer_graph_profile_fingerprint(report.support),
    )
    source = SourceProvenance.create(
        family=facts.source.family,
        family_version=facts.source.family_version,
        schema_version=facts.source.schema_version,
        problem_fingerprint=facts.source.problem_fingerprint,
        scope_id=facts.source.scope_id,
        operands=tuple(
            replace(value, logical_datatype="INT8") if value.key == OUTPUT_KEY else value
            for value in facts.source.operands
        ),
        semantics=encode_mvau_source_semantics(semantics),
    )
    with pytest.raises(ValueError, match="fixed values differ from authenticated payload"):
        derive_mvau_facts(facts.construction, source, semantics, facts.choices)

    snapshot = construct_mvau_snapshot(facts, inputs)
    broken = build_selected_snapshot(
        snapshot.model_copy(), replace(snapshot.declaration, source=source)
    )
    with pytest.raises(SelectedGraphError, match="fixed values differ"):
        decode_selected_graph(broken)


def test_selected_xnor_refuses_nonbinary_logical_operands() -> None:
    model = _mvau_model()
    model.graph.node[0].attribute.extend((helper.make_attribute("binaryXnorMode", 1),))
    _model, operation = _configured_mvau(model, supply=WeightSupply.EXTERNAL)
    assert isinstance(operation.network, Decided)
    answer = operation.selected_snapshot
    assert isinstance(answer, Absent)
    assert "requires BINARY" in answer.findings[0].message


def test_selected_integer_refuses_implicit_bipolar_popcount_operands() -> None:
    facts, _inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.BIPOLAR_POPCOUNT, WeightSupply.EXTERNAL
    )
    integer = MvauSourceSemantics(
        AccumulationMode.INTEGER,
        ActivationMode.NONE,
        "INT32",
        "INT32",
        None,
        "test-premise",
        "INT32",
    )
    with pytest.raises(ValueError, match="require bipolar popcount"):
        derive_mvau_facts(
            facts.construction,
            facts.source,
            integer,
            facts.choices,
        )


def test_selected_mvau_refuses_missing_input_logical_annotations() -> None:
    model = _mvau_model()
    output_annotations = [
        item for item in model.graph.quantization_annotation if item.tensor_name == "output"
    ]
    del model.graph.quantization_annotation[:]
    model.graph.quantization_annotation.extend(output_annotations)
    operation = MvauDataflowOp(model.graph.node[0]).bind(model, Build())
    assessment = operation.assess(MvauDataflowOp.source_accepts)
    assert assessment.verdict is False
    assert {
        finding.code
        for answer in assessment.answers.values()
        for finding in getattr(answer, "findings", ())
    } == {"integer-logical-datatype-annotation"}


def test_integer_selected_mvau_handles_r_not_equal_to_f_and_odd_height() -> None:
    activation = np.arange(18, dtype=np.float32).reshape(3, 6) - 7
    weight = np.arange(30, dtype=np.float32).reshape(6, 5) % 7 - 3
    facts, inputs, activation, weight, expected = _facts(
        AccumulationMode.INTEGER,
        WeightSupply.EXTERNAL,
        activation_override=activation,
        weight_override=weight,
        pe=1,
        simd=1,
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    actual_xr, actual = ReferenceEvaluator(snapshot.model_copy().model).run(
        ["XR", "Y"], {"X": activation, "W_source": weight}
    )
    assert np.array_equal(actual_xr, np.repeat(activation, 5, axis=0))
    assert np.array_equal(actual, expected)


def test_integer_selected_mvau_uses_int64_matmul_and_checked_int32_result() -> None:
    activation = np.asarray([[4097]], dtype=np.float32)
    weight = np.asarray([[4097]], dtype=np.float32)
    facts, inputs, _activation, _weight, _expected = _facts(
        AccumulationMode.INTEGER,
        WeightSupply.EXTERNAL,
        activation_override=activation,
        weight_override=weight,
        pe=1,
        simd=1,
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    model = snapshot.model_copy()
    nodes = {node.name: node for node in model.graph.node}
    assert nodes["source.activation.cast"].attribute[0].i == TensorProto.INT64
    assert nodes["weight.to_integer"].attribute[0].i == TensorProto.INT64
    assert nodes["compute.output.cast"].attribute[0].i == TensorProto.INT32
    (actual,) = ReferenceEvaluator(model.model).run(["Y"], {"X": activation, "W_source": weight})
    assert actual.dtype == np.int32
    assert actual.item() == 16_785_409


def test_selected_mvau_keeps_explicit_replay_at_one_fold() -> None:
    facts, inputs, activation, weight, expected = _facts(
        AccumulationMode.INTEGER, WeightSupply.EXTERNAL, pe=4, simd=3
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    actual_xr, actual = ReferenceEvaluator(snapshot.model_copy().model).run(
        ["XR", "Y"], {"X": activation, "W_source": weight}
    )
    assert np.array_equal(actual_xr, activation)
    assert np.array_equal(actual, expected)
    assert {item.op_type for item in snapshot.model_copy().graph.node} >= {
        "Unsqueeze",
        "Expand",
        "Reshape",
    }


def test_selected_mvau_flattens_and_restores_leading_dimensions() -> None:
    activation = np.arange(24, dtype=np.float32).reshape(2, 2, 6) - 5
    weight = np.arange(24, dtype=np.float32).reshape(6, 4) % 5 - 2
    facts, inputs, activation, weight, expected = _facts(
        AccumulationMode.INTEGER,
        WeightSupply.EXTERNAL,
        activation_override=activation,
        weight_override=weight,
        pe=2,
        simd=3,
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    actual_xr, actual = ReferenceEvaluator(snapshot.model_copy().model).run(
        ["XR", "Y_source"], {"X_source": activation, "W_source": weight}
    )
    assert np.array_equal(actual_xr, np.repeat(activation.reshape(4, 6), 2, axis=0))
    assert np.array_equal(actual, expected)


def test_selected_mvau_preserves_rank_one_source_shapes() -> None:
    activation = np.arange(6, dtype=np.float32) - 2
    weight = np.arange(24, dtype=np.float32).reshape(6, 4) % 5 - 2
    facts, inputs, activation, weight, expected = _facts(
        AccumulationMode.INTEGER,
        WeightSupply.EXTERNAL,
        activation_override=activation,
        weight_override=weight,
        pe=2,
        simd=3,
    )
    snapshot = construct_mvau_snapshot(facts, inputs)
    actual_xr, actual = ReferenceEvaluator(snapshot.model_copy().model).run(
        ["XR", "Y_source"], {"X_source": activation, "W_source": weight}
    )
    assert np.array_equal(actual_xr, np.repeat(activation.reshape(1, 6), 2, axis=0))
    assert np.array_equal(actual, expected)


def test_batch_interleaved_keeps_source_support_and_refuses_selected_construction() -> None:
    model = _mvau_model()
    operation = _interleaved_operation(model, interleave=2)
    assert isinstance(operation.network, Decided)
    answer = operation.selected_snapshot
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"selected-graph-unsupported-design"}


def test_step_4c_versions_change_only_the_migrated_semantics() -> None:
    assert MvauDataflowOp.family_version == "1"
    assert MvauDataflowOp.schema_version == 5
    assert MVAU_CONSTRUCTION_VERSION == "2"
    assert DotProductDesign.version == "3"
    assert ReplayBufferKernel.version == "2"
    assert ReplayBufferKernel.region.version == "2"
    assert DotpAxiKernel.version == "2"
    assert DotpAxiKernel.region.version == "2"
    assert BatchInterleavedDesign.version == "1"
    assert BatchInterleavedDotpAxiKernel.version == "1"
    assert BatchInterleavedDotpAxiKernel.region.version == "1"
    assert MemstreamKernel.version == "1"
    assert MemstreamKernel.region.version == "1"


def test_large_external_mvau_construction_and_decode_remain_compact(
    monkeypatch,
) -> None:
    height = 1_048_576
    activation_dtype = integer_type(DataType["INT8"])
    weight_dtype = integer_type(DataType["INT8"])
    result_dtype = integer_type(DataType["INT32"])
    premise = DotProductPremise(
        activation_dtype.value_range,
        activation_dtype,
        "FLOAT32",
        (2, 3),
        ACTIVATION_IDENTITY,
        RuntimeWeightPromise(
            IntegerRange(-128, 127),
            3 * height,
            WEIGHT_IDENTITY,
            (InvocationScope("large-scope"),),
            "large-runtime-weights",
            "FLOAT32",
        ),
        weight_dtype,
        "FLOAT32",
        (3, height),
        WEIGHT_IDENTITY,
        result_dtype,
        result_dtype,
        (2, height),
        InvocationScope("large-scope"),
    )
    support = check_integer_dot_product_support(
        premise=premise,
        selected_internal_bits=32,
        selected_output_carrier="INT32",
        target_max_accumulator_bits=64,
    )
    assert support.supported and support.support is not None
    semantics = MvauSourceSemantics(
        AccumulationMode.INTEGER,
        ActivationMode.NONE,
        "INT32",
        "INT32",
        None,
        integer_graph_profile_fingerprint(support.support),
        "INT32",
        encode_dot_product_premise(premise),
        None,
    )
    source = SourceProvenance.create(
        family="finn.dataflow.mvau",
        family_version="1",
        schema_version=5,
        problem_fingerprint="large-problem",
        scope_id="large-scope",
        operands=(
            SourceValueRef(ACTIVATION_KEY, (2, 3), TensorProto.FLOAT, "INT8", None),
            SourceValueRef(
                WEIGHT_KEY,
                (3, height),
                TensorProto.FLOAT,
                "INT8",
                None,
            ),
            SourceValueRef(
                OUTPUT_KEY,
                (2, height),
                TensorProto.FLOAT,
                "INT32",
                None,
            ),
        ),
        semantics=encode_mvau_source_semantics(semantics),
    )
    choices = (
        RecordedChoice("design.case", "dot_product"),
        RecordedChoice("design.dot_product.pe", 1),
        RecordedChoice("design.dot_product.simd", 3),
        RecordedChoice("design.dot_product.weight_supply", WeightSupply.EXTERNAL, "external"),
        RecordedChoice("design.dot_product.compute.kernel", "dotp_axi"),
    )
    facts = derive_mvau_facts(
        ConstructionIdentity(
            MVAU_CONSTRUCTION_FAMILY,
            MVAU_CONSTRUCTION_VERSION,
            "canonical",
        ),
        source,
        semantics,
        choices,
    )

    def refuse(*_args, **_kwargs):
        raise AssertionError("large selected MVAU entered an expansion iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)

    snapshot = construct_mvau_snapshot(facts, ConstructionInputs())
    decoded = decode_selected_graph(snapshot)
    replay = decoded.network.node("replay").region
    compute = decoded.network.node("compute").region
    assert (
        replay.output_interface("activation_out").port.operand.position_domain.cardinality
        == 2 * height * 3
    )
    assert compute.input("W").requirements.required_position_set.cardinality == height * 3
    assert len(snapshot.model_bytes) < 32_768


def test_all_profile_supply_artifacts_decode_source_free_in_a_fresh_process(tmp_path) -> None:
    paths = []
    for supply in WeightSupply:
        for mode in AccumulationMode:
            facts, inputs, _activation, _weight, _expected = _facts(mode, supply)
            path = tmp_path / f"{supply.value}-{mode.value}.onnx"
            path.write_bytes(construct_mvau_snapshot(facts, inputs).model_bytes)
            paths.append(str(path))
    script = f"""
from unittest.mock import patch
from finn.dataflow.ops.selected_registry import DEFAULT_SELECTED_CONSTRUCTIONS
from finn.dataflow.ops.selected import reconstruct_selected_graph

def forbidden(*args, **kwargs):
    raise AssertionError('detached decode touched source, occurrence, engine, or build state')

with patch('finn.dataflow.ops.source.read_source_node', forbidden), \\
     patch('finn.dataflow.space.occurrence.start_occurrence', forbidden), \\
     patch('finn.dataflow._engine.Engine.start', forbidden), \\
     patch('finn.dataflow.ops.base._build_value', forbidden):
    decoded = [
        reconstruct_selected_graph(path, constructions=DEFAULT_SELECTED_CONSTRUCTIONS)
        for path in {paths!r}
    ]
assert len(decoded) == 9
assert all(
    item.network.node('compute').region.output_interface('output').port.operand.shape == (2, 4)
    for item in decoded
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
