# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""G6: what an MVAU node *means*, as distinct from what is built for it.

Three claims, and they are checked separately because they fail separately:

1. the profile, the threshold operand and the narrowness analysis are read
   from the graph correctly, and disagreements are reported rather than
   assumed away;
2. the numbers are the ones the previous implementation produced -- checked
   against ``numpy.matmul``, ``xnorpopcountmatmul`` and ``multithreshold``
   directly, not against a recorded expectation that could have been wrong
   when it was recorded;
3. a fused-threshold node is a valid *problem* with no applicable Design here,
   which is a different outcome from an invalid node and must not be confused
   with one.
"""

from __future__ import annotations

from typing import Any

import numpy as np  # type: ignore[import-not-found]
import pytest
import qonnx.custom_op.general.xnorpopcount as xnor  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.general.multithreshold import (  # type: ignore[import-not-found]
    multithreshold,
)

from finn.dataflow._engine import Decided
from finn.dataflow.model.declarations import AuthoringError
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.dataflow.ops.mvau.computation import (
    MvauComputationProfile,
    execute_mvau,
    initializer_excludes_minimum,
)
from finn.dataflow.ops.mvau.designs.base import WeightedDotProductDesign
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.schema import Attribute, InputTensor, OutputTensor

from dataflow.ops.test_dataflow_op import Build, _replay_model, _unbound

MATRIX_WIDTH = 8
MATRIX_HEIGHT = 4
REPETITIONS = 2


def _value(name: str, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _model(
    *,
    no_activation: bool = True,
    binary_xnor: bool = False,
    activation_bias: int = 0,
    activation_type: str = "INT8",
    weight_type: str = "INT8",
    output_type: str = "INT32",
    weights: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
    threshold_shape: tuple[int, ...] | None = None,
    weight_initializer: bool = True,
) -> ModelWrapper:
    """One MVAU node, with whatever source semantics the test is about."""

    inputs = ["activation", "weight"]
    if thresholds is not None or threshold_shape is not None:
        inputs.append("threshold")
    node = helper.make_node(
        "MvauDataflowOp",
        inputs,
        ["output"],
        domain=DATAFLOW_DOMAIN,
        name="mvau0",
        noActivation=int(no_activation),
        binaryXnorMode=int(binary_xnor),
        ActVal=activation_bias,
    )
    shape = threshold_shape or (
        () if thresholds is None else tuple(int(extent) for extent in thresholds.shape)
    )
    extra = [_value("threshold", shape)] if shape else []
    graph = helper.make_graph(
        [node],
        "mvau",
        [_value("activation", (REPETITIONS, MATRIX_WIDTH))],
        [_value("output", (REPETITIONS, MATRIX_HEIGHT))],
        value_info=[_value("weight", (MATRIX_WIDTH, MATRIX_HEIGHT)), *extra],
    )
    model = ModelWrapper(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DATAFLOW_DOMAIN, 1)],
        )
    )
    model.set_tensor_datatype("activation", DataType[activation_type])
    model.set_tensor_datatype("weight", DataType[weight_type])
    model.set_tensor_datatype("output", DataType[output_type])
    if weight_initializer:
        model.set_initializer(
            "weight",
            np.zeros((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.float32)
            if weights is None
            else weights.astype(np.float32),
        )
    if shape:
        model.set_tensor_datatype("threshold", DataType["INT32"])
        if thresholds is not None:
            model.set_initializer("threshold", thresholds.astype(np.float32))
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def _bound(model: ModelWrapper) -> MvauDataflowOp:
    operation = _unbound(model, "mvau0").bind(model, Build())
    assert isinstance(operation, MvauDataflowOp)
    return operation


def _findings(answer: Any) -> set[str]:
    return {finding.code for finding in getattr(answer, "findings", ())}


def _accepts(operation: MvauDataflowOp) -> bool:
    """Whether this node's own semantics are consistent."""

    return operation.assess(MvauDataflowOp.source_accepts).verdict is True


def _refusals(operation: MvauDataflowOp) -> set[str]:
    """Every code the operation's own constraints reported."""

    assessment = operation.assess(MvauDataflowOp.source_accepts)
    return {
        finding.code
        for answer in assessment.answers.values()
        for finding in getattr(answer, "findings", ())
    }


# -- the profile ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("no_activation", "binary_xnor", "expected"),
    [
        (True, False, MvauComputationProfile.ACCUMULATOR_INTEGER),
        (True, True, MvauComputationProfile.BIPOLAR_XNOR_ACCUMULATOR),
        (False, False, MvauComputationProfile.FUSED_THRESHOLD),
        # A fused threshold outranks the XNOR mode: the threshold is what
        # decides the output type and the operand list.
        (False, True, MvauComputationProfile.FUSED_THRESHOLD),
    ],
    ids=["integer", "bipolar", "thresholded", "thresholded-bipolar"],
)
def test_the_profile_is_derived_from_the_two_attributes_that_decide_it(
    no_activation: bool, binary_xnor: bool, expected: MvauComputationProfile
) -> None:
    thresholds = None if no_activation else np.zeros((MATRIX_HEIGHT, 1), dtype=np.float32)
    model = _model(no_activation=no_activation, binary_xnor=binary_xnor, thresholds=thresholds)

    assert _bound(model).answer(MvauDataflowOp.profile) == Decided(expected)


# -- the threshold operand ------------------------------------------------------


def test_a_thresholded_node_reads_its_threshold_operand() -> None:
    thresholds = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    operation = _bound(_model(no_activation=False, thresholds=thresholds))

    assert operation.source.has("threshold")
    assert operation.answer(MvauDataflowOp.threshold__present) == Decided(True)
    assert operation.source.operand("threshold").shape == (MATRIX_HEIGHT, 1)


def test_a_plain_node_has_no_threshold_operand_and_that_is_not_a_refusal() -> None:
    operation = _bound(_model())

    assert not operation.source.has("threshold")
    assert operation.answer(MvauDataflowOp.threshold__present) == Decided(False)
    assert _accepts(operation)


def test_a_fused_activation_without_thresholds_is_refused() -> None:
    """Neither half of the disagreement is repairable, so it is a refusal."""

    assert _accepts(_bound(_model(no_activation=True)))

    operation = _bound(_model(no_activation=False))

    assert "mvau-threshold-presence-mismatch" in _refusals(operation)


def test_thresholds_on_a_node_that_fuses_nothing_are_refused() -> None:
    """Silently ignoring them would compute something the graph did not ask for."""

    thresholds = np.zeros((MATRIX_HEIGHT, 1), dtype=np.float32)
    operation = _bound(_model(no_activation=True, thresholds=thresholds))

    assert "mvau-threshold-presence-mismatch" in _refusals(operation)


@pytest.mark.parametrize(
    "shape", [(MATRIX_HEIGHT + 1, 1), (MATRIX_HEIGHT,), (1, MATRIX_HEIGHT, 1)], ids=str
)
def test_a_threshold_operand_is_one_row_per_output_channel(shape: tuple[int, ...]) -> None:
    operation = _bound(_model(no_activation=False, threshold_shape=shape))

    assert "mvau-threshold-shape" in _refusals(operation)


# -- narrowness is derived from the weights -------------------------------------


def test_narrow_weights_is_read_from_the_matrix_not_asserted_about_it() -> None:
    using_minimum = np.full((MATRIX_WIDTH, MATRIX_HEIGHT), -128.0, dtype=np.float32)
    avoiding_it = np.full((MATRIX_WIDTH, MATRIX_HEIGHT), -127.0, dtype=np.float32)

    assert _bound(_model(weights=avoiding_it)).answer(
        MvauDataflowOp.effective_narrow_weights
    ) == Decided(True)
    assert _bound(_model(weights=using_minimum)).answer(
        MvauDataflowOp.effective_narrow_weights
    ) == Decided(False)


def test_a_matrix_with_no_initializer_is_not_promised_to_be_narrow() -> None:
    """Absent is ``False``: hardware cannot be built on a promise nobody made."""

    operation = _bound(_model(weight_initializer=False))

    assert "weight_excludes_minimum" not in operation.source.analyses
    assert operation.answer(MvauDataflowOp.effective_narrow_weights) == Decided(False)


def test_the_analysis_keeps_only_its_scalar_result() -> None:
    """The array is read at binding time and does not survive into the point."""

    operation = _bound(_model(weights=np.full((MATRIX_WIDTH, MATRIX_HEIGHT), 3.0)))

    assert operation.source.analyses == {"weight_excludes_minimum": True}
    assert all(not isinstance(value, np.ndarray) for value in operation.source.analyses.values())


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([[-128.0, 0.0]]), False),
        (np.array([[-127.0, 5.0]]), True),
        (np.array([]), None),
    ],
    ids=["uses-minimum", "avoids-minimum", "empty"],
)
def test_the_analysis_says_nothing_when_it_cannot_ask(
    values: np.ndarray, expected: bool | None
) -> None:
    assert initializer_excludes_minimum(values, DataType["INT8"]) is expected


# -- what the operation is authoritative for ------------------------------------


def test_a_plain_node_derives_its_output_datatype_from_the_accumulator() -> None:
    assert _bound(_model()).expected_outputs()["output"] == (
        (REPETITIONS, MATRIX_HEIGHT),
        DataType["INT32"],
    )


def test_a_thresholded_node_does_not_claim_its_output_datatype() -> None:
    """Whoever wrote the thresholds chose it; this operation must not contradict them."""

    thresholds = np.zeros((MATRIX_HEIGHT, 1), dtype=np.float32)
    operation = _bound(_model(no_activation=False, thresholds=thresholds, output_type="UINT4"))

    shape, datatype = operation.expected_outputs()["output"]

    assert shape == (REPETITIONS, MATRIX_HEIGHT)
    assert datatype is None
    assert operation.reconciliation() == ()


# -- the Designs' applicability -------------------------------------------------


def test_a_fused_threshold_node_is_valid_and_has_no_applicable_design() -> None:
    """A different outcome from an invalid node, and reported differently."""

    thresholds = np.zeros((MATRIX_HEIGHT, 1), dtype=np.float32)
    operation = _bound(_model(no_activation=False, thresholds=thresholds))

    assert _accepts(operation)

    chosen = operation.design.select("dot_product").root
    assert "mvau-design-fuses-no-activation" in _findings(chosen.dataflow.accepted_answer)


def test_the_refusal_belongs_to_the_design_and_not_to_the_operation() -> None:
    """Named where it is: the mathematics is fine, this composition is not."""

    assert any(
        item is WeightedDotProductDesign.computes_a_bare_accumulator
        for item in WeightedDotProductDesign.dataflow_support.constraints
    )
    assert all(
        item is not WeightedDotProductDesign.computes_a_bare_accumulator
        for item in MvauDataflowOp.source_accepts.constraints
    )


# -- execution ------------------------------------------------------------------


def _executed(model: ModelWrapper, **values: np.ndarray) -> np.ndarray:
    operation = _unbound(model, "mvau0")
    operation.attach_model(model)
    context: dict[str, Any] = dict(values)
    operation.execute_node(context, model.graph)
    return np.asarray(context["output"])


def test_an_integer_node_computes_a_matrix_product() -> None:
    activation = np.arange(REPETITIONS * MATRIX_WIDTH, dtype=np.float32).reshape(
        REPETITIONS, MATRIX_WIDTH
    )
    weight = np.arange(MATRIX_WIDTH * MATRIX_HEIGHT, dtype=np.float32).reshape(
        MATRIX_WIDTH, MATRIX_HEIGHT
    )

    result = _executed(_model(), activation=activation, weight=weight)

    assert np.array_equal(result, np.matmul(activation, weight))


def test_an_xnor_node_computes_a_popcount_product() -> None:
    activation = (
        np.random.RandomState(0).randint(0, 2, (REPETITIONS, MATRIX_WIDTH)).astype(np.float32)
    )
    weight = (
        np.random.RandomState(1).randint(0, 2, (MATRIX_WIDTH, MATRIX_HEIGHT)).astype(np.float32)
    )
    model = _model(binary_xnor=True, activation_type="BINARY", weight_type="BINARY")

    result = _executed(model, activation=activation, weight=weight)

    assert np.array_equal(result, xnor.xnorpopcountmatmul(activation, weight))


def test_bipolar_operands_are_mapped_before_the_popcount() -> None:
    """The oracle's special case, and the one an integer matmul gets wrong."""

    activation = (
        np.random.RandomState(2).choice([-1.0, 1.0], (REPETITIONS, MATRIX_WIDTH)).astype(np.float32)
    )
    weight = (
        np.random.RandomState(3)
        .choice([-1.0, 1.0], (MATRIX_WIDTH, MATRIX_HEIGHT))
        .astype(np.float32)
    )
    model = _model(activation_type="BIPOLAR", weight_type="BIPOLAR")

    result = _executed(model, activation=activation, weight=weight)

    expected = xnor.xnorpopcountmatmul((activation + 1) / 2, (weight + 1) / 2)
    assert np.array_equal(result, expected)
    assert not np.array_equal(result, np.matmul(activation, weight))


def test_a_thresholded_node_applies_its_thresholds() -> None:
    activation = np.arange(REPETITIONS * MATRIX_WIDTH, dtype=np.float32).reshape(
        REPETITIONS, MATRIX_WIDTH
    )
    weight = np.ones((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.float32)
    thresholds = np.array([[10.0, 40.0]] * MATRIX_HEIGHT, dtype=np.float32)
    model = _model(no_activation=False, thresholds=thresholds, output_type="UINT4")

    result = _executed(model, activation=activation, weight=weight, threshold=thresholds)

    expected = multithreshold(np.matmul(activation, weight), thresholds, 1, 0)
    assert np.array_equal(result, expected)


def test_a_bipolar_output_scales_and_biases_the_threshold_result() -> None:
    activation = np.arange(REPETITIONS * MATRIX_WIDTH, dtype=np.float32).reshape(
        REPETITIONS, MATRIX_WIDTH
    )
    weight = np.ones((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.float32)
    thresholds = np.array([[40.0]] * MATRIX_HEIGHT, dtype=np.float32)

    result = execute_mvau(
        activation=activation,
        weight=weight,
        thresholds=thresholds,
        profile=MvauComputationProfile.FUSED_THRESHOLD,
        activation_type=DataType["INT8"],
        weight_type=DataType["INT8"],
        output_type=DataType["BIPOLAR"],
        activation_bias=0,
    )

    assert np.array_equal(result, multithreshold(np.matmul(activation, weight), thresholds, 2, -1))


def test_the_activation_bias_reaches_the_threshold_result() -> None:
    activation = np.ones((REPETITIONS, MATRIX_WIDTH), dtype=np.float32)
    weight = np.ones((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.float32)
    thresholds = np.array([[4.0]] * MATRIX_HEIGHT, dtype=np.float32)
    model = _model(no_activation=False, thresholds=thresholds, activation_bias=-3)

    result = _executed(model, activation=activation, weight=weight, threshold=thresholds)

    assert np.array_equal(result, multithreshold(np.matmul(activation, weight), thresholds, 1, -3))


def test_a_four_dimensional_result_is_transposed_around_multithreshold() -> None:
    """Channels-last in, channels-last out, channels-second in between."""

    activation = np.arange(2 * 3 * 3 * MATRIX_WIDTH, dtype=np.float32).reshape(
        2, 3, 3, MATRIX_WIDTH
    )
    weight = np.ones((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.float32)
    thresholds = np.array([[100.0, 400.0]] * MATRIX_HEIGHT, dtype=np.float32)

    result = execute_mvau(
        activation=activation,
        weight=weight,
        thresholds=thresholds,
        profile=MvauComputationProfile.FUSED_THRESHOLD,
        activation_type=DataType["INT8"],
        weight_type=DataType["INT8"],
        output_type=DataType["UINT4"],
        activation_bias=0,
    )

    product = np.matmul(activation, weight)
    expected = multithreshold(product.transpose((0, 3, 1, 2)), thresholds, 1, 0).transpose(
        (0, 2, 3, 1)
    )
    assert result.shape == product.shape
    assert np.array_equal(result, expected)


def test_execution_without_a_model_refuses_rather_than_guessing_a_datatype() -> None:
    """Constructed directly, as a caller outside QONNX's own path would."""

    model = _model()
    operation = MvauDataflowOp(model.graph.node[0], 1)

    with pytest.raises(DataflowOpError, match="no model attached"):
        operation.execute_node({}, model.graph)


def test_a_bound_occurrence_executes_from_its_frozen_reading() -> None:
    """Both states answer, and they answer the same."""

    activation = np.ones((REPETITIONS, MATRIX_WIDTH), dtype=np.float32)
    weight = np.ones((MATRIX_WIDTH, MATRIX_HEIGHT), dtype=np.float32)
    model = _model()
    operation = _bound(model)

    context: dict[str, Any] = {"activation": activation, "weight": weight}
    operation.execute_node(context, model.graph)

    assert np.array_equal(context["output"], np.matmul(activation, weight))


# -- verification ---------------------------------------------------------------


def test_verify_node_reports_what_the_projection_would_refuse() -> None:
    """One set of checks, two audiences -- never two sets that can disagree."""

    assert _bound(_model()).verify_node() == []

    broken = _bound(_model(no_activation=False))
    messages = broken.verify_node()

    assert messages and any("threshold" in message for message in messages)


# -- the graph names -------------------------------------------------------------


def test_the_node_attributes_keep_the_names_finns_graphs_already_use() -> None:
    declared = _unbound(_model(), "mvau0").get_nodeattr_types()

    assert {"noActivation", "binaryXnorMode", "ActVal", "accDataType"} <= set(declared)
    assert "no_activation" not in declared


def test_two_members_may_not_read_one_node_attribute() -> None:
    with pytest.raises(AuthoringError, match="one graph attribute is one source fact"):

        class Doubled(DataflowOp):
            family = "test.doubled"
            activation = InputTensor(index=0)
            result = OutputTensor(index=0)
            first = Attribute(int, default=0, onnx="shared")
            second = Attribute(int, default=0, onnx="shared")


# -- the second operation, at its own scale --------------------------------------


def test_the_replay_operation_executes_its_own_semantics() -> None:
    """Rows repeated consecutively, which is the order the buffer replays them."""

    model = _replay_model(repetitions=2, matrix_width=8, folds=4)
    operation = _unbound(model, "replay0")
    operation.attach_model(model)
    activation = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)

    context: dict[str, Any] = {"activation": activation}
    operation.execute_node(context, model.graph)

    assert np.array_equal(context["expanded"], np.repeat(activation, 4, axis=0))


def test_the_replay_operation_verifies_its_own_semantics() -> None:
    model = _replay_model()
    assert _unbound(model, "replay0").bind(model, Build()).verify_node() == []

    model.set_tensor_shape("activation", [8])
    broken = _unbound(model, "replay0").bind(model, Build())

    assert any("matrix-shaped" in message for message in broken.verify_node())


def test_a_stale_output_annotation_is_still_not_a_verification_failure() -> None:
    """The ruling from the review round holds for the second operation too.

    An output annotation the operation can repair must not be reported as a
    fault, or the repair can never be committed.  It is a reconciliation
    difference, and ``verify_node`` deliberately does not read it.
    """

    model = _replay_model()
    model.set_tensor_shape("expanded", [99, 8])
    operation = _unbound(model, "replay0").bind(model, Build())

    assert operation.verify_node() == []
    assert operation.reconciliation() != ()
