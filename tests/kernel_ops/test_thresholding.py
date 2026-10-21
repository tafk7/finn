# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: integer MultiThreshold bound to ``ThresholdingAxiKernel``."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import TensorRange
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from support.configure import configured
from support.models import THRESHOLDS, convert, schema_digest, thresholding_model
from support.refusals import refused

from finn.core.graph.errors import KernelOpError
from finn.core.onnx_exec import execute_onnx
from finn.custom_op.kernels.thresholding import Thresholding
from finn.kernels.library import library_module
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.transformation.kernels import InferKernelTensors

MEMORY = {"ram_style": "distributed", "block_stages": 1, "ultra_stages": 0}


def op(model: ModelWrapper) -> Thresholding:
    found = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(found, Thresholding)
    return found


def test_facts_and_the_output() -> None:
    model = thresholding_model(bias=-1)
    facts = op(model).facts()
    assert facts.space_type is ThresholdingAxiKernel and op(model).owned() == {}
    formals = facts.arguments
    assert formals["thresholds"] == IntegerTensorValue.of(
        (tuple(tuple(int(v) for v in row) for row in THRESHOLDS),)
    )
    assert formals["bias"] == -1
    (tensor,) = op(model).infer_output_tensors(model).values()
    # [-1, 2], in the RTL's output width, over that range
    assert tensor.shape == (3, 4) and tensor.element.dtype.name == "INT3"
    assert tensor.element.value_range == TensorRange(-1, 2)
    assert op(thresholding_model()).output_tensors()["y"].element.dtype.name == "UINT2"


def test_the_thresholds_must_be_an_initializer_with_one_row_or_a_row_per_channel() -> None:
    with refused(
        "initializer-absent",
        error=KernelOpError,
        owner="Thresholding",
        node="activate",
        tensor="t",
        message="are not an initializer: not a Thresholding",
    ):
        op(thresholding_model(stored=False, infer=False)).facts()
    with refused(
        "thresholds-shape-invalid",
        error=KernelOpError,
        owner="Thresholding",
        node="activate",
        tensor="t",
        message=r"\(2, 3\), neither one row .* its 4 channels",
    ):
        op(thresholding_model(thresholds=THRESHOLDS[:2], infer=False)).facts()


def test_one_row_for_every_channel_is_bound_as_the_graph_states_it() -> None:
    """The graph's (1, N) row stays one row; the kernel binds C = 1 and takes its
    channels and PE's domain from the input."""
    model = thresholding_model(thresholds=THRESHOLDS[:1], bias=-1)
    assert model.get_initializer("t").shape == (1, 3)
    stated = IntegerTensorValue.of(((tuple(int(v) for v in THRESHOLDS[0]),),))
    assert op(model).facts().arguments["thresholds"] == stated
    configured(
        model, {"activate": {"pe": 4, "use_axilite": False, "deep_pipeline": False, **MEMORY}}
    )
    activate = op(model).point()
    parameters = dict(library_module(activate).parameters)
    assert (parameters["C"], parameters["PE"]) == (1, 4)
    # One row in INT5, the type the facts derive, padded to four words.
    assert activate.thresholds_file.data == b"17\n01\n08\n00\n"
    # One row a lane: stage 1 holds two words, whatever the channels.
    assert parameters["DEPTH_TRIGGER_BRAM"] == 2
    with refused("domain-membership", owner="pe"):  # not a divisor of the input's 4 channels
        configured(model, {"activate": {"pe": 3}})
    assert op(model).choices()["pe"] == 4
    values = np.arange(-12, 12, dtype=np.float32).reshape(3, 8)[:, :4]
    produced = execute_onnx(model, {"x": values})["y"]
    expected = execute_onnx(multithreshold(model), {"x": values})["y"]
    assert np.array_equal(produced, expected)


def test_the_thresholds_type_holds_a_value_below_the_least_an_input_can_miss() -> None:
    """thresholding_axi saturates an INT8 input to the thresholds' type: with the least
    threshold at INT5's minimum, an input below it would count it. The type the facts
    derive holds one value below it (INT6), and the kernel admits the node; a least
    threshold at the input's own minimum, which every input meets, needs none."""
    table = np.array([[-16, -3, 2], [-9, 0, 15], [-4, -4, 7], [-1, 6, 14]])
    model = thresholding_model(thresholds=table)
    assert op(model).facts().arguments["threshold_dtype"] == DataType["INT6"]
    assert not op(model).verify_node()
    floor = table.copy()
    floor[0, 0] = -128
    formals = op(thresholding_model(thresholds=floor)).facts().arguments
    assert formals["threshold_dtype"] == DataType["INT8"]


def test_the_table_is_derived_from_the_stored_thresholds_and_the_input_type() -> None:
    """Rounded up and clipped to [min, max + 1] of the input's type, and typed by what it
    holds; the graph's table stays as it was, unannotated."""
    table = np.array([[-300.0, -2.5, 0.25], [-9.0, 1.0, 130.0], [-1.5, 3.0, 4.0], [0, 1, 2]])
    model = thresholding_model(thresholds=table, bias=-1, annotate=("x",))
    formals = op(model).facts().arguments
    assert formals["thresholds"] == IntegerTensorValue.of(
        (((-128, -2, 1), (-9, 1, 128), (-1, 3, 4), (0, 1, 2)),)
    )
    assert formals["threshold_dtype"] == DataType["INT9"]
    assert np.array_equal(model.get_initializer("t"), table.astype(np.float32))
    assert not model.has_tensor_datatype("t")
    source = multithreshold(model)
    for start in range(-128, 128, 12):  # every INT8 input
        values = np.clip(np.arange(start, start + 12), -128, 127).reshape(3, 4)
        feed = {"x": values.astype(np.float32)}
        assert np.array_equal(execute_onnx(model, feed)["y"], execute_onnx(source, feed)["y"])


def test_a_nan_threshold_is_refused_by_name() -> None:
    table = np.where(THRESHOLDS == 1, np.nan, THRESHOLDS)
    with refused(
        "thresholds-nan",
        error=KernelOpError,
        owner="Thresholding",
        node="activate",
        tensor="t",
        message="holds NaN",
    ):
        op(thresholding_model(thresholds=table, infer=False)).facts()


def test_a_float_thresholding_is_refused_while_its_choices_are_open() -> None:
    """The kernel's validity refuses FLOAT32 on the facts alone, though its folding and
    memories are still open."""
    model = thresholding_model(annotate=(), infer=False)
    for name in ("x", "t"):
        model.set_tensor_datatype(name, DataType["FLOAT32"])
    problems = op(model.transform(InferKernelTensors())).verify_node()
    assert [problem.split(": ")[:3] for problem in problems] == [
        ["activate", "table_supported", "threshold-type"],
        ["activate", "types_supported", "dtype-family"],
    ]


def test_an_empty_threshold_table_is_refused_by_name() -> None:
    # A row per channel, none of them holding a threshold: the empty initializer is
    # vacuously integral and has no range to check against its annotation.
    empty = np.zeros((len(THRESHOLDS), 0), dtype=np.float32)
    with refused(
        "initializer-empty",
        error=KernelOpError,
        owner="Thresholding",
        node="activate",
        tensor="t",
        message="is empty",
    ):
        op(thresholding_model(thresholds=empty, infer=False)).facts()


def test_the_schema_holds_the_memory_choices_and_bias_is_semantic() -> None:
    schema = Thresholding.schema()
    assert {"pe", "use_axilite", "deep_pipeline", *MEMORY} <= set(schema)
    assert schema["ram_style"] == ("s", ())
    assert schema["ultra_stages"] == ("i", ())
    # A channel's keys are its tensor's, never the node's; the table stays the kernel's.
    assert not any(name.partition(".")[0] in ("x", "y") for name in schema)
    assert len(schema) == 6
    types = op(thresholding_model()).get_nodeattr_types()
    assert types["bias"] == ("i", True, 0)
    assert "bias" not in schema


def test_the_schema_is_pinned_for_its_op_version() -> None:
    assert (Thresholding.op_version, schema_digest(Thresholding)) == (1, "7c279bb322d44f45")
    assert Thresholding.op_version == ThresholdingAxiKernel.version


def test_a_memory_choice_is_written_and_replayed() -> None:
    model = thresholding_model()
    configured(
        model, {"activate": {"pe": 2, "use_axilite": False, "deep_pipeline": False, **MEMORY}}
    )
    point = op(model).point()
    assert dict(library_module(point).parameters)["DEPTH_TRIGGER_BRAM"] == 4  # stage 1, PE 2
    with refused("domain-membership", owner="block_stages"):
        configured(model, {"activate": {"block_stages": 3}})
    assert op(model).choices()["block_stages"] == 1


def multithreshold(model: ModelWrapper) -> ModelWrapper:
    """The same operation as qonnx's MultiThreshold."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "MultiThreshold",
        ["x", "t"],
        ["y"],
        domain="qonnx.custom_op.general",
        out_dtype="INT2",
        out_bias=-1.0,
    )
    graph = helper.make_graph([node], "source", [x], [y])
    source = ModelWrapper(qonnx_make_model(graph))
    source.set_initializer("t", model.get_initializer("t"))
    return source


def test_execution_is_multithreshold() -> None:
    model = thresholding_model(bias=-1)
    values = np.arange(-12, 12, dtype=np.float32).reshape(3, 8)[:, :4]
    produced = execute_onnx(model, {"x": values})["y"]
    expected = execute_onnx(multithreshold(model), {"x": values})["y"]
    assert np.array_equal(produced, expected)


def layout_scenario(*, transposed: bool) -> ModelWrapper:
    """A MultiThreshold on a 4-D INT3 input whose channel axis the graph doesn't settle.

    ``transposed``: an NHWC input, a Transpose with perm [0, 2, 1, 3] (channels stay last),
    then thresholds shared by every channel, its output annotated NCHW (stale). Otherwise:
    an NCHW input and a row of thresholds for each of its 3 channels, ``data_layout``
    unset. Both from upstream FINN's tests of its threshold layout inference."""
    shape = [1, 3, 2, 4]
    nodes = []
    x = "x"
    if transposed:
        nodes.append(helper.make_node("Transpose", ["x"], ["xt"], name="swap", perm=[0, 2, 1, 3]))
        x = "xt"
    nodes.append(
        helper.make_node(
            "MultiThreshold",
            [x, "t"],
            ["y"],
            name="mt",
            domain="qonnx.custom_op.general",
            out_dtype="INT2",
            out_bias=-2.0,
        )
    )
    graph = helper.make_graph(
        nodes,
        "layout_scenario",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    rows = 1 if transposed else 3
    model.set_initializer("t", np.asarray([[-1.0, 0.0, 1.0]] * rows, dtype=np.float32))
    model.set_tensor_datatype("x", DataType["INT3"])
    model.set_tensor_layout("x", ["N", "H", "W", "C"] if transposed else ["N", "C", "H", "W"])
    model = model.transform(InferShapes()).transform(InferDataTypes())  # type: ignore[no-untyped-call]
    model = model.transform(InferDataLayouts())
    if transposed:
        model.set_tensor_layout("y", ["N", "C", "H", "W"])
    return model


@pytest.mark.parametrize("transposed", [True, False], ids=["shared-after-transpose", "nchw-rows"])
def test_a_layout_the_graph_does_not_settle_is_refused_never_reordered(transposed: bool) -> None:
    """Upstream FINN once reordered such values silently (a transpose on one side of the
    threshold only). Here the MultiThreshold stays on the host (``layout-unproven``), and
    the converted graph computes what qonnx computes."""
    source = layout_scenario(transposed=transposed)
    values = (np.arange(24, dtype=np.float32) % 5 - 2).reshape(1, 3, 2, 4)
    expected = execute_onnx(source, {"x": values})["y"]

    converted, conversion = convert(source)

    (outcome,) = [o for o in conversion.outcomes if o.nodes == ("mt",)]
    assert outcome.op is None
    assert "layout-unproven" in [finding.code for finding in outcome.findings]
    assert [n.op_type for n in converted.graph.node if n.name == "mt"] == ["MultiThreshold"]
    produced = execute_onnx(converted, {"x": values})["y"]
    assert np.array_equal(produced, expected)
