# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: integer MultiThreshold bound to ``ThresholdingAxiKernel``."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import KernelOpError
from finn.custom_op.kernels.thresholding import Thresholding
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.transformation.kernels import InferKernelTensors
from kernel_ops.models import THRESHOLDS, schema_digest, thresholding_model

MEMORY = {"ram_style": "distributed", "block_stages": 1, "ultra_stages": 0}


def op(model: ModelWrapper) -> Thresholding:
    found = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(found, Thresholding)
    return found


def test_facts_and_the_output() -> None:
    model = thresholding_model(bias=-1)
    facts = op(model).facts()
    assert facts.root is Thresholding.root() and facts.owned == ()
    formals = facts.formals()
    assert formals["thresholds"] == (tuple(tuple(int(v) for v in row) for row in THRESHOLDS),)
    assert formals["bias"] == -1
    ((dims, dtype),) = op(model).output_tensors().values()
    assert dims == (3, 4) and dtype.name == "INT3"  # [-1, 2], in the RTL's output width
    assert op(thresholding_model()).output_tensors()["y"][1].name == "UINT2"


def test_the_thresholds_must_be_an_initializer_with_one_row_or_a_row_per_channel() -> None:
    with pytest.raises(KernelOpError, match="are not an initializer: not a Thresholding"):
        op(thresholding_model(stored=False, infer=False)).facts()
    with pytest.raises(KernelOpError, match=r"\(2, 3\), neither one row .* its 4 channels"):
        op(thresholding_model(thresholds=THRESHOLDS[:2], infer=False)).facts()


def test_one_row_for_every_channel_is_bound_as_the_graph_states_it() -> None:
    """The graph's (1, N) row stays one row; the kernel binds C = 1 and takes its
    channels and PE's domain from the input."""
    model = thresholding_model(thresholds=THRESHOLDS[:1], bias=-1)
    assert model.get_initializer("t").shape == (1, 3)
    assert op(model).facts().formals()["thresholds"] == ((tuple(int(v) for v in THRESHOLDS[0]),),)
    op(model).save({"pe": 4, "use_axilite": False, "deep_pipeline": False, **MEMORY})
    activate = op(model).point().activate
    parameters = dict(activate.module.parameters)
    assert (parameters["C"], parameters["PE"]) == (1, 4)
    # One row, INT5 as normalized, padded to four words.
    assert activate.thresholds_file.data == b"17\n01\n08\n00\n"
    # One row a lane: stage 1 holds two words, whatever the channels.
    assert parameters["DEPTH_TRIGGER_BRAM"] == 2
    with pytest.raises(KernelOpError) as error:
        op(model).save({"pe": 3})  # not a divisor of the input's 4 channels
    assert error.value.keys == ("pe",)
    values = np.arange(-12, 12, dtype=np.float32).reshape(3, 8)[:, :4]
    produced = execute_onnx(model, {"x": values})["y"]
    expected = execute_onnx(multithreshold(model), {"x": values})["y"]
    assert np.array_equal(produced, expected)


def test_the_thresholds_type_holds_a_value_below_the_least_an_input_can_miss() -> None:
    """thresholding_axi saturates an INT8 input to the thresholds' type: with the least
    threshold at INT5's minimum, an input below it would count it. Normalized, the type
    holds one value below it (INT6), and the kernel admits the node; a least threshold
    at the input's own minimum, which every input meets, needs none."""
    table = np.array([[-16, -3, 2], [-9, 0, 15], [-4, -4, 7], [-1, 6, 14]])
    model = thresholding_model(thresholds=table)
    assert model.get_tensor_datatype("t").name == "INT6"
    assert not op(model).verify_node()
    floor = table.copy()
    floor[0, 0] = -128
    assert thresholding_model(thresholds=floor).get_tensor_datatype("t").name == "INT8"


def test_a_float_thresholding_is_refused_while_its_choices_are_open() -> None:
    """The kernel's admission refuses FLOAT32 on the facts alone, though its folding and
    memories are still open."""
    model = thresholding_model(annotate=(), infer=False)
    for name in ("x", "t"):
        model.set_tensor_datatype(name, DataType["FLOAT32"])
    problems = op(model.transform(InferKernelTensors())).verify_node()
    assert [problem.split(": ")[:3] for problem in problems] == [
        ["activate", "activate.table_supported", "threshold-type"],
        ["activate", "activate.types_supported", "dtype-family"],
    ]


def test_an_empty_threshold_table_is_refused_by_name() -> None:
    # A row per channel, none of them holding a threshold: the empty initializer is
    # vacuously integral and has no range to check against its annotation.
    empty = np.zeros((len(THRESHOLDS), 0), dtype=np.float32)
    with pytest.raises(KernelOpError, match="activate: t is empty"):
        op(thresholding_model(thresholds=empty, infer=False)).facts()


def test_the_schema_holds_the_memory_choices_and_bias_is_semantic() -> None:
    schema = Thresholding.schema()
    assert {"pe", "use_axilite", "deep_pipeline", *MEMORY} <= set(schema)
    assert schema["ram_style"] == ("s", ())
    assert schema["ultra_stages"] == ("i", ())
    assert {"x.transport", "y.transport"} <= set(schema)
    # Its input never carries a value: no source keys (the table stays the kernel's).
    assert not any(name.startswith("x.source") for name in schema)
    types = op(thresholding_model()).get_nodeattr_types()
    assert types["bias"] == ("i", True, 0)
    assert "bias" not in schema


def test_the_schema_is_pinned_for_its_op_version() -> None:
    assert (Thresholding.op_version, schema_digest(Thresholding)) == (1, "6f01bf0727989462")
    assert Thresholding.op_version == ThresholdingAxiKernel.version


def test_save_and_replay_a_memory_choice() -> None:
    model = thresholding_model()
    op(model).save({"pe": 2, "use_axilite": False, "deep_pipeline": False, **MEMORY})
    point = op(model).point()
    assert dict(point.activate.module.parameters)["DEPTH_TRIGGER_BRAM"] == 4  # stage 1, PE 2
    with pytest.raises(KernelOpError) as error:
        op(model).save({"block_stages": 3})
    assert error.value.keys == ("block_stages",)
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
