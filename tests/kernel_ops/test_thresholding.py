# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: integer MultiThreshold bound to ``ThresholdingAxiKernel``."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import KernelOpError
from finn.custom_op.kernels.roots import ThresholdingNode
from finn.custom_op.kernels.thresholding import Thresholding
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernel_ops.models import THRESHOLDS, schema_digest, thresholding_model

MEMORY = {"ram_style": "distributed", "block_stages": 1, "ultra_stages": 0}


def op(model: ModelWrapper) -> Thresholding:
    found = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(found, Thresholding)
    return found


def test_facts_and_the_output() -> None:
    model = thresholding_model(bias=-1)
    facts = op(model).facts()
    assert facts.root is ThresholdingNode
    formals = facts.formals()
    assert formals["thresholds"] == (tuple(tuple(int(v) for v in row) for row in THRESHOLDS),)
    assert formals["bias"] == -1
    ((dims, dtype),) = op(model).output_tensors().values()
    assert dims == (3, 4) and dtype.name == "INT3"  # [-1, 2], in the RTL's output width
    assert op(thresholding_model()).output_tensors()["y"][1].name == "UINT2"


def test_the_thresholds_must_be_an_initializer_with_a_row_per_channel() -> None:
    with pytest.raises(KernelOpError, match="must be an initializer"):
        op(thresholding_model(stored=False)).facts()
    with pytest.raises(KernelOpError, match=r"\(1, 3\), not one row for each of the 4 channels"):
        op(thresholding_model(thresholds=THRESHOLDS[:1])).facts()


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
    assert (Thresholding.op_version, schema_digest(Thresholding)) == (1, "77014c3c78204e2c")
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
