############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Shared fixtures for dataflow-kernel tests."""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model


def make_thresholding_model(num_channels=8, num_steps=7, pe=2, act_val=0,
                            idt="INT8", tdt="INT8", odt="UINT3", spatial=(4, 4)):
    """Build a 1-node Thresholding graph (NHWC) with initialized thresholds."""
    n = 1
    ishape = [n, *spatial, num_channels]
    # Monotonic *integer* thresholds per channel (hardware thresholds are ints).
    row = np.round(np.linspace(-40, 40, num_steps)).astype(np.float32)
    thr_data = np.tile(row, (num_channels, 1)).astype(np.float32)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ishape)
    thr = helper.make_tensor_value_info("thr", TensorProto.FLOAT, [num_channels, num_steps])
    node = helper.make_node(
        "Thresholding", ["inp", "thr"], ["out"], domain="finn.kernels", name="th0",
        num_steps=num_steps, act_val=act_val, PE=pe,
        input0Datatype=idt, input1Datatype=tdt, output0Datatype=odt,
    )
    graph = helper.make_graph(
        [node], "g", [inp, thr], [out],
        initializer=[helper.make_tensor("thr", TensorProto.FLOAT,
                     [num_channels, num_steps], thr_data.flatten())],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[idt])
    model.set_tensor_datatype("thr", DataType[tdt])
    model.set_tensor_datatype("out", DataType[odt])
    attrs = {
        "PE": pe, "input0Datatype": idt, "input1Datatype": tdt,
        "output0Datatype": odt, "num_steps": num_steps, "act_val": act_val,
    }
    return model, node, attrs, thr_data


@pytest.fixture
def thresholding_model():
    return make_thresholding_model()
