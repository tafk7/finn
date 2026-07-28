############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``is_fpgadataflow_node`` — kernel family membership derived from domain (C4/FU-1).

A ``finn.kernels`` node is a dataflow-family member by its DOMAIN alone (no ``backend``
stamp required), while a classic node still needs its ``backend=="fpgadataflow"`` token
(INV-L). This is the precondition for dropping the redundant kernel stamp (FU-1b).
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.kernels.adapter import InferKernels
from finn.kernels.ops.mvau.op import MvauKernelOp
from finn.kernels.ops.thresholding.op import ThresholdingKernelOp
from finn.util.fpgadataflow import is_fpgadataflow_node

MW, MH = 128, 64


def _kernel_node_no_stamp():
    """A finn.kernels node WITHOUT a backend nodeattr (post-FU-1b shape)."""
    return helper.make_node("MVAU", ["inp"], ["out"], domain="finn.kernels", name="kn")


def _classic_hls_with_stamp():
    return helper.make_node(
        "MVAU_hls", ["inp"], ["out"], domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow", name="classic",
    )


def _plain_node():
    return helper.make_node("Add", ["a", "b"], ["c"], name="plain")


def test_kernel_node_is_member_without_stamp():
    kn = _kernel_node_no_stamp()
    assert get_by_name(kn.attribute, "backend") is None
    assert is_fpgadataflow_node(kn) is True


def test_classic_node_with_stamp_is_member():
    assert is_fpgadataflow_node(_classic_hls_with_stamp()) is True


def test_classic_node_without_stamp_is_not_member():
    """INV-L: a classic node still needs its backend token — dropping it de-members it."""
    node = _classic_hls_with_stamp()
    node.attribute.remove(get_by_name(node.attribute, "backend"))
    assert is_fpgadataflow_node(node) is False


def test_plain_node_is_not_member():
    assert is_fpgadataflow_node(_plain_node()) is False


def test_inferred_kernel_node_has_no_stamp_but_is_member():
    """FU-1b end-state: a node straight out of Seam A's infer carries NO backend stamp,
    yet still sweeps into a dataflow partition (is_fpgadataflow_node True via domain)."""
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul],
        "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model = model.transform(InferKernels([MvauKernelOp, ThresholdingKernelOp]))
    kn = [n for n in model.graph.node if n.domain == "finn.kernels"][0]
    assert get_by_name(kn.attribute, "backend") is None
    assert is_fpgadataflow_node(kn) is True
