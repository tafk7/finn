############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``is_specialized_node`` — the ONE committed-check predicate (C1/C2).

The facade unions both populations: a classic ``…/hls``/``…/rtl`` node (committed by
its domain) and a ``finn.kernels`` node whose backend is selected. Its body is the
exact disjunction the 17 flow sites already compute, so its truth is byte-identical to
``is_hls_node(n) or is_rtl_node(n)`` BY CONSTRUCTION — the equivalence test below makes
that falsifiable rather than asserted, and holds identically pre- and post-migration.
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.kernels.compute.mvau.op import MvauKernelOp
from finn.kernels.compute.thresholding.op import ThresholdingKernelOp
from finn.util.fpgadataflow import is_hls_node, is_rtl_node, is_specialized_node

MW, MH = 128, 64
KERNEL_DOMAIN = "finn.kernels"


def _unresolved_mvau_node():
    """A ``finn.kernels`` MVAU node straight out of Seam A's infer (no ``implementation``)."""
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
    kn = [n for n in model.graph.node if n.domain == KERNEL_DOMAIN][0]
    return kn


def _set_impl(node, value):
    """Hand-stamp the ``implementation`` nodeattr (Seam B's job in production)."""
    existing = get_by_name(node.attribute, "backend")
    if existing is not None:
        node.attribute.remove(existing)
    node.attribute.append(helper.make_attribute("backend", value))


def _classic_hls_node():
    return helper.make_node(
        "MVAU_hls", ["inp"], ["out"], domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow", name="classic_hls",
    )


def _classic_rtl_node():
    return helper.make_node(
        "MVAU_rtl", ["inp"], ["out"], domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow", name="classic_rtl",
    )


def _plain_node():
    """A raw non-HW node — never committed."""
    return helper.make_node("Add", ["a", "b"], ["c"], name="plain_add")


def test_specialized_kernel_node_is_specialized():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "mvau_hls")
    assert is_specialized_node(kn) is True


def test_unspecialized_kernel_node_is_not_specialized():
    kn = _unresolved_mvau_node()
    assert is_specialized_node(kn) is False


def test_classic_hls_node_is_specialized():
    assert is_specialized_node(_classic_hls_node()) is True


def test_plain_node_is_not_specialized():
    assert is_specialized_node(_plain_node()) is False


def test_committed_check_equivalence():
    """FALSIFIABLE INV-L check: the facade's truth equals the raw disjunction over a
    REPRESENTATIVE node set. Depends on NO migrated site, so it holds identically pre-
    and post-migration — a site swap that ever changed truth could only break a
    genuinely non-equivalent edit, never this."""
    specialized_hls = _unresolved_mvau_node()
    _set_impl(specialized_hls, "mvau_hls")
    specialized_rtl = _unresolved_mvau_node()
    _set_impl(specialized_rtl, "mvau_dsp_softvec")
    unspecialized_kernel = _unresolved_mvau_node()

    nodes = [
        specialized_hls,
        specialized_rtl,
        unspecialized_kernel,
        _classic_hls_node(),
        _classic_rtl_node(),
        _plain_node(),
    ]
    for n in nodes:
        assert is_specialized_node(n) == (is_hls_node(n) or is_rtl_node(n))
