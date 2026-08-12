############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""ROUTE seam: is_specialized + kernel_hw_language (R1, R2).

Routing is a pure DERIVATION of the ``backend`` nodeattr — nothing extra stored that could
stale on re-resolve. ``is_specialized`` is the ONE definition of "a backend is committed"
(non-empty ``backend`` on a finn.kernels node); ``kernel_hw_language`` reads the selected
backend's static ``language``. Both are bare-node (a nodeattr read + a static pool lookup,
no model/getCustomOp). This module is the MERGE target for the old seam_c_routing +
specialized_node_facade + the routing half of seam_hone_specialized_contract.
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.kernels.compute.mvau.op import MvauDataflowOp
from finn.kernels.compute.thresholding.op import ThresholdingDataflowOp
from finn.kernels.ir.routing import is_specialized, kernel_hw_language
from finn.util.fpgadataflow import (
    is_fpgadataflow_node,
    is_hls_node,
    is_rtl_node,
    is_specialized_node,
)

MW, MH = 128, 64
KERNEL_DOMAIN = "finn.kernels"


def _unresolved_mvau_node():
    """A finn.kernels MVAU node straight out of Seam A's infer (no backend committed)."""
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
    model = model.transform(InferKernels([MvauDataflowOp, ThresholdingDataflowOp]))
    return [n for n in model.graph.node if n.domain == KERNEL_DOMAIN][0]


def _set_impl(node, value):
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
    return helper.make_node("Add", ["a", "b"], ["c"], name="plain_add")


# --- R1/R2: unspecialized routes nowhere but is still a family member -------


def test_unresolved_kernel_node_routes_nowhere():
    kn = _unresolved_mvau_node()
    backend = get_by_name(kn.attribute, "backend")
    assert backend is None or backend.s.decode("UTF-8") == ""
    assert is_specialized(kn) is False
    assert kernel_hw_language(kn) is None
    assert is_hls_node(kn) is False
    assert is_rtl_node(kn) is False
    # membership (domain-derived) holds even while not HW-ready — the partition-sweep invariant.
    assert is_fpgadataflow_node(kn) is True


def test_resolved_hls_kernel_routes_as_hls():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "mvau_hls")
    assert is_specialized(kn) is True
    assert kernel_hw_language(kn) == "hls"
    assert is_hls_node(kn) is True
    assert is_rtl_node(kn) is False


def test_resolved_rtl_kernel_routes_as_rtl():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "mvau_dsp_softvec")
    assert is_specialized(kn) is True
    assert kernel_hw_language(kn) == "rtl"
    assert is_rtl_node(kn) is True
    assert is_hls_node(kn) is False


def test_invalid_backend_derives_none_gracefully():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "not_a_real_backend")
    # is_specialized is True (backend nonempty) but the language lookup misses → None, no crash.
    assert is_specialized(kn) is True
    assert kernel_hw_language(kn) is None
    assert is_hls_node(kn) is False
    assert is_rtl_node(kn) is False


# --- classic path unchanged; non-kernel node → not specialized -------------


def test_classic_node_routing_unchanged():
    hls, rtl = _classic_hls_node(), _classic_rtl_node()
    assert is_hls_node(hls) is True and is_rtl_node(hls) is False
    assert is_rtl_node(rtl) is True and is_hls_node(rtl) is False
    # a kernel query is invisible to the classic-domain path.
    assert kernel_hw_language(hls) is None
    assert is_specialized(hls) is False  # not a finn.kernels node


def test_non_kernel_node_is_not_specialized():
    assert is_specialized(_plain_node()) is False
    assert kernel_hw_language(_plain_node()) is None


# --- is_specialized_node facade covers the kernel population ----------------
# (the merge keeps only the kernel-population case; the by-construction equivalence
# sweep from the old facade test is dropped as low-signal.)


def test_facade_covers_specialized_and_unspecialized_kernel():
    specialized = _unresolved_mvau_node()
    _set_impl(specialized, "mvau_hls")
    assert is_specialized_node(specialized) is True
    assert is_specialized_node(_unresolved_mvau_node()) is False
    assert is_specialized_node(_classic_hls_node()) is True
    assert is_specialized_node(_plain_node()) is False
