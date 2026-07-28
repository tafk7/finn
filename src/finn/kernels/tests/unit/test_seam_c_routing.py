############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam C — taxonomy + HW-readiness routing (handoff §4 gate).

Proves routing is a pure DERIVATION of the ``backend`` nodeattr — nothing extra
stored:
  * an UNRESOLVED kernel node (no ``backend``) routes NOWHERE (not HW-ready);
  * ``backend="mvau_hls"`` → ``is_hls_node`` True, ``is_rtl_node`` False;
  * ``backend="mvau_dsp_softvec"`` → ``is_rtl_node`` True, ``is_hls_node`` False;
  * a classic ``MVAU_hls`` node routes exactly as before (additive branch, no regression);
  * an INVALID ``backend`` string derives None gracefully (no crash).

Synthetic by design: Seam C hand-stamps ``backend`` (Seam B writes it in
production). The predicates are BARE-NODE — no model, no ``getCustomOp``.
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.kernels.adapter import InferKernels
from finn.kernels.ops.mvau.op import MvauKernelOp
from finn.kernels.ops.thresholding.op import ThresholdingKernelOp
from finn.kernels.routing import kernel_hw_language
from finn.util.fpgadataflow import is_fpgadataflow_node, is_hls_node, is_rtl_node

MW, MH = 128, 64
KERNEL_DOMAIN = "finn.kernels"


def _unresolved_mvau_node():
    """A ``finn.kernels`` MVAU node straight out of Seam A's infer (no ``backend``)."""
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
    """Hand-stamp the ``backend`` nodeattr (Seam B's job in production)."""
    existing = get_by_name(node.attribute, "backend")
    if existing is not None:
        node.attribute.remove(existing)
    node.attribute.append(helper.make_attribute("backend", value))


def test_unresolved_kernel_node_routes_nowhere():
    kn = _unresolved_mvau_node()
    impl = get_by_name(kn.attribute, "backend")
    # Seam A leaves implementation unset (or at its axis default ""); either way, not committed.
    assert impl is None or impl.s.decode("UTF-8") == ""
    assert kernel_hw_language(kn) is None
    assert is_hls_node(kn) is False
    assert is_rtl_node(kn) is False
    # Still a family member (backend token) even while not HW-ready.
    assert is_fpgadataflow_node(kn) is True


def test_resolved_hls_kernel_routes_as_hls():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "mvau_hls")
    assert kernel_hw_language(kn) == "hls"
    assert is_hls_node(kn) is True
    assert is_rtl_node(kn) is False


def test_resolved_rtl_kernel_routes_as_rtl():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "mvau_dsp_softvec")
    assert kernel_hw_language(kn) == "rtl"
    assert is_rtl_node(kn) is True
    assert is_hls_node(kn) is False


def test_invalid_implementation_derives_none():
    kn = _unresolved_mvau_node()
    _set_impl(kn, "not_a_real_backend")
    assert kernel_hw_language(kn) is None
    assert is_hls_node(kn) is False
    assert is_rtl_node(kn) is False


def test_classic_node_routing_unchanged():
    """A classic (domain-suffixed) node routes exactly as before — the kernel branch is
    additive and never touches the classic path."""
    hls = helper.make_node(
        "MVAU_hls", ["inp"], ["out"], domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow", name="classic_hls",
    )
    rtl = helper.make_node(
        "MVAU_rtl", ["inp"], ["out"], domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow", name="classic_rtl",
    )
    assert is_hls_node(hls) is True and is_rtl_node(hls) is False
    assert is_rtl_node(rtl) is True and is_hls_node(rtl) is False
    # A kernel node is invisible to the classic-domain path.
    assert kernel_hw_language(hls) is None
