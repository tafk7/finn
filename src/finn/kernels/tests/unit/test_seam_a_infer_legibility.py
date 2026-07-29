############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""F3 — the infer driver surfaces kernel BUGS instead of swallowing them (INV5).

The seam exists to prevent the invisible-lossy outcome: a defect that silently leaves a
classic MatMul on FINN's path. So:

  * a broken ``can_infer_from`` (a predicate that raises) PROPAGATES — a total predicate
    that crashes is a kernel bug, not a "no match".
  * a broken ``infer_from`` (raises after the claim) PROPAGATES — the claim promised a
    legal build the builder could not deliver.
  * a node that legitimately fails to INSTANTIATE (the ``ValueError``/``KeyError`` resolve
    surface) is the ONE case that skips-with-a-warning — that filter is the seam's purpose.
    An unexpected error type there (e.g. ``AttributeError`` from a typo) still propagates.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.kernels.compute.mvau.op import MvauKernelOp

MW, MH = 128, 64


def _matmul_model():
    """A quantized-weight MatMul the MVAU kernel legitimately claims."""
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul],
        "matmul_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


class _BrokenPredicate(MvauKernelOp):
    """A kernel whose CLAIM CHECK is broken (raises). A total predicate must not crash."""

    @classmethod
    def can_infer_from(cls, node, model):
        raise AttributeError("bug in can_infer_from")


class _BrokenBuilder(MvauKernelOp):
    """A kernel that claims correctly but whose BUILDER is broken (raises)."""

    @classmethod
    def infer_from(cls, node, model, insert_index):
        raise AttributeError("bug in infer_from")


def test_broken_can_infer_from_propagates():
    with pytest.raises(AttributeError, match="bug in can_infer_from"):
        _matmul_model().transform(InferKernels([_BrokenPredicate]))


def test_broken_infer_from_propagates():
    with pytest.raises(AttributeError, match="bug in infer_from"):
        _matmul_model().transform(InferKernels([_BrokenBuilder]))


def test_unexpected_validation_error_propagates(monkeypatch):
    """The inserted node instantiates via the registry (a real MvauKernelOp). An UNEXPECTED
    error type from its ``infer_node_datatype`` (an ``AttributeError``, not the resolve
    ValueError/KeyError surface) is a real bug — it PROPAGATES, not swallowed as a skip."""

    def _boom(self, model):
        raise AttributeError("unexpected bug during validation")

    monkeypatch.setattr(MvauKernelOp, "infer_node_datatype", _boom)
    with pytest.raises(AttributeError, match="unexpected bug during validation"):
        _matmul_model().transform(InferKernels([MvauKernelOp]))


def test_legitimate_instantiation_failure_skips_with_warning(monkeypatch, caplog):
    """A node that legitimately fails to resolve (a ``ValueError`` from _point/infer) is the
    ONE case that skips-with-a-warning, leaving the frontend node classic — the seam's
    filter, its narrow expected surface."""
    import logging

    def _illegal(self, model):
        raise ValueError("node cannot legally instantiate")

    monkeypatch.setattr(MvauKernelOp, "infer_node_datatype", _illegal)
    model = _matmul_model()
    with caplog.at_level(logging.WARNING):
        out = model.transform(InferKernels([MvauKernelOp]))
    # The frontend MatMul survives (kernel inference was skipped), no kernel node committed.
    assert [n.op_type for n in out.graph.node] == ["MatMul"]
    assert any("validation" in r.message for r in caplog.records)
