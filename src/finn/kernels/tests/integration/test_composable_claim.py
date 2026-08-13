############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``can_infer_from`` is composable truth — no hand-written escapes (engine hone Task 3.4).

MVAU's frontend claim used to hand-code two exceptions past the pool delegation:

    if model.get_tensor_sparsity(node.input[1]) is not None: return False
    if model.get_initializer(node.input[1]) is None:         return False

The second is exactly what ``constraints.IsStatic`` was built to express — and ``IsStatic``
was DEAD CODE, declared in the vocabulary and used nowhere. So the fact lived in Python, in
the frontend, where it could drift from the pool it was meant to describe, while the
mechanism for stating it declaratively sat unused.

Both are now constraints on the ``weights`` interface, and the claim is exactly
``op_type == "MatMul" AND some backend can build it``. The point is not that the rejections
still happen — it is that they happen FOR THE DECLARED REASON, so a backend that can consume
dynamic or sparse weights widens what infer accepts by declaring so, with no frontend edit.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.kernels.compute.mvau.op import MvauDataflowOp
from finn.kernels.engine.resolve import resolve


def _matmul_model(static=True, sparse=False, idt="INT4", wdt="INT4", mw=8, mh=8):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, mh])
    wt = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [mw, mh])
    node = helper.make_node("MatMul", ["inp", "weights"], ["out"])
    graph = helper.make_graph(
        [node], "g", [inp] + ([] if static else [wt]), [out],
        value_info=[wt] if static else [],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[idt])
    model.set_tensor_datatype("weights", DataType[wdt])
    model.set_tensor_datatype("out", DataType["INT16"])
    if static:
        model.set_initializer(
            "weights", np.random.default_rng(0).integers(-7, 8, (mw, mh)).astype(np.float32)
        )
    if sparse:
        model.set_tensor_sparsity("weights", {"dw": {"kernel_shape": [3, 3]}})
    return model, node


def _rejection_reasons(model, node):
    """Every reason the pool gives for this node, across all backends.

    Sources the Context the way the claim now does — from a CANDIDATE kernel node over the
    slots `infer_from` would use — rather than from a per-op trial builder. That is the F9
    point: one Context builder, so a given cannot be present for the claim and missing for
    the build (which is how `sparsity` came to be carried by one and dropped by the other)."""
    inputs, outputs = MvauDataflowOp._candidate_slots(node, model)
    ctx = MvauDataflowOp.candidate_op(model, inputs, outputs)._context()
    space = MvauDataflowOp.compile()
    reasons = []
    for backend in MvauDataflowOp.pool:
        try:
            r = resolve(space, ctx, {"backend": backend.name})
        except Exception:
            continue  # part-less probe context — the signal first_feasible_backend tolerates
        reasons.extend(getattr(r, "reasons", ()))
    return reasons


def test_dense_static_integer_matmul_is_claimed():
    model, node = _matmul_model()
    assert MvauDataflowOp.can_infer_from(node, model) is True


def test_dynamic_weight_matmul_rejected_by_IsStatic():
    """Was a hand-written `get_initializer(...) is None` escape; now the IsStatic constraint.
    The reason string is the evidence that the DECLARED rule did the rejecting."""
    model, node = _matmul_model(static=False)
    assert MvauDataflowOp.can_infer_from(node, model) is False
    assert any("initializer required" in r for r in _rejection_reasons(model, node))


def test_sparse_weight_matmul_rejected_by_SparsityFree():
    """Was a hand-written `get_tensor_sparsity(...) is not None` escape; now a constraint.
    Requires Context to carry sparsity — a given the trial context must not drop, or the
    rule silently cannot fire."""
    model, node = _matmul_model(sparse=True)
    assert MvauDataflowOp.can_infer_from(node, model) is False
    assert any("sparsity" in r for r in _rejection_reasons(model, node))


def test_float_matmul_still_rejected_by_datatype_support():
    """Unchanged behaviour, included so the rewrite is not silently widening the claim."""
    model, node = _matmul_model(idt="FLOAT32", wdt="FLOAT32")
    assert MvauDataflowOp.can_infer_from(node, model) is False
    assert any("not integer" in r for r in _rejection_reasons(model, node))


def test_claim_has_no_hand_written_escapes():
    """The structural point of the task: the claim body must not reach into the model for
    design-space facts. A future edit re-adding an escape should fail here and be made to
    justify itself as a constraint instead."""
    import inspect

    src = inspect.getsource(MvauDataflowOp.can_infer_from)
    body = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "get_tensor_sparsity" not in body
    assert "get_initializer" not in body


def test_context_carries_sparsity_from_model():
    """SparsityFree can only work if the given reaches Context at all."""
    from finn.kernels.engine.context import Context

    model, _ = _matmul_model(sparse=True)
    ctx = Context.from_model(model, "")
    assert ctx.tensor_sparsity("weights") is not None

    dense, _ = _matmul_model(sparse=False)
    assert Context.from_model(dense, "").tensor_sparsity("weights") is None


def test_the_claim_leaves_the_graph_unmodified():
    """The candidate node is built, wrapped and interrogated — never inserted. Asserted on
    SERIALIZED BYTES, not node identity: protobuf hands back a fresh Python wrapper on each
    access to a repeated field, so an id()-based check reports a change for an untouched
    graph."""
    model, node = _matmul_model()
    before = model.model.SerializeToString()
    MvauDataflowOp.can_infer_from(node, model)
    assert model.model.SerializeToString() == before


def test_sparsity_reaches_a_real_kernel_nodes_context():
    """The second F9-family defect, found by collapsing the two Context builders into one.

    `_context` (the BUILD path) did not carry sparsity while `_trial_context` (the CLAIM
    path) did — so the weights port's declared `SparsityFree` constraint held at the claim
    and was silently inert on the resulting kernel node. One builder now, and this pins the
    given that was being dropped."""
    model, matmul = _matmul_model(sparse=True)
    inputs, outputs = MvauDataflowOp._candidate_slots(matmul, model)
    ctx = MvauDataflowOp.candidate_op(model, inputs, outputs)._context()
    assert ctx.tensor_sparsity("weights") is not None
