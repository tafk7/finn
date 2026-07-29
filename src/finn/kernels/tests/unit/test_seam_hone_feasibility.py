############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""F2 / D-R5 — backend feasibility is the single source of truth, LEVERAGED at claim time.

The integer-i/w requirement lived in the frontend claim (``op.py``'s ``is_integer`` literal)
as pool feasibility wearing op identity. This proves the full split (scope b):

  * each integer backend's feasibility REJECTS a float32 i/w point (the pushed-down predicate
    is the SoT — D-R5's "push the constraint down FIRST");
  * ``Kernel.has_feasible_point`` is True for an integer MatMul ctx, False for a float one;
  * ``can_infer_from`` rejects a float MatMul FOR THE RIGHT REASON (no feasible backend), with
    NO ``is_integer`` literal surviving in ``op.py``;
  * the structural-match ∧ ¬feasible branch LOGS; a plain non-MatMul is silently skipped;
  * the operand→interface mapping lives in ONE shared helper both callers use.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

import finn.kernels.compute.mvau.op as mvau_op
from finn.kernels.compute.mvau.op import MvauKernelOp, mvau_kernel
from finn.kernels.space import Context, Illegal, Point, resolve

MW, MH = 128, 64
VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _ctx(idt="INT8", wdt="INT8", part=VERSAL):
    return Context(
        shapes={"inp": (1, MW), "weights": (MW, MH), "out": (1, MH)},
        datatypes={"inp": DataType[idt], "weights": DataType[wdt], "out": DataType["INT32"]},
        initializers={"weights": np.ones((MW, MH), dtype=np.float32)},
        fpgapart=part,
    )


def _matmul_model(idt="INT8", wdt="INT8"):
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul],
        "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[idt])
    model.set_tensor_datatype("weights", DataType[wdt])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


# ---------------------------------------------------------------------------
# Step 1 — each integer backend REJECTS a float32 i/w point (the SoT).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ["mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"])
def test_each_backend_rejects_float_iw(impl):
    k = mvau_kernel()
    float_pt = resolve(k.schema(), _ctx(idt="FLOAT32", wdt="FLOAT32"), {"backend": impl})
    assert isinstance(float_pt, Illegal)
    # And the SAME backend accepts a quantized-integer point (feasibility is dtype-specific,
    # not a blanket reject). softvec is the always-buildable DSP core; packed needs w<=8/a<=9;
    # hls builds anywhere — INT8/INT8 satisfies all three.
    int_pt = resolve(k.schema(), _ctx(idt="INT8", wdt="INT8"), {"backend": impl})
    assert isinstance(int_pt, Point)


# ---------------------------------------------------------------------------
# Step 2 — has_feasible_point is the pool query; can_infer_from delegates to it.
# ---------------------------------------------------------------------------


def test_has_feasible_point_true_for_int_false_for_float():
    k = mvau_kernel()
    assert k.has_feasible_point(_ctx(idt="INT8", wdt="INT8")) is True
    assert k.has_feasible_point(_ctx(idt="FLOAT32", wdt="FLOAT32")) is False


def test_can_infer_from_rejects_float_matmul_for_no_feasible_backend(caplog):
    with caplog.at_level(logging.INFO):
        claimed = MvauKernelOp.can_infer_from(
            _matmul_model(idt="FLOAT32", wdt="FLOAT32").graph.node[0],
            _matmul_model(idt="FLOAT32", wdt="FLOAT32"),
        )
    assert claimed is False
    # structural-match ∧ ¬feasible LOGS (distinct from a plain non-MatMul silent skip).
    assert any("no backend has a feasible point" in r.message for r in caplog.records)


def test_can_infer_from_claims_integer_matmul():
    model = _matmul_model(idt="INT8", wdt="INT8")
    assert MvauKernelOp.can_infer_from(model.graph.node[0], model) is True


def test_plain_non_matmul_is_silently_skipped(caplog):
    add = helper.make_node("Add", ["a", "b"], ["c"], name="add0")
    graph = helper.make_graph(
        [add],
        "g",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, MH])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    with caplog.at_level(logging.INFO):
        assert MvauKernelOp.can_infer_from(add, model) is False
    # a non-MatMul is legitimately "not mine" — no feasibility log.
    assert not any("feasible point" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Structural guarantees: no is_integer literal in op.py; one shared operand map.
# ---------------------------------------------------------------------------


def test_no_is_integer_literal_in_op_py():
    src = Path(mvau_op.__file__).read_text()
    assert "is_integer" not in src


def test_operand_map_shared_by_claim_and_build():
    model = _matmul_model()
    node = model.graph.node[0]
    # The same mapping feeds the trial context (claim) and the build (infer_from): a
    # mis-mapped operand would fail both identically.
    mapping = MvauKernelOp._operand_map(node)
    assert mapping == {"inp": "inp", "weights": "weights", "out": "out"}
    trial = MvauKernelOp._trial_context(node, model)
    assert set(trial.shapes) == {"inp", "weights", "out"}
