############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam B — AUTONOMOUS specialization end to end.

The seam that deletes the hand-pin: instead of a test calling
``set_nodeattr("backend", "mvau_hls")``, ``SpecializeKernels(PerNodePolicy(first_feasible))``
SELECTS and commits a backend across the graph on its own. These tests assert a node
specializes AUTONOMOUSLY (infer -> specialize, no hand-pin) and then folds/routes/publishes
its refined output dtype through the estimate tier. Coexistence with classic nodes, the
skip-specialized idempotence, and the no-feasible-backend path round out the seam.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.transformation.fpgadataflow.specialize_kernels import (
    PerNodePolicy,
    SpecializeKernels,
    first_feasible,
)
from finn.kernels.compute.mvau.op import MvauKernelOp
from finn.kernels.compute.thresholding.op import ThresholdingKernelOp
from finn.kernels.ir.routing import is_specialized, kernel_hw_language
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

pytestmark = pytest.mark.integration

MW, MH = 128, 64
NUM_STEPS = 7
KERNEL_DOMAIN = "finn.kernels"


def _infer():
    return InferKernels([MvauKernelOp, ThresholdingKernelOp])


def _specialize():
    return SpecializeKernels(PerNodePolicy(first_feasible))


def _thresholds(rows, seed=2, lo=0, hi=100):
    return np.sort(
        np.random.RandomState(seed).randint(lo, hi, size=(rows, NUM_STEPS)).astype(np.float32),
        axis=1,
    )


def _matmul_only_model(idt="INT8", wdt="INT8"):
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul], "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[idt])
    model.set_tensor_datatype("weights", DataType[wdt])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def _standalone_threshold_model(channels=MH):
    mt = helper.make_node(
        "MultiThreshold", ["inp", "thresholds"], ["out"],
        domain="qonnx.custom_op.general", out_dtype="INT4", out_scale=1.0,
        out_bias=-8.0, name="mt_solo",
    )
    graph = helper.make_graph(
        [mt], "standalone_threshold",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, channels])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, channels])],
        value_info=[helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [channels, NUM_STEPS])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_tensor_datatype("out", DataType["INT4"])
    model.set_initializer("thresholds", _thresholds(channels))
    return model


def _kernel_node(model):
    return [n for n in model.graph.node if n.domain == KERNEL_DOMAIN][0]


def _backend(node):
    attr = get_by_name(node.attribute, "backend")
    return attr.s.decode("UTF-8") if attr is not None else None


# --- autonomy: MVAU specializes with no hand-pin ---------------------------


def test_mvau_specializes_autonomously():
    model = _matmul_only_model().transform(_infer())
    node = _kernel_node(model)
    assert is_specialized(node) is False  # infer leaves it unspecialized

    model = model.transform(_specialize())
    node = _kernel_node(model)
    # SELECTED autonomously — no set_nodeattr("backend", ...) anywhere in this test.
    assert is_specialized(node) is True
    assert _backend(node) == "mvau_hls"  # first feasible pool member for the integer node
    assert kernel_hw_language(node) == "hls"
    assert is_hls_node(node) is True and is_rtl_node(node) is False


def test_mvau_folds_at_default_after_autonomous_specialize():
    # Once specialized (backend committed, folding unpinned = INCREMENTAL staging), the
    # impl-dependent getters resolve at the DEFAULT fold — the node is ready for SetFolding.
    model = _matmul_only_model().transform(_infer()).transform(_specialize())
    inst = model.get_customop_wrapper(_kernel_node(model))
    assert tuple(inst.get_folded_output_shape(0))  # resolves (no "unspecialized" raise)
    assert inst.get_instream_width(0) > 0
    assert inst.get_outstream_width(0) > 0
    assert inst.get_exp_cycles() > 0


def test_mvau_output_dtype_refined_by_autonomous_specialize():
    # Before specialize: unspecialized node publishes the raw graph output dtype.
    model = _matmul_only_model().transform(_infer())
    node = _kernel_node(model)
    inst = model.get_customop_wrapper(node)
    inst.infer_node_datatype(model)
    assert model.get_tensor_datatype(node.output[0]) == DataType["FLOAT32"]

    # After autonomous specialize: the backend-derived integer accumulator, refined by the
    # InferDataTypes the transform runs itself.
    model = model.transform(_specialize())
    node = _kernel_node(model)
    odt = model.get_tensor_datatype(node.output[0])
    assert odt.is_integer() and odt != DataType["FLOAT32"]


# --- autonomy: op-agnostic (Thresholding rides the same seam) ---------------


def test_thresholding_specializes_autonomously():
    model = _standalone_threshold_model().transform(_infer()).transform(_specialize())
    node = _kernel_node(model)
    assert node.op_type == "Thresholding"
    assert is_specialized(node) is True  # zero per-op branching in the transform
    assert kernel_hw_language(node) in ("hls", "rtl")


# --- coexistence: classic nodes untouched ----------------------------------


def test_mixed_graph_only_kernel_nodes_specialize():
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    pool = helper.make_node("MaxPool", ["mm_out"], ["out"], kernel_shape=[1, 1], name="pool0")
    graph = helper.make_graph(
        [matmul, pool], "mixed",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
            helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, MH]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model = model.transform(_infer()).transform(_specialize())

    kn = _kernel_node(model)
    assert is_specialized(kn) is True
    # The classic MaxPool is untouched — no `backend` axis committed on it.
    maxpool = [n for n in model.graph.node if n.op_type == "MaxPool"][0]
    assert get_by_name(maxpool.attribute, "backend") is None


# --- idempotence: skip-specialized -----------------------------------------


def test_specialize_is_idempotent_and_respects_prior_pin():
    model = _matmul_only_model().transform(_infer())
    node = _kernel_node(model)
    # Hand-pin a NON-default backend, then run the transform: it must skip an already-
    # specialized node and leave the pin intact (the policy gates on is_specialized).
    model.get_customop_wrapper(node).set_nodeattr("backend", "mvau_dsp_softvec")
    model = model.transform(_specialize())
    assert _backend(_kernel_node(model)) == "mvau_dsp_softvec"

    # Re-running on an autonomously-specialized graph is a no-op.
    fresh = _matmul_only_model().transform(_infer()).transform(_specialize())
    once = _backend(_kernel_node(fresh))
    fresh = fresh.transform(_specialize())
    assert _backend(_kernel_node(fresh)) == once


# --- no feasible backend: left unspecialized, logged -----------------------


def test_no_feasible_backend_leaves_node_unspecialized(caplog):
    # A float node is structurally a MatMul but has no feasible (integer-only) backend. It
    # never gets inferred to a kernel node (can_infer_from rejects it), so build a kernel node
    # via infer on an integer graph, then re-point its dtypes to float and re-specialize.
    import logging

    model = _matmul_only_model(idt="FLOAT32", wdt="FLOAT32")
    # can_infer_from rejects float -> no kernel node produced -> nothing to specialize.
    model = model.transform(_infer())
    assert not [n for n in model.graph.node if n.domain == KERNEL_DOMAIN]

    # Direct path: an integer-inferred node whose context later reads float has no feasible
    # backend; first_feasible logs and returns None, node stays unspecialized.
    model = _matmul_only_model().transform(_infer())
    node = _kernel_node(model)
    for t in ("inp", "weights", "out"):
        model.set_tensor_datatype(t, DataType["FLOAT32"])
    with caplog.at_level(logging.WARNING):
        model = model.transform(_specialize())
    assert is_specialized(_kernel_node(model)) is False
    assert any("no feasible backend" in r.message for r in caplog.records)
