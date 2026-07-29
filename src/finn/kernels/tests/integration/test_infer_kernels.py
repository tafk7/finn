############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""INFER seam — the frontend claim path, end to end (I1-I6).

The highest-value integration test: it most protects the frontend. InferKernels claims
MatMul[+MultiThreshold]→MVAU (thresholds absorbed, ONLY ActVal baked — no MW/MH/SIMD/PE/
dtypes), standalone MultiThreshold→Thresholding; the classic infer is bypassed (no
double-infer); InferShapes/InferDataTypes survive and publish the kernel output dtype; a
mixed graph claims only the kernel pattern. INV5 (legibility): a broken can_infer_from /
infer_from PROPAGATES; only the narrow ValueError/KeyError resolve surface skips-with-a-
warning; an unexpected error type still propagates. The DROPPED test_seam_a_verify_gate is
superseded by the ir bridge's unspecialized getter-state contract.
"""

import logging

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.kernels.compute.mvau.op import MvauKernelOp
from finn.kernels.compute.thresholding.op import ThresholdingKernelOp

pytestmark = pytest.mark.integration

MW, MH = 128, 64
NUM_STEPS = 7
KERNEL_DOMAIN = "finn.kernels"


def _pool():
    return InferKernels([MvauKernelOp, ThresholdingKernelOp])


def _thresholds(rows, seed=2, lo=0, hi=100):
    return np.sort(
        np.random.RandomState(seed).randint(lo, hi, size=(rows, NUM_STEPS)).astype(np.float32),
        axis=1,
    )


def _matmul_threshold_model(out_bias=-8):
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    mt = helper.make_node(
        "MultiThreshold", ["mm_out", "thresholds"], ["out"],
        domain="qonnx.custom_op.general", out_dtype="INT4", out_scale=1.0,
        out_bias=float(out_bias), name="mt0",
    )
    graph = helper.make_graph(
        [matmul, mt], "mm_threshold",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS]),
            helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, MH]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_tensor_datatype("out", DataType["INT4"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model.set_initializer("thresholds", _thresholds(MH))
    return model


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


def _node_by_domain(model, domain):
    return [n for n in model.graph.node if n.domain == domain]


def _attr_names(node):
    return {a.name for a in node.attribute}


# --- I3/I5: fused claim + bake contract ------------------------------------


def test_mvau_fused_infer_produces_kernel_node():
    model = _matmul_threshold_model(out_bias=-8).transform(_pool())
    kernel_nodes = _node_by_domain(model, KERNEL_DOMAIN)
    assert len(kernel_nodes) == 1
    kn = kernel_nodes[0]
    assert kn.op_type == "MVAU"
    assert [n.op_type for n in model.graph.node] == ["MVAU"]
    assert list(kn.input) == ["inp", "weights", "thresholds"]
    names = _attr_names(kn)
    assert "ActVal" in names
    for forbidden in ("MW", "MH", "SIMD", "PE", "numInputVectors", "mem_mode",
                      "inputDataType", "weightDataType", "outputDataType", "noActivation"):
        assert forbidden not in names, f"{forbidden} must not be baked on the kernel node"


def test_mvau_actval_carries_out_bias():
    model = _matmul_threshold_model(out_bias=-8).transform(_pool())
    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    assert get_by_name(kn.attribute, "ActVal").i == -8


def test_mvau_matmul_only_infer():
    model = _matmul_only_model().transform(_pool())
    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    assert kn.op_type == "MVAU"
    assert list(kn.input) == ["inp", "weights"]


def test_standalone_threshold_infer_produces_kernel_node():
    model = _standalone_threshold_model().transform(_pool())
    kernel_nodes = _node_by_domain(model, KERNEL_DOMAIN)
    assert len(kernel_nodes) == 1
    kn = kernel_nodes[0]
    assert kn.op_type == "Thresholding"
    assert [n.op_type for n in model.graph.node] == ["Thresholding"]
    assert list(kn.input) == ["inp", "thresholds"]
    names = _attr_names(kn)
    assert "ActVal" in names
    for forbidden in ("NumChannels", "numSteps", "PE", "numInputVectors",
                      "inputDataType", "outputDataType"):
        assert forbidden not in names


# --- I6: classic infer bypassed (no double-infer) --------------------------


def test_classic_mvau_infer_finds_nothing_after_kernel_infer():
    from finn.transformation.fpgadataflow.convert_to_hw_layers import (
        InferQuantizedMatrixVectorActivation,
    )

    model = _matmul_threshold_model().transform(_pool())
    assert not model.get_nodes_by_op_type("MatMul")
    before = [n.name for n in model.graph.node]
    model = model.transform(InferQuantizedMatrixVectorActivation())
    assert before == [n.name for n in model.graph.node]
    assert len(_node_by_domain(model, KERNEL_DOMAIN)) == 1
    assert not _node_by_domain(model, "finn.custom_op.fpgadataflow")


# --- I4/I6: cleanup passes survive; output dtype published -----------------


def test_infer_shapes_and_datatypes_survive():
    model = _matmul_threshold_model().transform(_pool())
    model = model.transform(InferShapes()).transform(InferDataTypes())
    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    assert tuple(model.get_tensor_shape(kn.output[0])) == (1, MH)
    assert model.get_tensor_datatype(kn.output[0]) == DataType["INT4"]


def test_matmul_only_publishes_accumulator_dtype():
    model = _matmul_only_model().transform(_pool())
    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    odt = model.get_tensor_datatype(kn.output[0])
    assert odt.is_integer() and odt != DataType["FLOAT32"]


# --- I1/I2: mixed graph — kernel claims only its pattern -------------------


def test_mixed_graph_kernel_claims_only_its_pattern():
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    mt = helper.make_node(
        "MultiThreshold", ["mm_out", "thresholds"], ["mt_out"],
        domain="qonnx.custom_op.general", out_dtype="INT4", out_scale=1.0, out_bias=-8.0, name="mt0",
    )
    pool = helper.make_node("MaxPool", ["mt_out"], ["out"], kernel_shape=[1, 1], name="pool0")
    graph = helper.make_graph(
        [matmul, mt, pool], "mixed",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS]),
            helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, MH]),
            helper.make_tensor_value_info("mt_out", TensorProto.FLOAT, [1, MH]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model.set_initializer("thresholds", _thresholds(MH))
    model = model.transform(_pool())
    op_types = [n.op_type for n in model.graph.node]
    assert "MVAU" in op_types and "MaxPool" in op_types
    assert "MatMul" not in op_types and "MultiThreshold" not in op_types
    assert len(_node_by_domain(model, KERNEL_DOMAIN)) == 1


# --- I4 (INV5): the driver surfaces kernel bugs, not swallows them ----------


class _BrokenPredicate(MvauKernelOp):
    @classmethod
    def can_infer_from(cls, node, model):
        raise AttributeError("bug in can_infer_from")


class _BrokenBuilder(MvauKernelOp):
    @classmethod
    def infer_from(cls, node, model, insert_index):
        raise AttributeError("bug in infer_from")


def test_broken_can_infer_from_propagates():
    with pytest.raises(AttributeError, match="bug in can_infer_from"):
        _matmul_only_model().transform(InferKernels([_BrokenPredicate]))


def test_broken_infer_from_propagates():
    with pytest.raises(AttributeError, match="bug in infer_from"):
        _matmul_only_model().transform(InferKernels([_BrokenBuilder]))


def test_unexpected_validation_error_propagates(monkeypatch):
    def _boom(self, model):
        raise AttributeError("unexpected bug during validation")

    monkeypatch.setattr(MvauKernelOp, "infer_node_datatype", _boom)
    with pytest.raises(AttributeError, match="unexpected bug during validation"):
        _matmul_only_model().transform(InferKernels([MvauKernelOp]))


def test_legitimate_instantiation_failure_skips_with_warning(monkeypatch, caplog):
    def _illegal(self, model):
        raise ValueError("node cannot legally instantiate")

    monkeypatch.setattr(MvauKernelOp, "infer_node_datatype", _illegal)
    model = _matmul_only_model()
    with caplog.at_level(logging.WARNING):
        out = model.transform(InferKernels([MvauKernelOp]))
    assert [n.op_type for n in out.graph.node] == ["MatMul"]
    assert any("validation" in r.message for r in caplog.records)


# --- the real build-step injection (folded from hardware/test_seam_a_step_convert) ---
# Drives the actual step_convert_to_hw build function (not just the transform sequence).
# Needs FINN's build-flow module graph (pinned onnx providing onnx.mapping) — skip if absent.


def test_step_convert_to_hw_kernel_claims_pattern():
    pytest.importorskip("onnx.mapping")
    try:
        from finn.builder.build_dataflow_config import DataflowBuildConfig
        from finn.builder.build_dataflow_steps import step_convert_to_hw
    except ImportError as exc:
        pytest.skip(f"build-flow import unavailable: {exc}")

    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    mt = helper.make_node(
        "MultiThreshold", ["mm_out", "thresholds"], ["mt_out"],
        domain="qonnx.custom_op.general", out_dtype="INT4", out_scale=1.0, out_bias=-8.0, name="mt0",
    )
    pool = helper.make_node("MaxPool", ["mt_out"], ["out"], kernel_shape=[1, 1], name="pool0")
    graph = helper.make_graph(
        [matmul, mt, pool], "mixed",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS]),
            helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, MH]),
            helper.make_tensor_value_info("mt_out", TensorProto.FLOAT, [1, MH]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model.set_initializer("thresholds", _thresholds(MH))

    cfg = DataflowBuildConfig(
        output_dir="/tmp/seam_a_step_convert",
        synth_clk_period_ns=10.0,
        generate_outputs=[],
        standalone_thresholds=True,
    )
    model = step_convert_to_hw(model, cfg)
    op_types = [n.op_type for n in model.graph.node]
    kernel_nodes = [n for n in model.graph.node if n.domain == KERNEL_DOMAIN]
    assert len(kernel_nodes) == 1 and kernel_nodes[0].op_type == "MVAU"
    assert "MaxPool" in op_types
    assert "MatMul" not in op_types and "MultiThreshold" not in op_types
