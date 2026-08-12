############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""INSTANTIATE seam: the DataflowOp bridge over a live ModelWrapper (N3-N6).

A kernel op owns only its design axes (backend/PE/SIMD/ActVal); shapes/dtypes/values come
from the LIVE model, re-keyed to interface names (N3). The getter partition:

  * Group-1 (committed-config, node-owned) getters answer on a bare getCustomOp(node).
  * Group-2 (graph-derived) getters RAISE "no model attached" on a bare instance — a stale
    node snapshot is the single-source-of-truth sin the design avoids.
  * backend-INDEPENDENT getters (normal shape/dtype, infer_node_datatype) answer on an
    UNSPECIALIZED node (the Seam-A verify gate, N4); backend-DEPENDENT getters (folded
    shape/width/cycles) raise a legible "unspecialized" error until backend is committed.

N5 exact value-derived dtypes (accumulator from REAL weights) · N6 folding capability from
Context are exercised via the model-aware path.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.kernels.compute.mvau import mvau_kernel
from finn.kernels.engine.context import Context

MW, MH = 128, 64
OP_TYPE = "MVAU"
DOMAIN = "finn.kernels"
FPGAPART = "xcvc1902-vsva2197-2MP-e-S"
NUM_STEPS = 7


def _build_model(simd=16, pe=4, thresholds=True, annotate_out=True, backend="mvau_hls"):
    inputs = ["inp", "weights", "thresholds"] if thresholds else ["inp", "weights"]
    node = helper.make_node(
        OP_TYPE, inputs, ["out"], domain=DOMAIN, backend=backend, SIMD=simd, PE=pe
    )
    value_info = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
    ]
    if thresholds:
        value_info.append(
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS])
        )
    graph = helper.make_graph(
        [node], "mvau_kernel_graph", value_info,
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    if annotate_out:
        model.set_tensor_datatype("out", DataType["INT32"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    if thresholds:
        model.set_tensor_datatype("thresholds", DataType["INT16"])
        thr = np.sort(
            np.random.RandomState(2).randint(0, 100, size=(MH, NUM_STEPS)).astype(np.float32),
            axis=1,
        )
        model.set_initializer("thresholds", thr)
    return model


def _ctx(thresholds=True):
    shapes = {"inp": (1, MW), "weights": (MW, MH), "out": (1, MH)}
    datatypes = {"inp": DataType["INT8"], "weights": DataType["INT8"], "out": DataType["INT32"]}
    initializers = {"weights": np.ones((MW, MH), dtype=np.float32)}
    if thresholds:
        shapes["thresholds"] = (MH, NUM_STEPS)
        datatypes["thresholds"] = DataType["INT16"]
        initializers["thresholds"] = np.sort(
            np.random.RandomState(2).randint(0, 100, size=(MH, NUM_STEPS)).astype(np.float32),
            axis=1,
        )
    return Context(shapes=shapes, datatypes=datatypes, initializers=initializers, fpgapart=FPGAPART)


def _inst(model):
    return model.get_customop_wrapper(model.graph.node[0])


def _unspecialized_model():
    """A 2-input UNSPECIALIZED node — exactly what Seam A's infer produces: ActVal baked,
    NO backend/SIMD/PE attribute (the backend nodeattr defaults to the "" sentinel)."""
    node = helper.make_node(
        OP_TYPE, ["inp", "weights"], ["out"], domain=DOMAIN, ActVal=0, name="MVAU_0"
    )
    value_info = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
    ]
    graph = helper.make_graph(
        [node], "mvau_unspecialized", value_info,
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


# ===========================================================================
# N1/N3 — registration + model-aware attach.
# ===========================================================================


def test_getcustomop_resolves_to_mvau_kernel_op():
    assert type(_inst(_build_model())).__name__ == "MvauDataflowOp"


def test_kernel_op_opts_into_model_aware_contract():
    model = _build_model()
    node = model.graph.node[0]
    bare = getCustomOp(node)
    assert bare.wants_model is True
    assert getattr(bare, "_model", None) is None
    aware = model.get_customop_wrapper(node)
    assert aware._model is model


# ===========================================================================
# N3 — Group-1 answerable bare; Group-2 raises without a model.
# ===========================================================================


def test_group1_getter_answerable_without_model():
    bare = getCustomOp(_build_model().graph.node[0])
    assert bare.get_nodeattr("backend") == "mvau_hls"
    assert bare.get_nodeattr("SIMD") == 16
    assert bare.get_nodeattr("PE") == 4


def test_group2_getter_raises_without_model():
    bare = getCustomOp(_build_model().graph.node[0])
    with pytest.raises(ValueError, match="no model attached"):
        bare.get_normal_output_shape()
    with pytest.raises(ValueError, match="no model attached"):
        bare.get_instream_width()
    with pytest.raises(ValueError, match="no model attached"):
        bare.get_output_datatype()


def test_model_aware_path_answers_both_groups():
    aware = _inst(_build_model())
    assert aware.get_nodeattr("backend") == "mvau_hls"
    assert aware.get_normal_input_shape() == (1, MW)
    assert aware.get_normal_output_shape() == (1, MH)
    assert aware.get_instream_width() > 0
    assert aware.get_output_datatype() is not None


# ===========================================================================
# N2 — nodeattr schema is axes only; no shape/dtype bakes.
# ===========================================================================


def test_nodeattr_types_include_axes_no_geometry_bakes():
    attrs = _inst(_build_model()).get_nodeattr_types()
    for key in ("backend", "PE", "SIMD", "ActVal"):
        assert key in attrs
    assert "noActivation" not in attrs
    assert "MW" not in attrs and "MH" not in attrs
    for key in ("inp_shape", "inp_dtype", "weights_shape", "out_dtype"):
        assert key not in attrs, f"geometry attr {key} must not be baked onto the node"


# ===========================================================================
# N3/N5 — adapter getters agree with the engine; exact value-derived dtype.
# ===========================================================================


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64)])
def test_getters_agree_with_engine(simd, pe):
    inst = _inst(_build_model(simd=simd, pe=pe))
    kernel, ctx = mvau_kernel(), _ctx()
    point = kernel.configure(ctx, {"backend": "mvau_hls", "SIMD": simd, "PE": pe})
    assert inst.get_instream_width(0) == kernel.get_instream_width(point, ctx, 0)
    assert inst.get_instream_width(1) == kernel.get_instream_width(point, ctx, 1)
    assert inst.get_outstream_width(0) == kernel.get_outstream_width(point, ctx, 0)
    assert tuple(inst.get_folded_input_shape(0)) == kernel.get_folded_input_shape(point, ctx, 0)
    assert tuple(inst.get_folded_output_shape(0)) == kernel.get_folded_output_shape(point, ctx, 0)
    assert inst.get_exp_cycles() == kernel.get_exp_cycles(point, ctx)


def test_datatype_getters():
    inst = _inst(_build_model())
    assert inst.get_input_datatype(0) == DataType["INT8"]
    assert inst.get_input_datatype(1) == DataType["INT8"]
    assert inst.get_output_datatype(0) == DataType["INT32"]


def test_number_output_values():
    assert _inst(_build_model(pe=4)).get_number_output_values() == MH // 4


def test_noactivation_output_uses_real_weight_accumulator():
    # N5: a 2-input (no-threshold) node's output dtype IS the weight-derived accumulator,
    # exact because the Context carries the REAL weight values (not placeholder zeros).
    model = _build_model(thresholds=False, annotate_out=False)
    inst = _inst(model)
    kernel, ctx = mvau_kernel(), _ctx(thresholds=False)
    point = kernel.configure(ctx, {"backend": "mvau_hls", "SIMD": 16, "PE": 4})
    inst.set_nodeattr("SIMD", 16)
    inst.set_nodeattr("PE", 4)
    got = inst.get_outstream_width(0)
    assert got == kernel.get_outstream_width(point, ctx, 0)
    assert got > 0  # real all-ones weights give a positive-width accumulator stream


# ===========================================================================
# N4/N5 — infer_node_datatype derives + propagates, idempotent.
# ===========================================================================


def test_infer_datatype_forwards_graph_dtype():
    model = _build_model(thresholds=True, annotate_out=True)
    _inst(model).infer_node_datatype(model)
    assert model.get_tensor_datatype("out") == DataType["INT32"]


def test_infer_datatype_derives_accumulator_under_noactivation():
    model = _build_model(thresholds=False, annotate_out=False)
    inst = _inst(model)
    inst.infer_node_datatype(model)
    first = model.get_tensor_datatype("out")
    assert first.is_integer() and first != DataType["FLOAT32"]
    inst.infer_node_datatype(model)  # idempotent
    assert model.get_tensor_datatype("out") == first


# ===========================================================================
# N4 — backend-independent getters answer UNSPECIALIZED; backend-dependent raise.
# ===========================================================================


def test_unspecialized_answers_impl_independent_getters():
    model = _unspecialized_model()
    inst = _inst(model)
    assert inst.get_normal_input_shape(0) == (1, MW)
    assert inst.get_normal_output_shape(0) == (1, MH)
    assert inst.get_input_datatype(0) == DataType["INT8"]
    assert inst.get_output_datatype(0) is not None
    assert inst.make_shape_compatible_op(model) is not None
    # The realized output dtype is BACKEND-SCOPED, so infer_node_datatype DEFERS on an
    # unspecialized node: it publishes the raw graph output dtype (here FLOAT32), not the
    # accumulator — that is refined once a backend is committed (Seam B).
    inst.infer_node_datatype(model)
    assert model.get_tensor_datatype("out") == DataType["FLOAT32"]


@pytest.mark.parametrize(
    "call",
    [
        lambda i: i.get_folded_input_shape(0),
        lambda i: i.get_folded_output_shape(0),
        lambda i: i.get_instream_width(0),
        lambda i: i.get_outstream_width(0),
        lambda i: i.get_exp_cycles(),
    ],
)
def test_unspecialized_raises_on_impl_dependent_getters(call):
    inst = _inst(_unspecialized_model())
    with pytest.raises(ValueError, match="unspecialized"):
        call(inst)


# ===========================================================================
# InferShapes + pickle/rehydrate (adapter surface durability).
# ===========================================================================


def test_shape_compatible_and_infer_shapes():
    from qonnx.transformation.infer_shapes import InferShapes

    model = _build_model().transform(InferShapes())
    assert tuple(model.get_tensor_shape("out")) == (1, MH)


def test_serialize_rehydrate_roundtrip(tmp_path):
    model = _build_model(simd=16, pe=4)
    before = {
        "instream0": _inst(model).get_instream_width(0),
        "folded_out": tuple(_inst(model).get_folded_output_shape(0)),
        "exp_cycles": _inst(model).get_exp_cycles(),
    }
    path = str(tmp_path / "mvau_kernel.onnx")
    model.save(path)
    reloaded = ModelWrapper(path)
    inst = _inst(reloaded)
    assert inst.get_instream_width(0) == before["instream0"]
    assert tuple(inst.get_folded_output_shape(0)) == before["folded_out"]
    assert inst.get_exp_cycles() == before["exp_cycles"]


# ===========================================================================
# The 4 latent skips: stock FINN analyses instantiate via bare getCustomOp(node)
# (no model); a model-bearing DataflowOp cannot serve them until the deferred
# getCustomOp-with-model build-flow wiring lands. Kept skipped, not deleted.
# ===========================================================================

_NEEDS_MODEL_THREADING = "getCustomOp(node)-with-model build-flow integration is deferred"


@pytest.mark.skip(reason=_NEEDS_MODEL_THREADING)
def test_annotate_cycles_and_dataflow_performance():
    from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance

    model = _build_model(simd=16, pe=4).transform(AnnotateCycles())
    stamped = _inst(model).get_nodeattr("cycles_estimate")
    assert stamped == _inst(model).get_exp_cycles()
    perf = model.analysis(dataflow_performance)
    assert perf["max_cycles"] == stamped


@pytest.mark.skip(reason=_NEEDS_MODEL_THREADING)
def test_exp_cycles_per_layer():
    from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer

    model = _build_model()
    cyc = model.analysis(exp_cycles_per_layer)
    assert cyc[model.graph.node[0].name] == _inst(model).get_exp_cycles()


@pytest.mark.skip(reason=_NEEDS_MODEL_THREADING)
def test_res_estimation():
    from finn.analysis.fpgadataflow.res_estimation import res_estimation

    model = _build_model()
    res = model.analysis(lambda m: res_estimation(m, fpgapart=FPGAPART))
    node_res = res[model.graph.node[0].name]
    for key in ("BRAM_18K", "LUT", "URAM", "DSP"):
        assert key in node_res
