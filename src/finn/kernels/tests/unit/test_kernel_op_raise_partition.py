############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam D — the Group-1/Group-2 getter-raise partition (design §7.3, D-D1).

A kernel op derives its shapes/dtypes/widths from live graph context, so those
getters (Group 2) MUST raise when the op is built WITHOUT a model — a bare
``getCustomOp(node)`` cannot answer them, and fabricating an answer from a stale
node snapshot is the single-source-of-truth sin the whole kernel design avoids.
Committed-config getters (Group 1, e.g. the ``implementation`` nodeattr) are
node-owned and answerable context-free.

The model-aware path (``model.get_customop_wrapper(node)``) attaches the model
(because ``wants_model=True``) and answers BOTH groups. This test locks that
partition in place."""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

MW, MH = 128, 64
OP_TYPE = "MVAU"
DOMAIN = "finn.kernels"


def _build_model():
    """A single-node MVAUKernel_hls graph (2-input, no-activation). Geometry lives on
    the tensors, not on nodeattrs; only the design axes (implementation/PE/SIMD) are
    node-owned."""
    node = helper.make_node(
        OP_TYPE,
        ["inp", "weights"],
        ["out"],
        domain=DOMAIN,
        backend="fpgadataflow",
        implementation="mvau_hls",
        SIMD=16,
        PE=4,
    )
    graph = helper.make_graph(
        [node],
        "mvau_kernel_graph",
        [
            helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def test_kernel_op_opts_into_model_aware_contract():
    """The op declares wants_model=True, so get_customop_wrapper attaches the model
    while bare getCustomOp does not."""
    model = _build_model()
    node = model.graph.node[0]

    bare = getCustomOp(node)
    assert bare.wants_model is True
    assert getattr(bare, "_model", None) is None

    aware = model.get_customop_wrapper(node)
    assert aware._model is model


def test_group1_getter_answerable_without_model():
    """A Group-1 (committed-config) getter reads a nodeattr and needs no graph
    context — a bare getCustomOp(node) answers it."""
    model = _build_model()
    bare = getCustomOp(model.graph.node[0])
    assert bare.get_nodeattr("implementation") == "mvau_hls"
    assert bare.get_nodeattr("SIMD") == 16
    assert bare.get_nodeattr("PE") == 4


def test_group2_getter_raises_without_model():
    """A Group-2 (graph-derived) getter has no model to source its Context from, so
    it raises loudly rather than fabricating an answer."""
    model = _build_model()
    bare = getCustomOp(model.graph.node[0])
    with pytest.raises(ValueError, match="no model attached"):
        bare.get_normal_output_shape()
    with pytest.raises(ValueError, match="no model attached"):
        bare.get_instream_width()
    with pytest.raises(ValueError, match="no model attached"):
        bare.get_output_datatype()


def test_model_aware_path_answers_both_groups():
    """The model-aware instance answers Group-1 AND Group-2 getters."""
    model = _build_model()
    aware = model.get_customop_wrapper(model.graph.node[0])
    # Group 1
    assert aware.get_nodeattr("implementation") == "mvau_hls"
    # Group 2 — resolves against the live graph, no raise
    assert aware.get_normal_input_shape() == (1, MW)
    assert aware.get_normal_output_shape() == (1, MH)
    assert aware.get_instream_width() > 0
    assert aware.get_output_datatype() is not None
