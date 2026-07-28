############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""F1 — the ONE ``is_specialized`` predicate + the three-state getter contract.

Proves the merge the hone delivers on the pre-resolution surface:

  * ``is_specialized(node)`` is the single definition of "a backend is committed"
    (``backend`` nodeattr non-empty). It agrees with ``kernel_hw_language`` on the
    same node in both states, and is DISTINCT from ``is_fpgadataflow_node`` (family
    membership — True even while unspecialized).
  * The getter contract's three states:
      - impl-INDEPENDENT getters (normal shape/dtype, make_shape_compatible_op,
        infer_node_datatype) SUCCEED on an unspecialized node — the Seam A verify gate.
      - impl-DEPENDENT getters (folded shape/width, exp_cycles) RAISE a legible
        "unspecialized" error, NOT a bare Illegal-ValueError or a silent default backend.
  * The F1 disagreement is GONE: an unspecialized node no longer answers
    ``get_folded_output_shape`` AS IF ``mvau_hls`` (the old schema-default backend).

The ``backend`` nodeattr default is now ``""`` (was ``mvau_hls``) — Seam B's
``set_nodeattr("backend", name)`` is the ONE write that specializes a node.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.kernels.ops.mvau.op import MvauKernelOp, mvau_kernel
from finn.kernels.routing import is_specialized, kernel_hw_language
from finn.util.fpgadataflow import is_fpgadataflow_node

MW, MH = 128, 64


def _matmul_only_model():
    """A 2-input UNSPECIALIZED ``finn.kernels`` MVAU node (bare MatMul, no thresholds) —
    exactly what Seam A's infer produces: ActVal baked, NO implementation/SIMD/PE."""
    node = helper.make_node(
        "MVAU",
        ["inp", "weights"],
        ["out"],
        domain="finn.kernels",
        ActVal=0,
        name="MVAU_0",
    )
    value_info = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
    ]
    graph = helper.make_graph(
        [node],
        "mvau_unspecialized",
        value_info,
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def _inst(model):
    return model.get_customop_wrapper(model.graph.node[0])


# ---------------------------------------------------------------------------
# The ONE predicate + the three-tier agreement.
# ---------------------------------------------------------------------------


def test_is_specialized_flips_only_on_backend():
    model = _matmul_only_model()
    node = model.graph.node[0]

    # Unspecialized out of infer: is_specialized False, but family membership holds.
    assert is_specialized(node) is False
    assert kernel_hw_language(node) is None
    assert is_fpgadataflow_node(node) is True  # partition-sweep invariant (Seam C)

    # Seam B's ONE write: set the backend axis -> specialized.
    _inst(model).set_nodeattr("backend", "mvau_hls")
    assert is_specialized(node) is True
    assert kernel_hw_language(node) == "hls"  # language agrees, only once specialized
    assert is_fpgadataflow_node(node) is True


def test_nodeattr_default_is_unspecialized_sentinel():
    attrs = _inst(_matmul_only_model()).get_nodeattr_types()
    dtype, required, default = attrs["backend"][:3]
    assert default == ""  # was "mvau_hls" (the removed second definition of resolved-ness)
    # T5 shadow proof: the schema `backend` axis fully overrides the classic-inherited
    # `backend` nodeattr (HWCustomOp default "fpgadataflow", required=True). On a kernel op
    # the axis definition wins — non-required, "" sentinel — so the two never collide.
    assert required is False
    assert default != "fpgadataflow"
    # The old axis name is fully gone: no `implementation` nodeattr survives the rename.
    assert "implementation" not in attrs


# ---------------------------------------------------------------------------
# Impl-INDEPENDENT getters answer on an unspecialized node (Seam A verify gate).
# ---------------------------------------------------------------------------


def test_unspecialized_answers_impl_independent_getters():
    model = _matmul_only_model()
    inst = _inst(model)

    assert inst.get_normal_input_shape(0) == (1, MW)
    assert inst.get_normal_output_shape(0) == (1, MH)
    assert inst.get_input_datatype(0) == DataType["INT8"]
    # get_output_datatype is a pure Context read of the (un-annotated) out tensor — it
    # succeeds with no committed backend (returns the graph dtype here).
    assert inst.get_output_datatype(0) is not None

    # The two InferShapes/InferDataTypes entrypoints succeed on the unspecialized node.
    assert inst.make_shape_compatible_op(model) is not None
    # infer_node_datatype resolves the impl-INDEPENDENT accumulator derived and publishes it:
    # no thresholds -> output IS the weight-derived accumulator, a real int type.
    inst.infer_node_datatype(model)
    assert model.get_tensor_datatype("out").is_integer()


# ---------------------------------------------------------------------------
# Impl-DEPENDENT getters RAISE a legible "unspecialized" error.
# ---------------------------------------------------------------------------


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
    inst = _inst(_matmul_only_model())
    with pytest.raises(ValueError, match="unspecialized"):
        call(inst)


def test_the_f1_disagreement_is_gone():
    # Regression: the old schema default silently supplied mvau_hls, so an UNSPECIALIZED
    # node answered get_folded_output_shape AS IF mvau_hls while kernel_hw_language said None.
    # They now agree — the folded-shape getter raises rather than fabricating a backend.
    model = _matmul_only_model()
    node = model.graph.node[0]
    assert kernel_hw_language(node) is None
    with pytest.raises(ValueError, match="unspecialized"):
        _inst(model).get_folded_output_shape(0)

    # Once specialized, the same getter answers (and language agrees).
    inst = _inst(model)
    inst.set_nodeattr("backend", "mvau_hls")
    inst.set_nodeattr("SIMD", 8)
    inst.set_nodeattr("PE", 8)
    assert kernel_hw_language(node) == "hls"
    # Re-instantiate to pick up the new nodeattrs, then the folded shape resolves.
    assert _inst(model).get_folded_output_shape(0)[-1] == 8  # PE folds MH -> last dim = PE
