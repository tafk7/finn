############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The frontend-bake round trip: nodeattr → ``_assignment`` → Point.

``KernelOp._assignment`` decides WHICH NODEATTRS REACH RESOLVE. Every other test of this
path either asserts the nodeattr (never resolving) or calls ``resolve`` with an explicit
assignment (never touching ``_assignment``), so the join between them was untested — and it
is exactly where a value can be lost SILENTLY: infer bakes ``ActVal``
(``compute/mvau/op.py``), emit reads ``point.ActVal`` (``emit_hls.py``), and a drop between
them produces working-but-wrong hardware with no exception and no ``Illegal``.

Measured before this test existed: narrowing ``_assignment``'s gate to ``axis_names`` alone
turned a baked ``ActVal=-8`` into ``point.ActVal == 0``, and the whole suite still passed —
the hardware byte gate uses ``ActVal=0`` in both golden cases, so ``0 == 0`` compared equal.

**Non-zero values are the point.** A test using the default value cannot distinguish
"carried" from "defaulted", which is precisely how this class of bug survives.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

MW, MH, NSTEPS = 8, 8, 15
DOMAIN = "finn.kernels"

# Deliberately NOT the schema default (0). See the module docstring.
BAKED_ACTVAL = -8
BAKED_MLO = 3


def _mvau_model(**attrs):
    """A specialized 3-input MVAU node — the shape infer produces, plus a committed
    backend so the impl-dependent getters resolve."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [MH, NSTEPS]
    )
    node = helper.make_node(
        "MVAU",
        ["inp", "weights", "thresholds"],
        ["out"],
        domain=DOMAIN,
        name="MVAU_0",
        backend="mvau_hls",
        **attrs,
    )
    graph = helper.make_graph(
        [node], "g", [inp], [out], value_info=[weights, thresholds]
    )
    model = ModelWrapper(helper.make_model(graph))
    rng = np.random.default_rng(0)
    model.set_initializer(
        "weights", rng.integers(-7, 8, (MW, MH)).astype(np.float32)
    )
    model.set_initializer(
        "thresholds",
        np.sort(rng.integers(-100, 100, (MH, NSTEPS)).astype(np.float32), axis=1),
    )
    model.set_tensor_datatype("inp", DataType["INT4"])
    model.set_tensor_datatype("weights", DataType["INT4"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_tensor_datatype("out", DataType["UINT4"])
    return model


def _op(model):
    return model.get_customop_wrapper(model.graph.node[0])


def test_a_baked_nonzero_actval_reaches_the_point():
    """THE regression. A baked value must arrive at the Point unchanged — emit reads it
    from there, so a drop here is silently wrong hardware."""
    op = _op(_mvau_model(ActVal=BAKED_ACTVAL))
    _, _, point = op._point()
    assert point.ActVal == BAKED_ACTVAL


def test_a_baked_nonzero_actval_appears_in_the_assignment():
    """One level lower, so a failure says WHICH half broke: the nodeattr gate or resolve."""
    op = _op(_mvau_model(ActVal=BAKED_ACTVAL))
    assert op._assignment().get("ActVal") == BAKED_ACTVAL


def test_an_unbaked_actval_falls_to_the_schema_default():
    """The other half of assignment-or-default: absent means default, not an error."""
    op = _op(_mvau_model())
    assert "ActVal" not in op._assignment()
    _, _, point = op._point()
    assert point.ActVal == 0


def test_every_node_owned_name_round_trips():
    """Generic over the whole surface rather than the two names we happen to have: anything
    the registry PUBLISHES must be readable back through ``_assignment``. A published
    nodeattr the assignment gate cannot see is the defect class this file exists for."""
    op = _op(_mvau_model(ActVal=BAKED_ACTVAL, mlo_max_iter=BAKED_MLO))
    schema = op.kernel().compile()
    from finn.kernels.ir.nodeattr_registry import axis_nodeattr_types

    published = set(axis_nodeattr_types(schema))
    readable = schema.axis_names | schema.attr_names
    assert published <= readable, (
        f"published but unreadable by _assignment: {sorted(published - readable)}"
    )

    assignment = op._assignment()
    assert assignment.get("ActVal") == BAKED_ACTVAL
    assert assignment.get("mlo_max_iter") == BAKED_MLO
    _, _, point = op._point()
    assert point.ActVal == BAKED_ACTVAL
    assert point["mlo_max_iter"] == BAKED_MLO


def test_thresholding_bakes_actval_through_the_same_path():
    """Thresholding's ``infer_from`` bakes the same ``out_bias`` residual, so it has the
    same exposure."""
    ch, steps = 8, 15
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, ch])
    thresholds = helper.make_tensor_value_info(
        "thresholds", TensorProto.FLOAT, [ch, steps]
    )
    node = helper.make_node(
        "Thresholding",
        ["inp", "thresholds"],
        ["out"],
        domain=DOMAIN,
        name="Thresholding_0",
        backend="thresholding_hls",
        ActVal=BAKED_ACTVAL,
    )
    graph = helper.make_graph([node], "g", [inp], [out], value_info=[thresholds])
    model = ModelWrapper(helper.make_model(graph))
    rng = np.random.default_rng(0)
    model.set_initializer(
        "thresholds",
        np.sort(rng.integers(-100, 100, (ch, steps)).astype(np.float32), axis=1),
    )
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_tensor_datatype("out", DataType["UINT4"])

    op = model.get_customop_wrapper(model.graph.node[0])
    _, _, point = op._point()
    assert point.ActVal == BAKED_ACTVAL
