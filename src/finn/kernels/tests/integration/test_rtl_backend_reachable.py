############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""F11 — an RTL backend is REACHABLE through the real specialize path.

The defect at ``0b05d3d82``: ``DataflowOp._fpgapart_from`` read an ``fpgapart`` nodeattr no op
declares. qonnx raises ``AttributeError`` on an undeclared name, the bare ``except`` swallowed
it, so every Context carried ``fpgapart=""``. Both MVAU DSP backends then raise "DSP block
needs a non-empty fpgapart" on every node, ``first_feasible_backend`` catches that as "not
feasible", and no RTL backend was reachable on ANY part.

Measured, and worth stating precisely because the headline understates it in one direction
and overstates it in another: the DEFAULT selection was `mvau_hls` either way, since HLS is
first in pool order and always feasible. So the reference build's committed backend does not
move. What was broken is every question about the other members — they did not merely lose,
they RAISED, on every part, including parts where they are the correct answer. A policy that
ranked backends, or any query about DSP feasibility, got a device-independent wrong answer.

These tests pin reachability, not a particular winner: asserting "Versal picks DSP" would
encode pool ORDER, which is a separate decision this pass does not make.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.specialize_kernels import (
    PerNodePolicy,
    SpecializeKernels,
    first_feasible,
)

pytestmark = pytest.mark.integration

MW, MH = 128, 64
VERSAL = "xcvc1902-vsva2197-2MP-e-S"  # DSP58 — both DSP backends feasible
ZYNQ7 = "xc7z020clg400-1"  # DSP48E1 — packed infeasible, softvec feasible


def _model():
    node = helper.make_node(
        "MVAU", ["inp", "weights"], ["out"], domain="finn.kernels", SIMD=2, PE=2
    )
    graph = helper.make_graph(
        [node],
        "mvau_reachability_graph",
        [
            helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT32"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def _op(part):
    """A node with device facts attached exactly as `SpecializeKernels` attaches them."""
    model = _model()
    op = model.get_customop_wrapper(model.graph.node[0])
    op.attach_device(part, clk_ns=5.0)
    return op


def test_device_facts_reach_the_context():
    """The direct read. Before the fix this was `""` for every node on every part, because
    the only source was a nodeattr that does not exist."""
    ctx = _op(VERSAL)._context()
    assert ctx.fpgapart == VERSAL
    assert ctx.clk_ns == 5.0


def test_a_dsp_backend_is_feasible_on_versal():
    """THE regression. Before the fix every DSP backend raised on its device probe, so this
    returned an empty set on every part."""
    op = _op(VERSAL)
    ctx = op._context()
    feasible = {
        b.name for b in type(op).pool if type(op).configure(ctx, {"backend": b.name})
    }
    assert "mvau_dsp_softvec" in feasible
    assert "mvau_dsp_packed" in feasible


def test_no_dsp_backend_is_feasible_without_device_facts():
    """The defect itself, pinned so a regression is visible as a behaviour change rather
    than a silent narrowing. Unknown-part is a legitimate state and the honest answer there
    is "cannot say" — the bug was that it was the ONLY state."""
    op = _op("")
    ctx = op._context()
    for member in ("mvau_dsp_softvec", "mvau_dsp_packed"):
        with pytest.raises(ValueError, match="fpgapart"):
            type(op).configure(ctx, {"backend": member})


def test_device_gating_discriminates_between_parts():
    """Reachability is not "DSP always works now": the packed backend needs DSP58, so a
    DSP48E1 part must still reject it. A fix that made every backend feasible everywhere
    would pass the test above and be just as wrong."""
    zynq = _op(ZYNQ7)
    ctx = zynq._context()
    from finn.kernels.engine.point import Illegal

    result = type(zynq).configure(ctx, {"backend": "mvau_dsp_packed"})
    assert isinstance(result, Illegal)
    assert any("DSP58" in r for r in result.reasons)


def test_specialize_kernels_threads_the_part_through():
    """End-to-end through the real transform, which is what the plan asks for: the part the
    build supplies must reach the node's Context, not be re-derived from the graph."""
    model = _model()
    model = model.transform(
        SpecializeKernels(
            PerNodePolicy(first_feasible), device=(VERSAL, 5.0, None)
        )
    )
    node = model.graph.node[0]
    op = model.get_customop_wrapper(node)
    op.attach_device(VERSAL, clk_ns=5.0)
    assert op._context().fpgapart == VERSAL
    # A backend WAS committed — the transform ran and selected.
    assert op.get_nodeattr("backend") in {b.name for b in type(op).pool}


def test_specialize_without_device_facts_still_specializes():
    """No device context is a legitimate call (a bare-node test, a part-less pre-pass). It
    must degrade to the part-independent members rather than fail — the transform's `device`
    is optional for exactly this."""
    model = _model()
    model = model.transform(SpecializeKernels(PerNodePolicy(first_feasible)))
    assert model.graph.node[0].domain == "finn.kernels"
    op = model.get_customop_wrapper(model.graph.node[0])
    assert op.get_nodeattr("backend") == "mvau_hls"
