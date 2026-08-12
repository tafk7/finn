############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""F3 — a published nodeattr domain comes from the SELECTED realization, not the pool union.

The defect: ``allowed_values`` for a pool-dispatched axis was unioned across every member,
so a DSP node advertised ``resType in {lut, dsp}`` even though its RTL backend rejects
``lut``. ``set_nodeattr("resType", "lut")`` therefore SUCCEEDED at the write and the node
failed later, at whichever getter first resolved a point — a diagnostic arriving one
mechanism away from the mistake.

Which union is correct is a two-part answer and both parts are pinned here. The NAME SET
must stay the union: ``get_nodeattr_types`` declares everything the node could carry, and a
node that re-specializes to another backend must not find its attribute undeclared. The
DOMAIN must not: it is a fact about the committed realization.

An UNSPECIALIZED node has no selection to read, so it keeps publishing the union — that is
the honest answer there, not a regression.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

MW, MH = 128, 64
DOMAIN = "finn.kernels"
# Versal: the part on which BOTH DSP backends are feasible, so a `resType` narrowing here
# is about the backend's own rule rather than a device rejection.
FPGAPART = "xcvc1902-vsva2197-2MP-e-S"


def _model(backend=None):
    """A 2-input MVAU node, optionally specialized onto ``backend``."""
    kwargs = {"SIMD": 2, "PE": 2}
    if backend is not None:
        kwargs["backend"] = backend
    node = helper.make_node(
        "MVAU", ["inp", "weights"], ["out"], domain=DOMAIN, **kwargs
    )
    graph = helper.make_graph(
        [node],
        "mvau_domain_graph",
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


def _op(backend=None):
    model = _model(backend)
    return model.get_customop_wrapper(model.graph.node[0])


def _res_type(op):
    return op.get_nodeattr_types()["resType"]


def test_dsp_node_publishes_only_its_own_restype():
    """The F3 fix. `mvau_dsp_softvec` rejects `lut` (rtl: "RTL rejects resType=lut"), so its
    published domain must not offer it."""
    dtype, required, default, allowed = _res_type(_op("mvau_dsp_softvec"))
    assert dtype == "s"
    assert required is False
    assert allowed == frozenset({"dsp"})
    assert default == "dsp"


def test_writing_an_out_of_domain_restype_raises_on_a_dsp_node():
    """THE regression. Before the fix this silently succeeded and the node failed later at a
    getter; now the write itself is rejected, which is where the mistake actually is."""
    op = _op("mvau_dsp_softvec")
    with pytest.raises(Exception):
        op.set_nodeattr("resType", "lut")


def test_the_same_write_is_legal_on_an_hls_node():
    """The narrowing must be the SELECTED backend's rule, not a blanket tightening: HLS
    genuinely accepts `lut`, so the identical write has to keep working."""
    op = _op("mvau_hls")
    op.set_nodeattr("resType", "lut")
    assert op.get_nodeattr("resType") == "lut"


def test_unspecialized_node_still_publishes_the_union():
    """No committed backend means no realization to read a domain from, so the union is the
    honest answer — and the only one that lets a not-yet-specialized node be written to."""
    dtype, _required, _default, allowed = _res_type(_op(None))
    assert dtype == "s"
    assert allowed == frozenset({"lut", "dsp"})


def test_the_name_set_stays_the_union_across_the_pool():
    """P8-adjacent, and the half a naive fix breaks: a specialized node must still DECLARE
    every name any member could carry, or re-specializing onto another backend would hit an
    undeclared attribute. Only the DOMAIN narrows."""
    specialized = set(_op("mvau_hls").get_nodeattr_types())
    unspecialized = set(_op(None).get_nodeattr_types())
    assert specialized == unspecialized
    # SEGMENTLEN-adjacent DSP-only dials are declared even on the HLS node.
    assert {"resType", "SIMD", "PE"} <= specialized


def test_the_published_default_is_the_realizations_own():
    """A second, quieter half of the same defect. The union picks whichever probe answered
    first, so the HLS node advertised default `dsp` while `impl_hls.py` declares `lut` (base
    FINN hls:58). Resolution never saw it — an unpinned axis takes the axis's own default,
    not the published one — but a host reading `get_nodeattr_def` did.

    Not a behaviour change in any resolved point: `points-before.txt` is byte-identical
    across this commit. It is the published surface catching up with the declaration."""
    _dtype, _required, hls_default, _allowed = _res_type(_op("mvau_hls"))
    assert hls_default == "lut"
    _dtype, _required, dsp_default, _allowed = _res_type(_op("mvau_dsp_softvec"))
    assert dsp_default == "dsp"


def test_fold_dials_publish_a_type_not_a_divisor_set():
    """The plan's stated risk, checked rather than assumed. A divisor domain is enumerable, so
    publishing it as `allowed_values` would newly reject a non-divisor SIMD/PE at the write —
    and would pin the node to ONE geometry's divisors, though geometry is a tensor fact an
    upstream reshape can change. `_classify_resolved` returns type-only for an
    OrderedParameter for exactly that reason."""
    for backend in ("mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"):
        types = _op(backend).get_nodeattr_types()
        for dial in ("SIMD", "PE"):
            assert types[dial] == ("i", False, 1), f"{backend}/{dial}"
