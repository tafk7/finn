############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Port-taxonomy tests (Phase 1 of the composition-stitch build).

Two layers:
  * the :class:`Port` type itself (shape-by-role invariant, the frozen fields);
  * each MVAU/memstream emit DECLARES the right role-tagged ports — the surface the
    stitch resolver (Phase 2) will bind. In particular: HLS-embedded exposes NO
    weight port (weights compiled into params.h); RTL exposes a WEIGHT_SINK (in1_V);
    memstream exposes the matching WEIGHT_SOURCE (m_axis_0) at the SAME width — the
    binding the stitch makes.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Artifacts,
    Context,
    Direction,
    Kind,
    Port,
    Role,
    STANDARD_BINDINGS,
    emit_point,
    resolve,
)
from finn.kernels.ops.mvau import (
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    mvau_pool,
    mvau_schema,
)
from finn.kernels.ops.parameters import DECOUPLED, EMBEDDED, parameters_pool
from finn.kernels.ops.parameters.emit_memstream import emit_memstream
from finn.kernels.ops.parameters.names import (
    PARAM_WIDTH,
    RAM_STYLE,
    RUNTIME_WRITEABLE,
    TOPOLOGY,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _ctx(mw=6, mh=8, wdt="INT8"):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": DataType[wdt], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def _decoupled_point(ctx, impl=MVAU_HLS, pe=2, simd=2, **extra):
    a = {
        "implementation": impl,
        "PE": pe,
        "SIMD": simd,
        "resType": "lut" if impl == MVAU_HLS else "dsp",
        "noActivation": 1,
        TOPOLOGY: DECOUPLED,
        RAM_STYLE: "block",
    }
    a.update(extra)
    return resolve(mvau_schema(), ctx, a)


def _embedded_point(ctx, impl=MVAU_HLS, pe=2, simd=2, **extra):
    a = {
        "implementation": impl,
        "PE": pe,
        "SIMD": simd,
        "resType": "lut" if impl == MVAU_HLS else "dsp",
        "noActivation": 1,
        TOPOLOGY: EMBEDDED,
    }
    a.update(extra)
    return resolve(mvau_schema(), ctx, a)


def _by_role(ports, role):
    return [p for p in ports if p.role == role]


# ============================================================= the Port type


def test_shaped_role_requires_shape_or_width():
    with pytest.raises(ValueError, match="must carry a shape or width"):
        Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V")  # no shape, no width


def test_nonshaped_role_rejects_shape():
    with pytest.raises(ValueError, match="must not carry a folded shape"):
        Port(Direction.IN, Kind.AXILITE, Role.CONFIG, "s_axilite", shape=(1, 2, 3))


def test_config_port_carries_width_or_nothing():
    # a CONFIG port with neither shape nor width is legal (bare register surface)
    p = Port(Direction.IN, Kind.AXILITE, Role.CONFIG, "s_axilite")
    assert p.shape is None and p.width is None


def test_port_is_frozen():
    p = Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk")
    with pytest.raises(Exception):
        p.pin = "other"


def test_standard_bindings_pairs_source_to_sink():
    pairs = dict(STANDARD_BINDINGS)
    assert pairs[Role.WEIGHT_SOURCE] == Role.WEIGHT_SINK
    assert pairs[Role.DATA_OUT] == Role.DATA_IN
    assert pairs[Role.INDEX_SOURCE] == Role.INDEX_SINK


# ============================================================= emit port surfaces


def test_rtl_emit_publishes_weight_sink():
    ctx = _ctx()
    arts = emit_point(mvau_pool(), _decoupled_point(ctx, impl=MVAU_DSP_SOFTVEC), ctx)
    roles = {p.role for p in arts.ports}
    assert Role.DATA_IN in roles
    assert Role.DATA_OUT in roles
    assert Role.WEIGHT_SINK in roles  # in1_V — the stitch binds this
    assert Role.CLOCK in roles and Role.RESET in roles
    # data edges are boundary (dataflow graph), weight sink is internal (binds sibling)
    (sink,) = _by_role(arts.ports, Role.WEIGHT_SINK)
    assert sink.pin == "in1_V"
    assert sink.boundary is False
    assert all(p.boundary for p in _by_role(arts.ports, Role.DATA_IN))


def test_hls_embedded_has_no_weight_port():
    ctx = _ctx()
    arts = emit_point(mvau_pool(), _embedded_point(ctx, impl=MVAU_HLS), ctx)
    roles = {p.role for p in arts.ports}
    # weights are compiled into params.h — the ABSENCE of a weight port IS embedded
    assert Role.WEIGHT_SINK not in roles
    assert Role.DATA_IN in roles and Role.DATA_OUT in roles


def test_memstream_publishes_weight_source_matching_compute_width():
    ctx = _ctx()
    p = _decoupled_point(ctx, impl=MVAU_DSP_SOFTVEC)
    compute = emit_point(mvau_pool(), p, ctx)
    delivery = emit_memstream(p, ctx)

    (source,) = _by_role(delivery.ports, Role.WEIGHT_SOURCE)
    (sink,) = _by_role(compute.ports, Role.WEIGHT_SINK)
    assert source.pin == "m_axis_0"
    assert source.direction == Direction.OUT and sink.direction == Direction.IN
    # THE binding invariant: complementary roles, equal width — the stitch's condition
    assert source.width == sink.width == p[PARAM_WIDTH]


def test_memstream_config_only_when_runtime_writable():
    ctx = _ctx()
    fixed = emit_memstream(_decoupled_point(ctx), ctx)
    assert not _by_role(fixed.ports, Role.CONFIG)

    writable = emit_memstream(
        _decoupled_point(ctx, **{RUNTIME_WRITEABLE: 1, RAM_STYLE: "block"}), ctx
    )
    (cfg,) = _by_role(writable.ports, Role.CONFIG)
    assert cfg.kind == Kind.AXILITE and cfg.boundary is True


def test_every_cell_has_primary_clock_and_one_reset():
    ctx = _ctx()
    p = _decoupled_point(ctx, impl=MVAU_DSP_SOFTVEC)
    for arts in (emit_point(mvau_pool(), p, ctx), emit_memstream(p, ctx)):
        clocks = {port.pin for port in _by_role(arts.ports, Role.CLOCK)}
        assert "ap_clk" in clocks  # primary clock always present
        assert "ap_clk2x" in clocks  # 2x clock port (non-pumped: tied to ap_clk)
        assert len(_by_role(arts.ports, Role.RESET)) == 1
