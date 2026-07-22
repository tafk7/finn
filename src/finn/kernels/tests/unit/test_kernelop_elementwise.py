############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Elementwise binary — the Tier-3 Kernel N-port stress test
(kernelop-tensor-block-stream.md §4).

Elementwise proves what a single-input op cannot:

  * ARBITRARY input ports — the interface list has TWO inputs (lhs, rhs); every getter
    indexes by position (``get_input_datatype(0|1)``, ``get_folded_input_shape(0|1)``,
    per-port widths). This is the "arbitrary number of input ports" headline.
  * BROADCAST — a size-1 rhs axis must NOT fold by PE; ``broadcast_aware`` returns 1 so
    the rhs streams a single (replicated/held) element rather than dividing a length-1
    axis by PE. (We model this as a 1-wide broadcast bus; FINN's rtl leaf instead
    replicates the value to a PE-wide fanout — a backend realization choice, noted.)
  * NO reduction ⇒ BLOCK = TENSOR ⇒ the GENERIC max-over-interfaces cost floor suffices
    (no ``cost_model`` needed) — the deliberate contrast to MVU, whose reduction coupling
    forced an op-level cost_model. FINN elementwise cost is ``prod(folded_out[:-1])``.

``input_pattern`` (dynamic_dynamic vs dynamic_static) is carried as a kernel_param; it
reassigns the rhs ROLE (DATA_IN vs WEIGHT_SINK) — but role only matters at Tier-4
(stitch/composition), not for any estimate-only getter, so the Tier-3 surface is
pattern-agnostic here.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Context,
    Direction,
    Illegal,
    Implementation,
    Interface,
    Kernel,
    Role,
    discrete_axis,
    fixed_axis,
)

N = 128  # last (channel) dim of lhs / output


def _elementwise_op() -> Kernel:
    from finn.kernels.space import Broadcast, Fold, Full

    # rhs_last is a context-fixed quantity the Broadcast spec reads (the rhs last-dim
    # length; 1 ⇒ broadcast). The PE dial is engine-derived from the lhs/output Fold specs
    # (divisors of the lhs channel count) — a broadcast rhs does NOT shrink its domain.
    rhs_last = fixed_axis("rhs_last", lambda p, ctx: ctx.tensor_shape("rhs")[-1])
    # func / input_pattern: structural op params (func affects output dtype rule at
    # Tier-4; input_pattern reassigns rhs role at Tier-4 — both inert for Tier-3 shapes).
    func = discrete_axis("func", {"Add", "Mul", "Sub"}, "Add")
    pattern = discrete_axis(
        "input_pattern", {"dynamic_dynamic", "dynamic_static"}, "dynamic_dynamic"
    )

    # rank-3 channel-last. lhs/output fold last by PE; rhs folds by PE UNLESS broadcast
    # (rhs_last == 1), then it streams 1 (the Broadcast spec).
    tiling = {
        "lhs": [Full(), Full(), Fold("PE")],
        "rhs": [Full(), Full(), Broadcast("rhs_last", "PE")],
        "output": [Full(), Full(), Fold("PE")],
    }
    hls = Implementation(name="elementwise_hls", tiling=tiling)
    rtl = Implementation(name="elementwise_rtl", tiling=tiling)

    return Kernel(
        name="ElementwiseBinary",
        interfaces=(
            Interface("lhs", "lhs", Direction.IN, Role.DATA_IN, index=0),
            Interface("rhs", "rhs", Direction.IN, Role.DATA_IN, index=1),
            Interface("output", "out", Direction.OUT, Role.DATA_OUT, index=0),
        ),
        pool=(hls, rtl),
        op_axes=(rhs_last, func, pattern),
    )


def _ctx(rhs_shape=(1, 56, N)) -> Context:
    # rhs_shape=(...,1) exercises the broadcast branch.
    return Context(
        shapes={"lhs": (1, 56, N), "rhs": rhs_shape, "out": (1, 56, N)},
        datatypes={
            "lhs": DataType["INT8"],
            "rhs": DataType["UINT4"],
            "out": DataType["INT16"],
        },
        fpgapart="xcvc1902-vsva2197-2MP-e-S",
    )


# ---------------------------------------------------------------------------
# N-port arity: every getter indexes by position.
# ---------------------------------------------------------------------------


def test_two_input_ports_indexed():
    op, ctx = _elementwise_op(), _ctx()
    assert len(op.inputs()) == 2
    assert op.get_input_datatype(ctx, 0) == DataType["INT8"]
    assert op.get_input_datatype(ctx, 1) == DataType["UINT4"]
    assert op.get_output_datatype(ctx, 0) == DataType["INT16"]
    assert op.get_normal_input_shape(ctx, 0) == (1, 56, N)
    assert op.get_normal_input_shape(ctx, 1) == (1, 56, N)


@pytest.mark.parametrize("pe", [1, 4, 16, 128])
def test_folded_shapes_and_widths_per_port(pe):
    op, ctx = _elementwise_op(), _ctx()
    pt = op.configure(ctx, {"PE": pe})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # Both inputs fold last dim by PE (rhs not broadcast here — full length N).
    assert op.get_folded_input_shape(pt, ctx, 0) == (1, 56, N // pe, pe)
    assert op.get_folded_input_shape(pt, ctx, 1) == (1, 56, N // pe, pe)
    assert op.get_folded_output_shape(pt, ctx, 0) == (1, 56, N // pe, pe)
    # Per-port widths use each port's own datatype.
    assert op.get_instream_width(pt, ctx, 0) == 8 * pe   # lhs INT8
    assert op.get_instream_width(pt, ctx, 1) == 4 * pe   # rhs UINT4
    assert op.get_outstream_width(pt, ctx, 0) == 16 * pe  # out INT16


# ---------------------------------------------------------------------------
# Broadcast: a size-1 rhs axis does NOT fold.
# ---------------------------------------------------------------------------


def test_broadcast_rhs_does_not_fold():
    op = _elementwise_op()
    ctx = _ctx(rhs_shape=(1, 1, 1))  # rhs fully broadcast -> rhs_last == 1
    pt = op.configure(ctx, {"PE": 16})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # lhs folds by PE=16; rhs (broadcast) streams 1 element/cycle, not folded by PE.
    assert op.get_folded_input_shape(pt, ctx, 0) == (1, 56, N // 16, 16)
    # rhs_last==1: broadcast_aware -> stream 1 -> width = 1 * rhs_bits (a 1-wide bus).
    assert op.get_instream_width(pt, ctx, 1) == 1 * 4  # UINT4, single held element


def test_non_broadcast_rhs_folds_normally():
    # Same op, rhs at full length -> folds by PE like lhs (regression vs broadcast path).
    op, ctx = _elementwise_op(), _ctx(rhs_shape=(1, 56, N))
    pt = op.configure(ctx, {"PE": 16})
    assert op.get_instream_width(pt, ctx, 1) == 4 * 16


# ---------------------------------------------------------------------------
# Cost: the GENERIC floor suffices (no reduction coupling, unlike MVU).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pe", [1, 4, 16, 128])
def test_exp_cycles_is_generic_floor(pe):
    op, ctx = _elementwise_op(), _ctx()
    pt = op.configure(ctx, {"PE": pe})
    # FINN elementwise: prod(folded_output[:-1]) = prod(out)//pe.
    expected = (1 * 56 * N) // pe
    assert op.get_exp_cycles(pt, ctx) == expected
    # And this op declares NO cost_model — the floor alone is correct here.
    assert op.cost_model is None


def test_exp_cycles_monotone():
    op, ctx = _elementwise_op(), _ctx()
    prev = None
    for pe in [1, 2, 4, 8, 16, 32, 64, 128]:
        c = op.get_exp_cycles(op.configure(ctx, {"PE": pe}), ctx)
        if prev is not None:
            assert c <= prev
        prev = c


# ---------------------------------------------------------------------------
# Structural params resolve; role-flip is a Tier-4 concern (noted, not built).
# ---------------------------------------------------------------------------


def test_input_pattern_and_func_resolve():
    op, ctx = _elementwise_op(), _ctx()
    for pat in ("dynamic_dynamic", "dynamic_static"):
        pt = op.configure(ctx, {"PE": 16, "input_pattern": pat, "func": "Mul"})
        assert not isinstance(pt, Illegal)
        assert pt["input_pattern"] == pat
        # Tier-3 surface is identical regardless of pattern (role/delivery is Tier-4).
        assert op.get_instream_width(pt, ctx, 0) == 8 * 16
