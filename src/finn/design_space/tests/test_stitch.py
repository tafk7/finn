############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Stitch-resolver tests (Phase 2 of the composition-stitch build).

Three layers:
  * the real composed MVAU: memstream WEIGHT_SOURCE binds to compute WEIGHT_SINK,
    clk/rst broadcast, data edges export as boundary pins; embedded composes to NO
    weight net (the sink exports up instead);
  * the op-agnostic GUARD: the resolver source reads no pin-name literal / op type /
    mem_mode — grepped, since that invariant is the whole point;
  * SYNTHETIC cardinality proofs: a 2-WEIGHT-by-index cell binds both params (proves
    the "one weight stream" assumption is gone), and a 2-compute / 0-memory shape
    stitches (proves compute⊗compute works even though we don't build shuffle).
"""

import inspect

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.design_space.space import (
    Cell,
    Context,
    Direction,
    Kind,
    Port,
    Role,
    StitchError,
    resolve,
    stitch,
)
from finn.design_space.space import stitch as stitch_module_ref  # for the guard grep
from finn.design_space.fixtures.mvau import MVAU_DSP_SOFTVEC, MVAU_HLS, mvau_schema
from finn.design_space.fixtures.mvau.compose_emit import emit_composed
from finn.design_space.fixtures.parameters import DECOUPLED, EMBEDDED
from finn.design_space.fixtures.parameters.names import RAM_STYLE, TOPOLOGY

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


def _point(ctx, topo=DECOUPLED, impl=MVAU_DSP_SOFTVEC, **extra):
    a = {
        "implementation": impl,
        "PE": 2,
        "SIMD": 2,
        "resType": "lut" if impl == MVAU_HLS else "dsp",
        "noActivation": 1,
        TOPOLOGY: topo,
    }
    if topo == DECOUPLED:
        a[RAM_STYLE] = "block"
    a.update(extra)
    return resolve(mvau_schema(), ctx, a)


# ============================================================= real composed MVAU


def test_decoupled_binds_weight_net():
    ctx = _ctx()
    arts = emit_composed(_point(ctx, DECOUPLED), ctx, module_name="mvau_top")
    cmds = arts.ipi.commands
    # two cells instantiated
    assert sum(c.startswith("create_bd_cell") for c in cmds) == 2
    # THE weight net: memstream m_axis_0 -> compute in1_V, by role, not by pin lookup
    net = [c for c in cmds if "connect_bd_intf_net" in c]
    assert any("mvau_top/in1_V" in c and "mvau_top_wstrm/m_axis_0" in c for c in net)
    # clk + rst broadcast: region-level bd_port fans out to each cell's clk pin
    clk = [c for c in cmds if "connect_bd_net" in c and "ap_clk" in c]
    assert any("get_bd_ports ap_clk" in c and "mvau_top/ap_clk" in c for c in clk)
    assert any("get_bd_ports ap_clk" in c and "mvau_top_wstrm/ap_clk" in c for c in clk)


def test_decoupled_exports_data_boundary():
    ctx = _ctx()
    arts = emit_composed(_point(ctx, DECOUPLED), ctx)
    ext = [c for c in arts.ipi.commands if "make_bd_intf_pins_external" in c]
    # in0_V / out0_V export up as dataflow edges; in1_V does NOT (it's bound)
    assert any("in0_V" in c for c in ext)
    assert any("out0_V" in c for c in ext)
    assert not any("in1_V" in c for c in ext)


def test_embedded_has_no_weight_net_and_exports_nothing_extra():
    ctx = _ctx()
    arts = emit_composed(_point(ctx, EMBEDDED, impl=MVAU_HLS), ctx)
    cmds = arts.ipi.commands
    # exactly one cell (no streamer), so no interface binding at all
    assert sum(c.startswith("create_bd_cell") for c in cmds) == 1
    assert not any("connect_bd_intf_net" in c for c in cmds)
    # HLS embedded has no weight port to export; data edges still export
    ext = [c for c in cmds if "make_bd_intf_pins_external" in c]
    assert any("in0_V" in c for c in ext) and any("out0_V" in c for c in ext)


def test_embedded_rtl_weight_sink_exports_as_boundary():
    # RTL embedded DOES have an in1_V port (the core reads a weight stream) but no
    # delivery sibling — so the sink exports up rather than binding. The topology-driven
    # role behaviour, decided by the resolver, not hardcoded.
    ctx = _ctx()
    arts = emit_composed(_point(ctx, EMBEDDED, impl=MVAU_DSP_SOFTVEC), ctx)
    cmds = arts.ipi.commands
    assert not any("connect_bd_intf_net" in c for c in cmds)
    ext = [c for c in cmds if "make_bd_intf_pins_external" in c]
    assert any("in1_V" in c for c in ext)  # unbound weight sink exported


# ============================================================= the op-agnostic guard


def test_resolver_reads_no_pin_op_or_memmode():
    # the resolver must not branch on any concrete pin name / op / mem_mode. Pin names
    # appear ONLY inside emitted command f-strings (via port.pin), never as a literal
    # the code compares against. Strip comments + docstrings first so the invariant is
    # tested on actual CODE (mentions in prose/lineage notes are fine).
    code = _strip_comments_and_docstrings(inspect.getsource(stitch_module_ref))
    lower = code.lower()
    for forbidden in ("in1_v", "m_axis_0", "mem_mode", "mvau", "implementation"):
        assert forbidden not in lower, f"resolver leaks a hardcoded token: {forbidden}"


def _strip_comments_and_docstrings(src: str) -> str:
    import io
    import tokenize

    out = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        # a STRING that stands alone as a statement is a docstring — drop it
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
        ):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            out.append(tok.string)
        prev_type = tok.type
    return " ".join(out)


# ============================================================= synthetic cardinality


def _clk_rst(inst):
    return (
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk"),
        Port(Direction.IN, Kind.RESET, Role.RESET, "ap_rst_n"),
    )


def test_two_weights_bind_by_index():
    # A compute cell consuming TWO param streams (weights=index 0, thresholds=index 1)
    # + two delivery cells. The resolver binds each by (role, index) with NO knowledge
    # that "MVAU has one weight stream" — proving the cardinality assumption is gone.
    compute = Cell(
        "compute", "compute_mod",
        (
            Port(Direction.IN, Kind.AXIS, Role.WEIGHT_SINK, "w_in", index=0, width=32),
            Port(Direction.IN, Kind.AXIS, Role.WEIGHT_SINK, "t_in", index=1, width=16),
            *_clk_rst("compute"),
        ),
    )
    w_src = Cell(
        "wstrm", "memstream",
        (Port(Direction.OUT, Kind.AXIS, Role.WEIGHT_SOURCE, "m_axis_0", index=0, width=32),
         *_clk_rst("wstrm")),
    )
    t_src = Cell(
        "tstrm", "threshstream",
        (Port(Direction.OUT, Kind.AXIS, Role.WEIGHT_SOURCE, "m_axis_0", index=1, width=16),
         *_clk_rst("tstrm")),
    )
    cmds = stitch((compute, w_src, t_src), "region").commands
    net = [c for c in cmds if "connect_bd_intf_net" in c]
    assert len(net) == 2
    assert any("compute/w_in" in c and "wstrm/m_axis_0" in c for c in net)
    assert any("compute/t_in" in c and "tstrm/m_axis_0" in c for c in net)


def test_two_compute_zero_memory_stitches():
    # A compute⊗compute shape (inner RTL -> outer HLS, shuffle-like): DATA_OUT of A
    # binds to DATA_IN of B, no memory cell at all. Proves the resolver handles the
    # 0-memory / N-compute shape we don't even build.
    a = Cell(
        "inner", "inner_mod",
        (Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V", index=0, width=8, boundary=True),
         Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0, width=8),
         *_clk_rst("inner")),
    )
    b = Cell(
        "outer", "outer_mod",
        (Port(Direction.IN, Kind.AXIS, Role.DATA_IN, "in0_V", index=0, width=8),
         Port(Direction.OUT, Kind.AXIS, Role.DATA_OUT, "out0_V", index=0, width=8, boundary=True),
         *_clk_rst("outer")),
    )
    cmds = stitch((a, b), "region").commands
    net = [c for c in cmds if "connect_bd_intf_net" in c]
    assert len(net) == 1
    assert any("inner/out0_V" in c and "outer/in0_V" in c for c in net)
    # the outer boundary edges export up
    ext = [c for c in cmds if "make_bd_intf_pins_external" in c]
    assert any("inner/in0_V" in c for c in ext)
    assert any("outer/out0_V" in c for c in ext)


def test_ambiguous_binding_raises():
    # two sources for the same (role, index) is an unresolvable stitch, not a silent pick
    a = Cell("a", "m", (Port(Direction.OUT, Kind.AXIS, Role.WEIGHT_SOURCE, "m", index=0, width=8),))
    b = Cell("b", "m", (Port(Direction.OUT, Kind.AXIS, Role.WEIGHT_SOURCE, "m", index=0, width=8),))
    sink = Cell("c", "m", (Port(Direction.IN, Kind.AXIS, Role.WEIGHT_SINK, "s", index=0, width=8),))
    with pytest.raises(StitchError, match="ambiguous"):
        stitch((a, b, sink), "region")


def test_width_mismatch_raises():
    src = Cell("a", "m", (Port(Direction.OUT, Kind.AXIS, Role.WEIGHT_SOURCE, "m", index=0, width=32),))
    sink = Cell("b", "m", (Port(Direction.IN, Kind.AXIS, Role.WEIGHT_SINK, "s", index=0, width=16),))
    with pytest.raises(StitchError, match="width mismatch"):
        stitch((src, sink), "region")
