############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Stitch resolver + port taxonomy (M5, M6).

The stitch is op-agnostic and role-driven: it binds complementary STANDARD_BINDINGS role
pairs by ``(role, index)`` across cells, broadcasts clock/reset, exports unbound ports as
boundary pins, and raises on an ambiguous (role,index) or a kind/width mismatch. It reads
ONLY port fields — never a pin name / op type / mem_mode (the invariant is the whole
point, guarded here on the resolver source). Port taxonomy: shaped roles carry a
shape/width, non-shaped roles must not carry a folded shape; a role implies its direction.
"""

import inspect

import pytest

from finn.kernels.emit import stitch as stitch_module
from finn.kernels.emit.stitch import Cell, StitchError, stitch
from finn.kernels.model.ports import (
    Direction,
    Protocol,
    Port,
    Role,
    STANDARD_BINDINGS,
    role_direction,
)


def _clk_rst(_inst):
    return (
        Port(Direction.IN, Protocol.Clock, Role.CLOCK, "ap_clk"),
        Port(Direction.IN, Protocol.Reset, Role.RESET, "ap_rst_n"),
    )


# --- M5: role-driven binding by (role, index) ------------------------------


def test_two_weights_bind_by_index():
    # A compute cell consuming TWO param streams (weights=index 0, thresholds=index 1)
    # + two delivery cells. The resolver binds each by (role, index) with NO knowledge
    # that "MVAU has one weight stream".
    compute = Cell(
        "compute", "compute_mod",
        (
            Port(Direction.IN, Protocol.Stream, Role.PARAM_SINK, "w_in", index=0, width=32),
            Port(Direction.IN, Protocol.Stream, Role.PARAM_SINK, "t_in", index=1, width=16),
            *_clk_rst("compute"),
        ),
    )
    w_src = Cell(
        "wstrm", "memstream",
        (Port(Direction.OUT, Protocol.Stream, Role.PARAM_SOURCE, "m_axis_0", index=0, width=32),
         *_clk_rst("wstrm")),
    )
    t_src = Cell(
        "tstrm", "threshstream",
        (Port(Direction.OUT, Protocol.Stream, Role.PARAM_SOURCE, "m_axis_0", index=1, width=16),
         *_clk_rst("tstrm")),
    )
    cmds = stitch((compute, w_src, t_src), "region").commands
    net = [c for c in cmds if "connect_bd_intf_net" in c]
    assert len(net) == 2
    assert any("compute/w_in" in c and "wstrm/m_axis_0" in c for c in net)
    assert any("compute/t_in" in c and "tstrm/m_axis_0" in c for c in net)


def test_two_compute_zero_memory_stitches():
    # A compute⊗compute shape (inner -> outer): DATA_OUT of A binds to DATA_IN of B, no
    # memory cell at all. Proves the resolver handles the 0-memory / N-compute shape.
    a = Cell(
        "inner", "inner_mod",
        (Port(Direction.IN, Protocol.Stream, Role.DATA_IN, "in0_V", index=0, width=8, boundary=True),
         Port(Direction.OUT, Protocol.Stream, Role.DATA_OUT, "out0_V", index=0, width=8),
         *_clk_rst("inner")),
    )
    b = Cell(
        "outer", "outer_mod",
        (Port(Direction.IN, Protocol.Stream, Role.DATA_IN, "in0_V", index=0, width=8),
         Port(Direction.OUT, Protocol.Stream, Role.DATA_OUT, "out0_V", index=0, width=8, boundary=True),
         *_clk_rst("outer")),
    )
    cmds = stitch((a, b), "region").commands
    net = [c for c in cmds if "connect_bd_intf_net" in c]
    assert len(net) == 1
    assert any("inner/out0_V" in c and "outer/in0_V" in c for c in net)
    ext = [c for c in cmds if "make_bd_intf_pins_external" in c]
    assert any("inner/in0_V" in c for c in ext)
    assert any("outer/out0_V" in c for c in ext)


def test_clock_and_reset_broadcast():
    a = Cell("a", "m", (Port(Direction.IN, Protocol.Stream, Role.DATA_IN, "in0_V", width=8, boundary=True),
                        *_clk_rst("a")))
    b = Cell("b", "m", (Port(Direction.OUT, Protocol.Stream, Role.DATA_OUT, "out0_V", width=8, boundary=True),
                        *_clk_rst("b")))
    cmds = stitch((a, b), "region").commands
    clk = [c for c in cmds if "connect_bd_net" in c and "ap_clk" in c]
    assert any("a/ap_clk" in c for c in clk)
    assert any("b/ap_clk" in c for c in clk)


# --- M5: error cases -------------------------------------------------------


def test_ambiguous_binding_raises():
    a = Cell("a", "m", (Port(Direction.OUT, Protocol.Stream, Role.PARAM_SOURCE, "m", index=0, width=8),))
    b = Cell("b", "m", (Port(Direction.OUT, Protocol.Stream, Role.PARAM_SOURCE, "m", index=0, width=8),))
    sink = Cell("c", "m", (Port(Direction.IN, Protocol.Stream, Role.PARAM_SINK, "s", index=0, width=8),))
    with pytest.raises(StitchError, match="ambiguous"):
        stitch((a, b, sink), "region")


def test_width_mismatch_raises():
    src = Cell("a", "m", (Port(Direction.OUT, Protocol.Stream, Role.PARAM_SOURCE, "m", index=0, width=32),))
    sink = Cell("b", "m", (Port(Direction.IN, Protocol.Stream, Role.PARAM_SINK, "s", index=0, width=16),))
    with pytest.raises(StitchError, match="width mismatch"):
        stitch((src, sink), "region")


# --- M5: op-agnostic by construction ---------------------------------------


def test_resolver_reads_no_pin_op_or_memmode():
    # The resolver must not branch on any concrete pin name / op / mem_mode. Pin names
    # appear ONLY inside emitted command f-strings (via port.pin), never as a literal the
    # code compares against. Strip comments + docstrings so the invariant is tested on
    # actual CODE, not prose.
    code = _strip_comments_and_docstrings(inspect.getsource(stitch_module))
    lower = code.lower()
    for forbidden in ("in1_v", "m_axis_0", "mem_mode", "mvau", "backend"):
        assert forbidden not in lower, f"resolver leaks a hardcoded token: {forbidden}"


def _strip_comments_and_docstrings(src: str) -> str:
    import io
    import tokenize

    out = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
        ):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            out.append(tok.string)
        prev_type = tok.type
    return " ".join(out)


# --- M6: port taxonomy invariants ------------------------------------------


def test_role_implies_direction():
    assert role_direction(Role.PARAM_SOURCE) == Direction.OUT
    assert role_direction(Role.DATA_OUT) == Direction.OUT
    assert role_direction(Role.PARAM_SINK) == Direction.IN
    assert role_direction(Role.DATA_IN) == Direction.IN
    assert role_direction(Role.CONFIG) == Direction.IN


def test_shaped_role_requires_shape_or_width():
    with pytest.raises(ValueError, match="shape or width"):
        Port(Direction.IN, Protocol.Stream, Role.DATA_IN, "in0_V")  # neither shape nor width


def test_non_shaped_role_rejects_folded_shape():
    with pytest.raises(ValueError, match="must not carry a folded shape"):
        Port(Direction.IN, Protocol.Config, Role.CONFIG, "cfg", shape=(1, 8))


def test_standard_bindings_are_complementary_source_sink_pairs():
    for src, sink in STANDARD_BINDINGS:
        assert role_direction(src) == Direction.OUT
        assert role_direction(sink) == Direction.IN
