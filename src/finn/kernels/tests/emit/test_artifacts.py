############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Typed template render + value binding (M1, M2).

The codegen vocabulary is typed, not string surgery. ``Template.render`` is an exact
match both directions — a slot with no binding AND a binding with no slot both raise —
which kills the silent ``$KEY$`` no-op-on-rename defect for good. ``RtlModule`` + ``bind``
add a TYPE per slot; each TypedValue's ``.render()`` reproduces the exact legacy string
(``BitWidth(14)->"14"``, ``Bool(True)->"1"``) so rendered bytes are unchanged.
"""

import pytest

from finn.kernels.emit.artifacts import (
    BindError,
    BitWidth,
    Bool,
    Dim,
    Raw,
    RtlModule,
    Template,
    TemplateError,
    bind,
)


# --- M1: Template.render exact-match both directions -----------------------


def test_template_missing_binding_raises():
    t = Template("val = $A$ + $B$;")
    with pytest.raises(TemplateError, match="no binding"):
        t.render({"A": 1})  # B missing


def test_template_renamed_token_raises():
    t = Template("val = $A$;")
    with pytest.raises(TemplateError, match="no matching slot"):
        t.render({"A": 1, "AA": 2})  # AA is a typo'd/renamed token -> loud error


def test_template_renders_when_exact_match():
    t = Template("$A$ and $B$")
    assert t.render({"A": "x", "B": "y"}) == "x and y"


def test_template_only_uppercase_tokens_are_slots():
    t = Template("wire [$clog2(N) : 0] x = $VAL$;")
    assert t.slots == frozenset({"VAL"})  # $clog2 lowercase is not a slot
    assert t.render({"VAL": 3}) == "wire [$clog2(N) : 0] x = 3;"


def test_template_stringifies_sequences_with_newlines():
    t = Template("$LINES$")
    assert t.render({"LINES": ["a", "b", "c"]}) == "a\nb\nc"


# --- M2: typed bind checks VALUE types -------------------------------------


def test_render_matches_untyped_stringification():
    # Each typed value renders EXACTLY the string the old int/str binding produced.
    assert BitWidth(14).render() == "14"
    assert Dim(6).render() == "6"
    assert Bool(True).render() == "1"
    assert Bool(False).render() == "0"
    assert Raw("mvau_top").render() == "mvau_top"


def test_bind_one_of_each_type():
    module = RtlModule("demo", {"W": BitWidth, "MW": Dim, "SIGNED": Bool, "NAME": Raw})
    out = bind(module, {"W": BitWidth(24), "MW": Dim(6), "SIGNED": Bool(True), "NAME": Raw("x")})
    assert out == {"W": "24", "MW": "6", "SIGNED": "1", "NAME": "x"}


def test_bind_type_mismatch_raises():
    module = RtlModule("demo", {"W": BitWidth})
    with pytest.raises(BindError):
        bind(module, {"W": Bool(True)})


def test_bind_missing_param_raises():
    module = RtlModule("demo", {"W": BitWidth, "MW": Dim})
    with pytest.raises(BindError):
        bind(module, {"W": BitWidth(8)})


def test_bind_extra_value_raises():
    module = RtlModule("demo", {"W": BitWidth})
    with pytest.raises(BindError):
        bind(module, {"W": BitWidth(8), "EXTRA": Dim(1)})


def test_bind_output_feeds_template_render():
    # bind() output is exactly the {slot: str} dict Template.render consumes — the two
    # halves compose without an adapter.
    module = RtlModule("m", {"W": BitWidth, "NAME": Raw})
    t = Template("wire [$W$-1:0] $NAME$;")
    rendered = t.render(bind(module, {"W": BitWidth(8), "NAME": Raw("sig")}))
    assert rendered == "wire [8-1:0] sig;"
