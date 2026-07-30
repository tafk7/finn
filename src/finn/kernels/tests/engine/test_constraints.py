############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The declarative constraint vocabulary + its compile-to-predicate path.

Covers the two Phase-3 wins: the optional-port skip (a constraint on an ABSENT port
auto-noops, deleting the ``if not has_tensor`` prelude) and the relational value rule
(``ValueNonNeg`` gated on ANOTHER tensor, skipping on the constrained port not the gate).
"""

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.constraints import (
    CustomConstraint,
    DatatypeConstraint,
    IsStatic,
    ShapeRank,
    ValueNonNeg,
    compile_constraint,
)
from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
from finn.kernels.engine.point import Point


def _ctx(**kw):
    return Context(**kw)


def test_shape_rank_checks_rank():
    c = ShapeRank("t", 2)
    ok = _ctx(shapes={"t": (4, 8)})
    bad = _ctx(shapes={"t": (4,)})
    assert c.check(Point({}), ok) is None
    assert "rank 2" in c.check(Point({}), bad)


def test_optional_port_skip_noops_when_tensor_absent():
    # The whole point of Finding 3a: a constraint on an ABSENT port never fires — no
    # has_tensor guard in the rule body.
    pred = compile_constraint(ShapeRank("thresholds", 2))
    absent = _ctx(shapes={"inp": (1, 8)})  # no "thresholds"
    assert pred.check(Point({}), absent) is None
    present_bad = _ctx(shapes={"thresholds": (4,)})
    assert pred.check(Point({}), present_bad) is not None


def test_value_nonneg_gated_on_other_tensor():
    # unsigned INPUT ⇒ THRESHOLDS >= 0. The gate reads INPUT; the rule constrains THRESHOLDS.
    def unsigned_input(p, ctx):
        return not ctx.tensor_datatype("inp").signed()

    c = ValueNonNeg("thresholds", when=unsigned_input)
    neg = np.array([[-1.0, 2.0]])
    unsigned = _ctx(
        shapes={"inp": (1, 2), "thresholds": (1, 2)},
        datatypes={"inp": DataType["UINT4"]},
        initializers={"thresholds": neg},
    )
    signed = _ctx(
        shapes={"inp": (1, 2), "thresholds": (1, 2)},
        datatypes={"inp": DataType["INT4"]},
        initializers={"thresholds": neg},
    )
    assert c.check(Point({}), unsigned) is not None  # negative + unsigned input => illegal
    assert c.check(Point({}), signed) is None  # signed input => rule does not apply


def test_value_nonneg_skip_fires_on_constrained_port_not_gate():
    # The optional-port skip must key on THRESHOLDS (constrained), even though the gate reads
    # INPUT. Thresholds absent ⇒ skip, regardless of the input.
    def unsigned_input(p, ctx):
        return not ctx.tensor_datatype("inp").signed()

    pred = compile_constraint(ValueNonNeg("thresholds", when=unsigned_input))
    no_thresholds = _ctx(shapes={"inp": (1, 2)}, datatypes={"inp": DataType["UINT4"]})
    assert pred.check(Point({}), no_thresholds) is None


def test_is_static_unless_exempts():
    c = IsStatic("weights", unless=lambda p: bool(p.get("rtw", 0)))
    dynamic = _ctx(shapes={"weights": (4, 8)})  # no initializer
    static = _ctx(shapes={"weights": (4, 8)}, initializers={"weights": np.ones((4, 8))})
    assert c.check(Point({}), dynamic) is not None  # dynamic + not exempt => illegal
    assert c.check(Point({"rtw": 1}), dynamic) is None  # runtime-writable exemption
    assert c.check(Point({}), static) is None


def test_datatype_constraint_adapts_support():
    c = DatatypeConstraint("inp", DatatypeSupport(kind=DatatypeKind.INTEGER))
    ints = _ctx(shapes={"inp": (1, 8)}, datatypes={"inp": DataType["INT8"]})
    floats = _ctx(shapes={"inp": (1, 8)}, datatypes={"inp": DataType["FLOAT32"]})
    assert c.check(Point({}), ints) is None
    assert c.check(Point({}), floats) is not None


def test_custom_constraint_without_iface_always_runs():
    # No iface => no optional-port skip; the closure always runs.
    pred = compile_constraint(CustomConstraint(lambda p, ctx: "always", desc="x"))
    assert pred.check(Point({}), _ctx()) == "always"
