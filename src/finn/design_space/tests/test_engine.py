############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Engine-level tests for the resolve core, independent of any fixture.

Covers the mechanics the MVAU fixture relies on: domain rejection, dependency
ordering, cycle detection, absent-axis semantics, and collect-all predicates.
"""

import pytest

from finn.design_space.space import (
    AbsentAxisError,
    Axis,
    Context,
    Derived,
    Illegal,
    Point,
    Predicate,
    Schema,
    SchemaError,
    discrete_axis,
    divisor_axis,
    resolve,
)


def _ctx():
    return Context(fpgapart="xc7z020clg400-1")


def test_point_absent_axis_raises():
    p = Point({"a": 1})
    assert "a" in p
    assert p.a == 1
    assert p.get("b") is None
    assert "b" not in p
    with pytest.raises(AbsentAxisError):
        _ = p.b
    with pytest.raises(AbsentAxisError):
        _ = p["b"]


def test_point_is_immutable():
    p = Point({"a": 1})
    with pytest.raises(AttributeError):
        p.a = 2


def test_domain_rejection_returns_illegal():
    schema = Schema(axes=(discrete_axis("x", {1, 2, 3}, 1),))
    r = resolve(schema, _ctx(), {"x": 9})
    assert isinstance(r, Illegal)
    assert "x" in r.reasons[0]
    assert not r  # Illegal is falsy


def test_default_used_when_unassigned():
    schema = Schema(axes=(discrete_axis("x", {1, 2, 3}, 2),))
    r = resolve(schema, _ctx(), {})
    assert isinstance(r, Point)
    assert r.x == 2


def test_dependency_order_is_topological():
    # b depends on a; a must be resolved first regardless of declaration order.
    a = discrete_axis("a", {4}, 4)
    b = divisor_axis("b", "a", 1, deps={"a"})
    schema = Schema(axes=(b, a))  # declared out of order on purpose
    order = [ax.name for ax in schema.ordered_axes()]
    assert order.index("a") < order.index("b")
    r = resolve(schema, _ctx(), {"b": 2})
    assert isinstance(r, Point)
    assert r.b == 2


def test_cycle_detection():
    a = Axis("a", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"b"})
    b = Axis("b", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"a"})
    with pytest.raises(SchemaError, match="cycle"):
        Schema(axes=(a, b))


def test_unknown_dependency_rejected():
    a = Axis("a", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"ghost"})
    with pytest.raises(SchemaError, match="unknown"):
        Schema(axes=(a,))


def test_duplicate_axis_name_rejected():
    with pytest.raises(SchemaError, match="Duplicate"):
        Schema(axes=(discrete_axis("x", {1}, 1), discrete_axis("x", {2}, 2)))


def test_guarded_out_axis_is_absent_not_defaulted():
    root = discrete_axis("root", {0, 1}, 0)
    guarded = discrete_axis(
        "child", {5, 6}, 5, guard=lambda p: p.root == 1, deps={"root"}
    )
    schema = Schema(axes=(root, guarded))
    r = resolve(schema, _ctx(), {"root": 0})
    assert isinstance(r, Point)
    assert "child" not in r
    with pytest.raises(AbsentAxisError):
        _ = r.child


def test_assigning_absent_axis_is_illegal():
    root = discrete_axis("root", {0, 1}, 0)
    guarded = discrete_axis(
        "child", {5, 6}, 5, guard=lambda p: p.root == 1, deps={"root"}
    )
    schema = Schema(axes=(root, guarded))
    r = resolve(schema, _ctx(), {"root": 0, "child": 6})
    assert isinstance(r, Illegal)
    assert "absent" in r.reasons[0]


def test_predicates_collect_all_reasons():
    schema = Schema(
        axes=(discrete_axis("x", {1}, 1),),
        predicates=(
            Predicate(lambda p, c: "reason one", "p1"),
            Predicate(lambda p, c: None, "p2"),
            Predicate(lambda p, c: "reason two", "p3"),
        ),
    )
    r = resolve(schema, _ctx(), {})
    assert isinstance(r, Illegal)
    assert set(r.reasons) == {"reason one", "reason two"}


def test_derived_computed_after_axes():
    schema = Schema(
        axes=(discrete_axis("x", {3}, 3),),
        derived=(Derived("double", lambda p, c: p.x * 2),),
    )
    r = resolve(schema, _ctx(), {})
    assert isinstance(r, Point)
    assert r.double == 6
