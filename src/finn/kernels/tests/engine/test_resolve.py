############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Engine resolve-walk contracts (E1-E6).

The generic resolve core is pure algebra over (schema, context, assignment). These
tests pin its walk directly with hand-built schemas — no ONNX, no op fixture — because
this is the one layer where a micro-test pins more than an integration test would: every
op in the system rides these semantics.

E1 dependency-order resolution · E2 guarded absence is not defaulting · E3 assigning an
absent axis is a caller error · E4 domain rejection across all three Domain kinds · E5
derived computed after axes, shared namespace · E6 predicates collect ALL reasons.
"""

import pytest

from finn.kernels.engine.axis import (
    Axis,
    PredicateDomain,
    discrete_axis,
    divisor_axis,
    predicate_axis,
)
from finn.kernels.engine.context import Context
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.ordered_parameter import OrderedParameter
from finn.kernels.engine.point import Illegal, Point
from finn.kernels.engine.predicate import Predicate
from finn.kernels.engine.resolve import resolve
from finn.kernels.engine.schema import Schema


def _ctx():
    return Context(fpgapart="xc7z020clg400-1")


# --- E1: dependency-order resolution ---------------------------------------


def test_dependency_order_is_topological():
    # b depends on a; a must be fixed first regardless of declaration order.
    a = discrete_axis("a", {4}, 4)
    b = divisor_axis("b", "a", 1, deps={"a"})
    schema = Schema(axes=(b, a))  # declared out of order on purpose
    order = [ax.name for ax in schema.ordered_axes()]
    assert order.index("a") < order.index("b")
    r = resolve(schema, _ctx(), {"b": 2})
    assert isinstance(r, Point)
    assert r.b == 2


def test_guard_reads_already_fixed_dep():
    # A guard reading an upstream axis always sees it fixed (E1 guarantee).
    root = discrete_axis("root", {0, 1}, 0)
    child = discrete_axis("child", {5, 6}, 5, guard=lambda p: p.root == 1, deps={"root"})
    schema = Schema(axes=(child, root))  # child declared before its dep
    r = resolve(schema, _ctx(), {"root": 1, "child": 6})
    assert isinstance(r, Point)
    assert r.child == 6


# --- E2: guarded absence is not defaulting ---------------------------------


def test_guarded_out_axis_is_absent_not_defaulted():
    root = discrete_axis("root", {0, 1}, 0)
    guarded = discrete_axis(
        "child", {5, 6}, 5, guard=lambda p: p.root == 1, deps={"root"}
    )
    schema = Schema(axes=(root, guarded))
    r = resolve(schema, _ctx(), {"root": 0})
    assert isinstance(r, Point)
    assert "child" not in r
    with pytest.raises(KeyError):
        _ = r.child


def test_default_used_when_present_and_unassigned():
    schema = Schema(axes=(discrete_axis("x", {1, 2, 3}, 2),))
    r = resolve(schema, _ctx(), {})
    assert isinstance(r, Point)
    assert r.x == 2


# --- E3: assigning an absent axis is a caller error ------------------------


def test_assigning_absent_axis_is_illegal():
    root = discrete_axis("root", {0, 1}, 0)
    guarded = discrete_axis(
        "child", {5, 6}, 5, guard=lambda p: p.root == 1, deps={"root"}
    )
    schema = Schema(axes=(root, guarded))
    r = resolve(schema, _ctx(), {"root": 0, "child": 6})
    assert isinstance(r, Illegal)
    assert "absent" in r.reasons[0]


# --- E4: domain rejection across all three Domain kinds --------------------


def test_frozenset_domain_rejection():
    schema = Schema(axes=(discrete_axis("x", {1, 2, 3}, 1),))
    r = resolve(schema, _ctx(), {"x": 9})
    assert isinstance(r, Illegal)
    assert "x" in r.reasons[0]
    assert not r  # Illegal is falsy


def test_ordered_parameter_domain_rejection():
    # divisor_axis builds an OrderedParameter domain (divisors of the dep dim).
    a = discrete_axis("a", {8}, 8)
    b = divisor_axis("b", "a", 1, deps={"a"})
    schema = Schema(axes=(a, b))
    # 3 is not a divisor of 8 → outside the ordered domain.
    r = resolve(schema, _ctx(), {"b": 3})
    assert isinstance(r, Illegal)
    # a divisor is accepted.
    r_ok = resolve(schema, _ctx(), {"b": 4})
    assert isinstance(r_ok, Point)
    assert r_ok.b == 4


def test_predicate_domain_rejection():
    even = predicate_axis("n", "even", lambda v: v % 2 == 0, 2)
    schema = Schema(axes=(even,))
    assert isinstance(resolve(schema, _ctx(), {"n": 3}), Illegal)
    assert resolve(schema, _ctx(), {"n": 4}).n == 4


def test_membership_is_uniform_across_domain_kinds():
    # frozenset, OrderedParameter and PredicateDomain all answer `val in dom`.
    assert 2 in frozenset({1, 2, 3})
    assert 2 in OrderedParameter("p", (1, 2, 4))
    assert 4 in PredicateDomain("even", lambda v: v % 2 == 0)
    assert 5 not in PredicateDomain("even", lambda v: v % 2 == 0)


# --- E5: derived computed after axes, shared namespace ---------------------


def test_derived_computed_after_axes():
    schema = Schema(
        axes=(discrete_axis("x", {3}, 3),),
        derived=(Derived("double", lambda p, c: p.x * 2),),
    )
    r = resolve(schema, _ctx(), {})
    assert isinstance(r, Point)
    assert r.double == 6


def test_derived_share_point_namespace_with_axes():
    schema = Schema(
        axes=(discrete_axis("x", {3}, 3),),
        derived=(Derived("y", lambda p, c: p.x + 1),),
    )
    r = resolve(schema, _ctx(), {})
    assert r["x"] == 3 and r["y"] == 4
    # both addressable the same way; derived never a search dimension.
    assert set(r) == {"x", "y"}


def test_derived_reads_context():
    schema = Schema(
        axes=(),
        derived=(Derived("part", lambda p, c: c.fpgapart),),
    )
    r = resolve(schema, Context(fpgapart="xcvc1902"), {})
    assert r.part == "xcvc1902"


# --- E6: predicates collect ALL reasons ------------------------------------


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


def test_all_none_predicates_is_legal():
    schema = Schema(
        axes=(discrete_axis("x", {1}, 1),),
        predicates=(Predicate(lambda p, c: None, "p"),),
    )
    assert isinstance(resolve(schema, _ctx(), {}), Point)


# --- Composition rides the SAME engine, no new primitive -------------------
# Ported refutation + mapping locks from the old test_composition_mapping: these
# stay to prevent re-adopting the refuted "Derived-returns-sub-Point" framing.


def _delivery_subspace() -> Schema:
    return Schema(
        axes=(
            discrete_axis("storage", {"onchip", "offchip"}, "onchip"),
            discrete_axis(
                "ram_style",
                {"block", "ultra"},
                "block",
                guard=lambda p: p["storage"] == "onchip",
                deps={"storage"},
            ),
        ),
        derived=(Derived("depth", lambda p, c: 128),),
    )


def test_naive_derived_cannot_reach_nondefault_subpoint():
    """REFUTED FRAMING: a Derived gets only (point, context) — no assignment channel —
    so resolving a sub-schema inside it can only ever produce the sub-space DEFAULTS.
    This is WHY composition does not ride a Derived-returns-sub-Point when the
    sub-kernel has free choices."""
    sub = _delivery_subspace()

    def weight_delivery_naive(p, ctx):
        return resolve(sub, _ctx())

    parent = Schema(
        axes=(discrete_axis("PE", {1, 2, 4}, 1),),
        derived=(Derived("weight_delivery", weight_delivery_naive),),
    )
    r_default = resolve(parent, _ctx(), {"PE": 2})
    r_offchip = resolve(parent, _ctx(), {"PE": 2, "storage": "offchip"})
    # Both collapse to the sub-space default — the explorer never reaches offchip.
    assert r_default.weight_delivery.storage == "onchip"
    assert r_offchip.weight_delivery.storage == "onchip"


def test_derived_returns_subpoint_ok_only_when_subkernel_is_parameter_free():
    """The Derived-returns-sub-Point idiom IS valid for a sub-kernel with no free axes."""
    param_free = Schema(axes=(), derived=(Derived("depth", lambda p, c: 64),))
    parent = Schema(
        axes=(discrete_axis("PE", {1, 2}, 1),),
        derived=(Derived("delivery", lambda p, c: resolve(param_free, c)),),
    )
    r = resolve(parent, _ctx(), {"PE": 2})
    assert isinstance(r, Point)
    assert r.delivery.depth == 64


def test_lifted_delivery_axes_are_explorable_and_guarded():
    """CORRECT mapping: the sub-space's FREE axes lift into the parent (namespaced),
    couplings become parent Derived, cross gates become parent Predicate — one flat
    schema, no new primitive."""
    parent = Schema(
        axes=(
            discrete_axis("PE", {1, 2, 4}, 1),
            discrete_axis("wd.storage", {"onchip", "offchip"}, "onchip"),
            discrete_axis(
                "wd.ram_style",
                {"block", "ultra"},
                "block",
                guard=lambda p: p["wd.storage"] == "onchip",
                deps={"wd.storage"},
            ),
        ),
        derived=(Derived("wd.depth", lambda p, c: 64 * p.PE),),
        predicates=(
            Predicate(
                check=lambda p, c: None
                if not (p.get("wd.ram_style") == "ultra" and p["wd.depth"] > 200)
                else "URAM depth over budget",
                description="capacity gate",
            ),
        ),
    )
    r_off = resolve(parent, _ctx(), {"PE": 2, "wd.storage": "offchip"})
    assert r_off.get("wd.storage") == "offchip"
    assert "wd.ram_style" not in r_off  # guarded absent under offchip
    assert r_off["wd.depth"] == 128  # coupling follows PE
    # capacity gate fires: ultra + PE=4 (depth 256 > 200) illegal, PE=2 fine.
    assert isinstance(
        resolve(parent, _ctx(), {"PE": 4, "wd.storage": "onchip", "wd.ram_style": "ultra"}),
        Illegal,
    )
    r_ok = resolve(parent, _ctx(), {"PE": 2, "wd.storage": "onchip", "wd.ram_style": "ultra"})
    assert r_ok.get("wd.ram_style") == "ultra"
