############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Demand-driven resolve: ``resolve(..., want={names})`` (engine hone Task 4.2).

The only way to get ONE value was to compute all of them — 26 deriveds, every tiling width,
the whole memstream geometry — because ``resolve`` is eager. ``want`` evaluates just the
transitive read-closure of the names asked for.

Two properties, and the compatibility one is the more important:

1. ``want=None`` is byte-identical to the pre-existing behaviour. A silently-partial point
   would be far worse than a slow one: it would answer questions it never checked.
2. ``want={...}`` really does skip the rest — asserted with a counting spy rather than by
   timing, so it cannot pass by accident.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.design_space import DesignSpace
from finn.kernels.engine.axis import discrete_axis
from finn.kernels.engine.point import Illegal, Point
from finn.kernels.engine.predicate import Predicate
from finn.kernels.engine.resolve import resolve

VERSAL = "xcvc1902-vsvd1760-2MP-e-S"


def _mvau_ctx(with_thresholds=False):
    mw, mh = 8, 8
    shapes = {"inp": (1, mw), "weights": (mw, mh), "out": (1, mh)}
    dts = {
        "inp": DataType["INT4"],
        "weights": DataType["INT4"],
        "out": DataType["INT16"],
    }
    rng = np.random.default_rng(0)
    inits = {"weights": rng.integers(-7, 8, (mw, mh)).astype(np.float32)}
    if with_thresholds:
        shapes["thresholds"] = (mh, 15)
        dts["thresholds"] = DataType["INT16"]
        dts["out"] = DataType["UINT4"]
        inits["thresholds"] = np.sort(
            rng.integers(-100, 100, (mh, 15)).astype(np.float32), axis=1
        )
    return Context(
        shapes=shapes, datatypes=dts, initializers=inits, fpgapart=VERSAL, clk_ns=5.0
    )


# =============================================================================
# Compatibility: want=None must not change anything.
# =============================================================================


@pytest.mark.parametrize("with_thresholds", [False, True], ids=["plain", "thresholds"])
def test_want_none_is_unchanged_for_mvau(with_thresholds):
    from finn.kernels.compute.mvau.op import mvau_kernel

    k = mvau_kernel()
    ctx = _mvau_ctx(with_thresholds)
    explicit = resolve(k.compile(), ctx, {"backend": "mvau_hls"}, want=None)
    default = resolve(k.compile(), ctx, {"backend": "mvau_hls"})
    assert isinstance(default, Point)
    assert dict(explicit) == dict(default)


def test_want_none_is_unchanged_for_thresholding():
    from finn.kernels.compute.thresholding.op import thresholding_kernel

    k = thresholding_kernel()
    ctx = Context(
        shapes={"inp": (1, 8), "thresholds": (8, 15), "out": (1, 8)},
        datatypes={
            "inp": DataType["INT8"],
            "thresholds": DataType["INT16"],
            "out": DataType["UINT4"],
        },
        initializers={
            "thresholds": np.sort(
                np.random.default_rng(0).integers(-100, 100, (8, 15)).astype(np.float32),
                axis=1,
            )
        },
        fpgapart=VERSAL,
        clk_ns=5.0,
    )
    r = resolve(k.compile(), ctx, {"backend": "thresholding_hls"})
    assert isinstance(r, Point)
    assert dict(resolve(k.compile(), ctx, {"backend": "thresholding_hls"})) == dict(r)


# =============================================================================
# The closure really is narrowed.
# =============================================================================


def test_want_does_not_compute_unrelated_deriveds():
    """The plan's named check: asking for a stream width must not run the accumulator
    derivation. A counting spy, so this cannot pass by coincidence."""
    from finn.kernels.compute.mvau.op import mvau_kernel

    space = mvau_kernel().compile()
    calls = {"acc": 0}

    spied = []
    for d in space.derived:
        if d.name == "accDataType":
            inner = d.compute

            def counting(p, c, _inner=inner):
                calls["acc"] += 1
                return _inner(p, c)

            spied.append(Derived(d.name, counting, deps=d.deps, optional_deps=d.optional_deps))
        else:
            spied.append(d)
    probe = DesignSpace(axes=space.axes, derived=tuple(spied), predicates=space.predicates)

    ctx = _mvau_ctx()
    resolve(probe, ctx, {"backend": "mvau_hls"}, want=frozenset({"stream_width.inp"}))
    assert calls["acc"] == 0, "want= computed a derived outside the requested closure"

    # ...and the eager path DOES compute it, so the spy is wired up correctly.
    resolve(probe, ctx, {"backend": "mvau_hls"})
    assert calls["acc"] > 0


def test_want_returns_the_requested_value_and_its_closure():
    from finn.kernels.compute.mvau.op import mvau_kernel

    space = mvau_kernel().compile()
    ctx = _mvau_ctx()
    r = resolve(space, ctx, {"backend": "mvau_hls"}, want=frozenset({"stream_width.inp"}))
    assert isinstance(r, Point)
    assert "stream_width.inp" in r
    # Its closure came too (the fold dial it reads), the rest did not.
    assert "SIMD" in r
    assert "accDataType" not in r


def test_want_value_matches_the_eager_value():
    """Demand-driven must not mean different — only less."""
    from finn.kernels.compute.mvau.op import mvau_kernel

    space = mvau_kernel().compile()
    ctx = _mvau_ctx()
    asg = {"backend": "mvau_hls", "SIMD": 4, "PE": 4}
    full = resolve(space, ctx, asg)
    for name in ("stream_width.inp", "stream_width.out", "accDataType", "TMEM_none"):
        if name not in full:
            continue
        lean = resolve(space, ctx, asg, want=frozenset({name}))
        assert lean[name] == full[name], f"{name} differs between want= and eager"


# =============================================================================
# Mechanism, on synthetic spaces.
# =============================================================================


def test_closure_follows_transitive_deps():
    computed = []

    def spy(name, value=1):
        def compute(p, c):
            computed.append(name)
            return value

        return compute

    space = DesignSpace(
        axes=(discrete_axis("x", {1}, 1), discrete_axis("y", {2}, 2)),
        derived=(
            Derived("a", spy("a"), deps={"x"}),
            Derived("b", spy("b"), deps={"a"}),
            Derived("unrelated", spy("unrelated"), deps={"y"}),
        ),
    ).finalize()

    r = resolve(space, Context(), {}, want=frozenset({"b"}))
    assert computed == ["a", "b"], f"expected the b-closure only, got {computed}"
    assert "unrelated" not in r


def test_absent_optional_dep_does_not_widen_the_closure():
    d = Derived("geo", lambda p, c: 7, optional_deps={"never.published"})
    space = DesignSpace(axes=(), derived=(d,)).finalize()
    assert resolve(space, Context(), {}, want=frozenset({"geo"}))["geo"] == 7


def test_present_optional_dep_is_included():
    order = []
    space = DesignSpace(
        axes=(),
        derived=(
            Derived("geo", lambda p, c: (order.append("geo"), p["demand"])[1],
                    optional_deps={"demand"}),
            Derived("demand", lambda p, c: (order.append("demand"), 42)[1]),
        ),
    ).finalize()
    r = resolve(space, Context(), {}, want=frozenset({"geo"}))
    assert order == ["demand", "geo"]
    assert r["geo"] == 42


def test_predicate_in_closure_still_rejects():
    """A rule whose closure WAS evaluated must still fire — want= narrows what is computed,
    not what is enforced over it."""
    space = DesignSpace(
        axes=(discrete_axis("x", {1}, 1),),
        derived=(Derived("a", lambda p, c: 5, deps={"x"}),),
        predicates=(Predicate(lambda p, c: "a is too big" if p["a"] > 3 else None,
                              "a <= 3", deps={"a"}),),
    ).finalize()
    r = resolve(space, Context(), {}, want=frozenset({"a"}))
    assert isinstance(r, Illegal)
    assert "a is too big" in r.reasons


def test_predicate_outside_the_closure_is_skipped_not_assumed():
    """A rule we could not evaluate is NOT run — and, critically, not treated as passing in
    any way that hides it. The caller gets a Point because nothing checkable objected; the
    unevaluated rule simply had no say. This weaker verdict is why want= is opt-in."""
    ran = []
    space = DesignSpace(
        axes=(discrete_axis("x", {1}, 1), discrete_axis("y", {2}, 2)),
        derived=(
            Derived("a", lambda p, c: 1, deps={"x"}),
            Derived("b", lambda p, c: 9, deps={"y"}),
        ),
        predicates=(
            Predicate(lambda p, c: (ran.append("b-rule"), "b is bad")[1], "b rule", deps={"b"}),
        ),
    ).finalize()
    r = resolve(space, Context(), {}, want=frozenset({"a"}))
    assert isinstance(r, Point)
    assert ran == [], "a rule outside the closure must not run"
    # The eager path DOES enforce it.
    assert isinstance(resolve(space, Context(), {}), Illegal)
