############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""DesignSpace-authoring error contracts (E7).

Malformed schemas are caught via the dependency topo-sort over one combined DAG of axes AND
deriveds: a duplicate name (across axes and deriveds), a dependency on an unknown name, and a
dependency cycle (including a derived cycle) all raise DesignSpaceError. Independent axes keep
declaration order so the schema reads predictably.

Validation is a WHOLE-SPACE property, run on the FINALIZED design space — not on an assembly
fragment. Construction is cheap and unvalidated; the check fires lazily the first time the
order is needed (``ordered_axes``/``ordered_derived``/any resolve) or eagerly via
``finalize()``. This lets ``compile`` build the space in stages where a compute derived
declares a dep on a name a later parameters fragment introduces — a cross-pool forward ref
that would be indistinguishable from a typo at fragment time. These tests therefore trigger
validation explicitly with ``finalize()``.
"""

import pytest

from finn.kernels.engine.axis import Axis, discrete_axis
from finn.kernels.engine.context import Context
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.design_space import DesignSpace, DesignSpaceError
from finn.kernels.engine.resolve import resolve

# The optional-dep tests resolve, but read nothing off the context.
_CTX = Context()


def test_duplicate_axis_name_rejected():
    with pytest.raises(DesignSpaceError, match="Duplicate"):
        DesignSpace(axes=(discrete_axis("x", {1}, 1), discrete_axis("x", {2}, 2))).finalize()


def test_unknown_dependency_rejected():
    a = Axis("a", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"ghost"})
    with pytest.raises(DesignSpaceError, match="unknown"):
        DesignSpace(axes=(a,)).finalize()


def test_cycle_detection():
    a = Axis("a", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"b"})
    b = Axis("b", domain=lambda p, c: frozenset({1}), default=lambda p, c: 1, deps={"a"})
    with pytest.raises(DesignSpaceError, match="cycle"):
        DesignSpace(axes=(a, b)).finalize()


def test_construction_does_not_validate_only_finalize_does():
    """A fragment with a cross-pool forward-ref dep constructs cleanly; validation fires only
    at finalize. This is the property that lets compile() stage the space across pools."""
    frag = DesignSpace(
        axes=(discrete_axis("x", {1}, 1),),
        derived=(Derived("acc", lambda p, c: 0, deps={"later.storage"}),),  # from a later fragment
    )
    # Construction did NOT raise on the not-yet-present dep.
    assert "acc" in {d.name for d in frag.derived}
    # Folding in the later fragment's derived, THEN finalizing, resolves the forward ref.
    whole = DesignSpace(
        axes=frag.axes,
        derived=frag.derived + (Derived("later.storage", lambda p, c: 1),),
    ).finalize()
    assert [d.name for d in whole.ordered_derived()] == ["later.storage", "acc"]


def test_finalize_still_catches_genuine_typo_on_the_whole():
    """A dep that names nothing in the COMPLETE space is a real error and still raises — the
    granularity loss is only WHICH fragment, not WHETHER it is caught."""
    with pytest.raises(DesignSpaceError, match="unknown"):
        DesignSpace(
            axes=(discrete_axis("x", {1}, 1),),
            derived=(Derived("acc", lambda p, c: 0, deps={"typpo"}),),
        ).finalize()


def test_independent_axes_preserve_declaration_order():
    x = discrete_axis("x", {1}, 1)
    y = discrete_axis("y", {2}, 2)
    z = discrete_axis("z", {3}, 3)
    schema = DesignSpace(axes=(x, y, z))
    assert [a.name for a in schema.ordered_axes()] == ["x", "y", "z"]


def test_axis_names_property():
    schema = DesignSpace(axes=(discrete_axis("a", {1}, 1), discrete_axis("b", {2}, 2)))
    assert schema.axis_names == frozenset({"a", "b"})


# --- A1: deriveds join the same DAG ----------------------------------------


def test_derived_cycle_detection():
    a = Derived("a", lambda p, c: 1, deps={"b"})
    b = Derived("b", lambda p, c: 1, deps={"a"})
    with pytest.raises(DesignSpaceError, match="cycle"):
        DesignSpace(axes=(), derived=(a, b)).finalize()


def test_derived_unknown_dependency_rejected():
    d = Derived("d", lambda p, c: 1, deps={"ghost"})
    with pytest.raises(DesignSpaceError, match="unknown"):
        DesignSpace(axes=(), derived=(d,)).finalize()


def test_derived_shadowing_axis_name_rejected():
    """A derived may not reuse an axis name — they share one point namespace, so a
    collision would silently let one clobber the other."""
    with pytest.raises(DesignSpaceError, match="Duplicate"):
        DesignSpace(
            axes=(discrete_axis("x", {1}, 1),),
            derived=(Derived("x", lambda p, c: 2),),
        ).finalize()


def test_ordered_derived_follows_dependency_order():
    schema = DesignSpace(
        axes=(discrete_axis("x", {1}, 1),),
        derived=(
            Derived("b", lambda p, c: 0, deps={"a"}),
            Derived("a", lambda p, c: 0, deps={"x"}),
        ),
    )
    # Axis order preserved; deriveds emerge after their deps regardless of list order.
    assert [a.name for a in schema.ordered_axes()] == ["x"]
    assert [d.name for d in schema.ordered_derived()] == ["a", "b"]


# --- Fragment assembly: forward refs, merge, optional deps ------------------


def test_fragment_forward_dep():
    """A fragment declaring a dep on a name a LATER fragment introduces constructs fine and
    finalizes fine after merge — the property compile() stages the space on. A genuine typo
    is still caught on the merged whole."""
    early = DesignSpace(
        axes=(discrete_axis("x", {1}, 1),),
        derived=(Derived("acc", lambda p, c: 0, deps={"later.storage"}),),
    )
    late = DesignSpace(axes=(), derived=(Derived("later.storage", lambda p, c: 1),))

    whole = DesignSpace.merge(early, late).finalize()
    assert [d.name for d in whole.ordered_derived()] == ["later.storage", "acc"]

    # Merge order does NOT carry the ordering — deps do.
    reversed_ = DesignSpace.merge(late, early).finalize()
    assert [d.name for d in reversed_.ordered_derived()] == ["later.storage", "acc"]

    # A dep no fragment defines is still a hard error.
    with pytest.raises(DesignSpaceError, match="unknown"):
        DesignSpace.merge(
            early, DesignSpace(axes=(), derived=(Derived("nothing.like.it", lambda p, c: 1),))
        ).finalize()


def test_optional_dep_absent_is_legal():
    """An optional dep naming a key the space does not define finalizes and resolves. This is
    the standalone-pool case: parameters_schema() has no composing op to publish a demand."""
    space = DesignSpace(
        axes=(discrete_axis("x", {1}, 1),),
        derived=(
            Derived(
                "geometry",
                lambda p, c: p.get("absent.demand", None) or 7,
                deps={"x"},
                optional_deps={"absent.demand"},
            ),
        ),
    ).finalize()
    assert [d.name for d in space.ordered_derived()] == ["geometry"]
    assert resolve(space, _CTX, {})["geometry"] == 7


def test_optional_dep_present_orders():
    """When the optional name IS present, it constrains the order exactly like a hard dep —
    this is the composed case, and it is what makes DEMAND→SOURCE structural."""
    space = DesignSpace(
        axes=(discrete_axis("x", {1}, 1),),
        derived=(
            # Declared BEFORE its optional dep, so list position would give the wrong answer.
            Derived(
                "geometry",
                lambda p, c: p.get("demand", None) or 7,
                deps={"x"},
                optional_deps={"demand"},
            ),
            Derived("demand", lambda p, c: 42),
        ),
    ).finalize()
    assert [d.name for d in space.ordered_derived()] == ["demand", "geometry"]
    # Ordered first => geometry actually SEES the value rather than defaulting.
    assert resolve(space, _CTX, {})["geometry"] == 42


def test_optional_dep_cycle_still_detected():
    """An optional edge is a real edge when present, so it can close a cycle — and must be
    caught rather than silently dropped to break the loop."""
    with pytest.raises(DesignSpaceError, match="cycle"):
        DesignSpace(
            axes=(),
            derived=(
                Derived("a", lambda p, c: 1, deps={"b"}),
                Derived("b", lambda p, c: 1, optional_deps={"a"}),
            ),
        ).finalize()


def test_dep_may_not_be_both_required_and_optional():
    """"Must exist" and "may be absent" are contradictory claims about one read. Purely local,
    so unlike the unknown-name check this fires at construction."""
    with pytest.raises(ValueError, match="BOTH deps and optional_deps"):
        Derived("d", lambda p, c: 1, deps={"x"}, optional_deps={"x"})
    with pytest.raises(ValueError, match="BOTH deps and optional_deps"):
        discrete_axis("a", {1}, 1, deps={"x"}, optional_deps={"x"})


# --- composition: `+` is the two-argument merge ------------------------------


def _frag(axis_name, derived_name, *, deps=frozenset()):
    return DesignSpace(
        axes=(discrete_axis(axis_name, {1}, 1),),
        derived=(Derived(derived_name, lambda p, c: 0, deps=deps),),
    )


def test_plus_equals_two_argument_merge():
    a, b = _frag("x", "dx"), _frag("y", "dy")
    assert a + b == DesignSpace.merge(a, b)


def test_plus_chain_and_variadic_merge_finalize_identically():
    """`(a + b) + c` builds an intermediate `merge(a, b, c)` does not, but the two finalize to
    the same evaluation order — both preserve the same left-to-right input order, so chaining
    `+` is a drop-in for the variadic call rather than a differently-ordered space."""
    a, b = _frag("x", "dx"), _frag("y", "dy")
    c = _frag("z", "dz", deps={"x"})  # a real edge, so order is not merely input order
    chained = ((a + b) + c).finalize()
    variadic = DesignSpace.merge(a, b, c).finalize()
    assert [n.name for n in chained.ordered_axes()] == [
        n.name for n in variadic.ordered_axes()
    ]
    assert [n.name for n in chained.ordered_derived()] == [
        n.name for n in variadic.ordered_derived()
    ]


def test_plus_is_commutative_in_the_resolved_point():
    """Composition order cannot change a resolved Point — the property `+` actually claims.

    Deliberately NOT a claim about the emitted sequence: among entries with no edge between
    them the sort preserves input order, so `ordered_derived()` genuinely differs between the
    two arms. Entries with no edge cannot read each other, so that difference is unobservable
    in the result, which is what makes the spelling honest."""
    a = _frag("x", "dx")
    b = _frag("y", "dy", deps={"x"})
    assert resolve((a + b).finalize(), _CTX, {}) == resolve((b + a).finalize(), _CTX, {})


def test_plus_honours_a_real_edge_from_either_side():
    """A dependent entry orders after its dep whichever fragment was composed first — the
    concrete reason order-in cannot leak into order-out anywhere it would be observable."""
    a = _frag("x", "dx")
    b = DesignSpace(
        axes=(),
        derived=(Derived("reader", lambda p, c: p["dx"] + 1, deps={"dx"}),),
    )
    for space in (a + b, b + a):
        names = [d.name for d in space.finalize().ordered_derived()]
        assert names.index("dx") < names.index("reader")


def test_plus_rejects_a_non_space():
    with pytest.raises(TypeError):
        _frag("x", "dx") + 3
