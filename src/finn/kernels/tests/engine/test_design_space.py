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
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.design_space import DesignSpace, DesignSpaceError


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
