############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Attr`` — the node-CONSTANT primitive, and its place in the design space.

An ``Attr`` is neither a choice nor a derivation: a value the op declares, the frontend fixes
once, and every consumer reads off the Point. These tests pin the three properties the
category exists for:

* it is NOT in ``axis_names`` — which is what keeps a frontend-fixed scalar from reading as a
  stratum-2 dial (the defect the category was introduced to fix);
* it IS a legal dep TARGET — a ``Derived`` may name one, which is the property that makes it a
  point-namespace citizen rather than a ``Context`` field (see ``engine/attr.py``);
* it contributes stratum 0 — a constant obliges nothing to be pinned.

The name-space checks matter more than they look: ``_topo_sort``'s unknown-name check is the
only thing standing between a real typo and a silently-dropped edge, so registering attrs must
not weaken it.
"""

import pytest

from finn.kernels.engine.attr import Attr, attr
from finn.kernels.engine.axis import Axis, discrete_axis
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.design_space import DesignSpace, DesignSpaceError
from finn.kernels.engine.point import Illegal, Point
from finn.kernels.engine.predicate import Predicate
from finn.kernels.engine.resolve import resolve


ACTVAL = attr("ActVal", "int", lambda v: isinstance(v, int), 0)
FOLD = discrete_axis("fold", {1, 2}, 1)


# =============================================================================
# The primitive.
# =============================================================================


def test_factory_mirrors_predicate_axis_argument_order():
    """The migration from ``predicate_axis`` should read as a rename, not a re-spelling."""
    a = attr("ActVal", "int", lambda v: isinstance(v, int), 0)
    assert (a.name, a.label, a.default) == ("ActVal", "int", 0)


def test_attr_declares_no_dependencies():
    """Structurally, not by convention: a constant cannot read anything, so it contributes no
    edge and can never sit in a cycle."""
    assert ACTVAL.deps == frozenset()
    assert ACTVAL.optional_deps == frozenset()
    assert Attr.__dataclass_fields__["deps"].init is False


def test_domain_answers_membership_like_an_axis():
    assert 7 in ACTVAL.domain
    assert "seven" not in ACTVAL.domain


def test_default_may_be_a_plain_value_or_read_context():
    """Source-agnosticism lives HERE — in the default — not in the category. One attr can be
    nodeattr-sourced and another Context-derived with no new mechanism."""
    assert ACTVAL.value(None) == 0
    from_ctx = attr("nvec", "int", lambda v: isinstance(v, int), lambda ctx: ctx["n"])
    assert from_ctx.value({"n": 4}) == 4


# =============================================================================
# Place in the design space.
# =============================================================================


def test_attrs_are_not_axes():
    """THE property. ``axis_names`` is the set of CHOICES, and it gates both the stratum
    inference and (via KernelOp) which nodeattrs reach resolve."""
    space = DesignSpace(axes=(FOLD,), attrs=(ACTVAL,)).finalize()
    assert space.axis_names == frozenset({"fold"})
    assert space.attr_names == frozenset({"ActVal"})


def test_attr_is_stratum_zero_while_a_plain_axis_is_two():
    space = DesignSpace(axes=(FOLD,), attrs=(ACTVAL,)).finalize()
    assert space.stratum_of("ActVal") == 0
    assert space.stratum_of("fold") == 2


def test_merge_unions_attrs():
    merged = DesignSpace.merge(
        DesignSpace(axes=(FOLD,), attrs=(ACTVAL,)), DesignSpace(axes=())
    )
    assert [a.name for a in merged.attrs] == ["ActVal"]


def test_describe_entry_and_origin_of_cover_attrs():
    """Both raise on an unknown name, so an entry kind they do not walk regresses to a
    KeyError for a name the space genuinely holds."""
    space = DesignSpace(axes=(FOLD,), attrs=(ACTVAL,)).finalize()
    assert "(attr)" in space.describe_entry("ActVal")
    assert space.origin_of("ActVal") == ""


# =============================================================================
# The dep edge — why this is a primitive and not a Context field.
# =============================================================================


def test_a_derived_may_depend_on_an_attr():
    """The live shape of this is ``narrow_weights`` -> ``mlo_max_iter``
    (``compute/mvau/dsp_common.py``). A REQUIRED dep, so if attrs were absent from the
    topo-sort's name space this would raise 'unknown name' at finalize — which is exactly
    what it did before attrs were registered."""
    reader = Derived("narrow", lambda p, c: 1, deps={"ActVal"})
    space = DesignSpace(axes=(FOLD,), derived=(reader,), attrs=(ACTVAL,)).finalize()
    assert space.stratum_of("narrow") == 0, "reading a CONSTANT pins nothing"


def test_registering_attrs_does_not_weaken_the_typo_check():
    """The unknown-name check is load-bearing: it is the only thing distinguishing a typo
    from a real edge. Widening the name space must not blunt it."""
    typo = Derived("bad", lambda p, c: 1, deps={"ActVaL"})  # wrong case
    with pytest.raises(DesignSpaceError, match="unknown name"):
        DesignSpace(axes=(FOLD,), derived=(typo,), attrs=(ACTVAL,)).finalize()


def test_an_attr_may_not_shadow_an_axis():
    """One name, one entry — a collision is the same authoring error as any other duplicate,
    and silently letting one win would make which-definition-applies depend on walk order."""
    with pytest.raises(DesignSpaceError, match="Duplicate name"):
        DesignSpace(axes=(discrete_axis("ActVal", {0}, 0),), attrs=(ACTVAL,)).finalize()


# =============================================================================
# Resolve behaviour.
# =============================================================================


def test_resolve_takes_the_assignment_or_falls_to_the_default():
    space = DesignSpace(axes=(FOLD,), attrs=(ACTVAL,)).finalize()
    assert resolve(space, None, {"ActVal": -8})["ActVal"] == -8
    assert resolve(space, None, {})["ActVal"] == 0


def test_resolve_rejects_an_out_of_domain_attr():
    """``validate`` is not vestigial — it is checked, exactly as an axis domain is."""
    space = DesignSpace(axes=(FOLD,), attrs=(ACTVAL,)).finalize()
    result = resolve(space, None, {"ActVal": "eight"})
    assert isinstance(result, Illegal)
    assert "ActVal" in result.reasons[0]


def test_attrs_are_written_before_axes():
    """Ordering with teeth: an axis DOMAIN that reads an attr would hit an absent key if
    attrs were written after the axis loop, so this pins the phase order rather than an
    incidental dict ordering."""
    scaled = Axis(
        name="width",
        domain=lambda p, ctx: frozenset({p["ActVal"] * 2}),
        default=lambda p, ctx: p["ActVal"] * 2,
        deps=frozenset({"ActVal"}),
    )
    space = DesignSpace(axes=(scaled,), attrs=(ACTVAL,)).finalize()
    assert resolve(space, None, {"ActVal": 5})["width"] == 10


def test_a_derived_reads_an_attr_off_the_point():
    """The ``narrow_weights`` shape, in miniature."""
    narrow = Derived("narrow", lambda p, c: 0 if p["ActVal"] else 1, deps={"ActVal"})
    space = DesignSpace(axes=(FOLD,), derived=(narrow,), attrs=(ACTVAL,)).finalize()
    assert resolve(space, None, {"ActVal": 3})["narrow"] == 0
    assert resolve(space, None, {"ActVal": 0})["narrow"] == 1


def test_a_predicate_may_read_an_attr_and_stays_stratum_zero():
    """A rule reading ONLY constants is decidable before any choice — the property that
    makes the capability envelope reachable without pinning a fold."""
    rule = Predicate(
        lambda pt, c: None if pt["ActVal"] >= 0 else "ActVal must be >= 0",
        "ActVal sign",
        deps={"ActVal"},
    )
    space = DesignSpace(axes=(FOLD,), predicates=(rule,), attrs=(ACTVAL,)).finalize()
    assert isinstance(resolve(space, None, {"ActVal": 1}), Point)
    assert isinstance(resolve(space, None, {"ActVal": -1}), Illegal)
    assert space.stratum_of(rule) == 0


def test_want_pulls_a_needed_attr_and_skips_an_unneeded_one():
    """Demand-driven resolve must see attrs in the closure — otherwise the walk would read a
    key it never wrote — while still computing nothing it was not asked for."""
    narrow = Derived("narrow", lambda p, c: 0 if p["ActVal"] else 1, deps={"ActVal"})
    space = DesignSpace(axes=(FOLD,), derived=(narrow,), attrs=(ACTVAL,)).finalize()

    wanted = resolve(space, None, {"ActVal": 3}, want={"narrow"})
    assert wanted["narrow"] == 0
    assert "ActVal" in wanted, "the attr its closure names must be pulled in"

    unrelated = resolve(space, None, {"ActVal": 3}, want={"fold"})
    assert "ActVal" not in unrelated, "an unrelated attr must not be computed"
