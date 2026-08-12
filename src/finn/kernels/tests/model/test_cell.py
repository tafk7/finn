############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Kernel`` — the named CELL tier (one pool + its design space).

Construction and pool identity only; :meth:`Kernel.space_for` (the per-realization space
builder) arrives in T3 and is tested beside it. What matters here is that the type holds
the tier's content without inferring any of it — in particular ``root_axis``, which the
old code recovered structurally from the merge and which is a declared field now.
"""

import pytest

from finn.kernels.model.backend import BACKEND_AXIS, Backend, PoolError
from finn.kernels.model.cell import Kernel


def _mvau_cell() -> Kernel:
    from finn.kernels.compute.mvau.op import mvau_pool

    return Kernel(name="compute", pool=mvau_pool(), root_axis=BACKEND_AXIS)


def test_holds_the_live_mvau_pool():
    """The live pool, not a fixture — the point of the tier is that it fits what exists."""
    cell = _mvau_cell()
    assert cell.root_axis == BACKEND_AXIS
    assert len(cell.pool) == 3
    assert cell.member_names == ("mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed")


def test_member_names_are_pool_order_because_order_is_precedence():
    """Pool order IS selection precedence, so the accessor must not sort."""
    cell = _mvau_cell()
    assert cell.member_names == tuple(b.name for b in cell.pool)


def test_member_looks_up_by_name():
    cell = _mvau_cell()
    assert cell.member("mvau_dsp_packed").name == "mvau_dsp_packed"


def test_member_raises_on_a_non_member():
    """A caller naming a member that is not in the pool has a bug — not a missing default."""
    cell = _mvau_cell()
    with pytest.raises(PoolError, match="not in the pool"):
        cell.member("mvau_nonesuch")


def test_empty_pool_rejected():
    with pytest.raises(PoolError, match="at least one"):
        Kernel(name="compute", pool=(), root_axis=BACKEND_AXIS)


def test_duplicate_member_names_rejected():
    """Two members answering to one name makes selection ambiguous — reject at construction,
    where the author is, rather than at whichever lookup wins."""
    dup = (Backend(name="a"), Backend(name="a"))
    with pytest.raises(PoolError, match="duplicate"):
        Kernel(name="compute", pool=dup, root_axis=BACKEND_AXIS)


def test_cell_level_entries_default_empty_and_are_tuples():
    """A minimal cell declares only a pool: MVAU's compute cell has no hand-authored dials
    (SIMD/PE are tiling-generated), so empty must be the no-argument case."""
    cell = _mvau_cell()
    assert (cell.axes, cell.derived, cell.predicates) == ((), (), ())


def test_root_axis_is_declared_not_inferred():
    """A memory cell's root is a namespaced topology key, not `backend`. Nothing about the
    pool says so — which is exactly why the field exists rather than a `_selection_roots()`
    style structural guess."""
    cell = Kernel(
        name="weights",
        pool=(Backend(name="embedded"), Backend(name="decoupled")),
        root_axis="parameters.weights.topology",
    )
    assert cell.root_axis == "parameters.weights.topology"


# --- space_for: one realization, no merge -----------------------------------


def _mvau_parts():
    """The live MVAU container, its compute cell, and the op interfaces `space_for` needs."""
    from finn.kernels.compute.mvau import mvau_kernel

    legacy = mvau_kernel()
    cell = Kernel(
        name="compute",
        pool=legacy.pool,
        root_axis=BACKEND_AXIS,
        axes=legacy.op_axes,
        derived=legacy.op_derived,
        predicates=legacy.op_predicates,
    )
    return legacy, cell


@pytest.mark.parametrize(
    "member", ["mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"]
)
def test_space_for_builds_each_member(member):
    legacy, cell = _mvau_parts()
    space = cell.space_for(member, interfaces=legacy.interfaces)
    assert space.axis_names  # non-empty: at minimum the root
    assert BACKEND_AXIS in space.axis_names


def test_root_domain_stays_the_whole_pool_but_default_is_the_selection():
    """Re-specialization must stay legal — `set_nodeattr("backend", other)` is a valid edit,
    not an error — so the DOMAIN is every member. Only the DEFAULT narrows to the selection,
    replacing the unspecialized "" sentinel."""
    from finn.kernels.engine.point import Point

    legacy, cell = _mvau_parts()
    space = cell.space_for("mvau_dsp_softvec", interfaces=legacy.interfaces)
    root = next(a for a in space.axes if a.name == BACKEND_AXIS)
    assert root.domain(Point({}), None) == frozenset(cell.member_names)
    assert root.default(Point({}), None) == "mvau_dsp_softvec"


@pytest.mark.parametrize(
    "member,own_plus_tiling",
    [("mvau_hls", 6), ("mvau_dsp_softvec", 8), ("mvau_dsp_packed", 9)],
)
def test_predicate_count_is_the_members_own_rules_not_the_pool(member, own_plus_tiling):
    """One member's rules, with nothing wrapped in a selection guard.

    Asserted as a COMPOSITION rather than a magic total: this member's own predicates, plus
    its tiling-generated divisibility rules, plus one gate per port declaring datatype
    support. `own_plus_tiling` is the first two — the augmented-pool sum (6/8/9) — pinned
    separately because it is the number the merge path also computes, so a drift between
    them would mean the two paths disagree about what a member declares.

    Contrast: the merged space compiles 37, because every member's rules are present at once
    and ~15 of them exist only to return None when their owner is not the selection."""
    legacy, cell = _mvau_parts()
    space = cell.space_for(member, interfaces=legacy.interfaces)
    backend = cell.member(member)
    tiling = cell._generated(backend, legacy.interfaces).predicates
    gates = cell._dtype_gates(backend)

    assert len(backend.predicates) + len(tiling) == own_plus_tiling
    assert len(space.predicates) == own_plus_tiling + len(gates)
    assert len(legacy.compile().predicates) == 37  # the merged count, for contrast


def test_no_predicate_is_wrapped_in_a_selection_guard():
    """F1 directly: a guarded predicate carries the root axis in its deps because the wrapper
    reads it. With one member there is no wrapper, so no predicate should acquire that dep
    for a reason it did not declare itself."""
    legacy, cell = _mvau_parts()
    merged = legacy.compile()
    guarded = [p for p in merged.predicates if BACKEND_AXIS in p.deps]
    assert guarded, "sanity: the merged space is the one with selection guards"

    space = cell.space_for("mvau_hls", interfaces=legacy.interfaces)
    for pred in space.predicates:
        assert "guarded on" not in (pred.origin or "")


def test_sources_is_a_constant_with_no_deps():
    """Under the merge, `sources` read the root axis to pick whose list to project. With one
    member there is nothing to pick, so a dep would be a lie the topo-sort still honours."""
    legacy, cell = _mvau_parts()
    space = cell.space_for("mvau_hls", interfaces=legacy.interfaces)
    sources = next(d for d in space.derived if d.name == "sources")
    assert sources.deps == frozenset()
    assert sources.compute(None, None) == cell.member("mvau_hls").sources


def test_non_owner_deriveds_are_absent_not_present_but_none():
    """THE recorded behaviour delta (T3), and a regression test that fails on 0b05d3d82.

    `_dispatch_compute`'s `else None` made a derived that a non-selected member owns resolve
    to None on every point. Per-realization construction has no non-selected member, so the
    key is simply not there — and reading it raises rather than yielding a None that a caller
    could mistake for a computed value. Audited: no production reader depends on the None."""
    legacy, cell = _mvau_parts()
    merged_names = {d.name for d in legacy.compile().derived}
    hls_names = {d.name for d in cell.space_for("mvau_hls", interfaces=legacy.interfaces).derived}

    # SEGMENTLEN is owned by the DSP backends only; HLS resolved it to None before.
    assert "SEGMENTLEN" in merged_names
    assert "SEGMENTLEN" not in hls_names


def test_a_members_own_axis_guard_survives_unwrapped():
    """P8: `space_for` splices a member's entries in UNGUARDED by selection, but a member's
    OWN `exists` guard is its declaration and passes through untouched."""
    legacy, cell = _mvau_parts()
    member = cell.member("mvau_dsp_softvec")
    guarded = {a.name for a in member.axes if a.exists is not None}
    space = cell.space_for("mvau_dsp_softvec", interfaces=legacy.interfaces)
    for axis in space.axes:
        if axis.name in guarded:
            own = next(a for a in member.axes if a.name == axis.name)
            assert axis.exists is own.exists
