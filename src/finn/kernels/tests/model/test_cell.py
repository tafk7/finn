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
    from finn.kernels.compute.mvau.kernel import mvau_pool

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
