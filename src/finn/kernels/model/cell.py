############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Kernel`` — the CELL tier: one pool of realizations plus its design space.

The system has three authored tiers and, before this module, named only two of them:

===============  =========================  ==============================================
tier             name                       selection
===============  =========================  ==============================================
node-level op    ``DataflowOp``             composes its cells (a PRODUCT)
**cell**         **``Kernel``** (here)      picks ONE pool member (a SUM)
realization      :class:`Backend`           the pool members themselves
===============  =========================  ==============================================

The cell tier having no name is what pushed its content outward in two directions: up
into the container as ``op_axes``/``op_derived``/``op_predicates`` (the ``op_`` prefix IS
the missing type), and sideways into
:class:`~finn.kernels.model.parameter_source.ParameterSource`, which is the same concept
— a pool plus the space around it — wearing a different shape because there was no
shared one to wear. Naming the tier is what lets both collapse onto it.

**One member selected, not a union of the pool.** The naive "superset of its pool" does
not construct: MVAU's three members share ``resType``/``SIMD``/``PE``, so unioning them
raises ``Duplicate name: 'resType'``. A superset therefore needs either a merge (the
~215 LOC this pass deletes) or one space per selected member. :meth:`space_for` is the
second — and because exactly one member is live, an axis domain becomes a **value**
rather than a dispatch closure, and a member's own rules need no selection guard.

``root_axis`` is a FIELD, not inferred. It replaces ``DesignSpace._selection_roots()``,
which recovered the same fact structurally ("an axis other axes depend on") — true only
because the merge added that dep to every merged entry. Delete the merge and the
inference breaks, and breaks asymmetrically: it silently returns a smaller set rather
than raising. One declared field costs a line; re-deriving it costs ~40 LOC plus a
lattice that is wrong the moment its premise changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend import Backend, PoolError


@dataclass(frozen=True)
class Kernel:
    """One pool of :class:`~finn.kernels.model.backend.Backend` realizations, plus the
    cell-level design space shared across them.

    Attributes:
        name: this cell's identity within its op — ``"compute"`` for the compute cell, the
            interface name (``"weights"``, ``"thresholds"``) for a memory cell. Names the
            cell's ROLE in the op, not its contents: a container holds one compute cell
            today and N for a multi-core op, plus one memory cell per delivered parameter.
        pool: the realizations, in declaration order. Order IS selection precedence —
            ``first_feasible_backend`` returns the first member that resolves.
        root_axis: the point key naming the selected member — ``"backend"`` for a compute
            cell, ``"parameters.<iface>.topology"`` for a memory cell. DECLARED rather than
            inferred; see the module docstring for why.
        axes: cell-level axes, shared by every member (the former ``op_axes``).
        derived: cell-level deriveds (the former ``op_derived``).
        predicates: cell-level legality rules (the former ``op_predicates``).

    A member's OWN axes/derived/predicates stay on its :class:`Backend` and are spliced in
    per-realization by :meth:`space_for`; they are deliberately not copied up here.
    """

    name: str
    pool: tuple[Backend, ...]
    root_axis: str
    axes: tuple = ()
    derived: tuple = ()
    predicates: tuple = ()
    _tiling_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "pool", tuple(self.pool))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "_tiling_cache", {})
        if not self.pool:
            raise PoolError(f"cell {self.name!r}: pool must contain at least one Backend")
        names = [b.name for b in self.pool]
        if len(names) != len(set(names)):
            raise PoolError(f"cell {self.name!r}: duplicate member names in pool: {names}")

    @property
    def member_names(self) -> tuple[str, ...]:
        """Every member's name, in pool (precedence) order. This is the root axis's DOMAIN
        under any realization: the node must be able to hold any of them, even though only
        one is live in a given space."""
        return tuple(b.name for b in self.pool)

    def member(self, name: str) -> Backend:
        """The pool member called ``name``. Raises :class:`PoolError` if absent — a caller
        asking for a member that is not in the pool has a bug, not a missing default."""
        for backend in self.pool:
            if backend.name == name:
                return backend
        raise PoolError(
            f"cell {self.name!r}: {name!r} is not in the pool (have {sorted(self.member_names)})"
        )
