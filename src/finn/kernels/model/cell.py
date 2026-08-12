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

from ..engine import provenance
from ..engine.axis import discrete_axis
from ..engine.datatype_spec import datatype_derived
from ..engine.derived import Derived
from ..engine.design_space import DesignSpace
from .backend import SOURCES_KEY, Backend, PoolError


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

    # -- per-realization space construction ---------------------------------

    def space_for(
        self,
        member_name: str,
        *,
        interfaces=(),
        sources_key: str = SOURCES_KEY,
        unspecialized_sentinel: bool = False,
    ) -> DesignSpace:
        """This cell's design space with ONE member selected — the alternative to merging
        the pool.

        Everything the merge layer exists to do is a consequence of siblings sharing one flat
        namespace. Select a member first and none of it is needed:

        * an axis ``domain``/``default`` is the member's own — a VALUE, not a closure that
          dispatches on the root at every read;
        * a member's predicates fire unconditionally, because there is no sibling whose rules
          must be suppressed. ~15 of MVAU's 37 compiled predicates existed only to return
          ``None`` when their owner was not selected;
        * ``sources`` is a constant with NO deps, because there is nothing to dispatch on.

        The root axis is still present and its domain is still the FULL member set: the node
        must be able to hold any member (``set_nodeattr("backend", other)`` has to remain
        legal — that is a re-specialization, not an error). What changes is the DEFAULT, which
        is the selected member rather than the unspecialized ``""`` sentinel.

        Args:
            member_name: the realization to build for. Must be in the pool.
            interfaces: the op's ``InterfaceSchema`` tuple, needed to join the member's
                ``stream`` folds against the op-owned ``block`` extents. Passed in rather
                than held because BLOCK structure is container-owned — a cell that stored a
                copy would be a second home for it.
            sources_key: the point key for the selected member's source list. A secondary
                pool in the same op namespaces it (``parameters.<iface>.sources``) so two
                cells' source lists cannot collide.
            unspecialized_sentinel: accepted for symmetry with ``pool_space`` and currently
                unused — under per-realization construction a member IS selected by
                construction, so there is no unspecialized space to build. Passing ``True``
                is not an error; it simply has nothing to change.

        Returns an UNFINALIZED space: the caller composes it with the op's other cells (a
        cross-cell dep such as ``accDataType`` → ``parameters.weights.datatype`` is
        unresolvable until they are all present) and finalizes the whole.
        """
        member = self.member(member_name)

        # The domain stays the whole pool; only the default narrows. See the docstring.
        root = discrete_axis(self.root_axis, frozenset(self.member_names), member_name)

        # UNWRAPPED and UNGUARDED — the entire point of the exercise. A member's own entries
        # go in exactly as declared, keeping their own `exists` guards (P8: a guarded-out axis
        # stays absent, and that is the member's own guard, not a selection wrapper).
        gen = self._generated(member, interfaces)
        axes = (root,) + tuple(self.axes) + tuple(member.axes) + tuple(gen.axes)

        derived = (
            tuple(self.derived)
            + tuple(member.derived)
            + tuple(gen.derived)
            + tuple(self._register_dtypes(member))
            + (self._sources(member, sources_key),)
        )

        predicates = (
            tuple(self.predicates)
            + tuple(self._attributed(member.predicates, member))
            + tuple(gen.predicates)
            + tuple(self._attributed(self._dtype_gates(member), member))
        )

        return DesignSpace(axes=axes, derived=derived, predicates=predicates)

    def _generated(self, member: Backend, interfaces):
        """The tiling engine's fragments for ONE member — its declared ``stream`` folds joined
        against the op interfaces' ``block`` extents. Memoized per member name, since it is
        pure over the (frozen) member and the interfaces are the same tuple every call."""
        from .tiling import generate_tiling

        got = self._tiling_cache.get(member.name)
        if got is None:
            dtypes = {n: p.derived_dtype for n, p in member.ports.items()}
            got = generate_tiling(interfaces, dict(member.stream), dtypes)
            self._tiling_cache[member.name] = got
        return got

    def _register_dtypes(self, member: Backend) -> tuple[Derived, ...]:
        """The member's INTERNAL-REGISTER dtype derivations (``accDataType``,
        ``weightDataType`` — produced dtypes with no port).

        Under the merge these were present-but-``None`` on every non-owning member. Here a
        register a member does not declare is simply ABSENT, which is the behaviour delta this
        commit records: keys DISAPPEAR from non-owners' points, they never change value."""
        return tuple(
            datatype_derived(
                name,
                spec,
                origin=provenance.realized("register dtype", member.name),
            )
            for name, spec in member.derived_dtypes.items()
        )

    def _attributed(self, predicates, member: Backend) -> tuple:
        """Stamp each of a member's rules with its owner, WITHOUT wrapping the check.

        Under the merge, a backend rule's attribution came from the selection guard's origin —
        so deleting the guard would silently drop "which backend's rule was this?" from every
        `Illegal` reason (`resolve._with_origin` appends it). That is a real diagnostic loss
        and none of it needs a wrapper: origin is metadata, so it can be set on a copy while
        the rule itself stays unguarded and fires unconditionally.

        A rule that already declares its own origin keeps it — an author's attribution beats a
        generated one."""
        from dataclasses import replace

        return tuple(
            p
            if p.origin
            else replace(p, origin=provenance.realized("rule", member.name))
            for p in predicates
        )

    def _dtype_gates(self, member: Backend) -> tuple:
        """The member's per-port datatype SUPPORT, compiled through the same
        :func:`~finn.kernels.engine.constraints.compile_constraint` path as every other
        constraint — so it inherits the optional-port skip. Unguarded: this member is the
        selected one, so its gate always applies."""
        from ..engine.constraints import DatatypeConstraint, compile_constraint

        return tuple(
            compile_constraint(DatatypeConstraint(iface, port.accepted_dtypes))
            for iface, port in member.ports.items()
            if port.accepted_dtypes is not None
        )

    def _sources(self, member: Backend, key: str) -> Derived:
        """The selected member's source-file list, as a constant on the point.

        Deliberately zero deps. Under the merge this read the root axis to pick whose sources
        to project; with one member there is nothing to pick, so the dep would be a lie the
        topo-sort still has to honour."""
        return Derived(
            key,
            lambda point, context, _v=member.sources: _v,
            origin=provenance.realized("sources", member.name),
        )
