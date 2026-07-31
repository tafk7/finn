############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``ParameterSource`` — the compute→source seam for one delivered parameter (design pitch §2).

A ``ParameterSource`` HOLDS the source backends for one parameter interface (a pool → one
is selected); it is the parameter-side MIRROR of a compute :class:`~finn.kernels.model.kernel.Kernel`,
NOT itself a source backend (that is a pool member — see
:func:`~finn.kernels.model.source_backend.source_backend`).

The design space has two independent selection pools that must nonetheless talk: the
**compute pool** (``implementation``) fixes PE·SIMD hence the parameter stream width; the
**source pool** (``parameters.<iface>.topology``) fixes how the parameter is stored and
must size its memory to match that width. The compute side produces a fact (how fast it
consumes a parameter); the source side consumes it (sizes memory to feed at that rate) —
a real, directional data dependency.

Historically that dependency was expressed three ways at once: a demand derived slotted
between the pools BY LIST POSITION, a mode-compatibility gate as ``replace()``-surgery on
the source root axis, and a ``compose`` union documenting the gap it refused to fill.
``ParameterSource`` collapses them into ONE per-parameter object that owns the seam:

* ``publishes`` — the compute→source DEMAND closure (a realization-free
  :class:`~finn.kernels.model.demand.ParamDemand` sized from the RESOLVED interface
  geometry). ``None`` for a live activation or an embedded-mode consumption.
* ``constrains`` — the topology-mode GUARD: the source pool's ``topology`` domain kept
  to just the modes the selected compute backend accepts for this interface.
* declared ``deps`` ``{implementation, topology.<iface>}`` — so the existing topo-sort
  orders COMPUTE → DEMAND → SOURCE structurally, not by list position.

It introduces NO new resolve mechanism: it reads only existing ``Backend`` fields
(``stream``, ``mem_modes``, topology ``mem_mode``) and existing point keys
(``stream_width.<iface>``, ``topology.<iface>``, ``demand.<iface>``), and its two-root
deps feed the topo-sort that already runs.

This module OWNS the assembly (:meth:`ParameterSource.to_subspaces`), reusing the
demand/guard COMPUTATION bodies (:func:`_demand_for`, :func:`_topology_domain`) kept in
:mod:`~finn.kernels.model.param_contract` beside the op-facing :class:`DeliveredParam`
declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from .backend import BACKEND_AXIS, Backend, pool_space
from .param_contract import DeliveredParam, _demand_for, _topology_default, _topology_domain
from ..engine.derived import Derived
from .param_names import demand_key, sources_key, topology_key
from ..engine.design_space import DesignSpace


@dataclass(frozen=True)
class ParameterSource:
    """The compute→source seam for one declared parameter interface. Built by
    :func:`parameter_source_for` from a
    :class:`~finn.kernels.model.param_contract.DeliveredParam` (the op's WHAT) plus the compute
    pool (whose members' ``mem_modes`` drive the guard). A ``ParameterSource`` HOLDS the pool
    of source backends (one is selected); it is the parameter-side mirror of a compute
    ``Kernel``, NOT itself a source backend.

    Attributes:
        schema: the op-side interface name this realizes (the ``DeliveredParam.iface`` —
            also the Context tensor key).
        pool: the CONCRETE source pool (storage-topology ``Backend``\\ s) for this
            interface. The selection space :meth:`to_subspaces` lowers to a ``DesignSpace``.
        stream: ``{compute_backend_name -> fold list}`` — the BLOCK→STREAM fold each
            compute backend declares for this port (a reference to ``Backend.stream[iface]``,
            not a copy of the fold math). Carried so the seam data lives in one object; the
            demand reads the RESOLVED ``stream_width.<iface>`` the tiling engine derives from
            it, so this map is not itself consumed by :meth:`to_subspaces`.
        mem_modes: ``{compute_backend_name -> frozenset[str] | None}`` — the modes each
            compute backend accepts for this port (``None`` = permissive, both modes). The
            per-backend variation is dispatched on the selected ``implementation`` inside
            ``constrains``; this map exposes the same facts declaratively.
        publishes: the DEMAND closure ``(point, context) -> ParamDemand | None`` sized from
            the resolved ``stream_width.<iface>``. ``None`` when the interface has no demand
            for any point is expressed inside the closure (returns ``None``), never as a
            missing closure.
        constrains: the topology-mode guard as ``(domain_closure, legal)`` — ``domain_closure``
            ``(point, context) -> frozenset`` overrides the delivery root-axis domain;
            ``legal`` ``(point) -> tuple[str, ...]`` is reused to guard the axis default.
        deps: the declared cross-pool dependencies ``{"backend", topology.<iface>}``
            that make the supply waterfall structural for the topo-sort.
    """

    schema: str
    pool: tuple[Backend, ...]
    stream: Mapping[str, Any]
    mem_modes: Mapping[str, Any]
    publishes: Callable[[Any, Any], Any]
    constrains: tuple[Callable[[Any, Any], Any], Callable[[Any], Any]]
    deps: frozenset[str]

    def to_subspaces(self) -> tuple[DesignSpace, DesignSpace]:
        """The ``(demand_schema, guarded source sub-schema)`` pair this interface
        contributes, in supply-waterfall order.

        The demand stage publishes ``parameters.<iface>.demand`` from resolved compute
        geometry (``publishes``); the source pool then sizes its own realization from
        that demand, its ``topology`` domain filtered by ``constrains`` to the modes the
        selected compute backend accepts."""
        return (self._demand_space(), self._source_subspace())

    def _demand_space(self) -> DesignSpace:
        """The DEMAND stage as a derived-only schema, ordered BETWEEN the compute pool and
        the source pool by its ``demand_key`` reading the tiling engine's resolved
        ``stream_width.<iface>`` (present only after compute tiling)."""
        return DesignSpace(
            axes=(),
            derived=(Derived(demand_key(self.schema), self.publishes),),
            predicates=(),
        )

    def _source_subspace(self) -> DesignSpace:
        """The source pool for this interface with its ``topology`` root-axis domain
        overridden by the mem-mode guard, so only topologies the selected compute
        backend can accept remain selectable. The default is likewise guarded so an
        out-of-domain default never makes an unpinned topology illegal."""
        schema = pool_space(
            topology_key(self.schema),
            (),
            (),
            (),
            self.pool,
            sources_key=sources_key(self.schema),
        )
        root = schema.axes[0]  # pool_space emits the root topology axis first
        domain, legal = self.constrains
        guarded = replace(root, domain=domain, default=_topology_default(root.default, legal))
        return replace(schema, axes=(guarded,) + tuple(schema.axes[1:]))


def parameter_source_for(dp: DeliveredParam, compute_pool) -> ParameterSource:
    """Assemble the :class:`ParameterSource` for one delivered parameter from its
    :class:`~finn.kernels.model.param_contract.DeliveredParam` declaration + the op's compute
    pool. The single place the compute→source contract is built — the demand schema and
    the guarded source sub-schema for this interface have one owner."""
    iface = dp.iface
    stream = {b.name: b.stream_of(iface) for b in compute_pool}
    mem_modes = {b.name: b.mem_modes_of(iface) for b in compute_pool}
    return ParameterSource(
        schema=iface,
        pool=tuple(dp.pool),
        stream=stream,
        mem_modes=mem_modes,
        publishes=_demand_for(dp),
        constrains=_topology_domain(compute_pool, dp),
        deps=frozenset({BACKEND_AXIS, topology_key(iface)}),
    )
