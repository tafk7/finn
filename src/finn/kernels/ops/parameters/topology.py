############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""``storage_topology`` — the parameters-pool member abstraction.

A **storage topology** is one way a kernel's parameters physically live and travel to
the compute core: ``embedded`` (compiled in), ``decoupled`` (on-chip replay via
memstream), and later ``external`` / ``off-chip-DMA`` / ``dynamic``. Each topology is
a self-contained bundle — its own storage/transport/selection axes, coupling derived,
feasibility predicates, RTL sources, and emit — exactly the shape of an
:class:`Implementation`. We do not introduce a parallel type: ``pool_schema`` reads an
``Implementation`` structurally, and the root axis name (``topology``) is what gives
the pool its meaning. This factory is a semantic alias so a topology reads as a
topology, not as a compute "implementation", while reusing the proven selection
machinery ([[impl-bundles]]).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finn.kernels.space import Artifacts, Derived, Implementation, Predicate
from finn.kernels.space.axis import Axis


def _feasible_ok(_point, _context) -> None:
    return None


def storage_topology(
    name: str,
    *,
    feasible: Callable[[Any, Any], str | None] = _feasible_ok,
    axes: tuple[Axis, ...] = (),
    derived: tuple[Derived, ...] = (),
    predicates: tuple[Predicate, ...] = (),
    sources: tuple[str, ...] = (),
    emit: Callable[[Any, Any], "Artifacts"] | None = None,
) -> Implementation:
    """Declare one storage topology as a parameters-pool member.

    Thin wrapper over :class:`Implementation` — same fields, topology-flavoured name.
    The pool's root axis is ``topology`` (see :func:`parameters_schema`); ``name`` is
    the value that selects this topology.
    """
    return Implementation(
        name=name,
        feasible=feasible,
        axes=axes,
        derived=derived,
        predicates=predicates,
        sources=sources,
        emit=emit,
    )
