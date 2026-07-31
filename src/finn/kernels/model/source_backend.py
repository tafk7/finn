############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``source_backend`` — a pool member of a :class:`~finn.kernels.model.parameter_source.ParameterSource`.

A ``source_backend`` is one pool member of a ``ParameterSource`` — the ``WEIGHT_SOURCE``
mirror of a compute :class:`~finn.kernels.model.backend.Backend`. A **storage topology** is
one way a kernel's parameters physically live and travel to the compute core: ``embedded``
(compiled in), ``decoupled`` (on-chip replay via memstream), and later ``external`` /
``off-chip-DMA`` / ``dynamic``. Each topology is a self-contained bundle — its own
storage/transport/selection axes, coupling derived, feasibility predicates, RTL sources, and
emit — exactly the shape of a :class:`Backend`. We do not introduce a parallel type:
``pool_space`` reads a ``Backend`` structurally, and the root axis name (``topology``) is
what gives the pool its meaning. This factory is a semantic alias so a source backend reads
as a source, not as a compute "backend", while reusing the proven selection
machinery ([[impl-bundles]]).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finn.kernels.model.artifacts import Artifacts, RtlModule
from finn.kernels.engine.axis import Axis
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.predicate import Predicate
from finn.kernels.model.backend import Backend


def source_backend(
    name: str,
    *,
    mem_mode: str,
    language: str | None = None,
    axes: tuple[Axis, ...] = (),
    derived: tuple[Derived, ...] = (),
    predicates: tuple[Predicate, ...] = (),
    sources: tuple[str, ...] = (),
    emit: Callable[[Any, Any], "Artifacts"] | None = None,
    schema: RtlModule | None = None,
) -> Backend:
    """Declare one storage topology as a source-pool member (a ``ParameterSource`` member).

    Thin wrapper over :class:`Backend` — same fields, topology-flavoured name. The pool's
    root axis is ``topology`` (see :func:`parameters_schema`); ``name`` is the value that
    selects this topology. ``mem_mode`` is the memory-realization mode this topology presents
    (``"embedded"``/``"decoupled"``) — carried on the ``Backend`` so the generic delivery
    guard reads it structurally (no string→mode side-table). ``language`` is the topology's
    realization language when it emits its own HDL (``"rtl"`` for the memstream streamer);
    ``None`` when it emits nothing of its own (``embedded`` — baked into the compute core).
    ``schema`` is the optional typed template contract this topology emits (a streaming
    topology references its wrapper schema; an embedded topology with no template leaves it
    ``None``).
    """
    return Backend(
        name=name,
        language=language,
        mem_mode=mem_mode,
        axes=axes,
        derived=derived,
        predicates=predicates,
        sources=sources,
        emit=emit,
        schema=schema,
    )
