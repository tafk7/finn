############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Attr`` — a node-instance CONSTANT: fixed before resolution, never explored.

The third design-space category, beside :class:`~finn.kernels.engine.axis.Axis` (a free
choice) and :class:`~finn.kernels.engine.derived.Derived` (a computed quantity). An ``Attr``
is neither: it is a value the op declares, the frontend fixes once, and every consumer reads
off the Point. MVAU's ``ActVal`` is the motivating case — the ``out_bias`` of a
``MultiThreshold`` that infer ABSORBS, so after the rewrite the graph no longer holds it.

**Attr vs Context — the line, and why it is not provenance.** The obvious rule ("Context comes
from the graph, an Attr is what the graph no longer holds") is already false in this engine:
``Context.fpgapart`` is sourced from a nodeattr (``ir/kernel_op.py``), and ``runtime_writeable``
from a model metadata prop. Provenance does not separate them.

The line that holds is CLOSED SCHEMA vs OPEN NAMESPACE. :class:`~finn.kernels.engine.context.Context`
is a frozen, engine-owned dataclass with a fixed field list: absorbing ``ActVal`` would mean a
field per op, the engine accreting op-specific knowledge in a struct it owns. An ``Attr`` lives
in the op-declared point namespace, where ``Axis`` and ``Derived`` already live — the engine
handles it generically and never names it.

The OPERATIONAL test is **nameability in deps**, and it is forced rather than chosen. ``deps``
is not "what do I read", it is "what must be computed BEFORE me": Context is complete before
resolution starts, so there is nothing to order against and ``_topo_sort`` rejects a dep naming
one. A point entry is produced DURING the walk, so reading one is a real edge. The deps auditor
(``tests/engine/test_deps_audit.py``) requires every point read to be declared — so a value a
``Derived``/``Predicate`` must order against, or that must appear in a ``resolve(want=)``
closure or contribute to a stratum, HAS to be a point-namespace citizen. ``narrow_weights ->
mlo_max_iter`` is exactly that edge. Hence: if a value must be nameable in a dep, it cannot be
a Context field.

**Source-agnostic by construction.** Where the value comes from is a property of ``default``,
not of the category: an ``Attr`` resolves assignment-or-default, and the default may read
Context. So one attr can be nodeattr-sourced and another Context-derived with no new mechanism.

**No deps, no exists, no point-dependent domain.** None is meaningful for a constant, and
omitting them is what keeps an ``Attr`` out of the DAG's ordering problem while still being a
legal dep TARGET (it is written to the point before any axis, so anything may read it). Its
``default`` takes ``(context)`` only — deliberately NOT ``(point, context)``: a point-reading
default would be a genuine dependency and would put attrs back in the DAG, defeating the point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .axis import PredicateDomain


@dataclass(frozen=True)
class Attr:
    """A frontend-fixed node constant carried on the Point.

    Attributes:
        name: the point key (and the nodeattr name).
        validate: membership test for the value — the ``test`` half of a
            :class:`~finn.kernels.engine.axis.PredicateDomain`. Kept rather than dropped as
            ceremony: it preserves today's domain check, and the nodeattr bridge needs the
            type information regardless. NOT vestigial.
        label: human description of the legal values ("int", "nonneg int"). Read by the
            nodeattr bridge to pick a storage type, and by diagnostics.
        default: the value when the assignment does not pin it — a plain value, or a
            ``(context) -> value`` callable for a Context-derived constant.
        origin: where this came from; see :mod:`finn.kernels.engine.provenance`.
    """

    name: str
    validate: Callable[[Any], bool]
    label: str = ""
    default: Any = None
    origin: str = ""

    # An Attr is not in the DAG's dependency problem: it is written to the point before any
    # axis, so it can be READ by anything without ordering against anything. These are frozen
    # empty so the uniform node walks (topo-sort, stratum, the deps auditor) need no isinstance
    # branch — an Attr simply contributes no edges.
    deps: frozenset[str] = field(default_factory=frozenset, init=False)
    optional_deps: frozenset[str] = field(default_factory=frozenset, init=False)

    @property
    def domain(self) -> PredicateDomain:
        """The membership test as a :class:`PredicateDomain`, so ``val in attr.domain`` reads
        the same as it does for an axis. A PROPERTY, not a stored callable: an attr's domain
        cannot depend on the point or the context, and making that structural is half of why
        the category exists."""
        return PredicateDomain(self.label, self.validate)

    def value(self, context) -> Any:
        """The default value for this Context — resolving a ``(context) -> value`` callable,
        or returning a plain value as-is."""
        return self.default(context) if callable(self.default) else self.default


def attr(
    name: str,
    label: str,
    test: Callable[[Any], bool],
    default: Any,
    *,
    origin: str = "",
) -> Attr:
    """A frontend-fixed node constant. The ``Attr`` counterpart of
    :func:`~finn.kernels.engine.axis.predicate_axis`, with the same
    ``(name, label, test, default)`` argument order so the migration reads as a rename."""
    return Attr(name=name, validate=test, label=label, default=default, origin=origin)
