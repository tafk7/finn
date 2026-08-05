############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Predicate`` — the one legality kind (design-space-model.md §1.4).

``check(point, context) -> reason | None`` where None means legal. There is no
``evaluation_phase``: what a predicate *happens* to read (config only / +device /
+toolchain / +data) is provenance an evaluator can inspect, not a category the
author declares. The reason string powers explain-style diagnostics.

``deps``/``optional_deps`` make that provenance actually inspectable, mirroring
:class:`~finn.kernels.engine.axis.Axis` and
:class:`~finn.kernels.engine.derived.Derived`. A predicate never enters the topo-sort
(predicates are evaluated last, over a complete point, so nothing orders them), but the
declaration is what lets the engine ask a cheaper question than "resolve everything":
a rule whose read-closure touches no axis is decidable from Context ALONE, before any
choice is pinned. That is the difference between answering "can this node be a kernel?"
with a full resolve per pool member and answering it from the node's datatypes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .derived import _reject_dep_overlap


@dataclass(frozen=True)
class Predicate:
    check: Callable[[Any, Any], str | None]
    description: str = ""
    deps: frozenset[str] = field(default_factory=frozenset)
    optional_deps: frozenset[str] = field(default_factory=frozenset)
    origin: str = ""  # where this came from; see engine.provenance

    def __post_init__(self):
        if not isinstance(self.deps, frozenset):
            object.__setattr__(self, "deps", frozenset(self.deps))
        if not isinstance(self.optional_deps, frozenset):
            object.__setattr__(self, "optional_deps", frozenset(self.optional_deps))
        _reject_dep_overlap(self)

    @property
    def name(self) -> str:
        """Predicates are anonymous; the description is the closest thing to an identity.

        Exists so a Predicate satisfies the same shape as Axis/Derived for the dep-overlap
        check and the sibling-coupling walk, which name the offending node in their errors."""
        return self.description

    def describe(self) -> str:
        return self.description


def predicate(
    description: str, *, deps=(), optional_deps=()
) -> Callable[[Callable[[Any, Any], str | None]], Predicate]:
    """Decorator: wrap a ``check(point, context)`` function as a ``Predicate``.

    The decorated function returns a reason string when illegal, or None when
    legal — the body typically ends ``return None if ok else "<why>"``.

    Declare in ``deps`` every POINT key the body reads. Context reads need no
    declaration — Context is a given, available before any choice is pinned; it is
    precisely a predicate that reads NO point key which can be evaluated early.
    """

    def wrap(fn: Callable[[Any, Any], str | None]) -> Predicate:
        return Predicate(
            check=fn,
            description=description,
            deps=frozenset(deps),
            optional_deps=frozenset(optional_deps),
        )

    return wrap
