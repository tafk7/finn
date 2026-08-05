############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Derived`` — a computed quantity, never a choice (design-space-model.md §1.3).

A ``Derived`` reads axes + context (including initializer VALUES) and produces a
value that is added to the point *after* all axes are fixed. It never enters the
search space — the model must never enumerate a derived value as a dimension.
Datatype-valued derivations reuse the primitives' range-builders directly rather
than reinventing them.

``deps`` names the axes and *other deriveds* this derivation's ``compute`` reads,
so the schema can order derived resolution (a derived may depend on another
derived, resolved in dependency order). Axes are always fixed before any derived,
so an axis dep is a prerequisite that needs no sequencing among the deriveds; a
derived dep does. Deps are explicitly declared (closures cannot be introspected),
mirroring :class:`~finn.kernels.engine.axis.Axis`.

``optional_deps`` names reads that are CONDITIONAL on composition: "order me after
this name if the space defines it; do not fail if it does not." A ``deps`` entry
naming an absent key is an authoring error (the typo check stays strict); an
``optional_deps`` entry naming an absent key is simply not an edge. This exists
because a dep set is static while a closure's reads are guarded — the same pool
resolves STANDALONE (where a composing op's key is absent by design) and COMPOSED
(where it is present and must order first). Reading such a key via
``p.get(name, None)`` and declaring it optional is the honest declaration; the
alternative is leaving the read undeclared and letting fold order carry the edge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Derived:
    name: str
    compute: Callable[[Any, Any], Any]
    deps: frozenset[str] = field(default_factory=frozenset)
    optional_deps: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.deps, frozenset):
            object.__setattr__(self, "deps", frozenset(self.deps))
        if not isinstance(self.optional_deps, frozenset):
            object.__setattr__(self, "optional_deps", frozenset(self.optional_deps))
        _reject_dep_overlap(self)


def _reject_dep_overlap(node) -> None:
    """A name may not be both required and optional — "must exist" and "may be absent"
    are contradictory claims about the same read. Purely LOCAL to one node (no whole-space
    knowledge), so unlike the unknown-name/cycle checks it is safe to run at construction."""
    both = node.deps & node.optional_deps
    if both:
        raise ValueError(
            f"{node.name!r} declares {sorted(both)!r} in BOTH deps and optional_deps — "
            f"a dependency is either required or optional, not both"
        )
