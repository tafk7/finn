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

    def __post_init__(self):
        if not isinstance(self.deps, frozenset):
            object.__setattr__(self, "deps", frozenset(self.deps))
