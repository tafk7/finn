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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Predicate:
    check: Callable[[Any, Any], str | None]
    description: str = ""

    def describe(self) -> str:
        return self.description


def predicate(description: str) -> Callable[[Callable[[Any, Any], str | None]], Predicate]:
    """Decorator: wrap a ``check(point, context)`` function as a ``Predicate``.

    The decorated function returns a reason string when illegal, or None when
    legal — the body typically ends ``return None if ok else "<why>"``.
    """

    def wrap(fn: Callable[[Any, Any], str | None]) -> Predicate:
        return Predicate(check=fn, description=description)

    return wrap
