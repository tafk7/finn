# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type's own copies of shapes other test modules declare too, so that its
tests import no test module (what two test modules share lives in a private module):

- ``IN_AN_OPEN_CASE`` (with ``even``): a refused ``when`` in a case of an open selector,
  the reference harness's spelling (as ``test_reference_cases`` has it);
- ``Ranked`` (with ``Source`` and ``ranked_by``): a choice whose listed values its domain's
  order ranks by another choice (as ``test_inspection``'s ``Listed`` has it).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.space.reference import Choice, Derived, Refuse, Scope, Selector, When, listed
from core.space.reference import Space as Description
from core.space.reference_harness import Case
from finn.core.space import Decision, Domain, Space, domain

# -- a refused ``when`` in a case of an open selector ----------------------------------------
#
# ``s`` is open; its case ``a`` holds ``g``, present when ``d`` is at least 1, and ``d``
# refuses an odd ``f``, read by nothing else. ``g``'s piece is unknown while ``a``'s presence
# is: validity is open, waiting on ``s``.


def even(read: Mapping[str, Any]) -> object:
    return Refuse("d-odd") if read["f"] % 2 else read["f"]


IN_AN_OPEN_CASE = Case(
    "a refused when in a case of an open selector",
    Description(
        ("f",),
        Scope(
            "",
            derived=(Derived("d", ("f",), even),),
            selectors=(
                Selector(
                    "s",
                    (
                        Scope(
                            "a",
                            choices=(Choice("p", listed(1, 2)),),
                            children=(
                                Scope(
                                    "g",
                                    when=When(("d",), lambda read: read["d"] >= 1),
                                    choices=(Choice("q", listed(1, 2)),),
                                ),
                            ),
                        ),
                        Scope("b", choices=(Choice("r", listed(1, 2)),)),
                    ),
                ),
            ),
        ),
    ),
)


# -- a choice ranked by another --------------------------------------------------------------


def preferred_first(values: tuple[int, ...], *, preferred: int) -> list[int]:
    return sorted(values, key=lambda value: value != preferred)


def ranked_by(preferred: object) -> Domain[int]:
    """Values 1 and 2, ranked with ``preferred`` first: an order reads a choice."""
    return domain(
        accepts=lambda *, candidate: candidate in (1, 2),
        candidates=lambda: (1, 2),
        order=preferred_first,
        preferred=preferred,
    )


class Source(Space):
    b: int = Decision(values=(1, 2))


class Ranked(Space):
    source = Source()
    a: int = Decision(domain=ranked_by(source.b))


__all__ = ["IN_AN_OPEN_CASE", "Ranked", "Source", "even", "preferred_first", "ranked_by"]
