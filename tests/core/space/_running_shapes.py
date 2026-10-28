# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type's own copies of shapes other test modules declare too, so that its
tests import no test module (what two test modules share lives in a private module):

- ``IN_AN_OPEN_CASE`` (with ``even``): a refused ``when`` in a case of an open selector,
  the reference harness's spelling (as ``test_reference_cases`` has it);
- ``ByArgument``: a chosen and a picked selector, each with a ``None`` case, given as a
  binding's argument, which reads the case's name (as ``test_select`` has it);
- ``PickedByAChoice`` (with ``Capped``): a choice in a case of a selector picked by a
  choice, its domain reading a value of that case (as ``test_reference_selectors`` has it,
  the domain rule's row).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.space.reference import Choice, Derived, Refuse, Scope, Selector, When, listed
from core.space.reference import Space as Description
from core.space.reference_harness import Case
from finn.core.space import (
    Decision,
    Domain,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    domain,
    reject,
    view,
)

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


# -- a selector given as a binding's argument reads its case name ----------------------------
#
# ``chosen`` is a chosen selector, ``picked`` one picked by ``chain`` (a derived value of the
# choice ``x``); each has the ``None`` case ``direct``. Given as a binding's argument (a
# derived value's, a view's, a constraint's, a guard's through a derived value), each reads
# its case's name, ``"direct"`` for the ``None`` case; in a body, ``self.chosen`` is the
# case's design space (``None`` for ``direct``).


class Lane(Space):
    width: int = Param()
    depth: int = Decision(domain=(1, 2, 4))


class Stage(Space):
    width: int = Param()
    stages: int = Decision(domain=(1, 2))


def _name(case: str) -> str:
    return case


def _is_lane(case: str) -> bool:
    return case == "lane"


def _no_stage(case: str) -> bool | Rejected:
    return reject("no-stage", "a stage refused") if case == "stage" else True


class ByArgument(Space):
    x: int = Decision(domain=(0, 1, 2))

    @derived
    def chain(self) -> str:
        return ("direct", "lane", "stage")[self.x]

    chosen = Select(direct=None, lane=Lane, stage=Stage, width=2)
    picked = Select(by=chain, direct=None, lane=Lane, stage=Stage, width=2)

    chosen_name: str = derived(case=chosen)(_name)
    picked_name: str = derived(case=picked)(_name)
    chosen_view = view(case=chosen)(_name)
    picked_view = view(case=picked)(_name)
    chosen_rule = constraint(case=chosen)(_no_stage)
    picked_rule = constraint(case=picked)(_no_stage)
    lane_chosen: bool = derived(case=chosen)(_is_lane)
    lane_only: int = Decision(domain=(1, 2), when=lane_chosen)

    @derived
    def body(self) -> str:
        return type(self.chosen).__name__


# -- a choice in a case of a selector picked by a choice ---------------------------------------
#
# ``sel`` is picked by ``pick``, which reads the choice ``x``. ``sel.a.z``'s domain reads
# ``sel.a.cap``, a derived value of the fact ``f`` in the same case: reading ``z``'s
# membership reads the case's guard, ``z``'s own existence guard, taken as holding, so it
# never reaches ``x`` (the domain rule holds, and the compile accepts it).


def _within(*, candidate: int, cap: int) -> bool:
    return candidate <= cap


def _three() -> tuple[int, ...]:
    return (1, 2, 3)


def capped(cap: object) -> Domain[int]:
    """Values 1 to 3, each at most ``cap``."""
    return domain(accepts=_within, candidates=_three, cap=cap)


class Capped(Space):
    f: int = Param()

    @derived
    def cap(self) -> int:
        return self.f + 1

    z: int = Decision(domain=capped(cap))


class PickedByAChoice(Space):
    f: int = Param()
    x: int = Decision(domain=(0, 1))

    @derived
    def pick(self) -> str:
        return ("a", "b")[self.x]

    sel = Select(by=pick, a=Capped(f=f), b=None)


__all__ = [
    "IN_AN_OPEN_CASE",
    "ByArgument",
    "Capped",
    "Lane",
    "PickedByAChoice",
    "Stage",
    "capped",
    "even",
]
