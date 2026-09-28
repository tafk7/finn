# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Settle a configuration: commit every Decision over nodes that compatibility decides.

``settle(point)`` looks at every applicable, undecided Decision over nodes. A
case is compatible when committing it alone is accepted and, if an
``admission`` is given, the candidate it places does not refuse the
configuration (``admission(candidate)`` is not ``Rejected``; ``None`` is no
rule). An admission still waiting on an open choice (``Unresolved``) does not
refuse: when every other case is refused, the remaining one is the only one
that can be built, whatever the open choice. A Decision with exactly one
compatible case is committed; one with several is a design choice and stays
open, as does one with none. Settling repeats until nothing
changes, since a commitment can open another Decision (an adapter follows the
core it feeds). Scalar Decisions are never settled: they have no candidate
that could refuse itself.

What admission means is the caller's: the engine knows no domain notion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from ._configuration import Space
from .occurrence import _attach, state
from .references import DecisionHandle
from .results import Available, QueryResult, Rejected

S = TypeVar("S", bound=Space)

Admission = Callable[[Space], "QueryResult[object] | None"]


@dataclass(frozen=True)
class Settlement(Generic[S]):
    """The settled point, what was committed, and each open Decision's compatible cases."""

    point: S
    committed: Mapping[str, str]
    open: Mapping[str, tuple[str, ...]]


def compatible_cases(point: S, key: str, admission: Admission | None = None) -> tuple[str, ...]:
    """The cases of the Decision over nodes keyed ``key`` that ``point`` admits."""
    linked = state(point).linked
    found = [item for item in linked.choices if item.key == key]
    if not found or linked.nodes[found[0].selector].kind != "decision":
        raise ValueError(f"{key}: not an open Decision over nodes of this configuration")
    choice = found[0]
    handle = DecisionHandle[str](linked, choice.selector)
    candidates = point.field(handle).candidates()
    if not isinstance(candidates, Available):
        return ()
    scopes = dict(choice.cases)
    result: list[str] = []
    for case in candidates.value:
        report = point.try_with_choices({handle: case})
        if not report.accepted:
            continue
        scope = scopes[case]
        if scope is not None and admission is not None:
            verdict = admission(_attach(state(report.instance), scope))
            if isinstance(verdict, Rejected):
                continue
        result.append(case)
    return tuple(result)


def settle(point: S, *, admission: Admission | None = None) -> Settlement[S]:
    """Commit every applicable, undecided Decision over nodes with exactly one compatible case."""
    committed: dict[str, str] = {}
    while True:
        linked = state(point).linked
        open_: dict[str, tuple[str, ...]] = {}
        progressed = False
        for choice in sorted(linked.choices, key=lambda item: item.key):
            if linked.nodes[choice.selector].kind != "decision":
                continue  # pinned by an enclosing body
            handle = DecisionHandle[str](linked, choice.selector)
            current = point.field(handle).state
            if not isinstance(current, Available) or current.value.status == "committed":
                continue  # inapplicable, guard unresolved, or already decided
            cases = compatible_cases(point, choice.key, admission)
            if len(cases) == 1:
                point = point.with_choices({handle: cases[0]})
                committed[choice.key] = cases[0]
                progressed = True
                break
            open_[choice.key] = cases
        if not progressed:
            return Settlement(point, committed, open_)


__all__ = ["Admission", "Settlement", "compatible_cases", "settle"]
