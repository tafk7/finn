# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forced Decisions: an open Decision with exactly one viable case reads as that case.

PROBE (design/stream-source, after the human's gate on implied decisions). A
forced Decision is derived at read time and never stored: it stays unassigned
(``DecisionState`` has no other status), ``selections.capture`` holds only
commitments, and a successor derives its own. Inspection reports it
(``inspection.forced``) with the reason: each other case and why it is refused,
or a domain of one value.

A case of a Decision over nodes is **viable** when the candidate it places does
not refuse the configuration through its ``admission`` member (a
``ConstraintGroup``, a constraint or a view; a group refuses as soon as one of
its constraints does); an admission still waiting on an open choice does not
refuse. Nothing else is consulted: the Decision's owner is not. A Decision over
values is forced only when its domain has one candidate. With several viable
cases the Decision stays open; with none it is refused (``decision-no-viable-case``),
each case with its reason.

A snapshot's forced Decisions are found once, on the first read of an open
Decision, in rounds over the open Decisions in rank order, each forced value
visible to the next (an adapter after the core it feeds is found in the same
round), on copies that do not force themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from ._configuration import Space
from ._runtime import Snapshot
from .declarations import Constraint, ConstraintGroup, View
from .occurrence import _attach, state
from .references import DecisionHandle
from .results import Available, Finding, FindingKind, QueryResult, Rejected


@dataclass(frozen=True)
class Found:
    """A snapshot's forced values and refused Decisions by node, and each Decision's
    refused cases with their reasons."""

    values: Mapping[int, object] = field(default_factory=dict)
    refused: Mapping[int, Rejected] = field(default_factory=dict)
    reasons: Mapping[int, Mapping[str, str]] = field(default_factory=dict)


NOTHING = Found()


@dataclass(frozen=True)
class Forced:
    """A forced Decision for inspection: its value, and why the other cases are not
    viable (empty for a Decision over values whose domain has one candidate)."""

    key: str
    value: object
    refused: Mapping[str, str]


def forced(snapshot: Snapshot) -> Found:
    """The snapshot's forced Decisions, found once."""
    found = snapshot.forced.get("found")
    if isinstance(found, Found):
        return found
    if snapshot.forced.get("busy"):
        return NOTHING
    snapshot.forced["busy"] = True
    try:
        result = _find(snapshot)
    finally:
        snapshot.forced.pop("busy", None)
    snapshot.forced["found"] = result
    return result


def _describe(result: QueryResult[object]) -> str:
    return (
        "; ".join(
            f"{finding.owner}: {finding.code}: {finding.message}" for finding in result.findings
        )
        if not isinstance(result, Available)
        else "refused"
    )


def admission(candidate: Space) -> QueryResult[object] | None:
    """A candidate's own refusal of its configuration: its ``admission`` member, if any."""
    member = getattr(type(candidate), "admission", None)
    if isinstance(member, ConstraintGroup):
        assessment = candidate.inspect(member)
        refused = [result for result in assessment.results.values() if isinstance(result, Rejected)]
        if refused:
            return Rejected(tuple(finding for result in refused for finding in result.findings))
        return cast("QueryResult[object]", assessment.result)
    if isinstance(member, Constraint):
        return cast("QueryResult[object]", candidate.inspect(member).result)
    if isinstance(member, View):
        return cast("QueryResult[object]", candidate.query(member))
    return None


def _viable(point: Space, index: int) -> tuple[tuple[str, ...], dict[str, str]] | None:
    """A Decision over nodes' viable cases, and each refused case's reason."""
    linked = state(point).linked
    choice = linked.choices[linked.selector_choices[index]]
    candidates = point.field(DecisionHandle[str](linked, index)).candidates()
    if not isinstance(candidates, Available) or candidates.value is None:
        return None
    scopes = dict(choice.cases)
    viable: list[str] = []
    reasons: dict[str, str] = {}
    base = state(point)
    for case in candidates.value:
        scope = scopes[case]
        if scope is not None:
            # The configuration plus the case, without revalidating the rest: a committed
            # choice nested under a selector not forced yet is not this case's refusal.
            trial = Snapshot(
                base.model, base.parameters, {**base.assignments, index: case}, forcing=False
            )
            verdict = admission(_attach(trial, scope))
            if isinstance(verdict, Rejected):
                reasons[case] = _describe(verdict)
                continue
        viable.append(case)
    return tuple(viable), reasons


def _find(snapshot: Snapshot) -> Found:
    linked = snapshot.linked
    values: dict[int, object] = {}

    def plain() -> Space:
        assignments = {**snapshot.assignments, **values}
        return _attach(Snapshot(snapshot.model, snapshot.parameters, assignments, forcing=False), 0)

    point = plain()
    progressed = True
    refused: dict[int, Rejected] = {}
    reasons: dict[int, Mapping[str, str]] = {}
    while progressed:
        progressed = False
        refused.clear()
        for index in sorted(linked.decisions, key=linked.ranks.__getitem__):
            if index in snapshot.assignments or index in values:
                continue
            handle = DecisionHandle[object](linked, index)
            current = point.field(handle).state
            if not isinstance(current, Available) or current.value.status != "unassigned":
                continue  # inapplicable, or its guard waits on an open choice
            cases: tuple[object, ...]
            if index in linked.selector_choices:
                found = _viable(point, index)
                if found is None:
                    continue
                cases, why = found
                reasons[index] = MappingProxyType(why)
            else:
                candidates = point.field(handle).candidates()
                if not isinstance(candidates, Available) or candidates.value is None:
                    continue
                cases = tuple(candidates.value)
            if len(cases) == 1:
                values[index] = cases[0]
                point = plain()
                progressed = True
            elif not cases and index in linked.selector_choices:
                detail = "; ".join(f"{case}: {why}" for case, why in reasons[index].items())
                refused[index] = Rejected(
                    (
                        Finding(
                            FindingKind.REJECTION,
                            "decision-no-viable-case",
                            linked.nodes[index].owner,
                            f"no case is viable: {detail}",
                        ),
                    )
                )
    return Found(MappingProxyType(values), MappingProxyType(refused), MappingProxyType(reasons))


def report(point: Space) -> tuple[Forced, ...]:
    """The configuration's forced Decisions, by key, each with its reason (inspection)."""
    snapshot = state(point)
    found = forced(snapshot) if snapshot.forcing else NOTHING
    return tuple(
        Forced(linked.nodes[index].key, value, dict(found.reasons.get(index, {})))
        for linked in (snapshot.linked,)
        for index, value in found.values.items()
    )


__all__ = ["Forced", "Found", "admission", "forced", "report"]
