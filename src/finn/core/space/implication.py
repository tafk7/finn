# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Implied decisions: an open Decision with exactly one viable case reads as that case.

PROTOTYPE (design/implied-decisions): the engine side of the implied-decisions
design pass. A configuration's implications are computed once per snapshot, on
the first read of an open Decision, and never committed: ``selections.capture``
holds only commitments, so nothing implied is persisted, and a successor
computes its own (an implication follows the facts and choices it was derived
from).

A case of a Decision over nodes is **viable** when it is in the Decision's
domain and neither the candidate it places nor the Decision's owner refuses
the configuration through its ``admission`` member (a ``ConstraintGroup``, a
constraint or a view; a group refuses as soon as one of its constraints does).
An admission still waiting on an open choice does not refuse. A Decision over
values is implied only when its domain has exactly one candidate. With one
viable case the Decision is implied; with several it stays open; with none it
is refused, each case with its reason.

Implications are found as ``settle`` found commitments, in rounds over the
open Decisions in rank order, each implied value visible to the next (a
Decision opened by an implied case, an adapter after the core it feeds, is
found in the same pass), on snapshots that do not imply themselves.
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
class Implications:
    """A snapshot's implied values, refused Decisions and open ones' viable cases."""

    implied: Mapping[int, object] = field(default_factory=dict)
    refused: Mapping[int, Rejected] = field(default_factory=dict)
    open: Mapping[int, tuple[object, ...]] = field(default_factory=dict)
    # Each Decision's refused cases and why: implied or refused, its reasons.
    reasons: Mapping[int, Mapping[str, str]] = field(default_factory=dict)


NONE = Implications()


def implications(snapshot: Snapshot) -> Implications:
    """The snapshot's implications, computed once."""
    found = snapshot.implication.get("result")
    if isinstance(found, Implications):
        return found
    if snapshot.implication.get("busy"):
        return NONE
    snapshot.implication["busy"] = True
    try:
        result = _compute(snapshot)
    finally:
        snapshot.implication.pop("busy", None)
    snapshot.implication["result"] = result
    return result


def _describe(results: list[QueryResult[object]]) -> str:
    return (
        "; ".join(
            f"{finding.owner}: {finding.code}: {finding.message}"
            for result in results
            if not isinstance(result, Available)
            for finding in result.findings
        )
        or "refused"
    )


def admission(candidate: Space) -> QueryResult[object] | None:
    """A Space's own refusal of its configuration: its ``admission`` member, if any."""
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
    handle = DecisionHandle[str](linked, index)
    candidates = point.field(handle).candidates()
    if not isinstance(candidates, Available) or candidates.value is None:
        return None
    scopes = dict(choice.cases)
    viable: list[str] = []
    reasons: dict[str, str] = {}
    base = state(point)
    for case in candidates.value:
        # The case among the candidates is a member of the domain; the trial takes it
        # without revalidating the rest: a committed choice nested under a selector not
        # implied yet is not a refusal of this case.
        trial = Snapshot(
            base.model, base.parameters, {**base.assignments, index: case}, implying=False
        )
        refusal = None
        scope = scopes[case]
        if scope is not None:
            verdict = admission(_attach(trial, scope))
            if isinstance(verdict, Rejected):
                refusal = verdict
        if refusal is None:
            verdict = admission(_attach(trial, choice.scope))
            if isinstance(verdict, Rejected):
                refusal = verdict
        if refusal is not None:
            reasons[case] = _describe([refusal])
            continue
        viable.append(case)
    return tuple(viable), reasons


def _compute(snapshot: Snapshot) -> Implications:
    linked = snapshot.linked
    implied: dict[int, object] = {}

    def plain() -> Space:
        assignments = {**snapshot.assignments, **implied}
        return _attach(
            Snapshot(snapshot.model, snapshot.parameters, assignments, implying=False), 0
        )

    point = plain()
    progressed = True
    refused: dict[int, Rejected] = {}
    open_: dict[int, tuple[object, ...]] = {}
    reasons: dict[int, Mapping[str, str]] = {}
    while progressed:
        progressed = False
        refused.clear()
        open_.clear()
        for index in sorted(linked.decisions, key=linked.ranks.__getitem__):
            if index in snapshot.assignments or index in implied:
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
                implied[index] = cases[0]
                point = plain()
                progressed = True
            elif not cases and index in linked.selector_choices:
                node = linked.nodes[index]
                detail = "; ".join(f"{case}: {why}" for case, why in reasons[index].items())
                refused[index] = Rejected(
                    (
                        Finding(
                            FindingKind.REJECTION,
                            "decision-no-viable-case",
                            node.owner,
                            f"no case is viable: {detail}",
                        ),
                    )
                )
            else:
                open_[index] = cases
    return Implications(
        MappingProxyType(implied),
        MappingProxyType(refused),
        MappingProxyType(open_),
        MappingProxyType(reasons),
    )


def implied(point: Space) -> Mapping[str, object]:
    """The configuration's implied Decisions by key (inspection)."""
    snapshot = state(point)
    found = implications(snapshot) if snapshot.implying else NONE
    return {snapshot.linked.nodes[index].key: value for index, value in found.implied.items()}


def why(point: Space, key: str) -> Mapping[str, str]:
    """Each refused case of the Decision keyed ``key`` and why (implied or refused)."""
    snapshot = state(point)
    found = implications(snapshot) if snapshot.implying else NONE
    return dict(found.reasons.get(snapshot.linked.keys[key], {}))


__all__ = ["Implications", "admission", "implications", "implied", "why"]
