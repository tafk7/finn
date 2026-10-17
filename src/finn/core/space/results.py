# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Query results, the findings they carry, and the shared assessment reducer."""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar, Union, cast, get_args, get_origin

from finn.core.reporting.findings import Finding, FindingKind, ordered_findings

from .semantics import NoTruthValue

T = TypeVar("T")


def merged_findings(answers: Iterable[NonValue]) -> tuple[Finding, ...]:
    """The findings of several answers, each identical finding once.

    One reason can reach a merge by two routes: a view that reads a value and
    also requires it, or a computation that reads two aliases of one blocked
    value. Findings are frozen values, so the same reason compares equal and is
    kept once; findings that differ in anything, owner included, are all kept.
    """

    return tuple(dict.fromkeys(finding for answer in answers for finding in answer.findings))


@dataclass(frozen=True, slots=True)
class Available(NoTruthValue, Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Inapplicable(NoTruthValue):
    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", ordered_findings(self.findings))


@dataclass(frozen=True, slots=True)
class Rejected(NoTruthValue):
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        if not self.findings:
            raise ValueError("Rejected requires at least one finding")
        object.__setattr__(self, "findings", ordered_findings(self.findings))


@dataclass(frozen=True, slots=True)
class Unresolved(NoTruthValue):
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        if not self.findings:
            raise ValueError("Unresolved requires at least one finding")
        object.__setattr__(self, "findings", ordered_findings(self.findings))


QueryResult: TypeAlias = Available[T] | Inapplicable | Rejected | Unresolved
NonValue: TypeAlias = Inapplicable | Rejected | Unresolved


def marked_value_type(annotation: object) -> object | None:
    """The value type ``T`` of an output annotated ``T | Rejected`` (any of the markers)."""
    if get_origin(annotation) not in (Union, types.UnionType):
        return None
    values = [
        arg for arg in get_args(annotation) if arg not in (Inapplicable, Rejected, Unresolved)
    ]
    return values[0] if len(values) == 1 else None


def reject(
    code: str,
    message: str,
    *,
    owner: str = "",
    values: Mapping[str, object] | None = None,
    causes: Iterable[Finding] = (),
) -> Rejected:
    return Rejected(
        (
            Finding(
                FindingKind.REJECTION,
                code,
                owner,
                message,
                tuple((values if values is not None else {}).items()),
                tuple(causes),
            ),
        )
    )


def owned_result(answer: QueryResult[T], owner: str) -> QueryResult[T]:
    """Attach the evaluator owner to authored findings without rebasing causes."""

    if isinstance(answer, Available):
        return answer
    findings = tuple(
        replace(finding, owner=owner) if not finding.owner else finding
        for finding in answer.findings
    )
    return type(answer)(findings)


def constraint_result(answer: bool | QueryResult[bool], owner: str) -> QueryResult[bool]:
    """Normalize bare false and explicit refusal to the same constraint meaning."""

    if type(answer) is bool:
        answer = Available(answer)
    if not isinstance(answer, (Available, Inapplicable, Rejected, Unresolved)):
        raise TypeError("a constraint must return bool or QueryResult[bool]")
    if isinstance(answer, Available):
        if type(answer.value) is not bool:
            raise TypeError("a constraint must return bool or QueryResult[bool]")
        if not answer.value:
            return reject("constraint-false", "constraint returned False", owner=owner)
    return owned_result(answer, owner)


@dataclass(frozen=True, slots=True)
class DecisionState(Generic[T]):
    owner: str
    status: Literal["unassigned", "committed"] = "unassigned"
    value: T | None = None

    def __post_init__(self) -> None:
        if self.status not in {"unassigned", "committed"}:
            raise ValueError("unknown decision state")
        if self.status == "unassigned" and self.value is not None:
            raise ValueError("an unassigned decision has no value")


def _unresolved(answers: Iterable[QueryResult[object]]) -> Unresolved | None:
    findings = merged_findings(answer for answer in answers if isinstance(answer, Unresolved))
    return Unresolved(findings) if findings else None


def _rejected(answers: Iterable[QueryResult[object]]) -> Rejected | None:
    findings = merged_findings(answer for answer in answers if isinstance(answer, Rejected))
    return Rejected(findings) if findings else None


@dataclass(frozen=True, slots=True)
class ConstraintAssessment:
    results: Mapping[str, QueryResult[bool]]
    result: QueryResult[bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(sorted(self.results.items()))))

    @property
    def verdict(self) -> bool | None:
        if isinstance(self.result, Available):
            return self.result.value
        return False if isinstance(self.result, Rejected) else None

    @property
    def refused(self) -> tuple[str, ...]:
        return tuple(
            owner for owner, answer in self.results.items() if isinstance(answer, Rejected)
        )

    @property
    def not_applicable(self) -> tuple[str, ...]:
        return tuple(
            owner for owner, answer in self.results.items() if isinstance(answer, Inapplicable)
        )


@dataclass(frozen=True, slots=True)
class ReadinessAssessment:
    results: Mapping[str, QueryResult[object]]
    result: QueryResult[bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(sorted(self.results.items()))))

    @property
    def ready(self) -> bool | None:
        return self.result.value if isinstance(self.result, Available) else None


def assess_constraints(answers: Mapping[str, QueryResult[bool]]) -> ConstraintAssessment:
    normalized = {owner: constraint_result(answer, owner) for owner, answer in answers.items()}
    values = tuple(cast(QueryResult[object], answer) for answer in normalized.values())
    unresolved = _unresolved(values)
    refused = _rejected(values)
    result: QueryResult[bool] = (
        refused
        if refused is not None
        else unresolved
        if unresolved is not None
        else Available(True)
    )
    return ConstraintAssessment(normalized, result)


@dataclass(frozen=True, slots=True)
class ViewAssessment(Generic[T]):
    output_result: QueryResult[T]
    readiness: ReadinessAssessment
    constraints: ConstraintAssessment
    accepted_result: QueryResult[T]

    def require_value(self) -> T:
        """Return the accepted value or raise with this assessment as context."""

        return require_value(self.accepted_result, context=self)


def require_value(result: QueryResult[T], *, context: object | None = None) -> T:
    """Return an available value and preserve the unavailable result on failure."""

    if isinstance(result, Available):
        return result.value
    from .errors import ValueUnavailableError  # noqa: PLC0415 - avoid an error/result cycle

    raise ValueUnavailableError(result, context=context)


def assess_view(
    output_result: QueryResult[T],
    *,
    owner: str,
    constraints: Mapping[str, QueryResult[bool]] | None = None,
    applicability: QueryResult[bool] = Available(True),
) -> ViewAssessment[T]:
    """Reduce raw output and obligations identically for every view form.

    Evaluators must resolve applicability before requesting the other answers.
    The reducer also preserves that precedence when given existing answers.
    """

    constraint_results = assess_constraints(constraints if constraints is not None else {})
    required = {
        name: cast(QueryResult[object], answer)
        for name, answer in constraint_results.results.items()
    }
    # The compiler owns the names and rejects member collisions. The raw view
    # output is always included even when no extra readiness was requested.
    required[owner] = cast(QueryResult[object], output_result)
    unresolved = _unresolved(required.values())
    readiness = ReadinessAssessment(
        required, unresolved if unresolved is not None else Available(True)
    )
    accepted: QueryResult[T]
    if not isinstance(applicability, Available):
        accepted = applicability
    elif applicability.value is False:
        accepted = Inapplicable()
    else:
        refusal = _rejected(required.values())
        if refusal is not None:
            accepted = refusal
        elif isinstance(readiness.result, Unresolved):
            accepted = readiness.result
        else:
            accepted = output_result
    return ViewAssessment(output_result, readiness, constraint_results, accepted)


__all__ = [
    "QueryResult",
    "marked_value_type",
    "ConstraintAssessment",
    "Available",
    "DecisionState",
    "Inapplicable",
    "NonValue",
    "ReadinessAssessment",
    "Rejected",
    "Unresolved",
    "ViewAssessment",
    "assess_constraints",
    "assess_view",
    "constraint_result",
    "merged_findings",
    "owned_result",
    "reject",
]
