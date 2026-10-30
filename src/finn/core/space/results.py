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

    One reason can reach a merge by two routes: a view whose output and whose
    Space's validity both read a refused value, or a computation that reads two
    aliases of one blocked value. Findings are frozen values, so the same reason
    compares equal and is kept once; findings that differ in anything, owner
    included, are all kept.
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
    """One view result with its presence, owning validity and attempted output.

    ``validity`` is None when presence prevented the gate. ``output_result`` is None
    when presence or owning refusal prevented output entry. Open validity permits a
    local result; its unrelated waits are not merged into that result. Completeness
    remains a separate operation-level question.
    """

    result: QueryResult[T]
    applicability: QueryResult[bool]
    validity: QueryResult[bool] | None
    output_result: QueryResult[T] | None

    def require_value(self) -> T:
        """Return the accepted value or raise with this assessment as context."""

        return require_value(self.result, context=self)


def require_value(result: QueryResult[T], *, context: object | None = None) -> T:
    """Return an available value and preserve the unavailable result on failure."""

    if isinstance(result, Available):
        return result.value
    from .errors import ValueUnavailableError  # noqa: PLC0415 - avoid an error/result cycle

    raise ValueUnavailableError(result, context=context)


def assess_view(
    output_result: QueryResult[T] | None,
    *,
    validity: QueryResult[bool] | None,
    applicability: QueryResult[bool] = Available(True),
) -> ViewAssessment[T]:
    """Presence first, then normalized owning validity, then the local output.

    The evaluator observes this order before entering a body. This reducer records
    what was attempted; open validity alone does not withhold an output.
    """

    if not isinstance(applicability, Available) or applicability.value is False:
        inactive: QueryResult[T] = (
            applicability if not isinstance(applicability, Available) else Inapplicable()
        )
        return ViewAssessment(inactive, applicability, None, None)
    if validity is None:
        raise ValueError("an applicable view requires its owning validity")
    if isinstance(validity, (Rejected, Inapplicable)):
        return ViewAssessment(validity, applicability, validity, None)
    if isinstance(validity, Available) and validity.value is not True:
        raise ValueError("owning validity must be normalized before assessing a view")
    if output_result is None:
        raise ValueError("an applicable view requires its local output")
    return ViewAssessment(output_result, applicability, validity, output_result)


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
