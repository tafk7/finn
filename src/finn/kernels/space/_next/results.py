# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Answers, deterministic causal findings and the shared assessment reducer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from .semantics import NoTruthValue

T = TypeVar("T")


class FindingKind(str, Enum):
    BLOCKER = "blocker"
    LIMITATION = "limitation"
    REJECTION = "rejection"
    AUTHORING = "authoring"
    REQUEST = "request"


def _freeze_detail(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_detail(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("finding detail mapping keys must be strings")
        return tuple(sorted((key, _freeze_detail(item)) for key, item in value.items()))
    raise TypeError("finding details must contain primitive values, sequences or string mappings")


def _detail_key(value: object) -> str:
    # Only canonical primitives survive _freeze_detail; no user repr is called.
    if isinstance(value, tuple):
        return "tuple:" + repr(tuple(_detail_key(item) for item in value))
    if type(value) is float:
        return "float:" + value.hex()
    return type(value).__name__ + ":" + repr(value)


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason, its semantic owner, and the reasons that caused it.

    An empty owner is allowed in authored callback results; evaluation assigns
    that owner before publishing the answer. Existing cause owners are retained.
    """

    kind: FindingKind
    code: str
    owner: str
    message: str
    details: tuple[tuple[str, object], ...] = ()
    causes: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("a finding requires a nonempty code")
        if any(not isinstance(key, str) for key, _ in self.details):
            raise TypeError("finding detail keys must be strings")
        details = tuple(
            sorted(
                ((key, _freeze_detail(value)) for key, value in self.details),
                key=lambda pair: pair[0],
            )
        )
        if len({key for key, _ in details}) != len(details):
            raise ValueError("finding detail keys must be unique")
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "causes", ordered_findings(self.causes))


def finding_sort_key(finding: Finding) -> tuple[object, ...]:
    return (
        finding.owner,
        finding.kind.value,
        finding.code,
        finding.message,
        tuple((key, _detail_key(value)) for key, value in finding.details),
        tuple(finding_sort_key(cause) for cause in finding.causes),
    )


def ordered_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=finding_sort_key))


@dataclass(frozen=True, slots=True)
class Decided(NoTruthValue, Generic[T]):
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


Answer: TypeAlias = Decided[T] | Inapplicable | Rejected | Unresolved
NonValue: TypeAlias = Inapplicable | Rejected | Unresolved


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


def owned_answer(answer: Answer[T], owner: str) -> Answer[T]:
    """Attach the evaluator owner to authored findings without rebasing causes."""

    if isinstance(answer, Decided):
        return answer
    findings = tuple(
        replace(finding, owner=owner) if not finding.owner else finding
        for finding in answer.findings
    )
    return type(answer)(findings)


def constraint_answer(answer: bool | Answer[bool], owner: str) -> Answer[bool]:
    """Normalize bare false and explicit refusal to the same constraint meaning."""

    if type(answer) is bool:
        answer = Decided(answer)
    if not isinstance(answer, (Decided, Inapplicable, Rejected, Unresolved)):
        raise TypeError("a constraint must return bool or Answer[bool]")
    if isinstance(answer, Decided):
        if type(answer.value) is not bool:
            raise TypeError("a constraint must return bool or Answer[bool]")
        if not answer.value:
            return reject("constraint-false", "constraint returned False", owner=owner)
    return owned_answer(answer, owner)


@dataclass(frozen=True, slots=True)
class MissingInput(NoTruthValue):
    """An optional external parameter was omitted, distinct from supplied None."""

    owner: str = ""


@dataclass(frozen=True, slots=True)
class NotApplicable(NoTruthValue):
    """An explicitly optional dependency does not apply in this specialization."""

    owner: str = ""


MISSING = MissingInput()
NOT_APPLICABLE = NotApplicable()


@dataclass(frozen=True, slots=True)
class DecisionState(Generic[T]):
    owner: str
    status: Literal["unassigned", "committed"] = "unassigned"
    value: T | None = None
    origin: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"unassigned", "committed"}:
            raise ValueError("unknown decision state")
        if (self.status == "committed") != (self.origin is not None):
            raise ValueError("only a committed decision state has an origin")
        if self.status == "unassigned" and self.value is not None:
            raise ValueError("an unassigned decision has no value")


def _unresolved(answers: Iterable[Answer[object]]) -> Unresolved | None:
    findings = tuple(
        finding
        for answer in answers
        if isinstance(answer, Unresolved)
        for finding in answer.findings
    )
    return Unresolved(findings) if findings else None


def _rejected(answers: Iterable[Answer[object]]) -> Rejected | None:
    findings = tuple(
        finding for answer in answers if isinstance(answer, Rejected) for finding in answer.findings
    )
    return Rejected(findings) if findings else None


@dataclass(frozen=True, slots=True)
class ConstraintAssessment:
    answers: Mapping[str, Answer[bool]]
    answer: Answer[bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(sorted(self.answers.items()))))

    @property
    def verdict(self) -> bool | None:
        if isinstance(self.answer, Decided):
            return self.answer.value
        return False if isinstance(self.answer, Rejected) else None

    @property
    def refused(self) -> tuple[str, ...]:
        return tuple(
            owner for owner, answer in self.answers.items() if isinstance(answer, Rejected)
        )

    @property
    def not_applicable(self) -> tuple[str, ...]:
        return tuple(
            owner for owner, answer in self.answers.items() if isinstance(answer, Inapplicable)
        )


@dataclass(frozen=True, slots=True)
class ReadinessAssessment:
    answers: Mapping[str, Answer[object]]
    answer: Answer[bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(sorted(self.answers.items()))))

    @property
    def ready(self) -> bool | None:
        return self.answer.value if isinstance(self.answer, Decided) else None


def assess_constraints(answers: Mapping[str, Answer[bool]]) -> ConstraintAssessment:
    normalized = {owner: constraint_answer(answer, owner) for owner, answer in answers.items()}
    values = tuple(cast(Answer[object], answer) for answer in normalized.values())
    unresolved = _unresolved(values)
    refused = _rejected(values)
    result: Answer[bool] = (
        unresolved if unresolved is not None else refused if refused is not None else Decided(True)
    )
    return ConstraintAssessment(normalized, result)


def assess_readiness(answers: Mapping[str, Answer[object]]) -> ReadinessAssessment:
    normalized: dict[str, Answer[object]] = {}
    for owner, answer in answers.items():
        if (
            isinstance(answer, Decided)
            and isinstance(answer.value, DecisionState)
            and answer.value.status == "unassigned"
        ):
            normalized[owner] = Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "decision-unassigned",
                        answer.value.owner,
                        "readiness requires a committed decision",
                    ),
                )
            )
        else:
            normalized[owner] = answer
    unresolved = _unresolved(normalized.values())
    return ReadinessAssessment(normalized, unresolved if unresolved is not None else Decided(True))


@dataclass(frozen=True, slots=True)
class ViewAssessment(Generic[T]):
    output_answer: Answer[T]
    readiness: ReadinessAssessment
    constraints: ConstraintAssessment
    accepted_answer: Answer[T]

    @property
    def raw_answer(self) -> Answer[T]:
        return self.output_answer


def assess_view(
    output_answer: Answer[T],
    *,
    owner: str,
    requires: Mapping[str, Answer[object]] | None = None,
    constraints: Mapping[str, Answer[bool]] | None = None,
    applicability: Answer[bool] = Decided(True),
) -> ViewAssessment[T]:
    """Reduce raw output and obligations identically for every view form.

    Evaluators must resolve applicability before requesting the other answers.
    The reducer also preserves that precedence when given existing answers.
    """

    constraint_results = assess_constraints(constraints if constraints is not None else {})
    required = dict(requires if requires is not None else {})
    required.update(
        (name, cast(Answer[object], answer)) for name, answer in constraint_results.answers.items()
    )
    # The compiler owns the names and rejects member collisions. The raw view
    # output is always included even when no extra readiness was requested.
    required[owner] = cast(Answer[object], output_answer)
    readiness = assess_readiness(required)
    accepted: Answer[T]
    if not isinstance(applicability, Decided):
        accepted = applicability
    elif applicability.value is False:
        accepted = Inapplicable()
    elif isinstance(readiness.answer, Unresolved):
        accepted = readiness.answer
    elif isinstance(output_answer, Inapplicable):
        accepted = output_answer
    else:
        refusal = _rejected(required.values())
        accepted = refusal if refusal is not None else output_answer
    return ViewAssessment(output_answer, readiness, constraint_results, accepted)


__all__ = [
    "Answer",
    "ConstraintAssessment",
    "Decided",
    "DecisionState",
    "Finding",
    "FindingKind",
    "Inapplicable",
    "MISSING",
    "MissingInput",
    "NOT_APPLICABLE",
    "NonValue",
    "NotApplicable",
    "ReadinessAssessment",
    "Rejected",
    "Unresolved",
    "ViewAssessment",
    "assess_constraints",
    "assess_readiness",
    "assess_view",
    "constraint_answer",
    "finding_sort_key",
    "ordered_findings",
    "owned_answer",
    "reject",
]
