# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic answers and compact operation reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from .primitives import NoTruthValue, QualifiedPath

T = TypeVar("T")


class FindingKind(str, Enum):
    BLOCKER = "blocker"
    REJECTION = "rejection"
    LIMITATION = "limitation"
    AUTHORING = "authoring"
    REQUEST = "request"


def _freeze_diagnostic(value: object) -> object:
    """Restrict diagnostic details to deterministic, immutable values."""

    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, QualifiedPath):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_diagnostic(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("finding-detail mapping keys must be strings")
        pairs = ((cast(str, key), _freeze_diagnostic(item)) for key, item in value.items())
        return tuple(sorted(pairs, key=lambda pair: pair[0]))
    raise TypeError(
        "finding details must use None, bool, int, float, str, QualifiedPath, "
        "or nested tuples/mappings of those values"
    )


def _diagnostic_key(value: object) -> tuple[object, ...]:
    if isinstance(value, QualifiedPath):
        return ("path", value.value)
    if isinstance(value, tuple):
        return ("tuple", *(_diagnostic_key(item) for item in value))
    if type(value) is float:
        return ("float", value.hex())
    return (type(value).__name__, value)


@dataclass(frozen=True, slots=True)
class Finding:
    """One immutable, deterministically ordered explanation."""

    kind: FindingKind
    code: str
    path: QualifiedPath
    message: str
    values: tuple[tuple[str, object], ...] = ()
    trace: tuple[QualifiedPath, ...] = ()

    def __post_init__(self) -> None:
        frozen = tuple(
            sorted(
                ((key, _freeze_diagnostic(value)) for key, value in self.values),
                key=lambda pair: pair[0],
            )
        )
        if len({key for key, _value in frozen}) != len(frozen):
            raise ValueError("finding detail keys must be unique")
        object.__setattr__(self, "values", frozen)


def finding_sort_key(finding: Finding) -> tuple[object, ...]:
    """Order findings without invoking adapter-defined comparison or representation."""

    return (
        finding.path.value.encode("ascii"),
        finding.kind.value,
        finding.code,
        tuple(path.value.encode("ascii") for path in finding.trace),
        tuple((key, _diagnostic_key(value)) for key, value in finding.values),
    )


def ordered_findings(findings: tuple[Finding, ...] | list[Finding]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=finding_sort_key))


class _AnswerBase(NoTruthValue):
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Decided(_AnswerBase, Generic[T]):
    value: T


@dataclass(frozen=True, slots=True)
class Absent(_AnswerBase):
    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", ordered_findings(self.findings))


@dataclass(frozen=True, slots=True)
class Unresolved(_AnswerBase):
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        if not self.findings:
            raise ValueError("Unresolved requires at least one finding")
        object.__setattr__(self, "findings", ordered_findings(self.findings))


Answer: TypeAlias = Decided[T] | Absent | Unresolved

AssignmentOrigin: TypeAlias = Literal["explicit", "proposal"]
DecisionStatus: TypeAlias = Literal["committed", "unassigned"]


@dataclass(frozen=True, slots=True)
class DecisionState:
    path: QualifiedPath
    status: DecisionStatus
    value: object | None = None
    origin: AssignmentOrigin | None = None
    has_proposal: bool = False

    def __post_init__(self) -> None:
        committed = self.status == "committed"
        if committed != (self.origin is not None):
            raise ValueError("only a committed decision state has an origin")


ItemDisposition: TypeAlias = Literal["committed", "unchanged", "skipped", "rejected", "unresolved"]
ItemSource: TypeAlias = Literal[
    "explicit", "applicability", "domain", "proposal", "conflict", "declaration"
]


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    path: QualifiedPath
    disposition: ItemDisposition
    source: ItemSource
    value: object | None = None
    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        if self.disposition == "unresolved" and not self.findings:
            raise ValueError("an unresolved item outcome requires findings")
        object.__setattr__(self, "findings", ordered_findings(self.findings))


class ProposalAdoptionMode(str, Enum):
    ONCE = "once"
    TO_FIXPOINT = "to_fixpoint"


@dataclass(frozen=True, slots=True)
class ConstraintAssessment:
    answers: Mapping[QualifiedPath, Answer[bool]]
    verdict: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(sorted(self.answers.items()))))

    @property
    def not_applicable(self) -> tuple[QualifiedPath, ...]:
        return tuple(path for path, answer in self.answers.items() if isinstance(answer, Absent))


@dataclass(frozen=True, slots=True)
class ReadinessAssessment:
    profile: str
    answers: Mapping[QualifiedPath, Answer[object]]
    ready: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "answers", MappingProxyType(dict(sorted(self.answers.items()))))


__all__ = [
    "Absent",
    "Answer",
    "AssignmentOrigin",
    "ConstraintAssessment",
    "Decided",
    "DecisionState",
    "DecisionStatus",
    "Finding",
    "FindingKind",
    "ItemDisposition",
    "ItemOutcome",
    "ItemSource",
    "ProposalAdoptionMode",
    "ReadinessAssessment",
    "Unresolved",
    "finding_sort_key",
    "ordered_findings",
]
