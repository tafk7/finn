# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A finding: one reason, who gave it, what it is about, and the reasons that caused it.

``owner`` names who refused or found it, where a reader looks to change the outcome: a
declaration's path in its space (``compute.packed.core_supported``), a setting's place in
the build's configuration (``configuration.target.part``), or a check's, an analysis's,
a transformation's or a strategy's name. What the finding is about goes in ``details``,
under ``node`` or ``tensor`` for a node or tensor of the graph.

A finding has three text forms: ``str(finding)``, one line (``owner: code: message``);
``finding.record()``, JSON's values; and ``by_code``, one line per code for many.

A ``Refusal`` is an error that carries the findings it refuses for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class FindingKind(str, Enum):
    """What a reader can do about a finding."""

    #: No answer until something is supplied: supply it (commit a choice, state a fact).
    BLOCKER = "blocker"
    #: Answered, with less than asked or less known: accept it, or change the request.
    LIMITATION = "limitation"
    #: The request is refused: change it (a setting, a graph, a choice).
    REJECTION = "rejection"
    #: The code is wrong, not the input: fix the code (a declaration, a step).
    AUTHORING = "authoring"


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
    """One reason, its owner, what it is about (``details``, frozen primitives sorted by
    key), and the reasons that caused it (``causes``, in a fixed order).

    An empty owner is allowed in a declaration's own result; the engine gives it the
    declaration's path before publishing the result. A cause keeps its owner.
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

    def __str__(self) -> str:
        """One line: ``owner: code: message``."""
        return f"{self.owner}: {self.code}: {self.message}"

    def about(self, **subjects: str) -> Finding:
        """This finding, its owner kept, naming in its details what it is about (``node``,
        ``tensor``) where they do not name it already: what a caller adds when it carries
        the finding out of where it was found."""
        stated = dict(self.details)
        added = tuple((key, value) for key, value in subjects.items() if key not in stated)
        return replace(self, details=(*self.details, *added)) if added else self

    def record(self) -> dict[str, Any]:
        """JSON's values: its kind, code, owner, message, details and causes, each cause
        a record of its own."""
        return {
            "kind": self.kind.value,
            "code": self.code,
            "owner": self.owner,
            "message": self.message,
            "details": dict(self.details),
            "causes": [cause.record() for cause in self.causes],
        }


def _finding_sort_key(finding: Finding) -> tuple[object, ...]:
    return (
        finding.owner,
        finding.kind.value,
        finding.code,
        finding.message,
        tuple((key, _detail_key(value)) for key, value in finding.details),
        tuple(_finding_sort_key(cause) for cause in finding.causes),
    )


def ordered_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """``findings`` in their fixed order: by owner, kind, code, message, details and
    causes."""
    return tuple(sorted(findings, key=_finding_sort_key))


def _first_message(found: Sequence[Finding]) -> str:
    more = f"; and {len(found) - 1} more" if len(found) > 1 else ""
    return found[0].message + more


def by_code(
    findings: Iterable[Finding], named: Callable[[Sequence[Finding]], str] = _first_message
) -> list[str]:
    """One line per finding code and kind, in the order each is first found: ``code
    (kind) count: text``, where the text is ``named`` of the code's findings, by default
    the first one's message and how many more there are. A caller adds its own prefix."""
    found: dict[tuple[str, FindingKind], list[Finding]] = {}
    for finding in findings:
        found.setdefault((finding.code, finding.kind), []).append(finding)
    return [
        f"{code} ({kind.value}) {len(each)}: {named(each)}" for (code, kind), each in found.items()
    ]


class Refusal(ValueError):
    """An error that refuses a request, carrying the findings that say why
    (``findings``, at least one). Its message is their lines, one per finding, unless the
    raiser gives its own (``message``). Every refusal is one, so a run tells a refusal
    from a bug by its type."""

    def __init__(self, findings: Iterable[Finding], message: str | None = None) -> None:
        self.findings = tuple(findings)
        if not self.findings:
            raise TypeError("a refusal carries at least one finding")
        super().__init__("\n".join(map(str, self.findings)) if message is None else message)


__all__ = [
    "Finding",
    "FindingKind",
    "Refusal",
    "by_code",
    "ordered_findings",
]
