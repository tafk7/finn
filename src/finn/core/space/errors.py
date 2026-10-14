# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Definition, request, refusal and contextual evaluator errors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Self

from finn.core.reporting.findings import Finding, FindingKind, Refusal

if TYPE_CHECKING:
    from .edits import ConfigurationResult
    from .results import NonValue


class SpaceError(Exception):
    """Base class for failures at Space boundaries."""


class DefinitionError(SpaceError, Refusal):
    """An authored definition cannot be compiled into a valid model: the declaration's
    code is wrong, not its input.

    It carries the findings that explain it, at least one, so every refusal names its
    rule and owner; its message is the detail. ``broken`` makes the error of one rule:
    an authoring finding owned by whoever checks it (the engine's module, such as
    ``space.declarations`` or ``space.names``, or a layer above it, such as ``Kernel``), the
    declaration it is about in its details where the rule knows it. A raise that rewraps
    another's refusal passes its ``findings``.
    """

    def __init__(self, detail: str, *, findings: Iterable[Finding]) -> None:
        super().__init__(findings, detail)

    @classmethod
    def broken(cls, code: str, owner: str, detail: str, *, declaration: str | None = None) -> Self:
        """The error of one broken rule, ``code`` by ``owner``, its message the
        detail, the ``declaration`` in its details."""
        details = () if declaration is None else (("declaration", declaration),)
        return cls(detail, findings=(Finding(FindingKind.AUTHORING, code, owner, detail, details),))


class RequestError(SpaceError, ValueError):
    """A binding or configuration request is malformed, before evaluation."""


class EvaluationError(SpaceError):
    """A callback failed, with its declaration owner and evaluator role.

    Raise this using ``raise ... from cause`` to preserve the programmer error.
    """

    def __init__(self, owner: str, role: str, detail: str) -> None:
        self.owner = owner
        self.role = role
        self.detail = detail
        super().__init__(f"{owner} ({role}): {detail}")


class ValueUnavailableError(SpaceError):
    """A value read reached a valid non-value query result."""

    def __init__(self, result: NonValue, *, context: object | None = None) -> None:
        self.result = result
        self.context = context
        super().__init__(f"value is unavailable: {type(result).__name__}")


class ReferenceUseError(SpaceError, TypeError):
    """A declaration reference was used as a value while a Space was declared.

    References are typed as the values they stand for; this error makes each
    value-like use fail at runtime, naming the reference and its source line.
    """


class ConfigurationError(SpaceError):
    """A well-formed configuration replacement could not be published."""

    def __init__(self, report: ConfigurationResult[Any]) -> None:
        self.report = report
        super().__init__("configuration change refused; inspect the report")


__all__ = [
    "DefinitionError",
    "EvaluationError",
    "ConfigurationError",
    "ReferenceUseError",
    "RequestError",
    "ValueUnavailableError",
]
