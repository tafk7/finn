# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Definition, request, refusal and contextual evaluator errors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .edits import ConfigurationResult
    from .results import Finding, NonValue


class SpaceError(Exception):
    """Base class for failures at Space boundaries."""


class DefinitionError(SpaceError, ValueError):
    """An authored definition cannot be compiled into a valid model.

    It carries the findings that explain it.
    """

    def __init__(self, detail: str, *, findings: Iterable[Finding] = ()) -> None:
        self.findings = tuple(findings)
        super().__init__(detail)


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
