# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Definition, request, refusal and contextual evaluator errors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import Finding


class SpaceError(Exception):
    """Base class for failures at Space boundaries."""


class DefinitionError(SpaceError, ValueError):
    """An authored definition cannot be compiled into a valid model."""

    def __init__(self, detail: str, *, findings: Iterable[Finding] = ()) -> None:
        self.findings = tuple(findings)
        super().__init__(detail)


AuthoringError = DefinitionError


class RequestError(SpaceError, ValueError):
    """A binding or refinement request is malformed, before evaluation."""

    def __init__(self, detail: str, *, findings: Iterable[Finding] = ()) -> None:
        self.findings = tuple(findings)
        super().__init__(detail)


class EvaluationError(SpaceError):
    """A callback failed, with its declaration owner and evaluator role.

    Raise this using ``raise ... from cause`` to preserve the programmer error.
    """

    def __init__(self, owner: str, role: str, detail: str) -> None:
        self.owner = owner
        self.role = role
        self.detail = detail
        super().__init__(f"{owner} ({role}): {detail}")


class RefinementError(SpaceError):
    """Strict assignment refused a well-formed candidate; inspect its report."""

    def __init__(self, report: object) -> None:
        self.report = report
        super().__init__("assignment refused; inspect the refinement report")


__all__ = [
    "AuthoringError",
    "DefinitionError",
    "EvaluationError",
    "RefinementError",
    "RequestError",
    "SpaceError",
]
