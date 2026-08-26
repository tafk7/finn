# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contextual exceptions at the engine's public boundaries."""

from __future__ import annotations

from collections.abc import Iterable

from .primitives import QualifiedPath
from .results import Finding, ordered_findings


class DesignSpaceError(Exception):
    """Base class for expected engine boundary failures."""


class _FindingsError(DesignSpaceError):
    label = "operation"

    def __init__(self, findings: Iterable[Finding]) -> None:
        self.findings = ordered_findings(list(findings))
        if not self.findings:
            raise ValueError(f"{type(self).__name__} requires at least one finding")
        super().__init__(f"{self.label} failed with {len(self.findings)} finding(s)")


class ValidationError(_FindingsError):
    label = "design-space validation"


class RequestError(_FindingsError):
    label = "engine request"


class EvaluationError(DesignSpaceError):
    """An evaluator or value-semantics callback violated its programming contract."""

    def __init__(self, owner: QualifiedPath, role: str, detail: str) -> None:
        self.owner = owner
        self.role = role
        self.detail = detail
        super().__init__(f"{owner} ({role}): {detail}")


__all__ = ["DesignSpaceError", "EvaluationError", "RequestError", "ValidationError"]
