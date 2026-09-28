# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit a configuration's choices named by their stable keys.

Facts are the root node's typed formals: ``design_space(MatMulKernel(rows=..., ...))``. Keys
are the ones ``inspection`` reports: ``"pe"``, ``"compute.compute_pumping"``, a
structural Decision such as ``"delivery"``, or a candidate-local choice such
as ``"delivery.cyclic.rom_style"``. All choices are committed in one atomic
batch. Refusals are raised as ``ValueError`` with their findings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from finn.core.space import (
    Available,
    ConfigurationError,
    QueryResult,
    RequestError,
    Space,
    inspection,
)

S = TypeVar("S", bound=Space)


def describe(results: Iterable[QueryResult[Any]]) -> str:
    """Owner, code and message of every finding in non-available results."""
    return "; ".join(
        f"{finding.owner}: {finding.code}: {finding.message}"
        for result in results
        if not isinstance(result, Available)
        for finding in result.findings
    )


def commit(point: S, choices: Mapping[str, object]) -> S:
    """Commit choices named by their stable decision keys, atomically."""
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    pinned = {item.key: item for item in inspection.pinned(point)}
    stale = sorted(choices.keys() & pinned.keys())
    if stale:
        raise ValueError(
            f"{type(point).__name__}: stale choices {stale}: "
            + "; ".join(pinned[key].text() for key in stale)
        )
    unknown = sorted(choices.keys() - owned.keys())
    if unknown:
        raise ValueError(f"{type(point).__name__}: unknown choices {unknown}")
    try:
        report = point.try_with_choices({owned[key]: value for key, value in choices.items()})
    except (RequestError, ConfigurationError) as error:
        raise ValueError(str(error)) from error
    if not report.accepted:
        raise ValueError(
            f"{type(point).__name__} choices are not accepted: "
            + describe(outcome.result for outcome in report.outcomes)
        )
    return report.instance


__all__ = ["commit", "describe"]
