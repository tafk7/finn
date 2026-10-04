# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit a configuration's choices named by their stable keys, and settle the rest.

Facts are the root node's typed formals: ``design_space(MatMulKernel(m=..., ...))``. Keys
are the ones ``inspection`` reports: a structural Decision such as ``"memory"``,
or a candidate-local choice such as ``"compute.packed.pe"`` or
``"memory.memstream.ram_style"``. All choices are committed in one atomic
batch. Refusals are raised as ``ValueError`` with their findings.

``settle`` commits every Decision over kernels that compatibility decides: the
engine's ``settle`` with the kernels' convention for a candidate's refusal,
its ``admission`` member (a constraint group, a constraint or a view).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from typing import Any, TypeVar

from finn.core.space import (
    Available,
    ConfigurationError,
    QueryResult,
    RequestError,
    Settlement,
    Space,
    inspection,
)
from finn.core.space import settle as settle_space
from finn.core.space.forcing import admission

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


def undecided(point: Space, pattern: str) -> list[str]:
    """Keys matching ``pattern`` (``fnmatch``) of applicable Decisions not yet committed."""
    found = []
    for item in inspection.decisions(point):
        if not fnmatchcase(item.key, pattern):
            continue
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status != "committed":
            found.append(item.key)
    return found


def settle(point: S) -> Settlement[S]:
    """Commit every Decision over kernels with exactly one compatible candidate.

    A candidate is compatible when committing it is accepted and its
    ``admission`` admits the configuration. Several compatible candidates are
    a design choice, left open in the settlement.
    """
    return settle_space(point, admission=admission)


__all__ = ["admission", "commit", "describe", "settle", "undecided"]
