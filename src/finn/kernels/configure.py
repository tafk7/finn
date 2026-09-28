# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit a configuration's choices named by their stable keys, and settle the rest.

Facts are the root node's typed formals: ``design_space(MatMulKernel(rows=..., ...))``. Keys
are the ones ``inspection`` reports: ``"pe"``, ``"compute.compute_pumping"``, a
structural Decision such as ``"delivery"``, or a candidate-local choice such
as ``"delivery.cyclic.rom_style"``. All choices are committed in one atomic
batch. Refusals are raised as ``ValueError`` with their findings.

``settle`` commits every Decision over kernels that compatibility decides: the
engine's ``settle`` with the kernels' convention for a candidate's refusal,
its ``admission`` member (a constraint group, a constraint or a view).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar, cast

from finn.core.space import (
    Available,
    ConfigurationError,
    Constraint,
    ConstraintGroup,
    QueryResult,
    RequestError,
    Settlement,
    Space,
    View,
    inspection,
)
from finn.core.space import settle as settle_space

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


def compatible(
    point: S, choice: str, requirement: Callable[[S], QueryResult[Any]]
) -> tuple[object, ...]:
    """The candidates of the Decision keyed ``choice`` under which ``requirement`` is accepted.

    Each candidate is committed on its own over ``point``, whose other choices
    are kept; ``requirement`` reads the resulting configuration. This filters
    by compatibility only: choosing among several compatible candidates is the
    caller's.
    """
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    if choice not in owned:
        raise ValueError(f"{type(point).__name__}: unknown choice {choice!r}")
    candidates = point.field(owned[choice]).candidates()
    if not isinstance(candidates, Available):
        found = describe([] if candidates is None else [candidates])
        raise ValueError(f"{choice}: candidates unavailable: {found}")
    accepted: list[object] = []
    for case in candidates.value:
        report = point.try_with_choices({owned[choice]: case})
        if report.accepted and isinstance(requirement(report.instance), Available):
            accepted.append(case)
    return tuple(accepted)


def admission(candidate: Space) -> QueryResult[object] | None:
    """A kernel's own refusal of its configuration: its ``admission`` member, if any."""
    member = getattr(type(candidate), "admission", None)
    if isinstance(member, (Constraint, ConstraintGroup)):
        return cast("QueryResult[object]", candidate.inspect(member).result)
    if isinstance(member, View):
        return cast("QueryResult[object]", candidate.query(member))
    return None


def settle(point: S) -> Settlement[S]:
    """Commit every Decision over kernels with exactly one compatible candidate.

    A candidate is compatible when committing it is accepted and its
    ``admission`` admits the configuration. Several compatible candidates are
    a design choice, left open in the settlement.
    """
    return settle_space(point, admission=admission)


__all__ = ["admission", "commit", "compatible", "describe", "settle"]
