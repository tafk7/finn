# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit a configuration's choices named by their stable keys; name the open ones.

Facts are the root node's typed formals: ``design_space(MatMulKernel(m=..., ...))``. Keys
are the ones ``inspection`` reports: a structural Decision such as ``"compute"``,
or a candidate-local choice such as ``"compute.packed.pe"`` or
``"w.source.memstream.ram_style"``. All choices are committed in one atomic
batch. Refusals are raised as ``ValueError`` with their findings.

A Decision whose one viable case is forced needs no commitment (the engine's
``inspection.forced``); ``inspection.admission`` reads a kernel's own refusal,
its ``admission`` member. ``undecided`` names the open Decisions: neither
committed nor forced; ``chosen`` the committed ones, the choices made on purpose.
In a root whose members are named by path (``MatMul_0``, ``x.adapter``), ``member_of``
names the member a key belongs to.
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping
from fnmatch import fnmatchcase
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


def undecided(point: Space, pattern: str) -> list[str]:
    """Keys matching ``pattern`` (``fnmatch``) of applicable Decisions that are open:
    neither committed nor forced."""
    forced = {item.key for item in inspection.forced(point)}
    found = []
    for item in inspection.decisions(point):
        if not fnmatchcase(item.key, pattern) or item.key in forced:
            continue
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status != "committed":
            found.append(item.key)
    return found


def chosen(point: Space) -> dict[str, object]:
    """Every Decision ``point`` commits, by key: the choices made on purpose (a forced
    Decision is never committed)."""
    found: dict[str, object] = {}
    for item in inspection.decisions(point):
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status == "committed":
            found[item.key] = state.value.value
    return found


def member_of(paths: Container[str], key: str) -> str | None:
    """The longest of ``paths``, member paths of a root (``MatMul_0``), that is ``key`` or
    one of its prefixes at a segment (``MatMul_0.compute.pe``), or None: the member a key
    belongs to."""
    parts = key.split(".")
    for end in range(len(parts), 0, -1):
        path = ".".join(parts[:end])
        if path in paths:
            return path
    return None


__all__ = ["chosen", "commit", "describe", "member_of", "undecided"]
