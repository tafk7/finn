# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit a configuration's choices named by their stable keys; name the open ones.

Facts are the formals of the declaration given to ``design_space``
(``design_space(MatMulKernel(x_tensor=..., ...))``). Keys are the ones ``inspection``
reports: a structural Decision such as ``"compute"``, or a candidate-local choice such as
``"compute.packed.pe"`` or ``"w.source.memstream.ram_style"``. All choices are committed
in one atomic batch. Refusals are raised as ``ValueError`` with their findings.

A Decision whose one viable case is forced needs no commitment (the engine's
``inspection.forced``); ``inspection.validity`` answers whether a configuration is valid,
``refusal`` why it is refused, and ``refusals`` the same with the causes of each Decision
left without a viable case. ``undecided`` names the open Decisions: neither
committed nor forced; ``chosen`` the committed ones, the choices made on purpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from fnmatch import fnmatchcase
from typing import Any, TypeVar

from finn.core.reporting.findings import Finding
from finn.core.space import (
    Available,
    ConfigurationError,
    QueryResult,
    Rejected,
    RequestError,
    Space,
    inspection,
)

S = TypeVar("S", bound=Space)


def describe(results: Iterable[QueryResult[Any]]) -> str:
    """Owner, code and message of every finding in non-available results."""
    return "; ".join(
        str(finding)
        for result in results
        if not isinstance(result, Available)
        for finding in result.findings
    )


def refusal(validity: inspection.Validity) -> tuple[Finding, ...]:
    """Why a configuration is refused (its ``inspection.validity``): the findings of its
    false pieces, each once; none where it is valid or undecided."""
    return validity.result.findings if isinstance(validity.result, Rejected) else ()


def refusals(point: Space) -> tuple[Finding, ...]:
    """Why ``point``'s configuration is refused (``refusal``), each open Decision over
    nodes with no viable case (``decision-no-viable-case``) given why each case was
    refused as its causes (``candidate_refusals``); none where it is valid or
    undecided."""
    found = refusal(inspection.validity(point))
    if not found:
        return ()
    cases = {item.key: tuple(item.refused) for item in inspection.viable(point) if not item.cases}
    return tuple(
        replace(finding, causes=candidate_refusals(point, finding.owner, cases[finding.owner]))
        if finding.code == "decision-no-viable-case" and finding.owner in cases
        else finding
        for finding in found
    )


def candidate_refusals(point: Space, key: str, cases: tuple[str, ...]) -> tuple[Finding, ...]:
    """Why each of ``cases`` of the selector ``key`` is refused on ``point``: the
    validity of the case's candidate once the case is committed, each finding once. A
    decision over values gives none: its own finding says why."""
    decisions = {item.key: item for item in inspection.decisions(point)}
    if not decisions[key].selector:
        return ()
    selectors = {name for name, item in decisions.items() if item.selector}
    found: list[Finding] = []
    for case in cases:
        report = point.try_with_choices({decisions[key].reference: case})
        if not report.accepted:
            for outcome in report.outcomes:
                if isinstance(outcome.result, Rejected):
                    found.extend(outcome.result.findings)
            continue
        candidate: Any = report.instance
        walked: list[str] = []
        for part in key.split("."):
            # A key names a case after its selector (compute.packed.pe): the selector's
            # member is already that case's candidate.
            if ".".join(walked) not in selectors:
                candidate = getattr(candidate, part)
            walked.append(part)
        found.extend(refusal(inspection.validity(candidate)))
    return tuple(dict.fromkeys(found))


def commit(point: S, choices: Mapping[str, object]) -> S:
    """Commit choices named by their stable decision keys, atomically."""
    owned = {item.key: item.reference for item in inspection.decisions(point)}
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
    """Every Decision ``point`` commits, by key, with its stored value: the choices made on
    purpose (a forced Decision is never committed), whatever a read of one answers. A choice
    kept unchecked (its own check cannot conclude yet: ``inspection.unchecked``) is one, a
    commitment whose evidence is pending: what saving writes."""
    found: dict[str, object] = {}
    for item in inspection.decisions(point):
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status == "committed":
            found[item.key] = state.value.value
    return found


__all__ = [
    "candidate_refusals",
    "chosen",
    "commit",
    "describe",
    "refusal",
    "refusals",
    "undecided",
]
