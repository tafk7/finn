# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Commit a configuration's choices named by their stable keys; name the open ones.

Facts are the formals of the declaration given to ``design_space``
(``design_space(MatMulKernel(x_tensor=..., ...))``). Keys are the ones ``inspection``
reports: a structural Decision such as ``"compute"``, or a candidate-local choice such as
``"compute.packed.pe"`` or ``"w.source.memstream.ram_style"``. All choices are committed
in one atomic batch. Refusals are raised as ``ValueError`` with their findings.

Nothing is set but what is committed: the engine fills nothing in.
``inspection.validity`` answers whether a configuration is valid, and ``refusal`` why it is
refused. ``undecided`` names the open choices in play; ``chosen`` the stored ones, those
out of play (dormant) included: what saving writes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    false pieces, each once; none where it is valid or open."""
    return validity.result.findings if isinstance(validity.result, Rejected) else ()


def commit(point: S, choices: Mapping[str, object]) -> S:
    """Commit choices named by their stable decision keys, atomically; a refusal names
    each value outside its domain, or each rule the choices would break with the stored
    choices standing in its way."""
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    unknown = sorted(choices.keys() - owned.keys())
    if unknown:
        raise ValueError(f"{type(point).__name__}: unknown choices {unknown}")
    try:
        report = point.try_with_choices({owned[key]: value for key, value in choices.items()})
    except (RequestError, ConfigurationError) as error:
        raise ValueError(str(error)) from error
    if not report.accepted:
        rules = "; ".join(
            f"{rule.rule}: "
            + "; ".join(map(str, rule.result.findings))
            + (f" (standing: {', '.join(rule.standing)})" if rule.standing else "")
            for rule in report.broken
        )
        raise ValueError(
            f"{type(point).__name__} choices are not accepted: "
            + (describe(outcome.result for outcome in report.outcomes) or rules)
        )
    return report.instance


def undecided(point: Space, pattern: str) -> list[str]:
    """Keys matching ``pattern`` (``fnmatch``) of the open choices in play: unset in a
    scope that is present (``inspection.Open``, waiting on nothing), in the flow order."""
    return [
        key
        for key, state in inspection.states(point)
        if isinstance(state, inspection.Open) and not state.waits and fnmatchcase(key, pattern)
    ]


def chosen(point: Space) -> dict[str, object]:
    """Every choice ``point`` stores, by key, with its stored value, whatever its state
    (set, waiting, or dormant: out of play, kept): what saving writes."""
    found: dict[str, object] = {}
    for item in inspection.decisions(point):
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status == "committed":
            found[item.key] = state.value.value
    return found


__all__ = [
    "chosen",
    "commit",
    "describe",
    "refusal",
    "undecided",
]
