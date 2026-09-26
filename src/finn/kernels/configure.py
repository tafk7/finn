# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configure a Space from facts and choices named by their stable keys.

Keys are the ones ``inspection`` reports: ``"pe"``, ``"compute.compute_pumping"``,
a structural choice's selector such as ``"implementation"``, or a case-local
choice such as ``"implementation.cyclic.rom_style"``. All choices are committed
in one atomic batch. Refusals are raised as ``ValueError`` with their findings.
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
    compile_space,
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


def configure(space_type: type[S], facts: Mapping[str, object], choices: Mapping[str, object]) -> S:
    model = compile_space(space_type)
    inputs = {
        item.key: item.reference for item in inspection.members(model) if item.kind == "param"
    }
    owned = {item.key: item.reference for item in inspection.decisions(model)}
    unknown = sorted(facts.keys() - inputs.keys()) + sorted(choices.keys() - owned.keys())
    if unknown:
        raise ValueError(f"{space_type.__name__}: unknown facts or choices {unknown}")
    try:
        point = model.bind({inputs[key]: value for key, value in facts.items()})
        report = point.try_with_choices(
            *(point.field(owned[key]).change(value) for key, value in choices.items())
        )
    except (RequestError, ConfigurationError) as error:
        raise ValueError(str(error)) from error
    if not report.accepted:
        raise ValueError(
            f"{space_type.__name__} choices are not accepted: "
            + describe(outcome.result for outcome in report.outcomes)
        )
    return report.instance


__all__ = ["configure", "describe"]
