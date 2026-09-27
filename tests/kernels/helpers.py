# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test harness using public model binding and typed discovery.

Supplied facts use exposed parameter keys. Required omissions are errors;
partial evaluation uses explicit optional-Param or unresolved-Decision fixtures."""

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from finn import resources

from finn.core.space import Space, compile_space
from finn.core.space import Constraint
from finn.core.space.errors import ConfigurationError, RequestError
from finn.core.space.inspection import decisions, members
from finn.core.space.results import QueryResult, Available

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def point_for(kernel: type[S], facts: Mapping[str, object], **choices: object) -> S:
    model = compile_space(kernel)
    parameters = {item.key: item.reference for item in members(model) if item.kind == "param"}
    unknown = facts.keys() - parameters.keys()
    if unknown:
        raise RequestError(f"unknown supplied facts: {sorted(unknown)}")
    point = model.bind({parameters[name]: value for name, value in facts.items()})
    owned = {item.key: item.reference for item in decisions(model)}
    unknown_choices = choices.keys() - owned.keys()
    if unknown_choices:
        raise RequestError(f"unknown choices: {sorted(unknown_choices)}")
    report = point.try_with_choices(
        *(point.field(owned[name]).change(value) for name, value in choices.items())
    )
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance


def value(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> QueryResult[bool]:
    return point.inspect(condition).result


def finnlib_root() -> Path:
    """FinnLib as FINN resolves it: FINN_RESOURCES_FINNLIB, a cached copy, or a fetch."""
    return Path(resources.path("finnlib"))
