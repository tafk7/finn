# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test harness using public model binding and typed discovery.

Supplied facts use exposed parameter keys. Required omissions are errors;
partial evaluation uses explicit optional-Param or unresolved-Decision fixtures."""

from collections.abc import Mapping
from typing import TypeVar

from finn.kernels.space import Space, compile_space
from finn.kernels.space.declarations import Constraint
from finn.kernels.space.errors import RefinementError, RequestError
from finn.kernels.space.inspection import decisions, members
from finn.kernels.space.results import Answer, Decided

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def point_for(kernel: type[S], facts: Mapping[str, object], **choices: object) -> S:
    model = compile_space(kernel)
    parameters = {item.key: item.reference for item in members(model) if item.kind == "param"}
    unknown = facts.keys() - parameters.keys()
    if unknown:
        raise RequestError(f"unknown supplied facts: {sorted(unknown)}")
    point = model.start({parameters[name]: value for name, value in facts.items()})
    owned = {item.key: item.reference for item in decisions(model)}
    unknown_choices = choices.keys() - owned.keys()
    if unknown_choices:
        raise RequestError(f"unknown choices: {sorted(unknown_choices)}")
    report = point.refine(*(point.edit(owned[name], value) for name, value in choices.items()))
    if not report.accepted:
        raise RefinementError(report)
    return report.point


def value(answer: Answer[T]) -> T:
    assert isinstance(answer, Decided), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> Answer[bool]:
    return point.assess(condition).answer
