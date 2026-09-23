# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Migration harness using public model binding and typed discovery.

Promoted to helpers.py at cutover. Supplied facts use the exposed parameter's
public relative key; production authoring examples bind declaration handles.
Required omissions are errors. Partial facts belong in explicit optional-Param
or unresolved-Decision composition fixtures, not an implicit Problem wrapper.
"""

from collections.abc import Mapping
from typing import TypeVar

from finn.kernels.space._next import Space, compile_space
from finn.kernels.space._next.declarations import Constraint
from finn.kernels.space._next.errors import RefinementError, RequestError
from finn.kernels.space._next.inspection import decisions, members
from finn.kernels.space._next.results import Answer, Decided

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
