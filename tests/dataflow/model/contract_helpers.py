# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test harness using the actual Space binding and assessment machinery."""

from collections.abc import Mapping
from typing import TypeVar, cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.model.logical.datatype_semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.space import ConstraintGroup, Decision, Input, Problem, Space, Subspace
from finn.dataflow.space.declarations import Constraint, declared_members

T = TypeVar("T")


def point_for(
    kernel: type[Space],
    facts: Mapping[str, object],
    **choices: object,
) -> Space:
    inputs: dict[str, Problem[object]] = {}
    for name, declaration in declared_members(kernel):
        if isinstance(declaration, Input):
            codec = (
                QONNX_DATATYPE_CODEC
                if (
                    declaration.value_semantics.type_token
                    is QONNX_DATATYPE_VALUE_SEMANTICS.type_token
                )
                else None
            )
            inputs[name] = Problem(declaration.value_semantics, required=False, canonical=codec)
    unknown = set(facts) - set(inputs)
    assert not unknown, unknown
    members: dict[str, object] = {
        "__module__": __name__,
        **inputs,
        "kernel": Subspace(kernel, **inputs),
    }
    host = type("ContractHost", (Space,), members)
    root = host.start({inputs[name]: value for name, value in facts.items()})
    point = cast("Space", getattr(root, "kernel"))
    for name, value in choices.items():
        declaration = getattr(kernel, name)
        assert isinstance(declaration, Decision)
        point = point.assign(declaration, value)
    return point


def value(answer: Answer[T]) -> T:
    assert isinstance(answer, Decided), answer
    return answer.value


def assess(point: Space, condition: Constraint) -> Answer[bool]:
    members = declared_members(type(point))
    name = next(name for name, member in members if member is condition)
    group = next(
        member
        for _, member in members
        if isinstance(member, ConstraintGroup) and condition in member.constraints
    )
    return next(
        answer
        for path, answer in point.assess(group).answers.items()
        if str(path).rsplit(".", 1)[-1] == name
    )
