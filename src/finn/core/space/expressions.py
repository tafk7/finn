# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A bounded integer expression vocabulary, without symbolic Python tracing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .declarations import ValueDecl, ValueRef
from .errors import DefinitionError
from .semantics import default_semantics

IntOperator = Literal["add", "sub", "mul", "floordiv", "mod", "neg"]
INTEGER_SEMANTICS = default_semantics(int)


class Expr(ValueDecl[int]):
    """An integer computation usable as a named descriptor or unbound reference.

    Reusing one expression shares its computation within each scope. Named
    expressions own their diagnostics. Anonymous shared expressions retain the
    first authored consumer as their source owner, while query evidence records
    each actual consumer's dependency path.
    """

    def __init__(self, operator: IntOperator, *operands: int | ValueRef[int]) -> None:
        if operator not in {"add", "sub", "mul", "floordiv", "mod", "neg"}:
            raise DefinitionError("unsupported integer expression operator")
        if len(operands) != (1 if operator == "neg" else 2):
            raise DefinitionError("wrong number of integer expression operands")
        for operand in operands:
            if type(operand) is int:
                continue
            if not isinstance(operand, ValueRef):
                raise DefinitionError(
                    "integer expressions require exact int literals or references"
                )
            semantics = operand.semantics
            if semantics is not None and semantics.type_token is not int:
                raise DefinitionError("integer expression operands require int value semantics")
        self.operator = operator
        self.operands = tuple(operands)
        self.semantics = INTEGER_SEMANTICS


def _apply_integer(operator: IntOperator, operands: tuple[int, ...]) -> int:
    """Evaluate the bounded integer vocabulary using ordinary Python arithmetic."""

    if any(type(value) is not int for value in operands):
        raise TypeError("integer expressions require exact int values")
    left = operands[0]
    if operator == "neg":
        return -left
    right = operands[1]
    if operator == "add":
        return left + right
    if operator == "sub":
        return left - right
    if operator == "mul":
        return left * right
    if operator == "floordiv":
        return left // right
    return left % right


def evaluator(operator: IntOperator) -> Callable[..., object]:
    """Lower the bounded operator to an ordinary dependency-bound callback."""

    if operator == "neg":

        def unary(*, operand: int) -> int:
            return _apply_integer(operator, (operand,))

        return unary

    def binary(*, left: int, right: int) -> int:
        return _apply_integer(operator, (left, right))

    return binary


__all__ = ["Expr"]
