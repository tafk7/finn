# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Executable typing contract; mypy checks this without a plugin or Any escapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import assert_type

from finn.core.space import (
    QueryResult,
    BoundDecision,
    BoundValue,
    BoundView,
    BoundViewField,
    ChoiceView,
    Const,
    Available,
    Decision,
    DecisionRef,
    Derived,
    Change,
    CommitmentReport,
    Param,
    Space,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    ValueSemantics,
    View,
    ViewAssessment,
    ViewKey,
    constraint,
    derived,
    refinement,
    view,
)


@dataclass(frozen=True)
class DType:
    bits: int


DTYPE = ValueSemantics.immutable_nominal(DType)
INT = ValueSemantics.immutable_nominal(int)
WIDTH = ValueKey("width", int)
PHYSICAL = ViewKey("physical", int)


class Fifo(Space):
    word_bits = Param(int)
    depth = Param(int)
    ram_style = Decision(str, values=("auto", "block"))
    banks = Decision(int, values=(1, 2))
    minimum_depth = Const(2)

    @constraint
    def supported(*, depth: int, minimum_depth: int) -> bool:
        return depth >= minimum_depth

    @derived
    def capacity(*, word_bits: int, depth: int) -> int:
        return word_bits * depth

    @view(constraints=(supported,))
    def physical(*, capacity: int, ram_style: str) -> int:
        return capacity + len(ram_style)

    detached = View(capacity, constraints=(supported,))
    exports = {WIDTH: word_bits, PHYSICAL: physical}


class Eltwise(Space):
    activation = Param(DTYPE)
    weight = Param(DTYPE)

    @derived(a=activation, b=weight, semantics=DTYPE)
    def result_dtype(*, a: DType, b: DType) -> QueryResult[DType]:
        return Available(DType(max(a.bits, b.bits) + 1))

    @view(semantics=INT)
    def physical(*, result_dtype: DType) -> QueryResult[int]:
        return Available(result_dtype.bits)

    @derived
    def width(*, result_dtype: DType) -> int:
        return result_dtype.bits

    exports = {WIDTH: width, PHYSICAL: physical}


class FifoInterface(Subspace[Fifo]):
    @property
    def width(self) -> ValueRef[int]:
        return self.ref(Fifo.word_bits)

    @property
    def style(self) -> DecisionRef[str]:
        return self.decision_ref(Fifo.ram_style)


class Assembly(Space):
    width = Param(int)
    first = FifoInterface(Fifo, word_bits=width, depth=Param(int))
    second = Subspace(Fifo, word_bits=8, depth=Decision(int, values=(4, 8)))
    implementation = SubspaceChoice(
        {
            "fifo": Subspace(Fifo, word_bits=width, depth=8),
            "eltwise": Subspace(Eltwise, activation=DType(8), weight=DType(8)),
        },
        exports=(WIDTH, PHYSICAL),
    )

    @derived(width=first.width)
    def twice_width(*, width: int) -> int:
        return width * 2


class GuardedAssembly(Space):
    enabled = Param(bool)
    slots = Decision(int, values=(1, 2), when=enabled)
    fifo = Subspace(Fifo, word_bits=8, depth=4, when=enabled)
    choice = SubspaceChoice(
        {"fifo": Subspace(Fifo, word_bits=8, depth=4)},
        exports=(PHYSICAL,),
        when=enabled,
    )

    @derived(when=enabled)
    def value(*, slots: int) -> int:
        return slots

    @derived(semantics=INT, when=enabled)
    def answer_value(*, slots: int) -> QueryResult[int]:
        return Available(slots)

    @constraint(when=enabled)
    def supported(*, slots: int) -> bool:
        return slots > 0

    physical = View(value, constraints=(supported,), when=enabled)

    @view(when=enabled)
    def decorated(*, slots: int) -> int:
        return slots

    @view(semantics=INT, when=enabled)
    def answer_view(*, slots: int) -> QueryResult[int]:
        return Available(slots)


def check(point: Fifo, assembly: Assembly, eltwise: Eltwise) -> None:
    assert_type(Fifo.word_bits, Param[int])
    assert_type(Fifo.ram_style, Decision[str])
    assert_type(Fifo.minimum_depth, Const[int])
    assert_type(Fifo.capacity, Derived[int])
    assert_type(Fifo.physical, View[int])
    assert_type(Fifo.detached, View[int])
    assert_type(point.word_bits, int)
    assert_type(point.ram_style, str)
    assert_type(point.minimum_depth, int)
    assert_type(point.capacity, int)
    assert_type(point.physical, BoundView[int])
    assert_type(point.physical(), ViewAssessment[int])
    assert_type(point.assess(Fifo.physical), ViewAssessment[int])
    assert_type(point.with_choices(ram_style="auto"), Fifo)
    assert_type(point.field(Fifo.capacity), BoundValue[int])
    assert_type(point.field(Fifo.ram_style), BoundDecision[str])
    assert_type(point.field(Fifo.physical), BoundViewField[int])
    assert_type(point.field(Fifo.ram_style).change("block"), Change[str])
    assert_type(refinement.change(point, Fifo.ram_style, "block"), Change[str])
    assert_type(
        refinement.commit(point, refinement.change(point, Fifo.ram_style, "block")),
        CommitmentReport[Fifo],
    )
    assert_type(
        refinement.commit(
            point,
            point.field(Fifo.ram_style).change("block"),
            point.field(Fifo.banks).change(2),
        ),
        CommitmentReport[Fifo],
    )
    assert_type(point.query(Fifo.capacity), QueryResult[int])
    assert_type(Eltwise.result_dtype, Derived[DType])
    assert_type(eltwise.result_dtype, DType)
    assert_type(Eltwise.physical, View[int])
    assert_type(eltwise.physical(), ViewAssessment[int])
    assert_type(Assembly.first, FifoInterface)
    assert_type(Assembly.first.width, ValueRef[int])
    assert_type(Assembly.first.style, DecisionRef[str])
    assert_type(Assembly.second.ref(Fifo.depth), ValueRef[int])
    assert_type(Assembly.second.decision_ref(Fifo.depth), DecisionRef[int])
    assert_type(assembly.first, Fifo)
    assert_type(assembly.first.with_choices(ram_style="block"), Fifo)
    assert_type(Assembly.first.accepted(Fifo.physical), ValueRef[int])
    assert_type(Assembly.implementation.ref(WIDTH), ValueRef[int])
    assert_type(Assembly.implementation.accepted(PHYSICAL), ValueRef[int])
    assert_type(assembly.implementation, ChoiceView)
    assert_type(Assembly(width=8), Assembly)
    assert_type(GuardedAssembly.value, Derived[int])
    assert_type(GuardedAssembly.answer_value, Derived[int])
    assert_type(GuardedAssembly.physical, View[int])
    assert_type(GuardedAssembly.decorated, View[int])
    assert_type(GuardedAssembly.answer_view, View[int])
