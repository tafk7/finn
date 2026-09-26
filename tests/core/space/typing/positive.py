# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Executable typing contract; mypy checks this without a plugin or Any escapes.

Declarations are typed as the values they stand for (option A): a node call
``Room(area=12)`` is a ``Room``, and ``kitchen.finish`` in a class body is an
``int``. Inside methods ``self`` is a configuration and every read is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import assert_type

from finn.core.space import (
    UNSUPPLIED,
    Available,
    BoundDecision,
    BoundValue,
    BoundView,
    Change,
    ConfigurationResult,
    Const,
    Decision,
    Derived,
    Located,
    LocatedParam,
    Members,
    Param,
    Present,
    QueryResult,
    Space,
    Users,
    ValueSemantics,
    View,
    ViewAssessment,
    ViewKey,
    configure,
    constraint,
    derived,
    selected,
    view,
)


@dataclass(frozen=True)
class DType:
    bits: int


DTYPE = ValueSemantics.immutable_nominal(DType)
INT = ValueSemantics.immutable_nominal(int)
COST = ViewKey("cost", int)


class Room(Space):
    area: Param[int] = Param(int)
    label: Param[str] = Param(str, default=UNSUPPLIED)
    finish = Decision(int, values=(1, 2, 3))

    @view
    def cost(self) -> int:
        return self.area * self.finish

    exports = {COST: cost}


class Boiler(Space):
    kw: Param[int] = Param(int)


class HeatPump(Space):
    kw: Param[int] = Param(int)
    cop = Decision(int, values=(3, 4))


class Thermostat(Space):
    kw: Param[int] = Param(int)


class Match(Space):
    a: LocatedParam[int] = Param(Located)
    b: LocatedParam[int] = Param(Located)

    @constraint
    def same(self) -> bool:
        assert_type(self.a, Located[int])
        return self.a.value == self.b.value


class House(Space):
    budget: Param[int] = Param(int)
    want_garage = Decision(bool, values=(False, True))
    hall = Room()  # a bare call: its area is assigned below
    kitchen = Room(area=12)
    dining = Room(area=budget, label="dining")
    garage = Room(area=kitchen.area, when=want_garage)
    heat_pump = HeatPump(kw=8)
    heating = Decision[Boiler | HeatPump](values={"boiler": Boiler(kw=24), "heat_pump": heat_pump})
    maybe = Decision(values={"none": None, "boiler": Boiler(kw=3)})
    thermostat = Thermostat(kw=heating.kw)
    hall.area = kitchen.area  # typed by Param.__set__: an int reference
    matched = Match(a=kitchen.finish, b=dining.finish)
    either = Room(area=Present(kitchen.area, dining.area))
    costs = Members(COST)
    which = selected(heating)

    # References in a class body are typed as the values they stand for.
    assert_type(kitchen, Room)
    assert_type(kitchen.finish, int)
    assert_type(kitchen.cost, BoundView[int])
    assert_type(heating, Boiler | HeatPump)
    assert_type(heating.kw, int)
    assert_type(maybe, Boiler | None)
    assert_type(heat_pump.cop, int)

    @view(requires=(costs, matched.same, kitchen.cost))
    def total(self) -> int:
        assert_type(self.kitchen, Room)
        assert_type(self.kitchen.finish, int)
        assert_type(self.kitchen.cost(), int)
        assert_type(self.heating, Boiler | HeatPump)
        assert_type(self.heating.kw, int)
        assert_type(self.maybe, Boiler | None)
        assert_type(self.which, str)
        assert_type(self.costs, tuple[Located[int], ...])
        return sum(member.value for member in self.costs)


class Estate(Space):
    """A reference input: the caller supplies the node (placed here if it is fresh)."""

    home: Param[House] = Param(House)

    @derived
    def finish(self) -> int:
        assert_type(self.home, House)
        return self.home.kitchen.finish


class Stream(Space):
    spec: Param[int] = Param(int)
    ends = Users(COST)

    @derived
    def users(self) -> int:
        assert_type(self.ends, tuple[Located[int], ...])
        return len(self.ends)


class Producer(Space):
    output: Param[Stream] = Param(Stream)
    feed: Param[Stream] = Param(Stream, default=UNSUPPLIED)

    @view
    def cost(self) -> int:
        # A reference input reads as the referenced node's configuration.
        assert_type(self.output, Stream)
        assert_type(self.output.spec, int)
        return self.output.spec

    exports = {COST: cost}


class Graph(Space):
    edge = Stream(spec=8)
    producer = Producer(output=edge)  # a placed node: a reference
    later = Producer()
    later.output = edge  # a reference input may be assigned too
    fresh = Producer(output=Stream(spec=4))  # a fresh node: placed at the input
    shared = Decision(int, values=(1, 2), name="shared")
    assert_type(shared, Decision[int])


class Fifo(Space):
    word_bits: Param[int] = Param(int)
    depth: Param[int] = Param(int)
    ram_style = Decision(str, values=("auto", "block"))
    banks = Decision(int, values=(1, 2))
    minimum_depth = Const(2)

    @constraint
    def supported(self) -> bool:
        return self.depth >= self.minimum_depth

    @derived
    def capacity(self) -> int:
        return self.word_bits * self.depth

    @view(requires=(supported,))
    def physical(self) -> int:
        return self.capacity + len(self.ram_style)

    detached = View(capacity, requires=(supported,))


class Eltwise(Space):
    activation: Param[DType] = Param(DTYPE)
    weight: Param[DType] = Param(DTYPE)

    @derived(a=activation, b=weight, semantics=DTYPE)
    def result_dtype(*, a: DType, b: DType) -> QueryResult[DType]:
        return Available(DType(max(a.bits, b.bits) + 1))

    @view(semantics=INT)
    def physical(*, result_dtype: DType) -> QueryResult[int]:
        return Available(result_dtype.bits)


class GuardedAssembly(Space):
    enabled: Param[bool] = Param(bool)
    slots = Decision(int, values=(1, 2), when=enabled)
    fifo = Fifo(word_bits=8, depth=4, when=enabled)

    @derived(when=enabled)
    def value(*, slots: int) -> int:
        return slots

    @constraint(when=enabled)
    def supported(*, slots: int) -> bool:
        return slots > 0

    physical = View(value, requires=(supported,), when=enabled)


def check(point: Fifo, house: House, eltwise: Eltwise) -> None:
    # Class access is the schema key: a declaration.
    assert_type(Fifo.word_bits, Param[int])
    assert_type(Fifo.ram_style, Decision[str])
    assert_type(Fifo.minimum_depth, Const[int])
    assert_type(Fifo.capacity, Derived[int])
    assert_type(Fifo.physical, View[int])
    assert_type(House.kitchen, Room)
    assert_type(House.kitchen.finish, int)
    # The compile step is typed as the family.
    assert_type(configure(House(budget=100)), House)
    assert_type(configure(Estate(home=House(budget=1))), Estate)
    assert_type(configure(Room()), Room)  # a bare call type-checks
    # Configuration reads are exact.
    assert_type(point.word_bits, int)
    assert_type(point.ram_style, str)
    assert_type(point.minimum_depth, int)
    assert_type(point.capacity, int)
    assert_type(point.physical, BoundView[int])
    assert_type(point.physical(), int)
    assert_type(point.physical.inspect(), ViewAssessment[int])
    assert_type(point.physical.query(), QueryResult[int])
    assert_type(point.view(Fifo.physical), BoundView[int])
    assert_type(point.field(Fifo.capacity), BoundValue[int])
    assert_type(point.field(Fifo.ram_style), BoundDecision[str])
    assert_type(point.field(Fifo.physical), BoundView[int])
    assert_type(point.field(Fifo.ram_style).change("block"), Change[str])
    assert_type(point.inspect(Fifo.physical), ViewAssessment[int])
    assert_type(point.query(Fifo.capacity), QueryResult[int])
    assert_type(house.query(House.kitchen.finish), QueryResult[int])
    assert_type(point.with_choices(ram_style="auto"), Fifo)
    assert_type(house.with_choices({House.kitchen.finish: 2, House.heating: "boiler"}), House)
    assert_type(
        point.try_with_choices(point.field(Fifo.ram_style).change("block")),
        ConfigurationResult[Fifo],
    )
    assert_type(eltwise.result_dtype, DType)
    assert_type(eltwise.physical(), int)
    assert_type(house.kitchen.finish, int)
    assert_type(house.heating, Boiler | HeatPump)
    assert_type(house.maybe, Boiler | None)
    assert_type(GuardedAssembly.value, Derived[int])
    assert_type(GuardedAssembly.physical, View[int])
