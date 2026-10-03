# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Executable typing contract; mypy checks this without a plugin or Any escapes.

Declarations are typed as the values they stand for (option A): a node call
``Room(area=12)`` is a ``Room``, and ``kitchen.finish`` in a class body is an
``int``. Formals and Decisions are annotated with their value type
(``area: int = Param()``), so a class-level member is typed as its value too,
and so is a read through a reference input in the class body
(``output.spec.payload_bits``). A view is no exception: ``kitchen.cost`` in a
class body and ``point.total`` on a configuration are typed as the view's
value; class access (``House.total``) is the view declaration, which
``point.inspect`` and ``point.query`` take. Inside methods ``self`` is a
configuration and every read is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type

from finn.core.space import (
    Available,
    BoundDecision,
    BoundValue,
    Change,
    ConfigurationResult,
    Const,
    ConstraintAssessment,
    Decision,
    DecisionHandle,
    Located,
    LocatedParam,
    Members,
    Param,
    Present,
    QueryResult,
    Space,
    Users,
    ValueHandle,
    ValueSemantics,
    View,
    ViewAssessment,
    ViewKey,
    constraint,
    derived,
    design_space,
    inspection,
    selected,
    view,
)
from finn.core.space.inspection import Provenance


@dataclass(frozen=True)
class DType:
    bits: int


DTYPE = ValueSemantics.immutable_nominal(DType)
INT = ValueSemantics.immutable_nominal(int)
COST = ViewKey("cost", int)


class Room(Space):
    area: int = Param()
    label: str = Param(required=False)
    finish: int = Decision(values=(1, 2, 3))

    @view
    def cost(self) -> int:
        return self.area * self.finish

    exports = {COST: cost}


class Boiler(Space):
    kw: int = Param()


class HeatPump(Space):
    kw: int = Param()
    cop: int = Decision(values=(3, 4))


class Thermostat(Space):
    kw: int = Param()


class Match(Space):
    a: LocatedParam[int] = LocatedParam()
    b: LocatedParam[int] = LocatedParam()

    @constraint
    def same(self) -> bool:
        assert_type(self.a, Located[int])
        return self.a.value == self.b.value


class House(Space):
    budget: int = Param()
    want_garage: bool = Decision(values=(False, True))
    hall = Room()  # a bare call: its area is assigned below
    kitchen = Room(area=12)
    dining = Room(area=budget, label="dining")
    garage = Room(area=kitchen.area, when=want_garage)
    heat_pump = HeatPump(kw=8)
    heating: Boiler | HeatPump = Decision({"boiler": Boiler(kw=24), "heat_pump": heat_pump})
    maybe: Boiler | None = Decision({"boiler": Boiler(kw=3)}, optional=True)
    thermostat = Thermostat(kw=heating.kw)
    hall.area = kitchen.area  # typed by the annotation: an int reference
    matched = Match(a=kitchen.finish, b=dining.finish)
    either = Room(area=Present(kitchen.area, dining.area))
    costs = Members(COST)
    which = selected(heating)

    # References in a class body are typed as the values they stand for.
    assert_type(kitchen, Room)
    assert_type(kitchen.finish, int)
    assert_type(kitchen.cost, int)  # a view reference: typed as its accepted value
    assert_type(heating, Boiler | HeatPump)
    assert_type(heating.kw, int)
    assert_type(maybe, Boiler | None)
    assert_type(heat_pump.cop, int)

    @view(requires=(costs, matched.same, kitchen.cost))
    def total(self) -> int:
        assert_type(self.kitchen, Room)
        assert_type(self.kitchen.finish, int)
        assert_type(self.kitchen.cost, int)
        assert_type(self.heating, Boiler | HeatPump)
        assert_type(self.heating.kw, int)
        assert_type(self.maybe, Boiler | None)
        assert_type(self.which, str)
        assert_type(self.costs, tuple[Located[int], ...])
        return sum(member.value for member in self.costs)


class Estate(Space):
    """A reference input: the caller supplies the node (placed here if it is fresh)."""

    home: House = Param()

    @derived
    def finish(self) -> int:
        assert_type(self.home, House)
        return self.home.kitchen.finish


class Stream(Space):
    spec: int = Param()
    ends = Users(COST)

    @derived
    def users(self) -> int:
        assert_type(self.ends, tuple[Located[int], ...])
        return len(self.ends)


class Producer(Space):
    output: Stream = Param()
    feed: Stream = Param(required=False)

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
    shared: int = Decision(values=(1, 2), name="shared")
    assert_type(shared, int)


class Fifo(Space):
    word_bits: int = Param()
    depth: int = Param()
    ram_style: str = Decision(values=("auto", "block"))
    banks: int = Decision(values=(1, 2))
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
    activation: DType = Param(semantics=DTYPE)
    weight: DType = Param(semantics=DTYPE)

    @derived(a=activation, b=weight, semantics=DTYPE)
    def result_dtype(*, a: DType, b: DType) -> QueryResult[DType]:
        return Available(DType(max(a.bits, b.bits) + 1))

    @view(semantics=INT)
    def physical(*, result_dtype: DType) -> QueryResult[int]:
        return Available(result_dtype.bits)


class GuardedAssembly(Space):
    enabled: bool = Param()
    slots: int = Decision(values=(1, 2), when=enabled)
    fifo = Fifo(word_bits=8, depth=4, when=enabled)

    @derived(when=enabled)
    def value(*, slots: int) -> int:
        return slots

    @constraint(when=enabled)
    def supported(*, slots: int) -> bool:
        return slots > 0

    physical = View(value, requires=(supported,), when=enabled)


def check(point: Fifo, house: House, eltwise: Eltwise) -> None:
    # Class access is the schema key, typed as its value like every reference.
    assert_type(Fifo.word_bits, int)
    assert_type(Fifo.ram_style, str)
    assert_type(Fifo.minimum_depth, Const[int])
    assert_type(Fifo.capacity, int)
    assert_type(Fifo.physical, View[int])
    assert_type(House.kitchen, Room)
    assert_type(House.kitchen.finish, int)
    # The compile step is typed as the family.
    assert_type(design_space(House(budget=100)), House)
    assert_type(design_space(Estate(home=House(budget=1))), Estate)
    assert_type(design_space(Room()), Room)  # a bare call type-checks
    # Configuration reads are exact.
    assert_type(point.word_bits, int)
    assert_type(point.ram_style, str)
    assert_type(point.minimum_depth, int)
    assert_type(point.capacity, int)
    assert_type(point.physical, int)  # a view reads as its accepted value
    assert_type(point.inspect(Fifo.physical), ViewAssessment[int])
    assert_type(point.query(Fifo.physical), QueryResult[int])
    # A key typed as its value binds as a decision accessor (a Param or a
    # derived value cannot be told from a Decision statically).
    assert_type(point.field(Fifo.capacity), BoundDecision[int])
    assert_type(point.field(Fifo.ram_style), BoundDecision[str])
    assert_type(point.field(Fifo.physical), BoundValue[int])  # a view binds as a value
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
    assert_type(eltwise.physical, int)
    assert_type(house.kitchen.finish, int)
    assert_type(house.heating, Boiler | HeatPump)
    assert_type(house.maybe, Boiler | None)
    assert_type(GuardedAssembly.value, int)
    assert_type(GuardedAssembly.physical, View[int])


# -- iteration 3: annotated formals, reads through reference inputs, overrides ----------


@dataclass(frozen=True)
class Spec:
    lanes: int
    bits: int

    @property
    def payload_bits(self) -> int:
        return self.lanes * self.bits


class Wire(Space):
    spec: Spec = Param()


class Buffer(Space):
    word_bits: int = Param()
    depth: int = Decision(values=(2, 4, 8))


class Kernel(Space):
    output: Wire = Param()  # a reference input, annotated with its family
    buffer = Buffer(word_bits=output.spec.payload_bits)  # typed in the class body
    assert_type(output, Wire)
    assert_type(output.spec, Spec)
    assert_type(output.spec.payload_bits, int)


class Board(Space):
    wire = Wire(spec=Spec(4, 8))
    kernel = Kernel(output=wire)
    kernel.buffer.depth = 4  # pin a Decision of a descendant
    spare = Kernel(output=wire)
    spare.buffer.depth = Decision(values=(2, 4))  # narrow it: same key
    spare.buffer = Buffer(word_bits=16)  # replace a child node (same family)
    lobby = Room(area=1)
    stages = Room(area=lobby.cost)  # a view's accepted value supplies a formal


def keys(board: Board, house: House) -> None:
    assert_type(Room(), Room)  # every member is optional at the call
    assert_type(Room(area=3, finish=2), Room)  # a Decision may be pinned at the call
    assert_type(Kernel.output, Wire)
    assert_type(Board.kernel.buffer.depth, int)
    assert_type(board.kernel.query(Kernel.output), QueryResult[Wire])
    assert_type(board.kernel.present(Kernel.output), bool)
    assert_type(board.kernel.output.spec.payload_bits, int)
    assert_type(inspection.decision_handle(house, House.kitchen.finish), DecisionHandle[int])
    assert_type(inspection.decision_handle(house, House.heating), DecisionHandle[str])
    assert_type(inspection.value_handle(board, Board.kernel.buffer.word_bits), ValueHandle[int])
    assert_type(inspection.provenance(board, Board.kernel.buffer.depth), Provenance | None)
    assert_type(house.with_choices({House.kitchen.finish: 2}), House)


# -- iteration 4: views read as values -------------------------------------------------


class Wing(Space):
    kitchen = Room(area=12)
    dining = Room(area=16)
    hall = Room()
    hall.area = kitchen.cost  # a view reference supplies a formal, typed int
    study = Room(area=dining.cost)
    either = Room(area=Present(kitchen.cost, dining.cost))
    assert_type(kitchen.cost, int)
    assert_type(Present(kitchen.cost, dining.cost), int)

    @view(requires=(kitchen.cost, dining.cost))  # view references as obligations
    def total(self) -> int:
        assert_type(self.kitchen.cost, int)  # inside a method: the accepted value
        assert_type(self.hall.cost, int)
        return self.kitchen.cost + self.dining.cost

    checked = View(kitchen.cost, requires=(dining.cost, total))
    assert_type(checked, View[int])


def views(point: Wing, house: House, fifo: Fifo) -> None:
    # On a configuration a view reads as its accepted value, like any member.
    assert_type(point.total, int)
    assert_type(point.checked, int)
    assert_type(point.kitchen.cost, int)
    assert_type(house.total, int)
    # Class access is the view declaration; assessment and query are explicit calls.
    assert_type(Wing.total, View[int])
    assert_type(point.inspect(Wing.total), ViewAssessment[int])
    assert_type(point.query(Wing.total), QueryResult[int])
    assert_type(house.inspect(House.total), ViewAssessment[int])
    assert_type(house.query(House.total), QueryResult[int])
    assert_type(point.inspect(Wing.checked).accepted_result, QueryResult[int])
    # A view of a child: on the child's configuration (precise), or through a path.
    assert_type(point.kitchen.inspect(Room.cost), ViewAssessment[int])
    assert_type(point.kitchen.query(Room.cost), QueryResult[int])
    assert_type(point.inspect(Wing.kitchen.cost), ViewAssessment[int])
    assert_type(point.query(Wing.kitchen.cost), QueryResult[int])
    # A constraint is still assessed by the same call.
    assert_type(fifo.inspect(Fifo.supported), ConstraintAssessment)
    # A view binds as a value accessor of its accepted value.
    assert_type(point.field(Wing.total), BoundValue[int])
    assert_type(point.field(Wing.total).get(), int)
