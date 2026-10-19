# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Possible values: an open choice stays open, whatever its possible values.

Two cores share a ``width``: ``packed`` refuses more than 64 lanes, ``stub`` an
odd width, each through a constraint, which counts toward the candidate's validity. A
value is possible unless a rule it reaches refuses it (``inspection.viable``): here over
a capability record. Nothing is filled in: a choice with one possible value reads open
until a change sets it. A successor takes from its base what its change does not reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from support.space_cores import CORES, Packed, Stub, built, commit
from support.space_walk import fresh

from core.space._choices_support import committed
from core.space._possible_support import broken, possible, refusals
from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    occurrence,
    reject,
)


def unassigned(answer: object) -> bool:
    return isinstance(answer, Unresolved) and {f.code for f in answer.findings} == {
        "decision-unassigned"
    }


class Unit(Space):
    width: int = Param()
    compute: Packed | Stub = Decision(CORES, width=width)


def test_one_possible_case_is_not_filled_in() -> None:
    point = design_space(Unit(width=128))
    assert unassigned(point.query(Unit.compute))
    state = point.field(Unit.compute).state
    assert isinstance(state, Available)
    assert (state.value.status, state.value.value) == ("unassigned", None)
    assert committed(point) == {}


def test_one_possible_case_and_why_the_other_is_refused() -> None:
    point = design_space(Unit(width=128))
    assert possible(point, Unit.compute) == ("stub",)
    assert refusals(point, Unit.compute) == {"packed": {"compute.packed.fits": set()}}


def test_inspection_answers_are_read_once_per_configuration_and_model() -> None:
    point = design_space(Unit(width=128))
    stub = inspection.candidate(point, Unit.compute, "stub")
    assert stub is not None
    assert inspection.validity(stub) is inspection.validity(stub)
    assert inspection.validity(point) is inspection.validity(point)
    assert inspection.decisions(point) is inspection.decisions(Unit)


def test_a_choice_stays_open_whatever_its_possible_values() -> None:
    both = design_space(Unit(width=8))
    assert unassigned(both.query(Unit.compute))
    # With no possible value, the choice is open too: the constraints of each case refuse.
    assert unassigned(design_space(Unit(width=129)).query(Unit.compute))


def test_no_possible_value_is_each_value_refused_naming_its_rule() -> None:
    assert possible(design_space(Unit(width=8)), Unit.compute) == ("packed", "stub")
    neither = design_space(Unit(width=129))
    assert possible(neither, Unit.compute) == ()
    assert refusals(neither, Unit.compute) == {
        "packed": {"compute.packed.fits": set()},
        "stub": {"compute.stub.even": set()},
    }


def test_a_choice_nested_under_an_open_selector_is_stored_and_waits_on_it() -> None:
    point = commit(design_space(Unit(width=128)), {"compute.stub.rows": 2})
    assert list(committed(point)) == ["compute.stub.rows"]
    assert unassigned(point.query(Unit.compute))
    waiting = point.query(Unit.compute["stub"].rows)  # type: ignore[index]
    assert isinstance(waiting, Unresolved)
    assert {(f.code, f.owner) for f in waiting.findings} == {("decision-unassigned", "compute")}
    assert commit(point, {"compute": "stub"}).compute.rows == 2


def test_a_validity_waiting_on_an_open_choice_does_not_refuse() -> None:
    class Waiting(Space):
        width: int = Param()
        pe: int = Decision(values=(1, 2, 8))

        @constraint
        def small(self) -> bool | Rejected:
            if self.pe > 4:
                return reject("waiting-pe", "at most 4 lanes a cycle")
            return True

    class Root(Space):
        width: int = Param()
        entries: dict[str, type[Space] | Space] = {"waiting": Waiting, "stub": Stub}
        compute: Waiting | Stub = Decision(entries, width=width)

    point = design_space(Root(width=7))  # odd: stub refuses; waiting waits on its pe
    candidate = inspection.candidate(point, Root.compute, "waiting")
    assert candidate is not None
    validity = inspection.validity(candidate)
    assert validity.status == "open" and isinstance(validity.result, Unresolved)
    assert [piece.owner for piece in validity.pieces] == ["compute.waiting.small"]


def test_a_validity_refused_beside_a_piece_waiting_refuses() -> None:
    class Both(Space):
        width: int = Param()
        pe: int = Decision(values=(1, 2))

        @constraint
        def waits(self) -> bool:
            return self.pe > 0

        @constraint
        def never() -> bool:
            return False

    class Root(Space):
        width: int = Param()
        entries: dict[str, type[Space] | Space] = {"both": Both, "stub": Stub}
        compute: Both | Stub = Decision(entries, width=width)

    point = design_space(Root(width=8))  # even: the stub core takes it; both always refuses
    assert unassigned(point.query(Root.compute))
    # While the selector is open, each case's pieces wait on it, the one never true too.
    validity = inspection.validity(point)
    assert validity.status == "open"
    assert [piece.owner for piece in validity.pieces] == [
        "compute.both.never",
        "compute.both.waits",
        "compute.stub.even",
    ]


def test_a_value_decision_with_one_case_is_open() -> None:
    class Lanes(Space):
        lanes: int = Decision(values=(4,))
        many: int = Decision(values=(1, 2))

    point = design_space(Lanes())
    assert unassigned(point.query(Lanes.lanes)) and unassigned(point.query(Lanes.many))
    assert point.with_choices(lanes=4).lanes == 4


@dataclass(frozen=True)
class Capabilities:
    fast: bool = True
    ports: int = 1


class Memory(Space):
    capabilities: Capabilities = Param(default=Capabilities())
    style: str = Decision(values=("auto", "fast"))
    writable: bool = Decision(values=(False, True))

    @constraint
    def fast_present(self) -> bool | Rejected:
        if self.style == "fast" and not self.capabilities.fast:
            return reject("fast-absent", "no fast memory here")
        return True

    @constraint
    def port_present(self) -> bool | Rejected:
        if self.writable and not self.capabilities.ports:
            return reject("port-absent", "no port to write through")
        return True


def test_a_case_a_constraint_refuses_is_refused_and_a_configuration_holding_it_reads_refused() -> (
    None
):
    open_ = design_space(Memory())
    assert commit(open_, {"style": "fast"}).style == "fast"
    point = design_space(Memory(capabilities=Capabilities(fast=False, ports=0)))
    assert unassigned(point.query(Memory.style)) and unassigned(point.query(Memory.writable))
    # The value passes its type check, and breaks the constraint on the facts alone: the
    # change is refused naming it, nothing standing in its way.
    report = point.try_with_choices({Memory.style: "fast"})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert outcome.result == Available(True) and outcome.status == "checked"
    assert broken(report) == {"fast_present": set()}
    # A configuration holding it, built directly, reads refused, naming the constraint.
    validity = inspection.validity(built(point, {"style": "fast"}))
    assert validity.status == "refused"
    assert [
        (piece.owner, [f.code for f in piece.result.findings]) for piece in validity.pieces
    ] == [("fast_present", ["fast-absent"])]
    # Enumeration stays the declared cases: the domain does not move with the facts.
    assert point.field(Memory.style).candidates() == Available(("auto", "fast"))


class Limited(Space):
    width: int = Param()
    limit: int = Decision(values=(1, 8))
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def within(self) -> bool | Rejected:
        if self.limit < 2:
            return reject("limited", "the limit leaves no lane")
        return True


class Hop(Space):
    """``burst`` reads the core's selector, which an odd width leaves one possible case."""

    width: int = Param()
    entries: dict[str, type[Space] | Space] = {"limited": Limited, "stub": Stub}

    @derived
    def stubbed(self) -> bool:
        return isinstance(self.compute, Stub)

    burst: int = Decision(values=(1, 2))
    compute: Limited | Stub = Decision(entries, width=width)

    @constraint
    def bursts(self) -> bool | Rejected:
        if self.burst == 2 and not self.stubbed:
            return reject("burst-source", "bursts need the stub core")
        return True


class Other(Space):
    size: int = Decision(values=(1, 2, 3))


class HopsFirst(Space):
    width: int = Param()
    a = Hop(width=width)
    b = Hop(width=width)
    other = Other()


class HopsLast(Space):
    width: int = Param()
    other = Other()
    a = Hop(width=width)
    b = Hop(width=width)


@pytest.mark.parametrize("space", [HopsFirst, HopsLast])
def test_a_successor_keeps_what_its_change_does_not_reach_whichever_order_declared(
    space: Any,
) -> None:
    base = commit(
        design_space(space(width=7)),
        {"a.compute": "limited", "b.compute": "limited", "a.burst": 1, "b.burst": 1},
    )
    base_world = occurrence.state(base).world
    nodes = occurrence.state(base).linked.nodes
    for node in nodes:
        base_world.entry(node.index)

    def kept(point: Any, prefix: str) -> bool:
        world = occurrence.state(point).world
        for node in nodes:
            world.entry(node.index)
        return all(
            world.cache[node.index] is base_world.cache[node.index]
            for node in nodes
            if node.key.startswith(prefix)
        )

    # Outside a and b, nothing of theirs is evaluated again.
    unreached = commit(base, {"other.size": 2})
    assert kept(unreached, "a.") and kept(unreached, "b.")
    # Inside a, what reads a's limit is evaluated again and nothing of b's.
    reached = commit(base, {"a.compute.limited.limit": 8})
    assert kept(reached, "b.") and not kept(reached, "a.")
    # Each answers every node as a configuration built from nothing.
    for point in (unreached, reached):
        got, want = occurrence.state(point).world, occurrence.state(fresh(point)).world
        for node in nodes:
            assert got.result(node.index) == want.result(node.index), node.key


def test_a_values_trial_keeps_what_the_value_does_not_reach() -> None:
    """Each value's trial is a world one choice away from the configuration's: a value the
    trials share is derived once, not again in every trial."""
    derivations: list[int] = []

    class Shared(Space):
        lanes: int = Param()

        @derived
        def width(self) -> int:
            derivations.append(self.lanes)
            return self.lanes

        # Read first, on the configuration itself: the constraint on ``wide`` reads ``width``.
        wide: bool = Decision(values=(False, True))
        compute: Packed | Stub = Decision(CORES, width=width)

        @constraint
        def lanes_present(self) -> bool | Rejected:
            if not self.width and self.wide:
                return reject("width-absent", "no lanes")
            return True

    point = design_space(Shared(lanes=128))
    assert possible(point, Shared.compute) == ("stub",)
    assert derivations == [128]


def test_viable_answers_each_listed_value_of_each_choice_asked() -> None:
    both = design_space(Unit(width=8))
    assert possible(both, Unit.compute) == ("packed", "stub")
    packed = commit(both, {"compute": "packed"})
    assert possible(packed, Unit.compute["packed"].pe) == (1, 2, 4, 8)  # type: ignore[index]
    wide = design_space(Unit(width=128))
    assert refusals(wide, Unit.compute) == {"packed": {"compute.packed.fits": set()}}
    assert possible(wide, Unit.compute["stub"].rows) == (1, 2, 4)  # type: ignore[index]
    narrow = design_space(Memory(capabilities=Capabilities(fast=False)))
    assert refusals(narrow, Memory.style) == {"fast": {"fast_present": set()}}
    assert possible(narrow, Memory.writable) == (False, True)
    portless = design_space(Memory(capabilities=Capabilities(ports=0)))
    assert refusals(portless, Memory.writable) == {True: {"port_present": set()}}
