# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral vertical slices for the replacement runtime."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from finn.core.space import (
    Const,
    Decision,
    Param,
    Space,
    View,
    compile_space,
    constraint,
    derived,
    view,
)
from finn.core.space.domains import divisors_of
from finn.core.space.errors import EvaluationError, RequestError
from finn.core.space.results import Available, QueryResult, Rejected, Unresolved


@dataclass(frozen=True)
class FifoShape:
    bits: int
    depth: int
    ram_style: str


class Fifo(Space):
    word_bits = Param(int)
    depth = Param(int)
    ram_style = Decision(str, values=("auto", "block", "shift"))
    family = Const("fifo")

    @constraint
    def supported(self) -> bool:
        return self.word_bits > 0 and self.depth >= 2

    @view(constraints=(supported,))
    def physical(self) -> FifoShape:
        return FifoShape(self.word_bits, self.depth, self.ram_style)


def test_fifo_compile_bind_replace_and_inspect_is_immutable() -> None:
    model = compile_space(Fifo)
    base = model.bind(word_bits=13, depth=8)
    assert base.family == "fifo"
    assert isinstance(base.physical.inspect().accepted_result, Unresolved)
    state = base.field(Fifo.ram_style).state
    assert isinstance(state, Available)
    assert state.value.status == "unassigned"
    assert isinstance(base.query(Fifo.ram_style), Unresolved)
    assert base.field(Fifo.ram_style).candidates() == Available(("auto", "block", "shift"))
    chosen = base.with_choices(ram_style="block")
    assert chosen.physical() == FifoShape(13, 8, "block")
    assert chosen.inspect(Fifo.physical) == chosen.physical.inspect()
    assert chosen.view(Fifo.physical)() == FifoShape(13, 8, "block")
    assert chosen.physical.query() == Available(FifoShape(13, 8, "block"))
    assert chosen.field(Fifo.ram_style).get() == "block"
    assert chosen.field(Fifo.ram_style).query() == Available("block")
    assert isinstance(base.query(Fifo.ram_style), Unresolved)
    assert chosen.with_choices(ram_style="block") is chosen
    assert chosen.with_choices(ram_style="shift").ram_style == "shift"
    other = model.bind(word_bits=7, depth=4).with_choices(ram_style="shift")
    assert other.physical() == FifoShape(7, 4, "shift")


def test_final_constraint_refusal_remains_visible_while_output_unresolved() -> None:
    base = Fifo(word_bits=0, depth=8)
    assessment = base.physical.inspect()
    assert isinstance(assessment.accepted_result, Unresolved)
    assert assessment.constraints.refused == ("supported",)
    ready = base.with_choices(ram_style="auto").physical.inspect()
    assert isinstance(ready.accepted_result, Rejected)


def test_atomic_refinement_follows_dependencies_and_never_publishes_partial_state() -> None:
    class Tiles(Space):
        extent = Decision(int, values=(8, 12))
        lanes = Decision(int, domain=divisors_of(extent))

    base = Tiles()
    report = base.try_with_choices(
        base.field(Tiles.lanes).change(4), base.field(Tiles.extent).change(12)
    )
    assert report.accepted
    assert report.instance.lanes == 4
    assert report.instance.extent == 12
    failure = base.try_with_choices(
        base.field(Tiles.lanes).change(5), base.field(Tiles.extent).change(12)
    )
    assert not failure.accepted
    assert failure.instance is base
    assert {x.status for x in failure.outcomes} == {"refused", "admissible"}
    assert isinstance(base.query(Tiles.extent), Unresolved)
    with pytest.raises(RequestError):
        base.try_with_choices(
            base.field(Tiles.extent).change(8), report.instance.field(Tiles.lanes).change(4)
        )


def test_binding_and_callback_values_are_snapshots_without_requiring_a_codec() -> None:
    calls: list[int] = []

    class Mutable(Space):
        payload: Param[list[object]] = Param(list)

        @derived
        def length(*, payload: list[object]) -> int:
            calls.append(len(payload))
            payload.append("local mutation")
            return len(payload)

        physical = View(length)

    original = [1, 2]
    model = compile_space(Mutable)
    base = model.bind(payload=original)
    original.append(3)
    assert base.length == 3
    assert base.payload == [1, 2]
    base.payload.append(4)
    assert base.payload == [1, 2]
    assert base.physical() == 3
    assert calls == [2]
    other = model.bind(payload=[8])
    assert other.length == 2
    assert calls == [2, 1]


def test_missing_required_parameters_and_malformed_commitments_fail_at_boundary() -> None:
    with pytest.raises(RequestError, match="depth"):
        Fifo(word_bits=13)
    with pytest.raises(RequestError):
        Fifo(word_bits=True, depth=8)
    with pytest.raises(RequestError):
        Fifo({Fifo.ram_style: "auto"}, word_bits=13, depth=8)


def test_programmer_failure_keeps_owner_and_cause() -> None:
    class Broken(Space):
        denominator = Param(int)

        @derived
        def quotient(*, denominator: int) -> int:
            return 10 // denominator

    with pytest.raises(EvaluationError) as caught:
        Broken(denominator=0).query(Broken.quotient)
    assert caught.value.owner == "quotient"
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


def test_concurrent_reads_and_successors_are_deterministic() -> None:
    base = Fifo(word_bits=13, depth=8)

    def run(style: str) -> QueryResult[FifoShape]:
        return base.with_choices(ram_style=style).physical.inspect().accepted_result

    styles = ["auto", "block", "shift"] * 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, styles))
    assert outputs == [Available(FifoShape(13, 8, style)) for style in styles]
    assert isinstance(base.query(Fifo.ram_style), Unresolved)


def test_generic_container_outputs_infer_nominal_snapshot_semantics() -> None:
    class Collections(Space):
        count = Param(int)

        @derived
        def indices(*, count: int) -> tuple[int, ...]:
            return tuple(range(count))

        @view
        def materialized(*, indices: tuple[int, ...]) -> list[int]:
            return list(indices)

    point = Collections(count=3)
    assert point.indices == (0, 1, 2)
    first = point.materialized()
    first.append(99)
    assert point.materialized() == [0, 1, 2]
    assert point.materialized.query() == Available([0, 1, 2])
