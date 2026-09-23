# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral vertical slices for the replacement runtime."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pytest

from finn.kernels.space._next import (
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
from finn.kernels.space._next.domains import divisors_of
from finn.kernels.space._next.errors import EvaluationError, RefinementError, RequestError
from finn.kernels.space._next.occurrence import candidates, decision_state, edit, refine
from finn.kernels.space._next.results import Decided, Rejected, Unresolved


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
    def supported(*, word_bits: int, depth: int) -> bool:
        return word_bits > 0 and depth >= 2

    @view(constraints=(supported,))
    def physical(*, word_bits: int, depth: int, ram_style: str) -> FifoShape:
        return FifoShape(word_bits, depth, ram_style)


def test_fifo_compile_start_commit_assess_is_immutable():
    model = compile_space(Fifo)
    base = model.start({Fifo.word_bits: 13, Fifo.depth: 8})
    assert base.family == "fifo"
    assert isinstance(base.physical().accepted_answer, Unresolved)
    assert decision_state(base, Fifo.ram_style).value.status == "unassigned"
    assert isinstance(base.answer(Fifo.ram_style), Unresolved)
    assert candidates(base, Fifo.ram_style) == Decided(("auto", "block", "shift"))
    chosen = base.assign(Fifo.ram_style, "block")
    assert chosen.physical().accepted_answer == Decided(FifoShape(13, 8, "block"))
    assert chosen.assess(Fifo.physical) == chosen.physical()
    assert isinstance(base.answer(Fifo.ram_style), Unresolved)
    assert chosen.assign(Fifo.ram_style, "block") is chosen
    with pytest.raises(RefinementError):
        chosen.assign(Fifo.ram_style, "shift")
    other = model.start({Fifo.word_bits: 7, Fifo.depth: 4}).assign(Fifo.ram_style, "shift")
    assert other.physical().accepted_answer == Decided(FifoShape(7, 4, "shift"))


def test_final_constraint_refusal_remains_visible_while_output_unresolved():
    base = Fifo.start({Fifo.word_bits: 0, Fifo.depth: 8})
    assessment = base.physical()
    assert isinstance(assessment.accepted_answer, Unresolved)
    assert assessment.constraints.refused == ("supported",)
    ready = base.assign(Fifo.ram_style, "auto").physical()
    assert isinstance(ready.accepted_answer, Rejected)


def test_atomic_refinement_follows_dependencies_and_never_publishes_partial_state():
    class Tiles(Space):
        extent = Decision(int, values=(8, 12))
        lanes = Decision(int, domain=divisors_of(extent))

    base = Tiles.start()
    report = refine(base, edit(base, Tiles.lanes, 4), edit(base, Tiles.extent, 12))
    assert report.accepted
    assert report.point.lanes == 4
    assert report.point.extent == 12
    failure = refine(base, edit(base, Tiles.lanes, 5), edit(base, Tiles.extent, 12))
    assert not failure.accepted
    assert failure.point is base
    assert {x.status for x in failure.outcomes} == {"refused", "provisional"}
    assert isinstance(base.answer(Tiles.extent), Unresolved)
    with pytest.raises(RequestError):
        refine(base, edit(base, Tiles.extent, 8), edit(report.point, Tiles.lanes, 4))


def test_binding_and_callback_values_are_snapshots_without_requiring_a_codec():
    calls = []

    class Mutable(Space):
        payload = Param(list)

        @derived
        def length(*, payload: list) -> int:
            calls.append(len(payload))
            payload.append("local mutation")
            return len(payload)

        physical = View(length)

    original = [1, 2]
    model = compile_space(Mutable)
    base = model.start({Mutable.payload: original})
    original.append(3)
    assert base.length == 3
    assert base.payload == [1, 2]
    base.payload.append(4)
    assert base.payload == [1, 2]
    assert base.physical().accepted_answer == Decided(3)
    assert calls == [2]
    other = model.start({Mutable.payload: [8]})
    assert other.length == 2
    assert calls == [2, 1]


def test_missing_required_parameters_and_malformed_commitments_fail_at_boundary():
    with pytest.raises(RequestError, match="depth"):
        Fifo.start({Fifo.word_bits: 13})
    with pytest.raises(RequestError):
        Fifo.start({Fifo.word_bits: True, Fifo.depth: 8})
    with pytest.raises(RequestError):
        Fifo.start({Fifo.word_bits: 13, Fifo.depth: 8, Fifo.ram_style: "auto"})


def test_programmer_failure_keeps_owner_and_cause():
    class Broken(Space):
        denominator = Param(int)

        @derived
        def quotient(*, denominator: int) -> int:
            return 10 // denominator

    with pytest.raises(EvaluationError) as caught:
        Broken.start({Broken.denominator: 0}).answer(Broken.quotient)
    assert caught.value.owner == "quotient"
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


def test_concurrent_reads_and_successors_are_deterministic():
    base = Fifo.start({Fifo.word_bits: 13, Fifo.depth: 8})

    def run(style):
        return base.assign(Fifo.ram_style, style).physical().accepted_answer

    styles = ["auto", "block", "shift"] * 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, styles))
    assert outputs == [Decided(FifoShape(13, 8, style)) for style in styles]
    assert isinstance(base.answer(Fifo.ram_style), Unresolved)


def test_generic_container_outputs_infer_nominal_snapshot_semantics():
    class Collections(Space):
        count = Param(int)

        @derived
        def indices(*, count: int) -> tuple[int, ...]:
            return tuple(range(count))

        @view
        def materialized(*, indices: tuple[int, ...]) -> list[int]:
            return list(indices)

    point = Collections.start({Collections.count: 3})
    assert point.indices == (0, 1, 2)
    first = point.materialized().accepted_answer
    assert isinstance(first, Decided)
    first.value.append(99)
    assert point.materialized().accepted_answer == Decided([0, 1, 2])
