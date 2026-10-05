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
    constraint,
    derived,
    design_space,
    inspection,
    view,
)
from finn.core.space.domains import divisors_of
from finn.core.space.errors import DefinitionError, EvaluationError, RequestError
from finn.core.space.results import Available, QueryResult, Rejected, Unresolved


@dataclass(frozen=True)
class FifoShape:
    bits: int
    depth: int
    ram_style: str


class Fifo(Space):
    word_bits: int = Param()
    depth: int = Param()
    ram_style: str = Decision(values=("auto", "block", "shift"))
    space_type = Const("fifo")

    @constraint
    def supported(self) -> bool:
        return self.word_bits > 0 and self.depth >= 2

    @view(requires=(supported,))
    def physical(self) -> FifoShape:
        return FifoShape(self.word_bits, self.depth, self.ram_style)


def test_fifo_configure_replace_and_inspect_is_immutable() -> None:
    base = design_space(Fifo(word_bits=13, depth=8))
    # Plain values are runtime inputs: every such root shares the Space class's one model.
    model = inspection.model(Fifo)
    assert inspection.model(base) is model
    assert base.space_type == "fifo"
    assert isinstance(base.inspect(Fifo.physical).accepted_result, Unresolved)
    state = base.field(Fifo.ram_style).state
    assert isinstance(state, Available)
    assert state.value.status == "unassigned"
    assert isinstance(base.query(Fifo.ram_style), Unresolved)
    assert base.field(Fifo.ram_style).candidates() == Available(("auto", "block", "shift"))
    chosen = base.with_choices(ram_style="block")
    assert chosen.physical == FifoShape(13, 8, "block")
    assert chosen.inspect(Fifo.physical).accepted_result == chosen.query(Fifo.physical)
    assert chosen.field(Fifo.physical).get() == FifoShape(13, 8, "block")
    assert chosen.query(Fifo.physical) == Available(FifoShape(13, 8, "block"))
    assert chosen.field(Fifo.ram_style).get() == "block"
    assert chosen.field(Fifo.ram_style).query() == Available("block")
    assert isinstance(base.query(Fifo.ram_style), Unresolved)
    assert chosen.with_choices(ram_style="block") is chosen
    assert chosen.with_choices(ram_style="shift").ram_style == "shift"
    other = design_space(Fifo(word_bits=7, depth=4)).with_choices(ram_style="shift")
    assert inspection.model(other) is model
    assert other.physical == FifoShape(7, 4, "shift")


def test_final_constraint_refusal_remains_visible_while_output_unresolved() -> None:
    base = design_space(Fifo(word_bits=0, depth=8))
    assessment = base.inspect(Fifo.physical)
    assert isinstance(assessment.accepted_result, Unresolved)
    assert assessment.constraints.refused == ("supported",)
    ready = base.with_choices(ram_style="auto").inspect(Fifo.physical)
    assert isinstance(ready.accepted_result, Rejected)


def test_atomic_refinement_follows_dependencies_and_never_publishes_partial_state() -> None:
    class Tiles(Space):
        extent: int = Decision(values=(8, 12))
        lanes: int = Decision(domain=divisors_of(extent))

    base = design_space(Tiles())
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
        payload: list[object] = Param()

        @derived
        def length(*, payload: list[object]) -> int:
            calls.append(len(payload))
            payload.append("local mutation")
            return len(payload)

        physical = View(length)

    original: list[object] = [1, 2]
    # The node call is the binding boundary: it snapshots the value.
    node = Mutable(payload=original)
    original.append(3)
    base = design_space(node)
    assert base.length == 3
    assert base.payload == [1, 2]
    base.payload.append(4)
    assert base.payload == [1, 2]
    assert base.physical == 3
    assert calls == [2]
    other = design_space(Mutable(payload=[8]))
    assert inspection.model(other) is inspection.model(base)
    assert other.length == 2
    assert calls == [2, 1]


def test_missing_required_parameters_and_malformed_commitments_fail_at_boundary() -> None:
    # A missing formal is refused when the root is configured; type errors at the call.
    with pytest.raises(DefinitionError, match="depth is not supplied"):
        design_space(Fifo(word_bits=13))
    with pytest.raises(DefinitionError, match="int"):
        Fifo(word_bits=True, depth=8)
    with pytest.raises(DefinitionError, match="keyword"):
        Fifo({Fifo.ram_style: "auto"}, word_bits=13, depth=8)  # type: ignore[arg-type, call-arg]
    # Commitments are made on a configuration and refused at that boundary.
    base = design_space(Fifo(word_bits=13, depth=8))
    with pytest.raises(RequestError, match="ram_style"):
        base.with_choices(ram_style=5)
    with pytest.raises(RequestError, match="bogus"):
        base.with_choices(bogus="auto")


def test_programmer_failure_keeps_owner_and_cause() -> None:
    class Broken(Space):
        denominator: int = Param()

        @derived
        def quotient(*, denominator: int) -> int:
            return 10 // denominator

    with pytest.raises(EvaluationError) as caught:
        design_space(Broken(denominator=0)).query(Broken.quotient)
    assert caught.value.owner == "quotient"
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


def test_concurrent_reads_and_successors_are_deterministic() -> None:
    base = design_space(Fifo(word_bits=13, depth=8))

    def run(style: str) -> QueryResult[FifoShape]:
        return base.with_choices(ram_style=style).inspect(Fifo.physical).accepted_result

    styles = ["auto", "block", "shift"] * 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, styles))
    assert outputs == [Available(FifoShape(13, 8, style)) for style in styles]
    assert isinstance(base.query(Fifo.ram_style), Unresolved)


def test_generic_container_outputs_infer_nominal_snapshot_semantics() -> None:
    class Collections(Space):
        count: int = Param()

        @derived
        def indices(*, count: int) -> tuple[int, ...]:
            return tuple(range(count))

        @view
        def materialized(*, indices: tuple[int, ...]) -> list[int]:
            return list(indices)

    point = design_space(Collections(count=3))
    assert point.indices == (0, 1, 2)
    first = point.materialized
    first.append(99)
    assert point.materialized == [0, 1, 2]
    assert point.query(Collections.materialized) == Available([0, 1, 2])
