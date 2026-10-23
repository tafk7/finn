# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from weakref import ReferenceType, ref

from finn.core.space import (
    Decision,
    Param,
    Space,
    derived,
    design_space,
    inspection,
)
from finn.core.space.results import Available


@dataclass(frozen=True)
class Payload:
    value: int


def test_concurrent_reads_evaluate_one_cached_output_per_design_space() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Example(Space):
        source: int = Param()

        @derived
        def output(*, source: int) -> Payload:
            payload = Payload(source)
            produced.append(ref(payload))
            return payload

    first = design_space(Example(source=1))
    second = design_space(Example(source=2))
    assert inspection.model(first) is inspection.model(second)

    def read_first(_: int) -> Payload:
        return first.output

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(read_first, range(64)))
    assert len(produced) == 1
    assert all(output is produced[0]() for output in outputs)
    assert second.output == Payload(2)
    assert len(produced) == 2
    assert produced[0]() is not produced[1]()


def test_concurrent_successors_keep_independent_commitments_and_caches() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Example(Space):
        factor: int = Decision(values=range(16))

        @derived
        def output(*, factor: int) -> Payload:
            payload = Payload(factor)
            produced.append(ref(payload))
            return payload

    base = design_space(Example())

    def explore(value: int) -> Example:
        point = base.with_choices(factor=value)
        assert point.output == Payload(value)
        return point

    with ThreadPoolExecutor(max_workers=8) as pool:
        candidates = list(pool.map(explore, range(16)))
    assert [point.output.value for point in candidates] == list(range(16))
    assert len(produced) == 16
    state = base.field(Example.factor).state
    assert isinstance(state, Available)
    assert state.value.status == "unassigned"


def test_independent_root_bindings_share_the_model_and_keep_their_own_facts() -> None:
    calls: list[tuple[int, ...]] = []

    class Example(Space):
        values: tuple[int, ...] = Param()

        @derived
        def total(*, values: tuple[int, ...]) -> int:
            calls.append(values)
            return sum(values)

    first = design_space(Example(values=(1, 2)))
    second = design_space(Example(values=(1, 2, 3)))
    assert inspection.model(first) is inspection.model(second)
    assert first.total == 3
    assert second.total == 6
    assert first.total == 3
    assert calls == [(1, 2), (1, 2, 3)]
