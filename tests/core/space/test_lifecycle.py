# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from weakref import ReferenceType, ref

from finn.core.space import (
    Decision,
    Param,
    Space,
    ValueSemantics,
    derived,
    design_space,
    inspection,
)
from finn.core.space.results import Available


@dataclass(frozen=True)
class Payload:
    value: int


PAYLOAD = ValueSemantics.immutable_nominal(Payload)


def test_discarded_candidate_snapshots_release_the_actual_cached_callback_outputs() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Example(Space):
        factor: int = Decision(values=range(16))

        @derived(semantics=PAYLOAD)
        def output(*, factor: int) -> Payload:
            payload = Payload(factor)
            produced.append(ref(payload))
            return payload

    model = inspection.model(Example)
    base = design_space(Example())
    assert inspection.model(base) is model  # a plain root reuses the Space class's model

    def population() -> list[Example]:
        points = [base.with_choices(factor=value) for value in range(16)]
        for point in points:
            point.query(Example.output)
        return points

    candidates = population()
    gc.collect()
    # References were captured inside the callback before any public reads.
    # Immutable semantics retain that exact output object in each local cache.
    assert len(produced) == 16
    assert all(reference() is not None for reference in produced)
    del candidates
    gc.collect()
    assert all(reference() is None for reference in produced)
    assert design_space(Example()).try_with_choices().accepted


def test_successor_does_not_retain_its_predecessors_output_cache() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Example(Space):
        factor: int = Decision(values=(1, 2))
        extra: int = Decision(values=(3, 4))

        @derived(semantics=PAYLOAD)
        def output(*, factor: int) -> Payload:
            payload = Payload(factor)
            produced.append(ref(payload))
            return payload

    earlier = design_space(Example()).with_choices(factor=1)
    earlier.query(Example.output)
    later = earlier.with_choices(extra=3)
    assert produced[0]() is not None
    del earlier
    gc.collect()
    assert produced[0]() is None
    assert later.query(Example.output) == Available(Payload(1))
    assert len(produced) == 2
    assert produced[1]() is not None


def test_concurrent_reads_evaluate_one_cached_output_per_snapshot() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Example(Space):
        source: int = Param()

        @derived(semantics=PAYLOAD)
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

        @derived(semantics=PAYLOAD)
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


def test_independent_root_bindings_remain_frozen_when_model_is_reused() -> None:
    calls: list[tuple[int, ...]] = []

    class Example(Space):
        values: list[int] = Param()

        @derived
        def total(*, values: list[int]) -> int:
            calls.append(tuple(values))
            return sum(values)

    original = [1, 2]
    first = design_space(Example(values=original))
    original.append(3)
    second = design_space(Example(values=original))
    original.clear()
    assert inspection.model(first) is inspection.model(second)
    assert first.total == 3
    assert second.total == 6
    assert first.total == 3
    assert calls == [(1, 2), (1, 2, 3)]
