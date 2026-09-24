# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
from weakref import ReferenceType, ref

from finn.kernels.space import Decision, Param, Space, ValueSemantics, compile_space, derived
from finn.kernels.space.results import Decided


@dataclass(frozen=True)
class Payload:
    value: int


PAYLOAD = ValueSemantics.immutable_nominal(Payload)


def test_discarded_candidate_snapshots_release_the_actual_cached_callback_outputs() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Family(Space):
        factor = Decision(int, values=range(16))

        @derived(semantics=PAYLOAD)
        def output(*, factor: int) -> Payload:
            payload = Payload(factor)
            produced.append(ref(payload))
            return payload

    model = compile_space(Family)
    base = model.start()

    def population() -> list[Family]:
        points = [base.assign(Family.factor, value) for value in range(16)]
        for point in points:
            point.answer(Family.output)
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
    assert model.start().refine().accepted


def test_successor_does_not_retain_its_predecessors_output_cache() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Family(Space):
        factor = Decision(int, values=(1, 2))
        extra = Decision(int, values=(3, 4))

        @derived(semantics=PAYLOAD)
        def output(*, factor: int) -> Payload:
            payload = Payload(factor)
            produced.append(ref(payload))
            return payload

    earlier = Family.start().assign(Family.factor, 1)
    earlier.answer(Family.output)
    later = earlier.assign(Family.extra, 3)
    assert produced[0]() is not None
    del earlier
    gc.collect()
    assert produced[0]() is None
    assert later.answer(Family.output) == Decided(Payload(1))
    assert len(produced) == 2
    assert produced[1]() is not None


def test_concurrent_reads_evaluate_one_cached_output_per_snapshot() -> None:
    produced: list[ReferenceType[Payload]] = []

    class Family(Space):
        source = Param(int)

        @derived(semantics=PAYLOAD)
        def output(*, source: int) -> Payload:
            payload = Payload(source)
            produced.append(ref(payload))
            return payload

    model = compile_space(Family)
    first = model.start({Family.source: 1})
    second = model.start({Family.source: 2})

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

    class Family(Space):
        factor = Decision(int, values=range(16))

        @derived(semantics=PAYLOAD)
        def output(*, factor: int) -> Payload:
            payload = Payload(factor)
            produced.append(ref(payload))
            return payload

    base = Family.start()

    def explore(value: int) -> Family:
        point = base.assign(Family.factor, value)
        assert point.output == Payload(value)
        return point

    with ThreadPoolExecutor(max_workers=8) as pool:
        candidates = list(pool.map(explore, range(16)))
    assert [point.output.value for point in candidates] == list(range(16))
    assert len(produced) == 16
    state = base.decision_state(Family.factor)
    assert isinstance(state, Decided)
    assert state.value.status == "unassigned"


def test_independent_root_bindings_remain_frozen_when_model_is_reused() -> None:
    calls: list[tuple[int, ...]] = []

    class Family(Space):
        values: Param[list[int]] = Param(list)

        @derived
        def total(*, values: list[int]) -> int:
            calls.append(tuple(values))
            return sum(values)

    model = compile_space(Family)
    original = [1, 2]
    first = model.start({Family.values: original})
    original.append(3)
    second = model.start({Family.values: original})
    original.clear()
    assert first.total == 3
    assert second.total == 6
    assert first.total == 3
    assert calls == [(1, 2), (1, 2, 3)]
