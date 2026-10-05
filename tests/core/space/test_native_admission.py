# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Production admission, demand traces, native work bounds, and lifetime regressions."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import permutations
from threading import Barrier
from typing import cast
from weakref import ReferenceType, ref

import pytest

from finn.core.space import (
    Available,
    Decision,
    EvaluationError,
    Param,
    Rejected,
    Space,
    ValueRef,
    ValueSemantics,
    _execution,
    _runtime,
    default_semantics,
    derived,
    design_space,
    divisors_of,
    domain,
    inspection,
)
from finn.core.space.occurrence import state


def _chain_space(reverse: bool) -> type[Space]:
    def plus_a(self: Space) -> int:
        return cast(int, getattr(self, "a")) + 1

    def plus_b(self: Space) -> int:
        return cast(int, getattr(self, "b")) + 1

    after_a, after_b = derived(plus_a), derived(plus_b)
    declarations = [
        ("a", Decision(values=(1,))),
        ("after_a", after_a),
        (
            "b",
            Decision(
                domain=domain(
                    accepts=lambda candidate, previous: candidate == previous, previous=after_a
                ),
            ),
        ),
        ("after_b", after_b),
        (
            "c",
            Decision(
                domain=domain(
                    accepts=lambda candidate, previous: candidate == previous, previous=after_b
                ),
            ),
        ),
    ]
    return cast(
        type[Space],
        type(
            "AdmissionChain",
            (Space,),
            {
                "__annotations__": {"a": int, "b": int, "c": int},
                **dict(reversed(declarations) if reverse else declarations),
            },
        ),
    )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("operation", ("keywords", "fields"))
def test_dependent_admission_all_request_and_declaration_orders(
    reverse: bool, operation: str
) -> None:
    space_type = _chain_space(reverse)
    for order in permutations(("a", "b", "c")):
        point = design_space(space_type())
        values = {"a": 1, "b": 2, "c": 3}
        for valid in (True, False):
            requested = {name: 99 if name == "a" and not valid else values[name] for name in order}
            if operation == "keywords":
                report = point.try_with_choices(**requested)
                accepted, instance, outcomes = report.accepted, report.instance, report.outcomes
            else:
                edits = [
                    point.field(cast(Decision[int], getattr(space_type, name))).change(value)
                    for name, value in requested.items()
                ]
                committed = point.try_with_choices(*edits)
                accepted, instance, outcomes = (
                    committed.accepted,
                    committed.instance,
                    committed.outcomes,
                )
            assert accepted is valid
            if valid:
                assert tuple(getattr(instance, name) for name in ("a", "b", "c")) == (1, 2, 3)
            else:
                assert instance is point
                assert all(isinstance(outcome.result, Rejected) for outcome in outcomes)


def test_invalid_candidate_never_crosses_a_getter_or_reaches_division() -> None:
    starts: list[int] = []
    divisions: list[int] = []

    class Example(Space):
        divisor: int = Decision(domain=domain(accepts=lambda candidate: candidate > 0))

        @derived
        def quotient(self) -> int:
            starts.append(1)
            value = self.divisor
            divisions.append(value)
            return 12 // value

        output: int = Decision(
            domain=domain(accepts=lambda candidate, value: candidate == value, value=quotient)
        )

    for order in (("output", "divisor"), ("divisor", "output")):
        report = design_space(Example()).try_with_choices(
            **{name: 0 if name == "divisor" else 3 for name in order}
        )
        assert not report.accepted
        assert all(isinstance(outcome.result, Rejected) for outcome in report.outcomes)
    assert len(starts) == 2 and divisions == []


def test_published_reads_do_not_repeat_admission_callbacks() -> None:
    calls: list[int] = []

    def accepts(*, candidate: int, extent: int) -> bool:
        calls.append(candidate)
        return 0 < candidate <= extent

    class Example(Space):
        extent: int = Param()
        choice: int = Decision(domain=domain(accepts=accepts, extent=extent))

        @derived
        def output(self) -> int:
            return self.choice + 1

    point = design_space(Example(extent=9)).with_choices(choice=3)
    assert calls == [3]
    for _ in range(3):
        assert point.choice == 3 and point.output == 4
        assert point.query(Example.choice) == Available(3)
        point.field(Example.choice).state
    assert calls == [3]


@pytest.mark.parametrize("raises", (False, True))
@pytest.mark.parametrize("operation", ("keywords", "fields"))
def test_equality_mutation_cannot_change_cached_or_caller_values(
    raises: bool, operation: str
) -> None:
    def equal(left: list[int], right: list[int]) -> bool:
        answer = left == right
        left.append(91)
        right.append(92)
        if raises:
            raise LookupError("equality failure")
        return answer

    semantics = ValueSemantics(list, "bag", lambda value: type(value) is list, equal, list)

    class Example(Space):
        choice: list[int] = Decision(
            domain=domain(accepts=lambda candidate: True), semantics=semantics
        )

    point = design_space(Example()).with_choices(choice=[1])
    candidate = [1]

    def update() -> object:
        if operation == "keywords":
            return point.try_with_choices(choice=candidate)
        return point.try_with_choices(point.field(Example.choice).change(candidate))

    if raises:
        with pytest.raises(EvaluationError) as caught:
            update()
        assert isinstance(caught.value.__cause__, LookupError)
    else:
        if operation == "keywords":
            assert point.try_with_choices(choice=candidate).instance is point
        else:
            assert (
                point.try_with_choices(point.field(Example.choice).change(candidate)).instance
                is point
            )
    assert point.choice == [1] and candidate == [1]


def test_finite_domain_equality_also_detaches_definition_values() -> None:
    def equal(left: list[int], right: list[int]) -> bool:
        answer = left == right
        left.append(91)
        right.append(92)
        return answer

    semantics = ValueSemantics(list, "bag", lambda value: type(value) is list, equal, list)

    class Example(Space):
        choice: list[int] = Decision(values=([1], [2]), semantics=semantics)

    for _ in range(2):
        point = design_space(Example()).with_choices(choice=[1])
        assert point.choice == [1]
        assert point.field(Example.choice).candidates() == Available(([1], [2]))


def test_observed_evidence_includes_cached_reads_and_omits_unselected_work() -> None:
    class Child(Space):
        extent: int = Param()
        lanes: int = Decision(domain=divisors_of(extent))

        @derived
        def cycles(self) -> int:
            return self.extent // self.lanes

    class Pair(Space):
        extent: int = Param()
        left = Child(extent=extent)
        right = Child(extent=extent)

        @derived
        def output(self) -> int:
            return max(self.left.cycles, self.right.cycles)

        @derived
        def unrelated(self) -> int:
            raise AssertionError("unreached")

    base = design_space(Pair(extent=12))
    point = base.with_choices(
        base.left.field(Child.lanes).change(3), base.right.field(Child.lanes).change(4)
    )
    assert point.left.cycles == 4
    evidence = inspection.explain(point, Pair.output)
    assert evidence.result == Available(4)
    labels = {node.declaration.reference: node.declaration.key for node in evidence.nodes}
    edges = {
        node.declaration.key: tuple(labels[handle] for handle in node.dependencies)
        for node in evidence.nodes
    }
    assert edges["output"] == ("left.cycles", "right.cycles")
    # left.extent forwards the root's extent: the read goes straight to it,
    # and the evidence names the formal it read through.
    assert edges["left.cycles"] == ("extent", "left.lanes")
    via = {node.declaration.key: [alias.key for alias in node.via] for node in evidence.nodes}
    assert via["left.cycles"] == ["left.extent"]
    assert "unrelated" not in edges
    starts = state(point).work.callback_starts
    assert inspection.explain(point, Pair.output) == evidence
    assert state(point).work.callback_starts == starts
    assert inspection.dependencies(point, Pair.output) == ()  # bodies are discovered when read


def declare(space_type: type[Space], bindings: Mapping[str, object]) -> Space:
    """Declare a node of a Space class built at runtime: its formals are not statically known."""
    node: Callable[..., Space] = space_type
    return node(**bindings)


def _deep_space(depth: int) -> tuple[type[Space], list[str]]:
    declarations: dict[str, object] = {"value_0": Param(semantics=default_semantics(int))}
    names = ["value_0"]

    def step(previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            return cast(int, getattr(self, previous)) + 1

        return compute

    for index in range(1, depth + 1):
        name = f"value_{index}"
        declarations[name] = derived(step(names[-1]))
        names.append(name)
    return cast(type[Space], type("DeepSelf", (Space,), declarations)), names


def test_native_chain_20000_has_one_start_per_callback() -> None:
    space_type, names = _deep_space(20_000)
    point = design_space(declare(space_type, {"value_0": 0}))
    assert getattr(point, names[-1]) == 20_000
    work = state(point).work
    assert work.callback_starts == 20_000
    assert work.getter_attempts == 20_001
    assert work.nodes_started == 20_001
    assert work.max_pending == 20_001
    assert getattr(point, names[-1]) == 20_000
    assert work.callback_starts == 20_000


def test_native_wide_fan_in_and_cached_prefix_deep_dependency() -> None:
    space_type, names = _deep_space(1_200)
    assert names[-1] == "value_1200"

    def output(self: Space) -> int:
        prefix = sum(cast(int, getattr(self, f"value_{index}")) for index in range(1, 401))
        return prefix + cast(int, getattr(self, "value_1200"))

    mixed = cast(type[Space], type("Mixed", (space_type,), {"output": derived(output)}))
    point = design_space(declare(mixed, {"value_0": 0}))
    assert getattr(point, "value_400") == 400
    before = state(point).work.callback_starts
    assert getattr(point, "output") == sum(range(1, 401)) + 1_200
    assert state(point).work.callback_starts - before == 801
    evidence = inspection.explain(point, cast(ValueRef[int], getattr(mixed, "output")))
    edges = next(node.dependencies for node in evidence.nodes if node.declaration.key == "output")
    assert len(edges) == 401

    leaves: dict[str, object] = {
        f"leaf_{index}": Param(semantics=default_semantics(int)) for index in range(1_000)
    }

    def total(self: Space) -> int:
        return sum(cast(int, getattr(self, f"leaf_{index}")) for index in range(1_000))

    leaves["total"] = derived(total)
    wide = cast(type[Space], type("WideSelf", (Space,), leaves))
    fan = design_space(declare(wide, {f"leaf_{index}": index for index in range(1_000)}))
    assert getattr(fan, "total") == sum(range(1_000))
    assert state(fan).work.callback_starts == 1
    assert state(fan).work.suspensions == 1_000


@dataclass(frozen=True)
class Payload:
    value: int


def test_contexts_continuations_trials_and_cached_outputs_are_reclaimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots: list[ReferenceType[_runtime.Snapshot]] = []
    contexts: list[ReferenceType[_execution.Context]] = []
    continuations: list[ReferenceType[_execution._Greenlet]] = []
    outputs: list[ReferenceType[Payload]] = []
    original_greenlet = _execution.greenlet.greenlet

    class ObservedSnapshot(_runtime.Snapshot):
        def __post_init__(self) -> None:
            super().__post_init__()
            snapshots.append(ref(self))

    class ObservedTrial(_runtime._TrialSnapshot):
        def __post_init__(self) -> None:
            super().__post_init__()
            snapshots.append(ref(self))

    class ObservedContext(_execution.Context):
        def __init__(
            self, snapshot: _runtime.Snapshot, index: int, native_parent: _execution._Greenlet
        ) -> None:
            super().__init__(snapshot, index, native_parent)
            contexts.append(ref(self))

    def continuation(
        function: Callable[..., object], *, parent: _execution._Greenlet
    ) -> _execution._Greenlet:
        result = original_greenlet(function, parent=parent)
        continuations.append(ref(result))
        return result

    monkeypatch.setattr(_runtime, "Snapshot", ObservedSnapshot)
    monkeypatch.setattr(_runtime, "_TrialSnapshot", ObservedTrial)
    monkeypatch.setattr(_execution, "Context", ObservedContext)
    monkeypatch.setattr(_execution.greenlet, "greenlet", continuation)

    def accepts(*, candidate: int) -> bool:
        if candidate == 99:
            raise LookupError("trial failure")
        return candidate > 0

    class Example(Space):
        fact: int = Param()
        choice: int = Decision(domain=domain(accepts=accepts))

        @derived
        def payload(self) -> Payload:
            return Payload(self.choice)

        @derived
        def cancelled(self) -> int:
            try:
                raise KeyboardInterrupt("cancel")
            finally:
                _ = self.fact

        @derived
        def broken(self) -> int:
            try:
                raise LookupError("failure")
            finally:
                _ = self.fact

    model = inspection.model(Example)
    survivor = design_space(Example(fact=1))

    def exercise() -> None:
        point = design_space(Example(fact=2)).with_choices(choice=1)
        assert point.payload == Payload(1)
        snapshot = state(point)
        answer = snapshot.cache[state(point).model.linked.keys["payload"]].result
        assert isinstance(answer, Available)
        outputs.append(ref(cast(Payload, answer.value)))
        successor = point.with_choices(choice=2)
        assert successor.payload == Payload(2)
        assert state(successor).parameters is snapshot.parameters
        assert state(successor).lock is not snapshot.lock
        failed = design_space(Example(fact=3))
        with pytest.raises(_execution.NativeEvaluationError):
            _ = failed.broken
        with pytest.raises(KeyboardInterrupt):
            _ = failed.cancelled
        assert failed.fact == 3
        assert not failed.try_with_choices(choice=-1).accepted
        with pytest.raises(_execution.NativeEvaluationError):
            failed.with_choices(choice=99)
        assert failed.fact == 3

    exercise()
    gc.collect()
    assert sum(item() is not None for item in snapshots) == 1
    assert all(item() is None for item in contexts)
    assert all(item() is None for item in continuations)
    assert all(item() is None for item in outputs)
    assert survivor.fact == 1 and inspection.model(survivor) is model
    assert inspection.model(Example) is model


def test_concurrent_independent_and_cold_shared_snapshots() -> None:
    barrier = Barrier(2)
    calls: list[int] = []

    class Independent(Space):
        fact: int = Param()

        @derived
        def output(self) -> int:
            value = self.fact
            barrier.wait(timeout=5)
            calls.append(value)
            return value

    first, second = design_space(Independent(fact=1)), design_space(Independent(fact=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda point: point.output, (first, second))) == [1, 2]
    assert sorted(calls) == [1, 2]

    class Shared(Space):
        fact: int = Param()

        @derived
        def output(self) -> int:
            calls.append(self.fact)
            return self.fact

    point = design_space(Shared(fact=3))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: point.output, range(64))) == [3] * 64
    assert sorted(calls) == [1, 2, 3]
    assert _execution.current() is None
