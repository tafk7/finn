# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
from itertools import permutations
from threading import Barrier
from typing import cast
from weakref import ref

import pytest

from finn.kernels.space._self_prototype import (
    Decision,
    Domain,
    Param,
    Space,
    Subspace,
    capture,
    constraint,
    derived,
    divisors_of,
    prepare,
    replay,
    using_scheduler,
    view,
)
from finn.kernels.space import _self_prototype as internal
from finn.kernels.space.errors import (
    DefinitionError,
    EvaluationError,
    RequestError,
    ValueUnavailableError,
)
from finn.kernels.space.results import Available, Inapplicable, Rejected, Unresolved
from finn.kernels.space.semantics import ValueSemantics


@pytest.fixture(params=("replay", "greenlet"))
def scheduler(request):
    if request.param == "greenlet":
        pytest.importorskip("greenlet")
    with using_scheduler(request.param):
        yield request.param


@dataclass(frozen=True)
class Shape:
    left: int
    right: int


class Tiles(Space):
    extent = Param(int)
    lanes = Decision(int, domain=divisors_of(extent))

    def quotient(self, value: int) -> int:
        return self.extent // value

    @derived
    def cycles(self) -> int:
        return self.quotient(self.lanes)


class InheritedTiles(Tiles):
    pass


class Pair(Space):
    extent = Param(int)
    left = Subspace(InheritedTiles, extent=extent)
    right = Subspace(Tiles, extent=extent)

    @derived
    def max_cycles(self) -> int:
        return max(self.left.cycles, self.right.cycles)

    @constraint
    def supported(self) -> bool:
        return self.max_cycles <= 4

    @view(constraints=(supported,))
    def shape(self) -> Shape:
        return Shape(self.left.cycles, self.right.cycles)


def paired() -> Pair:
    point = Pair(extent=12)
    return point.with_choices(
        point.left.field(Tiles.lanes).change(3), point.right.field(Tiles.lanes).change(4)
    )


def test_pair_inheritance_helper_plain_values_and_view(scheduler):
    point = paired()
    assert type(point.left) is InheritedTiles
    assert type(point.right) is Tiles
    assert type(point.max_cycles) is int and point.max_cycles == 4
    assert point.shape() == Shape(4, 3)
    assert point.shape.inspect().accepted_result == Available(Shape(4, 3))
    assert isinstance(point.left.query(Tiles.lanes), Available)
    assert point.left._scope != point.right._scope


def test_blocked_caught_unavailability_and_falsey_values(scheduler):
    completed = []

    class Family(Space):
        choice = Decision(int, domain=Domain(lambda self, candidate: candidate >= 0))
        maybe = Decision(type(None), domain=Domain(lambda self, candidate: True))
        flag = Decision(bool, domain=Domain(lambda self, candidate: True))

        @derived
        def fallback(self) -> int:
            try:
                return self.choice
            except ValueUnavailableError:
                completed.append("caught")
                return 99

    point = Family()
    assert isinstance(point.query(Family.fallback), Unresolved)
    with pytest.raises(ValueUnavailableError) as caught:
        _ = point.fallback
    assert isinstance(caught.value.result, Unresolved)
    chosen = point.with_choices(choice=0, maybe=None, flag=False)
    assert chosen.fallback == 0 and chosen.maybe is None and chosen.flag is False
    assert isinstance(point.query(Family.choice), Unresolved)


def test_branch_only_reads_selected_child_and_selector(scheduler):
    class Bomb(Space):
        @derived
        def value(self) -> int:
            raise AssertionError("unselected callback was reached")

    class Safe(Space):
        @derived
        def value(self) -> int:
            return 7

    class Family(Space):
        choose = Decision(bool, domain=Domain(lambda self, candidate: True))
        safe = Subspace(Safe)
        bomb = Subspace(Bomb)

        @derived
        def output(self) -> int:
            return self.safe.value if self.choose else self.bomb.value

    point = Family().with_choices(choose=True)
    assert point.output == 7
    assert point.explain(Family.output)["output"] == ("choose", "safe.value")
    assert "bomb.value" not in point.explain(Family.output)


def chain_family(reverse: bool):
    def plus_a(self) -> int:
        return self.a + 1

    def plus_b(self) -> int:
        return self.b + 1

    declarations = [
        ("a", Decision(int, domain=Domain(lambda self, candidate: candidate == 1))),
        ("after_a", derived(plus_a)),
        ("b", Decision(int, domain=Domain(lambda self, candidate: candidate == self.after_a))),
        ("after_b", derived(plus_b)),
        ("c", Decision(int, domain=Domain(lambda self, candidate: candidate == self.after_b))),
    ]
    return type(
        "AdmissionChain", (Space,), dict(reversed(declarations) if reverse else declarations)
    )


def test_dependent_admission_all_batch_and_declaration_orders(scheduler):
    for reverse in (False, True):
        family = chain_family(reverse)
        for order in permutations(("a", "b", "c")):
            point = family()
            values = {"a": 1, "b": 2, "c": 3}
            accepted = point.try_with_choices(**{name: values[name] for name in order})
            assert accepted.accepted
            assert (accepted.instance.a, accepted.instance.b, accepted.instance.c) == (1, 2, 3)
            refused = point.try_with_choices(
                **{name: 99 if name == "a" else values[name] for name in order}
            )
            assert not refused.accepted and refused.instance is point
            assert all(isinstance(answer, Rejected) for answer in refused.outcomes.values())


def test_invalid_prerequisite_never_crosses_getter_or_reaches_division(scheduler):
    starts, divisions = [], []

    class Family(Space):
        divisor = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))

        @derived
        def quotient(self) -> int:
            starts.append(1)
            divisor = self.divisor
            divisions.append(divisor)
            return 12 // divisor

        output = Decision(int, domain=Domain(lambda self, candidate: candidate == self.quotient))

    for order in (("output", "divisor"), ("divisor", "output")):
        report = Family().try_with_choices(
            **{name: 0 if name == "divisor" else 3 for name in order}
        )
        assert not report.accepted
        assert isinstance(report.outcomes["divisor"], Rejected)
        assert isinstance(report.outcomes["output"], Rejected)
    assert len(starts) == (3 if scheduler == "replay" else 2)
    assert divisions == []


def test_complete_request_validation_and_snapshot_before_callbacks(scheduler):
    calls = []
    payload = [1]

    def accept_first(self, candidate):
        payload.append(2)
        calls.append("first")
        return True

    def accept_second(self, candidate):
        calls.append(tuple(candidate))
        candidate.append(99)
        return True

    class Family(Space):
        first = Decision(int, domain=Domain(accept_first))
        second = Decision(list, domain=Domain(accept_second))

    base = Family()
    with pytest.raises(RequestError, match="nominal"):
        base.with_choices(first=1, second="bad")
    assert calls == []
    configured = base.with_choices(first=1, second=payload)
    assert calls == ["first", (1,)]
    assert configured.second == [1]


def test_root_child_replacement_revalidates_retained_inactive_and_atomic_clear(scheduler):
    class Child(Space):
        local = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))

    class Family(Space):
        facts = Param(list)
        enabled = Decision(bool, domain=Domain(lambda self, candidate: True))
        amount = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))
        sibling = Decision(
            int, domain=Domain(lambda self, candidate: candidate <= self.amount), when=enabled
        )
        child = Subspace(Child)

    facts = [1]
    original = Family(facts=facts).with_choices(enabled=True, amount=4, sibling=3)
    facts.append(2)
    child = original.child.with_choices(local=1)
    refused = child.try_with_choices(child.root.field(Family.amount).change(2))
    assert not refused.accepted and refused.instance is child
    assert isinstance(refused.outcomes["sibling"], Rejected)
    inactive = child.try_with_choices(child.root.field(Family.enabled).change(False))
    assert not inactive.accepted and isinstance(inactive.outcomes["sibling"], Inapplicable)
    updated = child.with_choices(
        child.root.field(Family.enabled).change(False), child.root.field(Family.sibling).clear()
    )
    assert type(updated) is Child and updated.local == 1
    assert updated.root.facts == [1] and original.amount == 4
    assert updated._snapshot.parameters is child._snapshot.parameters


def test_reached_direct_indirect_parent_and_admission_cycles(scheduler):
    class Direct(Space):
        @derived
        def cycle(self) -> int:
            return self.cycle

    class Indirect(Space):
        @derived
        def one(self) -> int:
            return self.two

        @derived
        def two(self) -> int:
            return self.one

    class Child(Space):
        source = Param(int)

        @derived
        def value(self) -> int:
            return self.source

    class Parent(Space):
        @derived
        def output(self) -> int:
            return self.child.value

        child = Subspace(Child, source=output)

    class Admission(Space):
        choice = Decision(int, domain=Domain(lambda self, candidate: self.choice == candidate))

    class Applicability(Space):
        @derived
        def enabled(self) -> bool:
            return self.choice > 0

        choice = Decision(int, domain=Domain(lambda self, candidate: True), when=enabled)

    for family, field in ((Direct, "cycle"), (Indirect, "one"), (Parent, "output")):
        point = family()  # Preparation deliberately does not execute these bodies.
        with pytest.raises(EvaluationError, match="dependency cycle") as caught:
            getattr(point, field)
        assert field in str(caught.value)
    for family, role in ((Admission, "admission/domain"), (Applicability, "applicability")):
        with pytest.raises(EvaluationError, match="dependency cycle") as caught:
            family().with_choices(choice=1)
        assert role in str(caught.value)


def test_unselected_cycle_is_lazy(scheduler):
    class Family(Space):
        choice = Decision(bool, domain=Domain(lambda self, candidate: True))

        @derived
        def cycle(self) -> int:
            return self.cycle

        @derived
        def value(self) -> int:
            return 1 if self.choice else self.cycle

    assert Family().with_choices(choice=True).value == 1
    with pytest.raises(EvaluationError, match="cycle"):
        _ = Family().with_choices(choice=False).value


def test_monotone_sequences_and_status_mutation_escape_refusal(scheduler):
    class Family(Space):
        a = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))
        b = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))

        @derived
        def stable(self) -> int:
            return self.a * 2

        @derived
        def inspect_escape(self) -> int:
            try:
                self.query(Family.b)
            except EvaluationError:
                return 99
            return 1

        @derived
        def mutate_escape(self) -> int:
            self.with_choices(b=2)
            return 0

    base = Family()
    first = base.commit(a=1).instance
    assert first.stable == 2
    second = first.commit(b=3).instance
    assert second.stable == first.stable
    assert second.explain(Family.stable) == first.explain(Family.stable)
    assert first.commit(a=1).instance is first
    assert not first.commit(a=2).accepted
    for reference in (Family.inspect_escape, Family.mutate_escape):
        with pytest.raises(EvaluationError, match="driver-only"):
            first.query(reference)
    with pytest.raises(AttributeError, match="immutable"):
        first.a = 7


def test_view_raw_rejected_unresolved_inapplicable_and_programmer_failure(scheduler):
    refused = paired().left.with_choices(lanes=1).root
    assert isinstance(refused, Pair)
    assessment = refused.shape.inspect()
    assert assessment.output_result == Available(Shape(12, 3))
    assert isinstance(assessment.accepted_result, Rejected)
    with pytest.raises(ValueUnavailableError):
        refused.shape()
    assert isinstance(Pair(extent=12).shape.inspect().accepted_result, Unresolved)

    class Family(Space):
        enabled = Param(bool)

        @view(when=enabled)
        def output(self) -> int:
            raise AssertionError("inactive view ran")

        @view()
        def broken(self) -> int:
            raise LookupError("original callback failure")

    assert isinstance(Family(enabled=False).output.inspect().accepted_result, Inapplicable)
    with pytest.raises(EvaluationError) as caught:
        Family(enabled=False).broken.inspect()
    assert isinstance(caught.value.__cause__, LookupError)


def test_cold_cached_evidence_and_cross_snapshot_read(scheduler):
    point = paired()
    cold = point.explain(Pair.max_cycles)
    calls = point._snapshot.work.callback_starts
    assert point.explain(Pair.max_cycles) == cold
    assert point._snapshot.work.callback_starts == calls
    revised = point.left.with_choices(lanes=2).root
    assert isinstance(revised, Pair) and revised.max_cycles == 6 and point.max_cycles == 4
    foreign = paired()

    class Family(Space):
        @derived
        def bad(self) -> int:
            try:
                return foreign.max_cycles
            except EvaluationError:
                return 100

    with pytest.raises(EvaluationError, match="cross-snapshot"):
        _ = Family().bad


def test_outputs_predecessors_and_trial_reclaimed(monkeypatch, scheduler):
    trials = []
    original = internal._Snapshot

    class ObservedSnapshot(original):
        def __post_init__(self):
            pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trials.append(ref(self))

    monkeypatch.setattr(internal, "_Snapshot", ObservedSnapshot)
    prepared = prepare(Pair)
    point = paired()
    point.shape()
    snapshot_ref = ref(point._snapshot)
    output = point._snapshot.cache["shape"].result
    assert isinstance(output, Available)
    output_ref = ref(output.value)
    del output
    successor = point.left.with_choices(lanes=2).root
    refused = point.left.try_with_choices(lanes=5)
    assert not refused.accepted
    del refused
    assert len(trials) >= 6
    gc.collect()
    assert sum(item() is not None for item in trials) == 2
    del point
    gc.collect()
    assert snapshot_ref() is None and output_ref() is None
    assert sum(item() is not None for item in trials) == 1
    assert prepared is prepare(Pair) and successor.extent == 12


def test_concurrent_independent_and_same_snapshot_reads(scheduler):
    barrier = Barrier(2)
    calls = []

    class Family(Space):
        source = Param(int)

        @derived
        def output(self) -> int:
            # One reached callback per thread; replay would enter twice after the read.
            value = self.source
            barrier.wait(timeout=5)
            calls.append(value)
            return value

    first, second = Family(source=1), Family(source=2)

    def read(point):
        with using_scheduler(scheduler):
            return point.output

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(read, (first, second))) == [1, 2]
    assert sorted(calls) == [1, 2]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, (first,) * 16)) == [1] * 16
    assert sorted(calls) == [1, 2]


def test_sparse_capture_replay(scheduler):
    point = paired()
    saved = capture(point)
    assert saved == (("left.lanes", 3), ("right.lanes", 4))
    restored = replay(Pair(extent=12), saved)
    assert restored.accepted and restored.instance.shape() == point.shape()
    assert restored.instance._snapshot is not point._snapshot


def test_prepare_rejects_known_structure_without_callbacks(scheduler):
    invoked = []

    class Other(Space):
        value = Param(int)

    class Broken(Space):
        child = Subspace(Tiles, extent=Other.value)

        @derived
        def output(self) -> int:
            invoked.append(1)
            return 1

    with pytest.raises(DefinitionError, match="foreign"):
        prepare(Broken)
    assert invoked == []


def test_greenlet_unwinds_suspended_finally_on_failure():
    pytest.importorskip("greenlet")
    cleaned = []

    class Child(Space):
        @derived
        def broken(self) -> int:
            raise LookupError("failure below suspended parent")

    class Family(Space):
        child = Subspace(Child)

        @derived
        def output(self) -> int:
            try:
                return self.child.broken
            finally:
                cleaned.append("parent cleanup")

    with using_scheduler("greenlet"), pytest.raises(EvaluationError) as caught:
        _ = Family().output
    assert isinstance(caught.value.__cause__, LookupError)
    assert cleaned == ["parent cleanup"] and internal._ACTIVE.get() is None


def test_public_reports_and_view_readiness_detach_mutable_values(scheduler):
    class Family(Space):
        choice = Decision(list, domain=Domain(lambda self, candidate: True))

        @view()
        def output(self) -> list:
            return self.choice

    report = Family().try_with_choices(choice=[1])
    answer = report.outcomes["choice"]
    assert isinstance(answer, Available)
    answer.value.append(2)
    assert report.instance.choice == [1]
    assessment = report.instance.output.inspect()
    raw = assessment.output_result
    ready = assessment.readiness.results["output"]
    accepted = assessment.accepted_result
    assert (
        isinstance(raw, Available)
        and isinstance(ready, Available)
        and isinstance(accepted, Available)
    )
    raw.value.append(3)
    ready.value.append(4)
    accepted.value.append(5)
    assert report.instance.output() == [1]


def test_equal_custom_semantics_recommit_is_noop(scheduler):
    bag = ValueSemantics(
        list,
        "bag",
        lambda value: type(value) is list,
        lambda left, right: sorted(left) == sorted(right),
        list,
    )

    class Family(Space):
        choice = Decision(bag, domain=Domain(lambda self, candidate: True))

    point = Family().with_choices(choice=[1, 2])
    assert point.commit(choice=[2, 1]).instance is point


def test_cold_same_snapshot_concurrency_and_cached_dependency_trace(scheduler):
    calls = []

    class Family(Space):
        fact = Param(int)

        @derived
        def output(self) -> int:
            value = self.fact
            calls.append(value)
            return value

    point = Family(fact=1)

    def read(_):
        with using_scheduler(scheduler):
            return point.output

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, range(64))) == [1] * 64
    assert calls == [1]
    pair = paired()
    _ = pair.left.cycles  # Existing cached prerequisite must still appear in later reads.
    evidence = pair.explain(Pair.max_cycles)
    assert evidence["max_cycles"] == ("left.cycles", "right.cycles")
    assert evidence["left.cycles"] == ("left.lanes", "left.extent")


def test_prepared_root_child_and_inherited_definitions_are_frozen(scheduler):
    point = paired()
    with pytest.raises(DefinitionError, match="finalized"):
        Pair.extent = Param(int)
    with pytest.raises(DefinitionError, match="finalized"):
        Tiles.cycles.function = lambda self: 99
    with pytest.raises(DefinitionError, match="finalized"):
        InheritedTiles.helper = lambda self: 99
    assert point.max_cycles == 4


def test_prepare_rejects_callback_shape_guard_types_and_literal_types(scheduler):
    class KeywordSelf(Space):
        @derived
        def output(*, self) -> int:
            raise AssertionError("preparation executed callback")

    class BadDomain(Space):
        choice = Decision(int, domain=Domain(lambda candidate: True))

    class BadGuard(Space):
        flag = Param(int)
        choice = Decision(
            int, domain=Domain(lambda self, candidate: True), when=cast(internal.Value[bool], flag)
        )

    class BadLiteral(Space):
        child = Subspace(Tiles, extent="bad")

    for family in (KeywordSelf, BadDomain, BadGuard, BadLiteral):
        with pytest.raises(DefinitionError):
            prepare(family)
