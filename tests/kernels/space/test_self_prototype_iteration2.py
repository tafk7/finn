# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Defensive equality and native failure-completion contract regressions."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import inspect
import sys
from weakref import ref

import greenlet
import pytest

from finn.kernels.space import _self_prototype as api
from finn.kernels.space._self_prototype import (
    Decision,
    Domain,
    Param,
    Space,
    ValueSemantics,
    constraint,
    derived,
    prepare,
    using_scheduler,
    view,
    NativeEvaluationError,
    CleanupFailure,
)
from finn.kernels.space.errors import EvaluationError
from finn.kernels.space.results import Available, Unresolved, Rejected, Inapplicable


@pytest.fixture
def native():
    with using_scheduler("greenlet"):
        yield
    assert api._ACTIVE.get() is None


@pytest.mark.parametrize("mode", ("replay", "greenlet"))
@pytest.mark.parametrize("operation", ("commit", "try_with_choices"))
@pytest.mark.parametrize("candidate", ([1], [2]))
@pytest.mark.parametrize("raises", (False, True))
def test_equality_receives_detached_arguments_and_contextualizes_errors(
    mode, operation, candidate, raises
):
    observed = []

    def equal(left, right):
        result = left == right
        left.append(91)
        right.append(92)
        observed.append((left, right))
        if raises:
            raise LookupError("equality adapter failure")
        return result

    semantics = ValueSemantics(list, "mutable bag", lambda value: type(value) is list, equal, list)

    class Family(Space):
        choice = Decision(semantics, domain=Domain(lambda self, candidate: True))

    original_candidate = list(candidate)
    with using_scheduler(mode):
        point = Family().with_choices(choice=[1])
        if raises:
            with pytest.raises(EvaluationError) as caught:
                getattr(point, operation)(choice=candidate)
            assert caught.value.owner == "choice" and caught.value.role == "configuration equality"
            assert isinstance(caught.value.__cause__, LookupError)
        else:
            report = getattr(point, operation)(choice=candidate)
            if original_candidate == [1]:
                assert report.accepted and report.instance is point
            elif operation == "commit":
                assert not report.accepted and report.instance is point
            else:
                assert report.accepted and report.instance.choice == [2]
        assert point.choice == [1]
        assert candidate == original_candidate
        assert observed


@pytest.mark.parametrize("warm", (False, True))
def test_nested_finally_and_context_manager_uncached_reads_finish(native, warm):
    events = []

    class Cleanup:
        def __init__(self, point):
            self.point = point

        def __enter__(self):
            return self

        def __exit__(self, *error):
            events.append(("manager", self.point.cleanup_two))
            return False

    class Family(Space):
        fact = Param(int)

        @derived
        def cleanup_one(self) -> int:
            return self.fact + 1

        @derived
        def cleanup_two(self) -> int:
            return self.cleanup_one + 1

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def middle(self) -> int:
            try:
                with Cleanup(self):
                    return self.broken
            finally:
                events.append(("middle", self.cleanup_two))

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(("outer", self.fact))

    point = Family(fact=7)
    if warm:
        assert point.cleanup_two == 9
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()
    assert events == [("manager", 9), ("middle", 9), ("outer", 7)]
    assert point.fact == 7 and api._ACTIVE.get() is None


@pytest.mark.parametrize("kind", ("unavailable", "inapplicable", "rejected"))
@pytest.mark.parametrize("warm", (False, True))
def test_nonvalue_cleanup_keeps_primary_and_outer_cleanup(native, kind, warm):
    events = []

    class Family(Space):
        fact = Param(int)
        enabled = Param(bool)
        unavailable = Decision(int, domain=Domain(lambda self, candidate: True))
        inapplicable = Decision(int, domain=Domain(lambda self, candidate: True), when=enabled)

        @constraint
        def refuses(self) -> bool:
            return False

        @view(constraints=(refuses,))
        def rejected(self) -> int:
            return 2

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def middle(self) -> int:
            try:
                return self.broken
            finally:
                events.append("middle started")
                value = getattr(self, kind)
                if kind == "rejected":
                    value()
                events.append("middle finished")

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact)

    point = Family(fact=7, enabled=False)
    if warm:
        if kind == "rejected":
            point.rejected.inspect()
        else:
            point.query(getattr(Family, kind))
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert isinstance(caught.value.__cause__, LookupError)
    issues = caught.value.cleanup_failures
    assert len(issues) == 1 and isinstance(issues[0], CleanupFailure)
    assert issues[0].owner == kind and issues[0].error is None
    assert isinstance(
        issues[0].result,
        {"unavailable": Unresolved, "inapplicable": Inapplicable, "rejected": Rejected}[kind],
    )
    assert events == ["middle started", 7]
    assert point.fact == 7


def test_secondary_exceptions_are_ordered_and_do_not_erase_primary(native):
    events = []

    class Family(Space):
        fact = Param(int)

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def middle(self) -> int:
            try:
                return self.broken
            finally:
                events.append(self.fact)
                raise ValueError("middle cleanup")

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact + 1)
                raise ZeroDivisionError("outer cleanup")

    point = Family(fact=7)
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.primary.__cause__, LookupError)
    assert [
        (issue.owner, type(issue.error.__cause__)) for issue in caught.value.cleanup_failures
    ] == [("middle", ValueError), ("outer", ZeroDivisionError)]
    assert events == [7, 8] and point.fact == 7


def test_local_body_and_cleanup_exception_context_is_preserved(native):
    class Family(Space):
        @derived
        def value(self) -> int:
            try:
                raise LookupError("body")
            finally:
                raise ValueError("cleanup")

    with pytest.raises(NativeEvaluationError) as caught:
        _ = Family().value
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    assert isinstance(caught.value.cleanup_failures[0].error.__cause__, ValueError)


@pytest.mark.parametrize("field", ("caught", "finally_return", "prohibited"))
def test_caught_failure_or_finally_return_cannot_publish_success(native, field):
    class Family(Space):
        fact = Param(int)

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def caught(self) -> int:
            try:
                return self.broken
            except EvaluationError:
                return 99

        @derived
        def finally_return(self) -> int:
            try:
                return self.broken
            finally:
                return 99

        @derived
        def prohibited(self) -> int:
            try:
                self.query(Family.fact)
            except EvaluationError:
                return 99
            return 1

    point = Family(fact=7)
    with pytest.raises(NativeEvaluationError):
        getattr(point, field)
    assert field not in point._snapshot.cache
    assert point.fact == 7 and api._ACTIVE.get() is None


def test_failed_prerequisite_rereads_are_query_local_and_do_not_restart(native):
    starts, events = [], []

    class Family(Space):
        fact = Param(int)

        @derived
        def broken(self) -> int:
            starts.append(1)
            raise LookupError("primary")

        @derived
        def output(self) -> int:
            try:
                return self.broken
            finally:
                for _ in range(3):
                    try:
                        _ = self.broken
                    except EvaluationError:
                        events.append("caught")
                events.append(self.fact)

    point = Family(fact=7)
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.output
    assert starts == [1] and events == ["caught", "caught", "caught", 7]
    assert caught.value.cleanup_failures == ()
    assert "broken" not in point._snapshot.cache
    with pytest.raises(NativeEvaluationError):
        _ = point.broken
    assert starts == [1, 1]  # failures do not remain in published caches
    assert point.fact == 7


def test_cycle_during_cleanup_keeps_primary_and_finishes_outer_parent(native):
    events = []

    class Family(Space):
        fact = Param(int)

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def cycle(self) -> int:
            return self.cycle

        @derived
        def middle(self) -> int:
            try:
                return self.broken
            finally:
                _ = self.cycle

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact)

    point = Family(fact=7)
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    cycle = caught.value.cleanup_failures[0].error
    assert cycle.owner == "cycle" and cycle.role == "dependency cycle"
    assert events == [7] and point.fact == 7


@dataclass(frozen=True)
class Payload:
    value: int


def test_native_continuations_contexts_trials_and_outputs_are_released(native, monkeypatch):
    snapshots, contexts, continuations, outputs = [], [], [], []
    original_snapshot, original_context, original_greenlet = (
        api._Snapshot,
        api._Context,
        greenlet.greenlet,
    )

    class Snapshot(original_snapshot):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            snapshots.append(ref(self))

    class Context(original_context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            contexts.append(ref(self))

    def continuation(*args, **kwargs):
        result = original_greenlet(*args, **kwargs)
        continuations.append(ref(result))
        return result

    monkeypatch.setattr(api, "_Snapshot", Snapshot)
    monkeypatch.setattr(api, "_Context", Context)
    monkeypatch.setattr(greenlet, "greenlet", continuation)

    def accepts(self, candidate):
        if candidate == 99:
            raise LookupError("domain failure")
        return candidate > 0

    class Family(Space):
        fact = Param(int)
        choice = Decision(int, domain=Domain(accepts))

        @derived
        def payload(self) -> Payload:
            return Payload(self.choice)

        @derived
        def cancelled(self) -> int:
            try:
                raise KeyboardInterrupt("control-flow cancellation")
            finally:
                _ = self.fact

        @derived
        def broken(self) -> int:
            try:
                raise LookupError("body failure")
            finally:
                _ = self.fact

    model = prepare(Family)
    surviving = model.bind(fact=1)

    def exercise():
        point = Family(fact=2).with_choices(choice=1)
        _ = point.payload
        answer = point._snapshot.cache["payload"].result
        assert isinstance(answer, Available)
        outputs.append(ref(answer.value))
        missing = Family(fact=3)
        assert isinstance(missing.query(Family.payload), Unresolved)
        assert missing.fact == 3
        failed = Family(fact=4)
        with pytest.raises(NativeEvaluationError):
            _ = failed.broken
        assert failed.fact == 4 and api._ACTIVE.get() is None
        with pytest.raises(KeyboardInterrupt):
            _ = failed.cancelled
        assert failed.fact == 4 and api._ACTIVE.get() is None
        refused = Family(fact=5).try_with_choices(choice=-1)
        assert not refused.accepted and refused.instance.fact == 5
        bad_trial = Family(fact=6)
        with pytest.raises(NativeEvaluationError):
            bad_trial.commit(choice=99)
        assert bad_trial.fact == 6 and api._ACTIVE.get() is None

    exercise()
    gc.collect()
    assert sum(item() is not None for item in snapshots) == 1
    assert all(item() is None for item in contexts)
    assert all(item() is None for item in continuations)
    assert all(item() is None for item in outputs)
    assert surviving.fact == 1 and prepare(Family) is model


@pytest.mark.parametrize("hook", ("recognition", "snapshot"))
@pytest.mark.parametrize("operation", ("commit", "try_with_choices"))
def test_request_adapter_errors_are_contextual_and_leave_receiver_intact(native, hook, operation):
    def recognizes(value):
        if hook == "recognition" and value == [2]:
            raise LookupError("recognition failure")
        return type(value) is list

    def snapshot(value):
        if hook == "snapshot" and value == [2]:
            raise LookupError("snapshot failure")
        return list(value)

    semantics = ValueSemantics(list, "bag", recognizes, lambda left, right: left == right, snapshot)

    class Family(Space):
        choice = Decision(semantics, domain=Domain(lambda self, candidate: True))

    point = Family().with_choices(choice=[1])
    with pytest.raises(EvaluationError) as caught:
        getattr(point, operation)(choice=[2])
    assert caught.value.owner == "choice"
    assert isinstance(caught.value.__cause__, LookupError)
    assert point.choice == [1]


@pytest.mark.parametrize("cancellation", (KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("warm", (False, True))
def test_control_flow_cancellation_drains_and_retains_original_type(native, cancellation, warm):
    events = []
    original = cancellation("cancel")

    class Family(Space):
        fact = Param(int)

        @derived
        def cancelled(self) -> int:
            raise original

        @derived
        def middle(self) -> int:
            try:
                return self.cancelled
            except cancellation:
                events.append("caught cancellation")
                return 99
            finally:
                events.append(self.fact)

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact + 1)
                raise ValueError("secondary cleanup")

    point = Family(fact=7)
    if warm:
        assert point.fact == 7
    with pytest.raises(cancellation) as caught:
        _ = point.outer
    assert caught.value is original
    details = api.cancellation_details(caught.value)
    assert details is not None and details.primary.owner == "cancelled"
    assert details.primary.__cause__ is original
    assert len(details.cleanup_failures) == 1
    assert isinstance(details.cleanup_failures[0].error.__cause__, ValueError)
    assert events == ["caught cancellation", 7, 8]
    assert "middle" not in point._snapshot.cache and point.fact == 7
    assert api._ACTIVE.get() is None


def test_cancellation_in_cleanup_keeps_the_earlier_programmer_failure(native):
    original = KeyboardInterrupt("cancel during cleanup")

    class Family(Space):
        @derived
        def broken(self) -> int:
            raise LookupError("first")

        @derived
        def cancelled(self) -> int:
            raise original

        @derived
        def output(self) -> int:
            try:
                return self.broken
            finally:
                _ = self.cancelled

    with pytest.raises(KeyboardInterrupt) as caught:
        _ = Family().output
    details = api.cancellation_details(caught.value)
    assert details is not None and isinstance(details.primary.__cause__, LookupError)
    assert len(details.cleanup_failures) == 1
    assert details.cleanup_failures[0].error.__cause__ is original


@pytest.mark.parametrize("boundary", ("allocated", "first-switch", "suspended"))
def test_driver_cancellation_respects_invocation_state(native, boundary):
    events = []
    original = KeyboardInterrupt("driver boundary")

    class Family(Space):
        fact = Param(int)
        cleanup = Param(int)

        @derived
        def output(self) -> int:
            events.append("entered")
            try:
                return self.fact
            finally:
                events.append(("cleanup", self.cleanup))

    markers = {
        "allocated": "task.continuation.gr_context = copy_context()",
        "first-switch": "demanded = continuation.switch(context, call)",
        "suspended": "dependency = cast(str, demanded)",
    }
    source, start = inspect.getsourcelines(api._native_evaluate)
    target = next(start + offset for offset, line in enumerate(source) if markers[boundary] in line)
    fired = False

    def trace(frame, event, _arg):
        nonlocal fired
        if (
            not fired
            and event == "line"
            and frame.f_code is api._native_evaluate.__code__
            and frame.f_lineno == target
        ):
            fired = True
            raise original
        return trace

    point = Family(fact=7, cleanup=8)
    prior = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            _ = point.output
    finally:
        sys.settrace(prior)
    assert fired and caught.value is original
    details = api.cancellation_details(caught.value)
    assert details is not None and details.cleanup_failures == ()
    if boundary == "suspended":
        assert events == ["entered", ("cleanup", 8)]
        assert point._snapshot.work.callback_starts == 1
    else:
        assert events == []
        assert point._snapshot.work.callback_starts == 0
    assert point.fact == 7 and api._ACTIVE.get() is None
