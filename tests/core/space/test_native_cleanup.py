# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Native failure cleanup and cancellation at every completion boundary."""

from __future__ import annotations

import sys
from types import FrameType
from typing import TYPE_CHECKING

import pytest
from core.space._native_support import source_line

from finn.core.space import (
    Decision,
    EvaluationError,
    Inapplicable,
    NativeEvaluationError,
    Param,
    Rejected,
    Space,
    Unresolved,
    _execution,
    cancellation_details,
    constraint,
    derived,
    design_space,
    view,
)
from finn.core.space.occurrence import state

if TYPE_CHECKING:
    from _typeshed import TraceFunction


@pytest.mark.parametrize("warm", (False, True))
def test_nested_finally_and_manager_finish_uncached_reads(warm: bool) -> None:
    events: list[tuple[str, int]] = []

    class Example(Space):
        fact: int = Param()

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

    class Cleanup:
        def __init__(self, point: Example) -> None:
            self.point = point

        def __enter__(self) -> Cleanup:
            return self

        def __exit__(self, *error: object) -> None:
            events.append(("manager", self.point.cleanup_two))

    point = design_space(Example(fact=7))
    if warm:
        assert point.cleanup_two == 9
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()
    assert events == [("manager", 9), ("middle", 9), ("outer", 7)]
    assert point.fact == 7


@pytest.mark.parametrize("kind", ("unavailable", "inapplicable", "rejected"))
@pytest.mark.parametrize("warm", (False, True))
def test_nonvalue_cleanup_preserves_primary_and_outer_cleanup(kind: str, warm: bool) -> None:
    events: list[object] = []

    class Example(Space):
        fact: int = Param()
        enabled: bool = Param()
        unavailable: int = Decision(values=(1, 2))
        inapplicable: int = Decision(values=(1,), when=enabled)

        @constraint
        def refuses(self) -> bool:
            return False

        @view(requires=(refuses,))
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
                if kind == "rejected":
                    _ = self.rejected
                else:
                    _ = self.unavailable if kind == "unavailable" else self.inapplicable
                events.append("middle finished")

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact)

    point = design_space(Example(fact=7, enabled=False))
    if warm:
        if kind == "rejected":
            point.inspect(Example.rejected)
        else:
            point.query(Example.unavailable if kind == "unavailable" else Example.inapplicable)
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert isinstance(caught.value.__cause__, LookupError)
    issues = caught.value.cleanup_failures
    assert len(issues) == 1 and issues[0].owner == kind and issues[0].error is None
    assert isinstance(
        issues[0].result,
        {"unavailable": Unresolved, "inapplicable": Inapplicable, "rejected": Rejected}[kind],
    )
    assert events == ["middle started", 7]


def test_secondary_exceptions_keep_order_and_primary() -> None:
    events: list[int] = []

    class Example(Space):
        fact: int = Param()

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

    point = design_space(Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert [
        (issue.owner, type(issue.error.__cause__))
        for issue in caught.value.cleanup_failures
        if issue.error is not None
    ] == [("middle", ValueError), ("outer", ZeroDivisionError)]
    assert events == [7, 8]


def test_local_body_and_cleanup_exception_chain() -> None:
    class Example(Space):
        @derived
        def value(self) -> int:
            try:
                raise LookupError("body")
            finally:
                raise ValueError("cleanup")

    with pytest.raises(NativeEvaluationError) as caught:
        _ = design_space(Example()).value
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    secondary = caught.value.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)


def test_failed_prerequisite_rereads_are_query_local() -> None:
    starts: list[int] = []
    events: list[object] = []

    class Example(Space):
        fact: int = Param()

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

    point = design_space(Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.output
    assert starts == [1] and events == ["caught", "caught", "caught", 7]
    assert caught.value.cleanup_failures == ()
    with pytest.raises(NativeEvaluationError):
        _ = point.broken
    assert starts == [1, 1]


def test_cleanup_cycle_keeps_primary_and_finishes_parent() -> None:
    events: list[int] = []

    class Example(Space):
        fact: int = Param()

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

    point = design_space(Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.outer
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    cycle = caught.value.cleanup_failures[0].error
    assert cycle is not None and cycle.owner == "cycle" and cycle.role == "dependency cycle"
    assert events == [7]


@pytest.mark.parametrize("cancellation", (KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("warm", (False, True))
def test_cancellation_drains_and_retains_identity(
    cancellation: type[BaseException], warm: bool
) -> None:
    events: list[object] = []
    original = cancellation("cancel")

    class Example(Space):
        fact: int = Param()

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

    point = design_space(Example(fact=7))
    if warm:
        assert point.fact == 7
    with pytest.raises(cancellation) as caught:
        _ = point.outer
    assert caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and details.primary.owner == "cancelled"
    assert details.primary.__cause__ is original and len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)
    assert events == ["caught cancellation", 7, 8] and point.fact == 7


def test_cancellation_cleanup_keeps_earlier_programmer_failure() -> None:
    original = KeyboardInterrupt("cleanup cancellation")

    class Example(Space):
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
        _ = design_space(Example()).output
    assert caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and isinstance(details.primary.__cause__, LookupError)
    assert len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and secondary.__cause__ is original


@pytest.mark.parametrize("boundary", ("allocated", "first-switch", "suspended"))
def test_driver_cancellation_respects_callback_start(boundary: str) -> None:
    events: list[object] = []
    original = KeyboardInterrupt("driver boundary")

    class Example(Space):
        fact: int = Param()
        cleanup: int = Param()

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
        "suspended": "dependency = cast(int, demanded)",
    }
    target = source_line(_execution.run, markers[boundary])
    fired = False

    def trace(frame: FrameType, event: str, argument: object) -> TraceFunction:
        nonlocal fired
        if (
            not fired
            and event == "line"
            and frame.f_code is _execution.run.__code__
            and frame.f_lineno == target
        ):
            fired = True
            raise original
        return trace

    point = design_space(Example(fact=7, cleanup=8))
    prior = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            _ = point.output
    finally:
        sys.settrace(prior)
    assert fired and caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and details.cleanup_failures == ()
    assert details.primary.role == "derived"
    if boundary == "suspended":
        assert events == ["entered", ("cleanup", 8)]
        assert state(point).work.callback_starts == 1
    else:
        assert events == [] and state(point).work.callback_starts == 0
    assert point.fact == 7


@pytest.mark.parametrize(
    "boundary",
    (
        "construct",
        "publish",
        "close",
        "retire",
        "active",
        "deliver",
        "returned",
    ),
)
@pytest.mark.parametrize("primary_failure", (False, True))
def test_interrupted_completion_drains_parent_and_preserves_failure(
    boundary: str,
    primary_failure: bool,
) -> None:
    events: list[object] = []
    original = KeyboardInterrupt("completion boundary")
    first = LookupError("original callback failure")

    class Example(Space):
        cleanup: int = Param()

        @derived
        def fact(self) -> int:
            events.append("fact")
            if primary_failure:
                raise first
            return 7

        @derived
        def middle(self) -> int:
            events.append("entered")
            try:
                return self.fact
            finally:
                events.append(("middle cleanup", self.cleanup))

        @derived
        def output(self) -> int:
            try:
                return self.middle
            finally:
                events.append(("outer cleanup", self.cleanup + 1))
                raise ValueError("outer cleanup failure")

    markers = {
        "construct": "if isinstance(outcome, _Failure):"
        if primary_failure
        else "outcome = _runtime.Evaluation(",
        "publish": "failures[task.identity] = outcome"
        if primary_failure
        else "snapshot.cache[task.identity] = outcome",
        "close": "task.frame.close()",
        "retire": "tasks.pop()",
        "active": "active.pop(task.identity)",
        "deliver": "waiter.incoming = (",
    }
    target = None if boundary == "returned" else source_line(_execution.run, markers[boundary])
    fired = False

    def trace(frame: FrameType, event: str, argument: object) -> TraceFunction:
        nonlocal fired
        task = frame.f_locals.get("task")
        at_boundary = (
            event == "return"
            if boundary == "returned"
            else event == "line" and frame.f_lineno == target
        )
        if (
            not fired
            and at_boundary
            and frame.f_code.co_name == "finish"
            and frame.f_code.co_filename == _execution.__file__
            and isinstance(task, _execution._Task)
            and task.context.key == "fact"
        ):
            fired = True
            raise original
        return trace

    point = design_space(Example(cleanup=8))
    prior = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            _ = point.output
    finally:
        sys.settrace(prior)
    assert fired and caught.value is original
    assert events == ["entered", "fact", ("middle cleanup", 8), ("outer cleanup", 9)]
    details = cancellation_details(caught.value)
    assert details is not None and details.primary.owner == "fact"
    if primary_failure:
        assert details.primary.__cause__ is first
        assert len(details.cleanup_failures) == 2
        interrupted = details.cleanup_failures[0].error
        assert interrupted is not None and interrupted.__cause__ is original
    else:
        assert details.primary.__cause__ is original
        assert len(details.cleanup_failures) == 1
    last = details.cleanup_failures[-1].error
    assert last is not None and isinstance(last.__cause__, ValueError)
    snapshot = state(point)
    for name in ("middle", "output"):
        assert state(point).model.linked.keys[name] not in snapshot.cache
    assert point.cleanup == 8 and _execution.current() is None


def test_interruption_before_failure_delivery_keeps_pending_primary() -> None:
    events: list[int] = []
    first = LookupError("provider failure")
    original = KeyboardInterrupt("before failure delivery")

    class Example(Space):
        cleanup: int = Param()

        @view
        def failed(self) -> int:
            raise first

        @derived
        def output(self) -> int:
            try:
                return self.failed
            finally:
                events.append(self.cleanup)

    target = source_line(_execution.run, "finish(task, _remember_failure(context, task.incoming))")
    fired = False

    def trace(frame: FrameType, event: str, argument: object) -> TraceFunction:
        nonlocal fired
        task = frame.f_locals.get("task")
        if (
            not fired
            and event == "line"
            and frame.f_code is _execution.run.__code__
            and frame.f_lineno == target
            and isinstance(task, _execution._Task)
            and task.context.snapshot.linked.nodes[task.context.index].kind == "view"
        ):
            fired = True
            raise original
        return trace

    point = design_space(Example(cleanup=8))
    prior = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt) as caught:
            _ = point.output
    finally:
        sys.settrace(prior)
    assert fired and caught.value is original and events == [8]
    details = cancellation_details(original)
    assert details is not None and details.primary.__cause__ is first
    assert len(details.cleanup_failures) == 1
    interrupted = details.cleanup_failures[0].error
    assert interrupted is not None and interrupted.__cause__ is original
