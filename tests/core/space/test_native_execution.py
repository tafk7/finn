# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Native failure completion, cancellation boundaries, and sticky controlled reads."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextvars import ContextVar
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

import pytest

from finn.core.space import (
    Decision,
    EvaluationError,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Subspace,
    Unresolved,
    ValueSemantics,
    ValueUnavailableError,
    constraint,
    derived,
    domain,
    view,
)
from finn.core.space import _execution, inspection, selections
from finn.core.space._execution import NativeEvaluationError, cancellation_details
from finn.core.space.occurrence import state

if TYPE_CHECKING:
    from _typeshed import TraceFunction


@pytest.fixture(autouse=True)
def native_context_is_reset() -> Iterator[None]:
    assert _execution.current() is None
    yield
    assert _execution.current() is None


@pytest.mark.parametrize("warm", (False, True))
def test_nested_finally_and_manager_finish_uncached_reads(warm: bool) -> None:
    events: list[tuple[str, int]] = []

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

    class Cleanup:
        def __init__(self, point: Family) -> None:
            self.point = point

        def __enter__(self) -> Cleanup:
            return self

        def __exit__(self, *error: object) -> None:
            events.append(("manager", self.point.cleanup_two))

    point = Family(fact=7)
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

    class Family(Space):
        fact = Param(int)
        enabled = Param(bool)
        unavailable = Decision(int, values=(1,))
        inapplicable = Decision(int, values=(1,), when=enabled)

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
                if kind == "rejected":
                    self.rejected()
                else:
                    _ = self.unavailable if kind == "unavailable" else self.inapplicable
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
            point.query(Family.unavailable if kind == "unavailable" else Family.inapplicable)
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
    assert isinstance(caught.value.__cause__, LookupError)
    assert [
        (issue.owner, type(issue.error.__cause__))
        for issue in caught.value.cleanup_failures
        if issue.error is not None
    ] == [("middle", ValueError), ("outer", ZeroDivisionError)]
    assert events == [7, 8]


def test_local_body_and_cleanup_exception_chain() -> None:
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
    secondary = caught.value.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)


@pytest.mark.parametrize("field", ("caught", "finally_return", "blocked"))
def test_caught_failure_and_nonvalue_cannot_publish_fallback(field: str) -> None:
    class Family(Space):
        choice = Decision(int, values=(1,))

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
        def blocked(self) -> int:
            try:
                return self.choice
            except ValueUnavailableError:
                return 99

    point = Family()
    if field == "blocked":
        assert isinstance(point.query(Family.blocked), Unresolved)
        assert point.with_choices(choice=1).blocked == 1
    else:
        with pytest.raises(NativeEvaluationError):
            getattr(point, field)
        index = state(point).model.linked.keys[field]
        assert index not in state(point).snapshot.cache


@pytest.mark.parametrize(
    "operation",
    (
        "query",
        "inspect",
        "state",
        "candidates",
        "change",
        "clear",
        "update",
        "capture",
        "restore",
        "explain",
        "assign",
        "delete",
        "foreign",
    ),
)
def test_driver_operations_remain_sticky_when_caught(operation: str) -> None:
    class Family(Space):
        fact = Param(int)
        choice = Decision(int, values=(1,))

        @view
        def output(self) -> int:
            return self.fact

        @derived
        def invalid(self) -> int:
            try:
                actions[operation](self)
            except (EvaluationError, AttributeError):
                return 99
            return 0

    foreign = Family(fact=8)
    saved = selections.capture(foreign)
    actions: dict[str, Callable[[Family], object]] = {
        "query": lambda point: point.query(Family.choice),
        "inspect": lambda point: point.output.inspect(),
        "state": lambda point: point.field(Family.choice).state,
        "candidates": lambda point: point.field(Family.choice).candidates(),
        "change": lambda point: point.field(Family.choice).change(1),
        "clear": lambda point: point.field(Family.choice).clear(),
        "update": lambda point: point.with_choices(choice=1),
        "capture": selections.capture,
        "restore": lambda point: selections.restore(point, saved),
        "explain": lambda point: inspection.explain(point, Family.choice),
        "assign": lambda point: setattr(point, "fact", 99),
        "delete": lambda point: delattr(point, "fact"),
        "foreign": lambda point: foreign.fact,
    }
    point = Family(fact=7)
    with pytest.raises(NativeEvaluationError, match="cross-snapshot|driver-only"):
        _ = point.invalid
    assert point.fact == 7 and isinstance(point.query(Family.choice), Unresolved)


def test_failed_prerequisite_rereads_are_query_local() -> None:
    starts: list[int] = []
    events: list[object] = []

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
    with pytest.raises(NativeEvaluationError):
        _ = point.broken
    assert starts == [1, 1]


def test_cleanup_cycle_keeps_primary_and_finishes_parent() -> None:
    events: list[int] = []

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
    assert cycle is not None and cycle.owner == "cycle" and cycle.role == "dependency cycle"
    assert events == [7]


@pytest.mark.parametrize("cancellation", (KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("warm", (False, True))
def test_cancellation_drains_and_retains_identity(
    cancellation: type[BaseException], warm: bool
) -> None:
    events: list[object] = []
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
    details = cancellation_details(caught.value)
    assert details is not None and details.primary.owner == "cancelled"
    assert details.primary.__cause__ is original and len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)
    assert events == ["caught cancellation", 7, 8] and point.fact == 7


def test_cancellation_cleanup_keeps_earlier_programmer_failure() -> None:
    original = KeyboardInterrupt("cleanup cancellation")

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
        "suspended": "dependency = cast(int, demanded)",
    }
    source, start = inspect.getsourcelines(_execution.run)
    target = next(start + offset for offset, line in enumerate(source) if markers[boundary] in line)
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

    point = Family(fact=7, cleanup=8)
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
        assert state(point).snapshot.work.callback_starts == 1
    else:
        assert events == [] and state(point).snapshot.work.callback_starts == 0
    assert point.fact == 7


def test_context_variables_survive_suspension_without_leaking() -> None:
    marker: ContextVar[str] = ContextVar("test_native_marker", default="driver")
    observed: list[str] = []

    class Family(Space):
        fact = Param(int)

        @derived
        def inner(self) -> int:
            observed.append(marker.get())
            marker.set("inner")
            return self.fact

        @derived
        def outer(self) -> int:
            token = marker.set("outer")
            try:
                value = self.inner
                observed.append(marker.get())
                return value
            finally:
                marker.reset(token)

    point = Family(fact=7)
    assert point.outer == 7 and observed == ["driver", "outer"]
    assert marker.get() == "driver"


def test_contextual_domain_error_is_one_primary_failure() -> None:
    def accepts(*, candidate: int) -> bool:
        raise LookupError("membership failure")

    class Family(Space):
        choice = Decision(int, domain=domain(accepts=accepts))

    with pytest.raises(NativeEvaluationError) as caught:
        Family().with_choices(choice=1)
    assert caught.value.owner == "choice" and caught.value.role == "domain membership"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()


def test_public_snapshot_failure_cannot_be_caught_into_success() -> None:
    armed = False

    def snapshot(value: list[int]) -> list[int]:
        if armed:
            raise LookupError("snapshot failure")
        return list(value)

    semantics = ValueSemantics(
        list, "list", lambda value: type(value) is list, lambda left, right: left == right, snapshot
    )

    class Family(Space):
        fact = Param(semantics)

        @derived
        def output(self) -> int:
            try:
                return len(self.fact)
            except EvaluationError:
                return 99

    point = Family(fact=[1])
    armed = True
    with pytest.raises(NativeEvaluationError) as caught:
        _ = point.output
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.primary.owner == "fact"
    assert caught.value.cleanup_failures == ()


@pytest.mark.parametrize("hook", ("recognition", "snapshot", "equality"))
def test_semantic_transformations_cannot_read_configuration_even_when_caught(hook: str) -> None:
    armed = False

    class Facts(Space):
        value = Param(int)

    facts = Facts(value=7)

    def probe(role: str) -> None:
        if armed and hook == role:
            try:
                _ = facts.value
            except EvaluationError:
                pass

    def recognizes(value: object) -> bool:
        probe("recognition")
        return type(value) is list

    def snapshot(value: list[int]) -> list[int]:
        probe("snapshot")
        return list(value)

    def equal(left: list[int], right: list[int]) -> bool:
        probe("equality")
        return left == right

    semantics = ValueSemantics(list, "list", recognizes, equal, snapshot)

    class Family(Space):
        choice = Decision(semantics, domain=domain(accepts=lambda candidate: True))

    point = Family().with_choices(choice=[1])
    armed = True
    with pytest.raises(EvaluationError, match="pure value transformation"):
        point.with_choices(choice=[1])
    armed = False
    assert point.choice == [1] and facts.value == 7


def test_blocked_self_constraint_keeps_its_inspectable_assessment() -> None:
    class Family(Space):
        choice = Decision(int, values=(1,))

        @constraint
        def positive(self) -> bool:
            return self.choice > 0

    assessment = Family().inspect(Family.positive)
    assert isinstance(assessment.result, Unresolved)
    assert isinstance(assessment.results["positive"], Unresolved)


def test_membership_only_domain_enumeration_keeps_applicability_and_blockers() -> None:
    class Family(Space):
        enabled = Param(bool)
        prerequisite = Decision(int, values=(1,))
        choice = Decision(
            int,
            domain=domain(accepts=lambda candidate, value: candidate == value, value=prerequisite),
            when=enabled,
        )

    assert isinstance(Family(enabled=False).field(Family.choice).candidates(), Inapplicable)
    point = Family(enabled=True)
    assert isinstance(point.field(Family.choice).candidates(), Unresolved)
    assert point.with_choices(prerequisite=1).field(Family.choice).candidates() is None


def test_self_cycles_are_reached_through_scopes_admission_and_guards() -> None:
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
        @derived
        def extent(self) -> int:
            return self.choice

        choice = Decision(
            int, domain=domain(accepts=lambda candidate, value: candidate == value, value=extent)
        )

    class Guard(Space):
        @derived
        def enabled(self) -> bool:
            return self.choice > 0

        choice = Decision(int, values=(1,), when=enabled)

    point = Parent()
    with pytest.raises(EvaluationError, match="dependency cycle") as scoped:
        _ = point.output
    assert "child.value" in str(scoped.value)
    for family in (Admission, Guard):
        with pytest.raises(EvaluationError, match="dependency cycle"):
            family().with_choices(choice=1)
