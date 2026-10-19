# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The failure contract on the evaluation core (``finn.core.space._world``, ``_failure``).

Each test mirrors a public test that pins the contract (named in its docstring), on a
world's entries. An authored function raising is its owner's error, never cached; a
failure reaches a reading method through its read, so its cleanup runs and may read; the
first failure is primary and sticky; later ones are cleanup failures, in order; a
cancellation keeps its identity. Where an interruption lands is one of the core's own
points: before a call starts, a read inside a method, publishing an entry, recording a
failure, delivering one.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from types import FrameType
from typing import TYPE_CHECKING

import pytest
from support.refusals import refused

from core.space._world_support import node, open_world, source_line, successor
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    ValueUnavailableError,
    _failure,
    _world,
    constraint,
    derived,
    domain,
    view,
)
from finn.core.space._failure import NativeEvaluationError, cancellation_details
from finn.core.space._world import World
from finn.core.space.results import ConstraintAssessment

if TYPE_CHECKING:
    from _typeshed import TraceFunction


@pytest.fixture(autouse=True)
def no_call_is_left_active() -> object:
    assert _failure.current() is None
    yield None
    assert _failure.current() is None


def value(world: World, key: str) -> object:
    result = world.result(node(world, key))
    assert isinstance(result, Available), result
    return result.value


def read(world: World, key: str) -> None:
    world.entry(node(world, key))


@pytest.mark.parametrize("warm", (False, True))
def test_nested_finally_and_manager_finish_uncached_reads(warm: bool) -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name."""
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

    world = open_world(Example(fact=7))
    if warm:
        assert value(world, "cleanup_two") == 9
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "outer")
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()
    assert events == [("manager", 9), ("middle", 9), ("outer", 7)]
    assert value(world, "fact") == 7
    assert node(world, "outer") not in world.cache and node(world, "broken") not in world.cache


@pytest.mark.parametrize("kind", ("unavailable", "inapplicable", "rejected"))
@pytest.mark.parametrize("warm", (False, True))
def test_nonvalue_cleanup_preserves_primary_and_outer_cleanup(kind: str, warm: bool) -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name."""
    events: list[object] = []

    class Example(Space):
        fact: int = Param()
        enabled: bool = Param()
        unavailable: int = Decision(values=(1, 2))
        inapplicable: int = Decision(values=(1,), when=enabled)

        @constraint
        def refuses(self) -> bool:
            return False

        @view
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

    world = open_world(Example(fact=7, enabled=False))
    if warm:
        read(world, kind)
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "outer")
    assert isinstance(caught.value.__cause__, LookupError)
    issues = caught.value.cleanup_failures
    assert len(issues) == 1 and issues[0].owner == kind and issues[0].error is None
    assert isinstance(
        issues[0].result,
        {"unavailable": Unresolved, "inapplicable": Inapplicable, "rejected": Rejected}[kind],
    )
    assert events == ["middle started", 7]


def test_secondary_exceptions_keep_order_and_primary() -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name."""
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

    world = open_world(Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "outer")
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert [
        (issue.owner, type(issue.error.__cause__))
        for issue in caught.value.cleanup_failures
        if issue.error is not None
    ] == [("middle", ValueError), ("outer", ZeroDivisionError)]
    assert events == [7, 8]


def test_local_body_and_cleanup_exception_chain() -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name."""

    class Example(Space):
        @derived
        def value(self) -> int:
            try:
                raise LookupError("body")
            finally:
                raise ValueError("cleanup")

    with pytest.raises(NativeEvaluationError) as caught:
        read(open_world(Example()), "value")
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    secondary = caught.value.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)


def test_failed_prerequisite_rereads_are_query_local() -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name: within one demand a
    failed node is not run again; the next demand runs it again."""
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

    world = open_world(Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "output")
    assert starts == [1] and events == ["caught", "caught", "caught", 7]
    assert caught.value.cleanup_failures == ()
    with pytest.raises(NativeEvaluationError):
        read(world, "broken")
    assert starts == [1, 1]


def test_cleanup_cycle_keeps_primary_and_finishes_parent() -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name."""
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

    world = open_world(Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "outer")
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
    """Mirrors ``test_native_cleanup``'s test of the same name."""
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

    world = open_world(Example(fact=7))
    if warm:
        assert value(world, "fact") == 7
    with pytest.raises(cancellation) as caught:
        read(world, "outer")
    assert caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and details.primary.owner == "cancelled"
    assert details.primary.__cause__ is original and len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)
    assert events == ["caught cancellation", 7, 8] and value(world, "fact") == 7


def test_cancellation_cleanup_keeps_earlier_programmer_failure() -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name."""
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
        read(open_world(Example()), "output")
    assert caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and isinstance(details.primary.__cause__, LookupError)
    assert len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and secondary.__cause__ is original


def _interrupt_at(code: object, line: int, original: BaseException, when: object = None) -> object:
    """A trace function raising ``original`` once, at ``line`` of ``code``."""
    fired: list[bool] = []

    def trace(frame: FrameType, event: str, argument: object) -> TraceFunction:
        if (
            not fired
            and event == "line"
            and frame.f_code is code
            and frame.f_lineno == line
            and (when is None or when(frame))  # type: ignore[operator]
        ):
            fired.append(True)
            raise original
        return trace

    trace.fired = fired  # type: ignore[attr-defined]
    return trace


def _traced(trace: object, run: object) -> None:
    prior = sys.gettrace()
    try:
        sys.settrace(trace)  # type: ignore[arg-type]
        run()  # type: ignore[operator]
    finally:
        sys.settrace(prior)


@pytest.mark.parametrize("boundary", ("before-call", "suspended-read"))
def test_driver_cancellation_respects_callback_start(boundary: str) -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name, at the core's points: an
    interruption before the call starts (the dispatcher's "allocated" and "first-switch")
    never enters it; one inside a read of the running method (its "suspended") runs the
    method's cleanup."""
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

    world = open_world(Example(fact=7, cleanup=8))
    if boundary == "before-call":
        function = _world._authored
        line = source_line(function, "outcome = run.context.copy().run(")
    else:
        function = _world.MethodReader._take  # type: ignore[assignment]
        line = source_line(function, "entry = self.world._get(index)")
    trace = _interrupt_at(function.__code__, line, original)
    with pytest.raises(KeyboardInterrupt) as caught:
        _traced(trace, lambda: read(world, "output"))
    assert trace.fired and caught.value is original  # type: ignore[attr-defined]
    details = cancellation_details(caught.value)
    assert details is not None and details.cleanup_failures == ()
    assert details.primary.role == "derived" and details.primary.owner == "output"
    calls = world.evaluator.counters.calls.get("derived", 0)
    if boundary == "suspended-read":
        assert events == ["entered", ("cleanup", 8)] and calls == 1
    else:
        assert events == [] and calls == 1  # counted, never entered
    assert value(world, "fact") == 7


@pytest.mark.parametrize(("boundary", "primary_failure"), (("publish", False), ("record", True)))
def test_interrupted_completion_drains_parent_and_preserves_failure(
    boundary: str, primary_failure: bool
) -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name, at the core's points (the
    dispatcher's completion lines): an evaluation's entry published (``_get``'s cache),
    or, where it failed, its failure recorded for the demand."""
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

    world = open_world(Example(cleanup=8))
    fact = node(world, "fact")
    marker = (
        "return self.cache.setdefault(node, entry)"
        if boundary == "publish"
        else "run.failures[key] = failing.failure"
    )
    line = source_line(World._get, marker)
    trace = _interrupt_at(
        World._get.__code__, line, original, lambda frame: frame.f_locals.get("node") == fact
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        _traced(trace, lambda: read(world, "output"))
    assert trace.fired and caught.value is original  # type: ignore[attr-defined]
    assert events == ["entered", "fact", ("middle cleanup", 8), ("outer cleanup", 9)]
    details = cancellation_details(caught.value)
    assert details is not None
    if primary_failure:
        assert details.primary.owner == "fact" and details.primary.__cause__ is first
        assert len(details.cleanup_failures) == 2
        interrupted = details.cleanup_failures[0].error
        assert interrupted is not None and interrupted.__cause__ is original
    else:
        assert details.primary.owner == "fact" and details.primary.__cause__ is original
        assert len(details.cleanup_failures) == 1
    last = details.cleanup_failures[-1].error
    assert last is not None and isinstance(last.__cause__, ValueError)
    for key in ("fact", "middle", "output"):
        assert node(world, key) not in world.cache
    assert value(world, "cleanup") == 8


def test_interruption_before_failure_delivery_keeps_pending_primary() -> None:
    """Mirrors ``test_native_cleanup``'s test of the same name: interrupted as a read
    delivers a failure to its method, the failure stays primary."""
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

    world = open_world(Example(cleanup=8))
    line = source_line(_world.MethodReader._take, "remember_failure(self.call, failing.failure)")
    trace = _interrupt_at(_world.MethodReader._take.__code__, line, original)
    with pytest.raises(KeyboardInterrupt) as caught:
        _traced(trace, lambda: read(world, "output"))
    assert trace.fired and caught.value is original and events == [8]  # type: ignore[attr-defined]
    details = cancellation_details(original)
    assert details is not None and details.primary.__cause__ is first
    assert len(details.cleanup_failures) == 1
    interrupted = details.cleanup_failures[0].error
    assert interrupted is not None and interrupted.__cause__ is original


@pytest.mark.parametrize("field", ("caught", "finally_return", "blocked"))
def test_caught_failure_and_nonvalue_cannot_publish_fallback(field: str) -> None:
    """Mirrors ``test_native_execution``'s test of the same name."""

    class Example(Space):
        choice: int = Decision(values=(1, 2))

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
                return 99  # noqa: B012 - the case under test

        @derived
        def blocked(self) -> int:
            try:
                return self.choice
            except ValueUnavailableError:
                return 99

    world = open_world(Example())
    if field == "blocked":
        assert isinstance(world.result(node(world, "blocked")), Unresolved)
        assert value(successor(world, {"choice": 1}), "blocked") == 1
    else:
        with pytest.raises(NativeEvaluationError):
            read(world, field)
        assert node(world, field) not in world.cache


def test_driver_only_and_foreign_reads_remain_sticky_when_caught() -> None:
    """Mirrors ``test_native_execution``'s ``test_driver_operations_remain_sticky_when_caught``
    for what the core owns: its driver-only guard, and a read through a receiver of
    another world. (The public driver APIs reach the core's guard once they run on it.)"""
    receivers: list[Space] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def kept(self) -> int:
            receivers.append(self)
            return self.fact

        @derived
        def driver(self) -> int:
            try:
                _failure.driver_only("query inspection")
            except EvaluationError:
                return 99
            return 0

        @derived
        def foreign(self) -> int:
            try:
                return int(getattr(receivers[0], "fact"))
            except EvaluationError:
                return 99

    other = open_world(Example(fact=8))
    assert value(other, "kept") == 8
    world = open_world(Example(fact=7))
    with pytest.raises(NativeEvaluationError, match="driver-only"):
        read(world, "driver")
    with pytest.raises(NativeEvaluationError, match="cross-configuration"):
        read(world, "foreign")
    assert value(world, "fact") == 7
    with pytest.raises(EvaluationError, match="after it returned"):
        _ = getattr(receivers[0], "fact")


def test_context_variables_survive_suspension_without_leaking() -> None:
    """Mirrors ``test_native_execution``'s test of the same name: each call runs in a copy
    of its driver's context, never in its reader's."""
    marker: ContextVar[str] = ContextVar("test_core_marker", default="driver")
    observed: list[str] = []

    class Example(Space):
        fact: int = Param()

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

    world = open_world(Example(fact=7))
    assert value(world, "outer") == 7 and observed == ["driver", "outer"]
    assert marker.get() == "driver"


def test_contextual_domain_error_is_one_primary_failure() -> None:
    """Mirrors ``test_native_execution``'s test of the same name: a stored value's type
    check whose membership raises."""

    def accepts(*, candidate: int) -> bool:
        raise LookupError("membership failure")

    class Example(Space):
        choice: int = Decision(domain=domain(accepts=accepts))

    world = open_world(Example(), {"choice": 1})
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "choice")
    assert caught.value.owner == "choice" and caught.value.role == "domain membership"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()


def test_public_snapshot_failure_cannot_be_caught_into_success() -> None:
    """Mirrors ``test_native_execution``'s test of the same name."""
    armed = False

    def snapshot(value: list[int]) -> list[int]:
        if armed:
            raise LookupError("snapshot failure")
        return list(value)

    semantics = ValueSemantics(
        list, "list", lambda value: type(value) is list, lambda left, right: left == right, snapshot
    )

    class Example(Space):
        fact: list[int] = Param(semantics=semantics)

        @derived
        def output(self) -> int:
            try:
                return len(self.fact)
            except EvaluationError:
                return 99

    world = open_world(Example(fact=[1]))
    armed = True
    with pytest.raises(NativeEvaluationError) as caught:
        read(world, "output")
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.primary.owner == "fact"
    assert caught.value.cleanup_failures == ()


def test_value_transformations_cannot_read_a_world_even_when_caught() -> None:
    """Mirrors ``test_native_execution``'s
    ``test_semantic_transformations_cannot_read_configuration_even_when_caught`` for the
    core's guard: a read inside a value transformation fails the transformation, caught
    or not. (Value semantics enter the core's guard once the core is wired.)"""

    class Facts(Space):
        value: int = Param()

    facts = open_world(Facts(value=7))
    with pytest.raises(EvaluationError, match="pure value transformation"):
        with _failure.transformation("value snapshot"):
            try:
                read(facts, "value")
            except EvaluationError:
                pass
    assert value(facts, "value") == 7


def test_blocked_self_constraint_keeps_its_inspectable_assessment() -> None:
    """Mirrors ``test_native_execution``'s test of the same name."""

    class Example(Space):
        choice: int = Decision(values=(1, 2))

        @constraint
        def positive(self) -> bool:
            return self.choice > 0

    world = open_world(Example())
    assessment = world.entry(node(world, "positive")).assessment
    assert isinstance(assessment, ConstraintAssessment)
    assert isinstance(assessment.result, Unresolved)
    assert isinstance(assessment.results["positive"], Unresolved)


def test_self_cycles_are_reached_through_scopes_domains_and_guards() -> None:
    """Mirrors ``test_native_execution``'s
    ``test_self_cycles_are_reached_through_scopes_admission_and_guards``: a cycle through
    a child's formal, through a guard. One through a choice's domain cannot be built: a
    domain reading its own choice reads a choice, refused when the model is made."""

    class Child(Space):
        source: int = Param()

        @derived
        def value(self) -> int:
            return self.source

    class Parent(Space):
        @derived
        def output(self) -> int:
            return self.child.value

        child = Child(source=output)

    class Domain(Space):
        @derived
        def extent(self) -> int:
            return self.choice

        choice: int = Decision(
            domain=domain(accepts=lambda candidate, value: candidate == value, value=extent)
        )

    class Guard(Space):
        @derived
        def enabled(self) -> bool:
            return self.choice > 0

        choice: int = Decision(values=(1,), when=enabled)

    world = open_world(Parent())
    with pytest.raises(EvaluationError, match="dependency cycle") as scoped:
        read(world, "output")
    assert "child.value" in str(scoped.value)
    with refused("domain-reads-choice", error=DefinitionError):
        open_world(Domain(), {"choice": 1})
    cyclic = open_world(Guard(), {"choice": 1})
    with pytest.raises(EvaluationError, match="dependency cycle"):
        read(cyclic, "choice")


def test_a_failure_in_a_successor_runs_its_readers_cleanup() -> None:
    """Early cutoff never stands in for a reader whose read now fails: the reader runs,
    and its cleanup with it, as in a world evaluated from nothing."""
    events: list[int] = []

    class Example(Space):
        count: int = Decision(values=(1, 2, 3))

        @derived
        def inner(self) -> int:
            if self.count == 3:
                raise LookupError("three")
            return self.count

        @derived
        def outer(self) -> int:
            try:
                return self.inner
            finally:
                events.append(self.count)

    world = open_world(Example(), {"count": 1})
    assert value(world, "outer") == 1
    with pytest.raises(NativeEvaluationError) as caught:
        read(successor(world, {"count": 3}), "outer")
    assert caught.value.primary.owner == "inner" and events == [1, 3]
