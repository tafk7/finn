# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One native dispatcher for self methods and explicitly bound providers.

Runs authored callbacks on greenlets for the evaluator: engine frames
(``_runtime``) stay generators and only authored calls take a native stack.
Guards value transformations and driver-only APIs; failures and cancellation
return through waiting reads, and cleanup failures are collected
(``CleanupFailure``, ``NativeCancellationDetails``).
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeAlias, cast

from .errors import EvaluationError, ValueUnavailableError
from .results import Inapplicable, NonValue, QueryResult, Rejected, Unresolved

if TYPE_CHECKING:
    from ._runtime import Evaluation, Snapshot


class _Greenlet(Protocol):
    gr_context: object
    dead: bool

    def switch(self, *arguments: object) -> object: ...


class _GreenletModule(Protocol):
    """The native dependency's deliberately small typed boundary."""

    def getcurrent(self) -> _Greenlet: ...

    def greenlet(self, run: Callable[..., object], *, parent: _Greenlet) -> _Greenlet: ...


greenlet = cast(_GreenletModule, import_module("greenlet"))


@dataclass(slots=True)
class Work:
    callback_starts: int = 0
    getter_attempts: int = 0
    nodes_started: int = 0
    suspensions: int = 0
    max_pending: int = 0


@dataclass(slots=True)
class Context:
    """One node's actual reads and sticky failures, shared with its native call."""

    snapshot: Snapshot
    index: int
    native_parent: _Greenlet
    native_started: bool = False
    dependencies: dict[int, None] = field(default_factory=dict)
    # Reads of a forwarding alias served by its source: alias -> source.
    via: dict[int, int] = field(default_factory=dict)
    blocked: dict[int, NonValue] = field(default_factory=dict)
    failure: _Failure | None = None
    fault: EvaluationError | None = None
    role: str = "value"

    @property
    def key(self) -> str:
        return self.snapshot.linked.nodes[self.index].owner


_ACTIVE: ContextVar[Context | None] = ContextVar("space_active_evaluation", default=None)


@dataclass(slots=True)
class _Transformation:
    role: str
    parent: _Transformation | None
    error: EvaluationError | None = None


_TRANSFORM: ContextVar[_Transformation | None] = ContextVar("space_transformation", default=None)


@contextmanager
def transformation(role: str) -> Iterator[None]:
    """Value adapters may transform supplied values but cannot read configurations."""
    context = _Transformation(role, _TRANSFORM.get())
    token = _TRANSFORM.set(context)
    try:
        yield
        if context.error is not None:
            raise context.error
    finally:
        _TRANSFORM.reset(token)


def _check_transformation() -> None:
    transform = _TRANSFORM.get()
    if transform is None:
        return
    active = current()
    error = EvaluationError(
        active.key if active is not None else "value semantics",
        transform.role,
        "configuration access during pure value transformation",
    )
    while transform is not None:
        transform.error = error
        transform = transform.parent
    if active is not None:
        _native_fault(active, error)
    raise error


def current() -> Context | None:
    return _ACTIVE.get()


def driver_only(role: str) -> None:
    _check_transformation()
    active = current()
    if active is not None:
        error = EvaluationError(active.key, role, "driver-only API during computation")
        _native_fault(active, error)
        raise error


def check_snapshot(snapshot: Snapshot) -> None:
    _check_transformation()
    active = current()
    if active is not None and active.snapshot is not snapshot:
        error = EvaluationError(active.key, "field read", "cross-snapshot access")
        _native_fault(active, error)
        raise error


@dataclass(frozen=True, slots=True)
class _Halt:
    results: tuple[NonValue, ...]


@dataclass(frozen=True, slots=True)
class CleanupFailure:
    """One secondary cleanup error/nonvalue, in deterministic encounter order."""

    owner: str
    role: str
    error: EvaluationError | None = None
    result: QueryResult[object] | None = None


@dataclass(slots=True)
class _Failure:
    primary: EvaluationError
    cancellation: BaseException | None = None
    cleanup: list[CleanupFailure] = field(default_factory=list)
    observed: set[tuple[str, int]] = field(default_factory=set)

    def add(self, issue: CleanupFailure) -> None:
        identity = (
            issue.owner,
            id(issue.error if issue.error is not None else issue.result),
        )
        if identity not in self.observed:
            self.observed.add(identity)
            self.cleanup.append(issue)


class NativeEvaluationError(EvaluationError):
    """Sticky native query failure; primary cause and cleanup issues stay separate."""

    def __init__(self, failure: _Failure) -> None:
        primary = failure.primary
        super().__init__(primary.owner, primary.role, primary.detail)
        self.primary = primary
        self.cleanup_failures = tuple(failure.cleanup)
        self._failure = failure
        self.__cause__ = primary.__cause__


@dataclass(frozen=True, slots=True)
class NativeCancellationDetails:
    """Structured diagnostics attached to an original control-flow exception."""

    primary: EvaluationError
    cleanup_failures: tuple[CleanupFailure, ...]


def cancellation_details(error: BaseException) -> NativeCancellationDetails | None:
    details = getattr(error, "space_cancellation", None)
    return details if isinstance(details, NativeCancellationDetails) else None


def _raise_native_failure(failure: _Failure) -> NoReturn:
    if failure.cancellation is not None:
        cancellation = failure.cancellation
        setattr(
            cancellation,
            "space_cancellation",
            NativeCancellationDetails(failure.primary, tuple(failure.cleanup)),
        )
        raise cancellation
    raise NativeEvaluationError(failure)


def _remember_failure(context: Context, incoming: _Failure) -> _Failure:
    if context.failure is None:
        context.failure = incoming
    elif context.failure is not incoming:
        if context.failure.cancellation is None:
            context.failure.cancellation = incoming.cancellation
        context.failure.add(
            CleanupFailure(incoming.primary.owner, incoming.primary.role, error=incoming.primary)
        )
        for issue in incoming.cleanup:
            context.failure.add(issue)
    return context.failure


def _native_fault(context: Context, error: EvaluationError) -> None:
    context.fault = error
    _remember_failure(context, _Failure(error))


def _remember_exception(context: Context, cause: BaseException) -> _Failure | None:
    # Python retains an exception raised by the body as __context__ when finally
    # raises another error. Preserve that chronological chain, including semantic
    # cleanup blockers, without repeatedly attaching transported engine failures.
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = cause
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        previous = current.__context__
        # A contextual engine error wraps its cause; it is not another cleanup
        # failure. Still follow the cause's earlier context (e.g. a finally body).
        if isinstance(current, EvaluationError) and current.__cause__ is previous:
            current = None if previous is None else previous.__context__
        else:
            current = previous
    for item in reversed(chain):
        if isinstance(item, NativeEvaluationError):
            _remember_failure(context, item._failure)
        elif isinstance(item, ValueUnavailableError):
            if context.failure is not None:
                context.failure.add(
                    CleanupFailure(str(item.context), "cleanup read", result=item.result)
                )
        else:
            if context.failure is not None and item is context.failure.cancellation:
                continue
            if item is context.fault and context.failure is not None:
                continue
            if isinstance(item, EvaluationError):
                error = item
            else:
                error = EvaluationError(context.key, context.role, str(item))
                error.__cause__ = item
            cancellation = (
                item if isinstance(item, (KeyboardInterrupt, SystemExit, GeneratorExit)) else None
            )
            _remember_failure(context, _Failure(error, cancellation=cancellation))
    return context.failure


@dataclass(frozen=True, slots=True)
class Call:
    function: Callable[..., object]
    arguments: tuple[object, ...]
    keywords: Mapping[str, object] = field(default_factory=dict)
    role: str = "computation"


@dataclass(frozen=True, slots=True)
class _Returned:
    value: object


# Frames yield dependencies or native calls and finish with an Evaluation.
# Dependency requests receive QueryResult values; calls receive CallOutcome.
# Native descriptor reads receive complete Evaluations or transported failures.
Frame: TypeAlias = Generator[int | Call, object, "Evaluation"]
CallOutcome: TypeAlias = _Returned | _Halt | _Failure


def _invoke(context: Context, call: Call) -> CallOutcome:
    """Only authored execution and its exception boundary occupy a native stack."""
    context.native_started = True
    context.role = call.role
    token = _ACTIVE.set(context)
    try:
        if context.failure is not None:
            return context.failure
        context.snapshot.work.callback_starts += 1
        try:
            value = call.function(*call.arguments, **call.keywords)
        except BaseException as cause:
            failure = _remember_exception(context, cause)
            if failure is not None:
                return failure
            if context.blocked:
                return _Halt(tuple(context.blocked.values()))
            # A manually raised unavailable exception is a programmer failure.
            error = EvaluationError(context.key, context.role, str(cause))
            error.__cause__ = cause
            return _remember_failure(context, _Failure(error))
        if context.failure is not None:
            return context.failure
        if context.blocked:
            return _Halt(tuple(context.blocked.values()))
        return _Returned(value)
    finally:
        _ACTIVE.reset(token)


@dataclass(slots=True)
class _Task:
    """A suspended node frame and, while its callback runs, a native continuation."""

    identity: object
    context: Context
    frame: Frame
    continuation: _Greenlet | None = None
    call: Call | None = None
    incoming: object = None
    started: bool = False


def accept_read(context: Context, index: int, outcome: object) -> Evaluation:
    if isinstance(outcome, _Failure):
        _remember_failure(context, outcome)
        _raise_native_failure(outcome)
    entry = cast("Evaluation", outcome)
    if isinstance(entry.result, (Unresolved, Rejected, Inapplicable)):
        context.blocked[index] = entry.result
        if context.failure is not None:
            context.failure.add(
                CleanupFailure(
                    context.snapshot.linked.nodes[index].owner, "cleanup read", result=entry.result
                )
            )
    return entry


def run(snapshot: Snapshot, index: int, frame: Frame | None = None) -> Evaluation:
    """Drain a query's dependency stack, including native cleanup after failure.

    Integer identities represent cacheable nodes. A supplied domain-operation
    frame gets a unique identity so its result cannot replace the decision value.
    Successful nodes enter the snapshot cache; failures live only for this run.
    Completion remains inside the exception boundary through retirement/delivery.
    """
    from . import _runtime  # noqa: PLC0415 - execution/runtime record cycle

    parent = greenlet.getcurrent()
    tasks: list[_Task] = []
    active: dict[object, _Task] = {}
    failures: dict[object, _Failure] = {}
    root_identity: object = index if frame is None else object()
    final: Evaluation | _Failure | None = None

    def begin(
        current_index: int,
        identity: object,
        initial: Frame | None = None,
    ) -> None:
        context = Context(snapshot, current_index, parent)
        task = _Task(
            identity,
            context,
            initial
            if initial is not None
            else _runtime._frame(snapshot, snapshot.linked.nodes[current_index]),
        )
        tasks.append(task)
        active[identity] = task
        snapshot.work.nodes_started += 1
        snapshot.work.max_pending = max(snapshot.work.max_pending, len(tasks))

    def finish(task: _Task, outcome: Evaluation | _Failure) -> None:
        nonlocal final
        if isinstance(outcome, _Failure):
            failures[task.identity] = outcome
        else:
            dependencies, via = tuple(task.context.dependencies), tuple(task.context.via.items())
            outcome = _runtime.Evaluation(
                outcome.result,
                dependencies,
                outcome.assessment,
                via,
                _runtime.read_decisions(snapshot, dependencies, via),
            )
            outcome = _runtime.supplied_provenance(snapshot, outcome, task.context.index)
            if isinstance(task.identity, int):
                snapshot.cache[task.identity] = outcome
        task.frame.close()
        tasks.pop()
        active.pop(task.identity)
        if tasks:
            waiter = tasks[-1]
            waiter.incoming = (
                outcome
                if waiter.continuation is not None or isinstance(outcome, _Failure)
                else outcome.result
            )
        else:
            final = outcome

    begin(index, root_identity, frame)
    while tasks:
        task = tasks[-1]
        context = task.context
        try:
            if task.continuation is not None:
                continuation = task.continuation
                if not context.native_started:
                    call = task.call
                    assert call is not None
                    demanded = continuation.switch(context, call)
                    task.call = None
                else:
                    demanded = continuation.switch(task.incoming)
                task.incoming = None
                if continuation.dead:
                    # A frame may make another native call after this one.
                    task.continuation = None
                    task.incoming = demanded
                    context.native_started = False
                    continue
                snapshot.work.suspensions += 1
            else:
                if isinstance(task.incoming, _Failure):
                    finish(task, _remember_failure(context, task.incoming))
                    continue
                completed: Evaluation | None = None
                try:
                    if task.started:
                        demanded = task.frame.send(task.incoming)
                    else:
                        task.started = True
                        demanded = next(task.frame)
                except StopIteration as completion:
                    completed = cast("Evaluation", completion.value)
                if completed is not None:
                    # Completion must remain inside the failure boundary, and
                    # outside the StopIteration handler's exception context.
                    finish(task, completed)
                    continue
                task.incoming = None
                if isinstance(demanded, Call):
                    task.call = demanded
                    context.role = demanded.role
                    task.continuation = greenlet.greenlet(_invoke, parent=parent)
                    task.continuation.gr_context = copy_context()
                    continue
            dependency = cast(int, demanded)
            context.dependencies[dependency] = None
            if dependency in snapshot.cache:
                entry = snapshot.cache[dependency]
                task.incoming = entry if task.continuation is not None else entry.result
            elif dependency in failures:
                task.incoming = failures[dependency]
            elif dependency in active:
                roles = [f"{item.context.key} ({item.context.role})" for item in tasks]
                owner = snapshot.linked.nodes[dependency].owner
                task.incoming = _Failure(
                    EvaluationError(owner, "dependency cycle", " -> ".join((*roles, owner)))
                )
            else:
                begin(dependency, dependency)
        except BaseException as cause:
            if isinstance(task.incoming, _Failure):
                # An engine-only waiter may not yet have consumed the earlier
                # failure when interrupted immediately before its delivery.
                _remember_failure(context, task.incoming)
            failure = _remember_exception(context, cause)
            assert failure is not None
            if not any(pending is task for pending in tasks):
                # Completion may already have retired this task. Its safely
                # published result can remain cached, but its parent must see
                # the interruption instead of the pending successful delivery.
                active.pop(task.identity, None)
            if tasks:
                waiter = tasks[-1]
                waiter.incoming = _remember_failure(waiter.context, failure)
            else:
                final = failure
    if isinstance(final, _Failure):
        _raise_native_failure(final)
    assert final is not None
    return final
