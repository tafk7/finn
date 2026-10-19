# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Authored calls, and how their failures complete, for the evaluation core.

Each authored function (a ``self`` method, an explicit function, a domain's
membership or listing) runs as a ``Call``: while it runs it is the **active call**, in
a copy of its driver's context. A driver-only API used during a call, a read through
another call's receiver, or a read during a value transformation is that call's
failure.

**The failure contract.** An authored function raising is its owner's
``EvaluationError`` (role: what was running), never cached. A failure reaching a
method through one of its reads is raised in the method as ``NativeEvaluationError``
(or as the original cancellation), so its ``finally`` blocks and context managers
run, and may read values. The **first** failure a call meets is its primary and is
sticky: a method that catches it, or a read that halted it, still fails or halts. A
later error, or a non-value read in cleanup, is a ``CleanupFailure`` beside the
primary, in the order met. ``KeyboardInterrupt``, ``SystemExit`` and ``GeneratorExit``
keep their identity: the original is raised at the driver, with
``NativeCancellationDetails`` attached.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import NoReturn

from .errors import EvaluationError, ValueUnavailableError
from .ir import NodeId
from .results import NonValue, QueryResult

_CANCELLATIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)


@dataclass(frozen=True, slots=True)
class CleanupFailure:
    """One secondary error or non-value read met after the primary, in encounter order."""

    owner: str
    role: str
    error: EvaluationError | None = None
    result: QueryResult[object] | None = None


@dataclass(slots=True, eq=False)
class Failure:
    """A failure as it completes: its primary error, the cancellation that came with it
    (or after it), and the cleanup failures met since, each once."""

    primary: EvaluationError
    cancellation: BaseException | None = None
    cleanup: list[CleanupFailure] = field(default_factory=list)
    observed: set[tuple[str, int]] = field(default_factory=set)

    def add(self, issue: CleanupFailure) -> None:
        identity = (issue.owner, id(issue.error if issue.error is not None else issue.result))
        if identity not in self.observed:
            self.observed.add(identity)
            self.cleanup.append(issue)


class NativeEvaluationError(EvaluationError):
    """A failure of authored code, as the driver (or a reading method) receives it: its
    primary cause and its cleanup failures kept apart."""

    def __init__(self, failure: Failure) -> None:
        primary = failure.primary
        super().__init__(primary.owner, primary.role, primary.detail)
        self.primary = primary
        self.cleanup_failures = tuple(failure.cleanup)
        self.failure = failure
        self.__cause__ = primary.__cause__


@dataclass(frozen=True, slots=True)
class NativeCancellationDetails:
    """What a cancellation passed through, attached to the original exception."""

    primary: EvaluationError
    cleanup_failures: tuple[CleanupFailure, ...]


def cancellation_details(error: BaseException) -> NativeCancellationDetails | None:
    details = getattr(error, "space_cancellation", None)
    return details if isinstance(details, NativeCancellationDetails) else None


class Failing(BaseException):
    """A failure on its way through the evaluator's own frames (never seen by authored
    code: a read turns it into the failure's exception)."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.primary)
        self.failure = failure


@dataclass(slots=True, eq=False)
class Call:
    """One running authored function: who and what (``owner``, ``role``), the first
    failure it met and every later one (``failure``, sticky), an engine refusal raised
    in it (``fault``), and the reads of non-values that halt it (``blocked``).
    ``outer``: the exception its caller was handling when it started (it runs on its
    caller's stack, so what it raises chains to that one, which is not its own).
    ``abandoned``: the depth guard's unwinding passed through it, so whatever it does
    next is discarded and it runs again."""

    owner: str
    role: str
    failure: Failure | None = None
    fault: EvaluationError | None = None
    blocked: dict[NodeId, NonValue] = field(default_factory=dict)
    outer: BaseException | None = None
    abandoned: BaseException | None = None


ACTIVE: ContextVar[Call | None] = ContextVar("space_active_call", default=None)


def current() -> Call | None:
    """The authored call running in this context, if any."""
    return ACTIVE.get()


def exception(failure: Failure) -> BaseException:
    """What a failure raises: its cancellation, with the details attached, or a
    ``NativeEvaluationError``."""
    if failure.cancellation is not None:
        setattr(
            failure.cancellation,
            "space_cancellation",
            NativeCancellationDetails(failure.primary, tuple(failure.cleanup)),
        )
        return failure.cancellation
    return NativeEvaluationError(failure)


def raise_failure(failure: Failure) -> NoReturn:
    raise exception(failure)


def remember_failure(call: Call, incoming: Failure) -> Failure:
    """``incoming`` met by ``call``: its primary where the call has none yet, else a
    cleanup failure of the call's (and a cancellation it carries is kept)."""
    if call.failure is None:
        call.failure = incoming
    elif call.failure is not incoming:
        if call.failure.cancellation is None:
            call.failure.cancellation = incoming.cancellation
        call.failure.add(
            CleanupFailure(incoming.primary.owner, incoming.primary.role, error=incoming.primary)
        )
        for issue in incoming.cleanup:
            call.failure.add(issue)
    return call.failure


def fault(call: Call, error: EvaluationError) -> None:
    """An engine refusal raised in ``call`` (a driver-only API, a foreign read): it fails
    the call whether or not the call catches it."""
    call.fault = error
    remember_failure(call, Failure(error))


def remember_exception(call: Call, cause: BaseException) -> Failure | None:
    """Fold what ``call`` raised into its failure, oldest first.

    Python keeps the exception a body raised as ``__context__`` of the one its
    ``finally`` raised, so the chain is followed back to the first (and no further than
    the call's ``outer``, its caller's): a failure carried through a read is that
    failure; a non-value read is a cleanup read where a failure is already held; any
    other exception is the call's own error (a cancellation keeps its identity).
    ``None`` where nothing failed (only halting reads)."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    item: BaseException | None = cause
    while item is not None and id(item) not in seen and item is not call.outer:
        seen.add(id(item))
        chain.append(item)
        previous = item.__context__
        # An engine error wrapping its cause is one failure: follow the cause's context.
        if isinstance(item, EvaluationError) and item.__cause__ is previous:
            item = None if previous is None else previous.__context__
        else:
            item = previous
    for item in reversed(chain):
        if isinstance(item, NativeEvaluationError):
            remember_failure(call, item.failure)
        elif isinstance(item, Failing):
            remember_failure(call, item.failure)
        elif isinstance(item, ValueUnavailableError):
            if call.failure is not None:
                call.failure.add(
                    CleanupFailure(str(item.context), "cleanup read", result=item.result)
                )
        else:
            if call.failure is not None and item is call.failure.cancellation:
                continue
            if item is call.fault and call.failure is not None:
                continue
            if isinstance(item, EvaluationError):
                error = item
            else:
                error = EvaluationError(call.owner, call.role, str(item))
                error.__cause__ = item
            cancellation = item if isinstance(item, _CANCELLATIONS) else None
            remember_failure(call, Failure(error, cancellation=cancellation))
    return call.failure


def from_exception(
    cause: BaseException, *, owner: str, role: str, outer: BaseException | None
) -> Failure:
    """The failure of an evaluation that raised ``cause`` outside any authored call
    (``outer``: the exception being handled when it started)."""
    call = Call(owner, role, outer=outer)
    found = remember_exception(call, cause)
    if found is not None:
        return found
    error = EvaluationError(owner, role, str(cause))
    error.__cause__ = cause
    return Failure(error)


def driver_only(role: str) -> None:
    """Refuse a driver-only API during an authored call (sticky: it fails the call)."""
    check_transformation()
    active = current()
    if active is not None:
        error = EvaluationError(active.owner, role, "driver-only API during computation")
        fault(active, error)
        raise error


@dataclass(slots=True)
class _Transformation:
    role: str
    parent: _Transformation | None
    error: EvaluationError | None = None


_TRANSFORM: ContextVar[_Transformation | None] = ContextVar(
    "space_core_transformation", default=None
)


@contextmanager
def transformation(role: str) -> Iterator[None]:
    """A pure value transformation (recognition, equality, a snapshot): it may transform
    the values it is given but never read a configuration; a read refused in it fails
    the transformation even where the adapter catches the error."""
    context = _Transformation(role, _TRANSFORM.get())
    token = _TRANSFORM.set(context)
    try:
        yield
        if context.error is not None:
            raise context.error
    finally:
        _TRANSFORM.reset(token)


def check_transformation() -> None:
    """Refuse a configuration read during a value transformation."""
    transform = _TRANSFORM.get()
    if transform is None:
        return
    active = current()
    error = EvaluationError(
        active.owner if active is not None else "value semantics",
        transform.role,
        "configuration access during pure value transformation",
    )
    while transform is not None:
        transform.error = error
        transform = transform.parent
    if active is not None:
        fault(active, error)
    raise error


__all__ = [
    "ACTIVE",
    "Call",
    "CleanupFailure",
    "Failing",
    "Failure",
    "NativeCancellationDetails",
    "NativeEvaluationError",
    "cancellation_details",
    "check_transformation",
    "current",
    "driver_only",
    "exception",
    "fault",
    "from_exception",
    "raise_failure",
    "remember_exception",
    "remember_failure",
    "transformation",
]
