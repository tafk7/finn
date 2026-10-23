# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit value recognition, equality and snapshot policies."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, NoReturn, TypeVar, cast

from .errors import EvaluationError

T = TypeVar("T")


class NoTruthValue:
    """Semantic results require an explicit inspection of their variant."""

    __slots__ = ()

    def __bool__(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} has no truth value; inspect its fields")


@dataclass(frozen=True, slots=True)
class ValueSemantics(Generic[T]):
    """Adapter-owned recognition, equality and snapshots for one value type.

    The token denotes compatibility; it is deliberately independent of a Python
    class or any persisted form. Custom snapshots must detach mutable state.
    """

    type_token: object
    name: str
    recognizes: Callable[[object], bool]
    equal: Callable[[T, T], bool]
    snapshot: Callable[[T], T]

    @classmethod
    def immutable_nominal(
        cls, value_type: type[T], *, name: str | None = None
    ) -> ValueSemantics[T]:
        """Use exact nominal recognition for a caller-guaranteed immutable type."""

        return cls(
            type_token=value_type,
            name=name or value_type.__qualname__,
            recognizes=lambda value: type(value) is value_type,
            equal=lambda left, right: left == right,
            snapshot=_itself,
        )

    def accepts(self, value: object) -> bool:
        from ._failure import transformation  # noqa: PLC0415 - semantics/evaluation

        with transformation("value recognition"):
            return bool(self.recognizes(value))

    def freeze(self, value: object) -> T:
        from ._failure import transformation  # noqa: PLC0415 - semantics/evaluation

        with transformation("value snapshot"):
            if not self.accepts(value):
                raise TypeError(unrecognized(self))
            frozen = self.snapshot(cast(T, value))
            if not self.accepts(frozen):
                raise TypeError(f"snapshot for {self.name} changed its nominal value type")
            return frozen

    def values_equal(self, left: object, right: object) -> bool:
        """Whether ``left`` and ``right`` are both values of this type, equal after each
        side's snapshot: a value is compared as it is stored (``freeze``), as the engine
        compares the values it holds. Where the snapshot is the value itself, none is
        taken."""
        from ._failure import transformation  # noqa: PLC0415 - semantics/evaluation

        with transformation("value equality"):
            if not self.accepts(left) or not self.accepts(right):
                return False
            if self.snapshot is not _itself:
                left, right = self.freeze(left), self.freeze(right)
            return bool(self.equal(cast(T, left), cast(T, right)))

    def is_compatible_with(self, other: ValueSemantics[object]) -> bool:
        return self.type_token is other.type_token


def default_semantics(value_type: type[T]) -> ValueSemantics[T]:
    """Nominal Python values with detached snapshots and structural equality.

    An immutable value is its own snapshot, shared by every read: an immutable
    builtin (an int, a str, ...), a tuple or frozenset of immutable values, and an
    instance of a frozen dataclass, whose fields cannot be rebound (the contract is
    that a frozen value class holds immutable values; one that holds a mutable field
    states its own semantics). Anything else is deep-copied.
    """

    snapshot: Callable[[T], T]
    if value_type in _ATOMS or _frozen(value_type):
        snapshot = _itself
    elif value_type in (tuple, frozenset):
        snapshot = _detached
    else:
        snapshot = deepcopy
    return ValueSemantics(
        type_token=value_type,
        name=value_type.__qualname__,
        recognizes=lambda value: type(value) is value_type,
        equal=lambda left, right: left == right,
        snapshot=snapshot,
    )


_ATOMS: frozenset[type] = frozenset(
    (bool, int, float, complex, str, bytes, type(None), type(Ellipsis), range)
)


def _frozen(value_type: type) -> bool:
    params = getattr(value_type, "__dataclass_params__", None)
    return params is not None and bool(params.frozen)


def _immutable(value: object) -> bool:
    kind = type(value)
    if kind in _ATOMS or _frozen(kind):
        return True
    if kind is tuple or kind is frozenset:
        return all(_immutable(item) for item in cast("tuple[object, ...]", value))
    return False


def _detached(value: T) -> T:
    """A tuple or frozenset as it is where everything it holds is immutable, else a deep
    copy."""
    return value if _immutable(value) else deepcopy(value)


def _itself(value: T) -> T:
    return value


@dataclass(frozen=True, slots=True)
class OptionalSemantics(ValueSemantics[T]):
    """The semantics of an optional value, ``T | None`` (``optional``): ``None``, the absent
    value, checked first and by identity, then ``inner``'s, ``T``'s, for every other value.
    Its type token is ``inner``'s, so a binding of a ``T`` to it (or of it to a ``T``) is
    compatible."""

    inner: ValueSemantics[T]


def optional(inner: ValueSemantics[T]) -> OptionalSemantics[T]:
    """``inner``'s semantics over ``inner``'s values or ``None``, where every function sees
    ``None`` first, by identity: it recognizes ``None``, holds ``None`` equal to ``None``
    alone, and keeps ``None`` as its own snapshot. ``inner``'s functions never see it. The
    snapshot is ``inner``'s itself where that keeps every value as it is."""

    def recognizes(value: object) -> bool:
        return value is None or inner.recognizes(value)

    def equal(left: T, right: T) -> bool:
        if left is None or right is None:
            return left is right
        return inner.equal(left, right)

    def detached(value: T) -> T:
        return value if value is None else inner.snapshot(value)

    return OptionalSemantics(
        type_token=inner.type_token,
        name=f"{inner.name} | None",
        recognizes=recognizes,
        equal=equal,
        snapshot=_itself if inner.snapshot is _itself else detached,
        inner=inner,
    )


BOOL = cast(ValueSemantics[object], default_semantics(bool))
STRING = cast(ValueSemantics[object], default_semantics(str))


def unrecognized(semantics: ValueSemantics[T]) -> str:
    """The one wording for a value ``semantics`` does not recognize."""
    return f"expected value of nominal type {semantics.name}"


def recognize(semantics: ValueSemantics[T], value: object, *, owner: str, role: str) -> bool:
    """Whether ``semantics`` recognizes ``value``; an adapter that raises fails ``owner``."""
    return _owned(lambda: semantics.accepts(value), owner, role)


def snapshot(semantics: ValueSemantics[T], value: object, *, owner: str, role: str) -> T:
    """``semantics``' detached snapshot of ``value``; any failure fails ``owner``. A value
    of the nominal type of semantics whose snapshot is the value itself is returned as
    it is: no adapter runs for it."""
    if semantics.snapshot is _itself and type(value) is semantics.type_token:
        return cast(T, value)
    return _owned(lambda: semantics.freeze(value), owner, role)


def equal(
    semantics: ValueSemantics[T], left: object, right: object, *, owner: str, role: str
) -> bool:
    """Whether ``semantics`` holds ``left`` and ``right`` equal; an adapter that raises
    fails ``owner``."""
    return _owned(lambda: semantics.values_equal(left, right), owner, role)


R = TypeVar("R")


def _owned(call: Callable[[], R], owner: str, role: str) -> R:
    """``call``'s result; a failure of the adapter it runs is ``owner``'s, in ``role``.

    An ``EvaluationError`` passes unchanged: the adapter read a configuration
    (``_failure.transformation``), and that error already names who and where.
    """
    try:
        return call()
    except EvaluationError:
        raise
    except Exception as cause:
        raise EvaluationError(owner, role, str(cause)) from cause


def semantics_for(value_type: type[T] | ValueSemantics[T]) -> ValueSemantics[T]:
    return value_type if isinstance(value_type, ValueSemantics) else default_semantics(value_type)


__all__ = [
    "NoTruthValue",
    "OptionalSemantics",
    "ValueSemantics",
    "default_semantics",
    "optional",
    "semantics_for",
]
