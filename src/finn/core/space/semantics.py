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
            snapshot=lambda value: value,
        )

    def accepts(self, value: object) -> bool:
        from . import _execution  # noqa: PLC0415 - semantics/execution boundary

        with _execution.transformation("value recognition"):
            return bool(self.recognizes(value))

    def freeze(self, value: object) -> T:
        from . import _execution  # noqa: PLC0415 - semantics/execution boundary

        with _execution.transformation("value snapshot"):
            if not self.accepts(value):
                raise TypeError(unrecognized(self))
            frozen = self.snapshot(cast(T, value))
            if not self.accepts(frozen):
                raise TypeError(f"snapshot for {self.name} changed its nominal value type")
            return frozen

    def values_equal(self, left: object, right: object) -> bool:
        from . import _execution  # noqa: PLC0415 - semantics/execution boundary

        with _execution.transformation("value equality"):
            if not self.accepts(left) or not self.accepts(right):
                return False
            return bool(self.equal(cast(T, left), cast(T, right)))

    def is_compatible_with(self, other: ValueSemantics[object]) -> bool:
        return self.type_token is other.type_token


def default_semantics(value_type: type[T]) -> ValueSemantics[T]:
    """Nominal Python values with detached snapshots and structural equality."""

    return ValueSemantics(
        type_token=value_type,
        name=value_type.__qualname__,
        recognizes=lambda value: type(value) is value_type,
        equal=lambda left, right: left == right,
        snapshot=deepcopy,
    )


def unrecognized(semantics: ValueSemantics[T]) -> str:
    """The one wording for a value ``semantics`` does not recognize."""
    return f"expected value of nominal type {semantics.name}"


def recognize(semantics: ValueSemantics[T], value: object, *, owner: str, role: str) -> bool:
    """Whether ``semantics`` recognizes ``value``; an adapter that raises fails ``owner``."""
    return _owned(lambda: semantics.accepts(value), owner, role)


def snapshot(semantics: ValueSemantics[T], value: object, *, owner: str, role: str) -> T:
    """``semantics``' detached snapshot of ``value``; any failure fails ``owner``."""
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
    (``_execution.transformation``), and that error already names who and where.
    """
    try:
        return call()
    except EvaluationError:
        raise
    except Exception as cause:
        raise EvaluationError(owner, role, str(cause)) from cause


def semantics_for(value_type: type[T] | ValueSemantics[T]) -> ValueSemantics[T]:
    return value_type if isinstance(value_type, ValueSemantics) else default_semantics(value_type)


__all__ = ["NoTruthValue", "ValueSemantics", "default_semantics", "semantics_for"]
