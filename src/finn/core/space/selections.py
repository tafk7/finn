# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Detached sparse choices, captured and replayed through ordinary configuration updates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from . import _execution
from ._changes import try_with_choices
from ._configuration import Space
from .compiler import Model
from .declarations import Decision
from .edits import Change, ConfigurationResult
from .errors import EvaluationError, RequestError
from .occurrence import state
from .references import DecisionHandle, decision_key
from .semantics import ValueSemantics, recognize, snapshot, unrecognized

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def _check_nodes(model: Model[Space], indices: Iterable[int]) -> None:
    seen: set[int] = set()
    for index in indices:
        if (
            type(index) is not int
            or not 0 <= index < len(model.linked.nodes)
            or model.linked.nodes[index].kind != "decision"
        ):
            raise RequestError("selection entry must identify an owning decision")
        if index in seen:
            raise RequestError(f"duplicate selection key {decision_key(model.linked, index)!r}")
        seen.add(index)


def _semantics(model: Model[Space], index: int) -> ValueSemantics[object]:
    semantics = model.linked.nodes[index].semantics
    assert semantics is not None
    return semantics


def _recognize(model: Model[Space], index: int, value: object) -> None:
    semantics = _semantics(model, index)
    owner = model.linked.nodes[index].owner
    if not recognize(semantics, value, owner=owner, role="selection recognition"):
        raise RequestError(f"{decision_key(model.linked, index)}: {unrecognized(semantics)}")


def _snapshot(model: Model[Space], index: int, value: object) -> object:
    owner = model.linked.nodes[index].owner
    return snapshot(_semantics(model, index), value, owner=owner, role="selection snapshot")


@dataclass(frozen=True, slots=True)
class SelectionEntry:
    """A detached public copy of one owning decision's value."""

    key: str
    reference: DecisionHandle[object]
    value: object


@dataclass(frozen=True, slots=True)
class _StoredEntry:
    node: int
    value: object = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class Selection:
    """Read-only sparse choices for one model; edit a configuration, then capture it.

    Internal entries contain owning node indices and detached values. Public
    entries, equality operands, and replay values receive defensive snapshots.
    """

    _model: Model[Space] = field(repr=False)
    _entries: tuple[_StoredEntry, ...] = field(repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection) or self._model.linked is not other._model.linked:
            return False
        if tuple(entry.node for entry in self._entries) != tuple(
            entry.node for entry in other._entries
        ):
            return False
        for left, right in zip(self._entries, other._entries):
            try:
                equal = _semantics(self._model, left.node).values_equal(
                    _snapshot(self._model, left.node, left.value),
                    _snapshot(self._model, right.node, right.value),
                )
            except EvaluationError:
                raise
            except Exception as cause:
                raise EvaluationError(
                    self._model.linked.nodes[left.node].owner, "selection equality", str(cause)
                ) from cause
            if not equal:
                return False
        return True

    @classmethod
    def _from_values(cls, model: Model[Space], values: Iterable[tuple[int, object]]) -> Selection:
        pending = tuple(values)
        _check_nodes(model, (index for index, _ in pending))
        for index, value in pending:
            _recognize(model, index, value)
        return cls(
            model,
            tuple(
                _StoredEntry(index, _snapshot(model, index, value))
                for index, value in sorted(
                    pending, key=lambda entry: decision_key(model.linked, entry[0])
                )
            ),
        )

    @property
    def entries(self) -> tuple[SelectionEntry, ...]:
        return tuple(
            SelectionEntry(
                decision_key(self._model.linked, entry.node),
                DecisionHandle(self._model.linked, entry.node),
                _snapshot(self._model, entry.node, entry.value),
            )
            for entry in self._entries
        )

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(decision_key(self._model.linked, entry.node) for entry in self._entries)

    @overload
    def value(self, reference: Decision[T] | DecisionHandle[T]) -> T: ...

    @overload
    def value(self, reference: Space | None) -> str: ...

    @overload
    def value(self, reference: T) -> T: ...

    def value(self, reference: object) -> Any:
        """The captured value of a decision: a Decision member, a reference, or a handle.

        A Decision over nodes (typed as its candidates) captures its key; a
        reference is typed as its value, so its captured value has that type.
        """
        try:
            index = self._model.decision(0, reference)
        except RequestError as cause:
            raise RequestError(f"selection lookup requires an owning Decision: {cause}") from cause
        _check_nodes(self._model, (index,))
        for entry in self._entries:
            if entry.node == index:
                return _snapshot(self._model, index, entry.value)
        raise KeyError(decision_key(self._model.linked, index))


def capture(point: Space) -> Selection:
    """Capture every committed owner in the root, without evaluating other work."""
    _execution.driver_only("selection capture")
    current = state(point)
    with current.lock:
        return Selection._from_values(current.model, current.assignments.items())


def restore(base: S, selection: Selection) -> ConfigurationResult[S]:
    """Atomically replay choices on a root with no existing assignments.

    Bind a new root to restore different facts. Edit configured points with
    with_choices; restore never implicitly merges or overwrites their choices.
    """
    _execution.driver_only("selection restore")
    current = state(base)
    if base._scope != 0:
        raise RequestError("selection restore requires a root configuration")
    if current.assignments:
        raise RequestError("selection restore requires a root with no committed choices")
    if not isinstance(selection, Selection):
        raise RequestError("restore requires a Selection")
    if selection._model.linked is not current.linked:
        raise RequestError("selection belongs to a different compiled model")
    _check_nodes(current.model, (entry.node for entry in selection._entries))
    return try_with_choices(
        base,
        *(
            Change(current, current.linked.nodes[entry.node].scope, entry.node, entry.value)
            for entry in selection._entries
        ),
    )


__all__ = ["Selection", "SelectionEntry", "capture", "restore"]
