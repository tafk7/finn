# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Detached sparse choices, captured and replayed through ordinary configuration updates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from . import _execution
from ._changes import try_with_choices
from ._configuration import Space
from .compiler import SpaceModel
from .declarations import Decision, DecisionRef
from .edits import Change, ConfigurationResult
from .errors import EvaluationError, RequestError
from .occurrence import state
from .references import DecisionHandle, decision_key
from .semantics import ValueSemantics

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def _check_nodes(model: SpaceModel[Space], indices: Iterable[int]) -> None:
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


def _semantics(model: SpaceModel[Space], index: int) -> ValueSemantics[object]:
    semantics = model.linked.nodes[index].semantics
    assert semantics is not None
    return semantics


def _recognize(model: SpaceModel[Space], index: int, value: object) -> None:
    semantics = _semantics(model, index)
    node = model.linked.nodes[index]
    try:
        accepted = semantics.accepts(value)
    except Exception as cause:
        raise EvaluationError(node.owner, "selection recognition", str(cause)) from cause
    if not accepted:
        raise RequestError(
            f"{decision_key(model.linked, index)}: "
            f"expected selection value of type {semantics.name}"
        )


def _snapshot(model: SpaceModel[Space], index: int, value: object) -> object:
    try:
        return _semantics(model, index).freeze(value)
    except Exception as cause:
        raise EvaluationError(
            model.linked.nodes[index].owner, "selection snapshot", str(cause)
        ) from cause


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

    _model: SpaceModel[Space] = field(repr=False)
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
    def _from_values(
        cls, model: SpaceModel[Space], values: Iterable[tuple[int, object]]
    ) -> Selection:
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

    def value(self, reference: Decision[T] | DecisionRef[T]) -> T:
        if not isinstance(reference, (Decision, DecisionRef)):
            raise RequestError("selection lookup requires an owning Decision")
        index = self._model.resolve(0, reference)
        _check_nodes(self._model, (index,))
        for entry in self._entries:
            if entry.node == index:
                return cast(T, _snapshot(self._model, index, entry.value))
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


def replace_owned(
    mapping: Mapping[str, object], updates: Mapping[str, object], *, owned_keys: Iterable[str]
) -> dict[str, object]:
    """Return a mapping with exactly the current owned sparse entries replaced.

    Callers supply both current and obsolete owned keys; unrelated entries keep
    their values. This helper performs no persistence or external mutation.
    """
    owned = frozenset(owned_keys)
    if any(type(key) is not str or not key for key in owned):
        raise RequestError("owned mapping keys must be nonempty strings")
    if any(type(key) is not str or not key for key in updates):
        raise RequestError("updated mapping keys must be nonempty strings")
    unknown = updates.keys() - owned
    if unknown:
        raise RequestError(f"updates contain unowned keys: {sorted(unknown)}")
    result = {key: value for key, value in mapping.items() if key not in owned}
    result.update(updates)
    return result


__all__ = ["Selection", "SelectionEntry", "capture", "replace_owned", "restore"]
