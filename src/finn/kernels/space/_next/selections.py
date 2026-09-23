# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Detached sparse commitments and checked replay through atomic refinement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from .compiler import SpaceModel
from .declarations import Decision, DecisionRef, Space
from .edits import RefinementReport
from .errors import EvaluationError, RequestError
from .inspection import DecisionInfo, decision_info, decisions
from .occurrence import state
from .references import DecisionHandle
from .results import Decided
from .semantics import ValueSemantics

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def _semantics(info: DecisionInfo[object]) -> ValueSemantics[object]:
    semantics = info.reference.semantics
    if semantics is None:
        raise RequestError(f"{info.key}: decision has no value semantics")
    return semantics


def _recognize(info: DecisionInfo[object], value: object) -> None:
    semantics = _semantics(info)
    try:
        accepted = semantics.accepts(value)
    except Exception as cause:
        raise EvaluationError(info.owner, "selection recognition", str(cause)) from cause
    if not accepted:
        raise RequestError(f"{info.key}: expected selection value of type {semantics.name}")


def _snapshot(info: DecisionInfo[object], value: object) -> object:
    try:
        return _semantics(info).freeze(value)
    except Exception as cause:
        raise EvaluationError(info.owner, "selection snapshot", str(cause)) from cause


@dataclass(frozen=True, slots=True)
class SelectionEntry:
    """A detached public copy of one committed owning decision."""

    key: str
    reference: DecisionHandle[object]
    value: object


@dataclass(frozen=True, slots=True)
class _StoredEntry:
    info: DecisionInfo[object]
    value: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class SelectionChange:
    """A detached typed request, created by Selection.edit or Selection.remove."""

    _model: SpaceModel[Space] = field(repr=False)
    _info: DecisionInfo[object] = field(repr=False)
    _remove: bool
    _value: object = field(default=None, repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class Selection:
    """Immutable sparse choices for exactly one compiled model.

    Private payloads belong to this value. Every public value read takes a new
    declared snapshot, including entry iteration and replay requests.
    Declaration references are relative to the model root; children use scoped
    references or model-bound handles, never a guessed placement.
    """

    _model: SpaceModel[Space] = field(repr=False)
    _entries: tuple[_StoredEntry, ...] = field(repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection) or self._model.linked is not other._model.linked:
            return False
        if self.keys != other.keys:
            return False
        for left, right in zip(self._entries, other._entries):
            try:
                equal = _semantics(left.info).values_equal(
                    _snapshot(left.info, left.value), _snapshot(right.info, right.value)
                )
            except EvaluationError:
                raise
            except Exception as cause:
                raise EvaluationError(left.info.owner, "selection equality", str(cause)) from cause
            if not equal:
                return False
        return True

    @classmethod
    def _from_values(
        cls, model: SpaceModel[Space], values: Iterable[tuple[DecisionInfo[object], object]]
    ) -> Selection:
        pending = tuple(values)
        seen: set[str] = set()
        for info, _ in pending:
            actual = decision_info(model, info.reference)
            if actual.key != info.key or info.key in seen:
                raise RequestError(f"duplicate or incompatible selection key {info.key!r}")
            seen.add(info.key)
        for info, value in pending:
            _recognize(info, value)
        return cls(
            model,
            tuple(
                _StoredEntry(info, _snapshot(info, value))
                for info, value in sorted(pending, key=lambda entry: entry[0].key)
            ),
        )

    @property
    def entries(self) -> tuple[SelectionEntry, ...]:
        return tuple(
            SelectionEntry(entry.info.key, entry.info.reference, _snapshot(entry.info, entry.value))
            for entry in self._entries
        )

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.info.key for entry in self._entries)

    def value(self, reference: Decision[T] | DecisionRef[T]) -> T:
        info = decision_info(self._model, reference)
        for entry in self._entries:
            if entry.info.reference == info.reference:
                return cast(T, _snapshot(entry.info, entry.value))
        raise KeyError(info.key)

    def edit(self, reference: Decision[T] | DecisionRef[T], value: T) -> SelectionChange:
        info = cast(DecisionInfo[object], decision_info(self._model, reference))
        _recognize(info, value)
        return SelectionChange(self._model, info, False, _snapshot(info, value))

    def remove(self, reference: Decision[T] | DecisionRef[T]) -> SelectionChange:
        info = cast(DecisionInfo[object], decision_info(self._model, reference))
        return SelectionChange(self._model, info, True)

    def with_changes(self, changes: Iterable[SelectionChange]) -> Selection:
        pending = tuple(changes)
        seen: set[str] = set()
        # Validate the complete change list before copying values or running adapters.
        for change in pending:
            if not isinstance(change, SelectionChange):
                raise RequestError("selection changes must come from edit() or remove()")
            if change._model.linked is not self._model.linked:
                raise RequestError("selection change belongs to a different compiled model")
            info = decision_info(self._model, change._info.reference)
            if info.key != change._info.key or info.key in seen:
                raise RequestError(f"duplicate or incompatible selection change {info.key!r}")
            seen.add(info.key)
        values = {entry.info.key: (entry.info, entry.value) for entry in self._entries}
        for change in pending:
            if change._remove:
                values.pop(change._info.key, None)
            else:
                values[change._info.key] = (change._info, change._value)
        # A changed selector deliberately retains old case commitments until removed.
        return Selection._from_values(self._model, values.values())


def capture(point: Space) -> Selection:
    """Capture the shared root's committed owners, without querying unrelated work."""
    current = state(point)
    root = point.root
    values: list[tuple[DecisionInfo[object], object]] = []
    with current.snapshot.lock:
        committed = set(current.snapshot.assignments)
        for info in decisions(current.model):
            if current.model.resolve(0, info.reference) not in committed:
                continue
            answer = root.decision_state(info.reference)
            if isinstance(answer, Decided) and answer.value.status == "committed":
                values.append((info, answer.value.value))
    return Selection._from_values(current.model, values)


def restore(base: S, selection: Selection) -> RefinementReport[S]:
    """Validate a detached request and atomically replay it on a root checkpoint."""
    current = state(base)
    if base._scope != 0:
        raise RequestError("selection restore requires a root occurrence")
    if not isinstance(selection, Selection):
        raise RequestError("restore requires a Selection")
    if selection._model.linked is not current.model.linked:
        raise RequestError("selection belongs to a different compiled model")
    seen: set[str] = set()
    for entry in selection._entries:
        info = decision_info(current.model, entry.info.reference)
        if info.key != entry.info.key or info.key in seen:
            raise RequestError(f"duplicate or incompatible selection key {info.key!r}")
        seen.add(info.key)
    entries = selection.entries
    return base.refine(*(base.edit(entry.reference, entry.value) for entry in entries))


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


__all__ = [
    "Selection",
    "SelectionChange",
    "SelectionEntry",
    "capture",
    "replace_owned",
    "restore",
]
