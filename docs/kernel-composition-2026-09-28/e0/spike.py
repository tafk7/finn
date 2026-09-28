# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""E0 spike: narrowing and pinning a Decision over nodes by key, as E1 would.

Today an enclosing body narrows a Decision over nodes only by a *new*
Decision over *fresh* nodes, whose bindings are read in the enclosing body.
A refined Decision's shared bindings name the inner body's streams, which the
enclosing body cannot reach (a reference input names a node placed beside
it), so that form cannot narrow it. The spike adds what E1 needs, treating the
selector as what it is, a ``str`` Decision over the keys:

- ``mm.compute = "packed"`` pins the key: the selector becomes a constant,
  its key is listed as pinned and refused as stale, the pinned candidate is
  selected;
- ``mm.compute = Decision(values=("packed", "stub"))`` narrows the key
  domain; the key stays, under the same name.

Either keeps every declared candidate node, with its bindings read in the body
that declared it; candidates outside the narrowed keys stay placed and are
never selectable (inapplicable). The two engine functions are patched only
inside ``engine_spike()``; ``src/`` is unchanged.
"""

# ruff: noqa: SLF001

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, cast
from unittest import mock

from finn.core.space import Decision as EngineDecision
from finn.core.space import _linker, _nodes
from finn.core.space.domains import finite
from finn.core.space.errors import DefinitionError
from finn.core.space.ir import Argument


@dataclass(frozen=True)
class KeySelection:
    """An enclosing body's pin (one key) or narrowing (several) of a Decision over nodes."""

    keys: tuple[str, ...]
    pin: bool

    def __repr__(self) -> str:
        return repr(self.keys[0]) if self.pin else f"Decision(values={self.keys!r})"


_original_choice_supplier = _nodes._choice_supplier
_original_choice = _linker._Linker.choice


def _choice_supplier(original: _nodes.NodeDecision, value: object, label: str) -> object:
    if isinstance(value, str):
        keys: tuple[str, ...] = (value,)
        pin = True
    elif isinstance(value, EngineDecision) and not isinstance(value, _nodes.NodeDecision):
        found = value.domain._finite_values
        if found is None or any(type(item) is not str for item in found):
            raise DefinitionError(
                f"{label}: narrow a Decision over nodes with Decision(values=(<keys>...))"
            )
        keys, pin = tuple(cast(tuple[str, ...], found)), False
    else:
        return _original_choice_supplier(original, value, label)
    unknown = [key for key in keys if key not in original.candidates]
    if unknown:
        raise DefinitionError(
            f"{label}: {unknown} are not cases of {original.describe()}; an override narrows "
            "the cases, it does not add one"
        )
    return KeySelection(keys, pin)


def _choice(self: Any, scope: Any, name: str, declared: _nodes.NodeDecision) -> None:
    """``_Linker.choice`` with a key selection: keep the declared candidates."""
    slot = scope.slots.get(name)
    if slot is None or not isinstance(slot.supplier, KeySelection):
        _original_choice(self, scope, name, declared)
        return
    selection = cast(KeySelection, slot.supplier)
    key = _linker._key(scope.name, name)
    decision, source, writer = declared, scope.index, scope.index
    guard = self.guarded(
        scope.index,
        scope.guard,
        scope.effective.guards.get(declared),
        key + ".$guard",
        owner=key,
    )
    selector = self.reserve(
        scope.index,
        key,
        "const" if selection.pin else "decision",
        _linker._STRING,
        guard=guard,
        origin=decision.origin,
    )
    if selection.pin:
        self.nodes[selector] = replace(self.nodes[selector], value=selection.keys[0])
        self.pinned[key] = slot.provenance
    else:
        self.nodes[selector] = replace(
            self.nodes[selector], domain=finite(selection.keys, _linker._STRING)
        )
    self.provenance[selector] = slot.provenance
    scope.named_members[name] = selector
    choice = _linker._ChoiceDraft(len(self.choice_drafts), scope.index, key, guard, selector)
    self.choice_drafts.append(choice)
    for alias in self.aliases[scope.effective.space_type][name]:
        scope.choices[alias] = choice.index
        scope.members[alias] = selector
    for case, record in decision.candidates.items():
        case_key = _linker._key(key, case)
        if record is None:
            choice.cases.append((case, None))
            continue
        case_guard = self.reserve(
            scope.index,
            case_key + ".$selected",
            "derived",
            _linker._BOOL,
            guard=guard,
            source_owner=case_key,
        )
        self.nodes[case_guard] = replace(
            self.nodes[case_guard],
            function=_linker._matches(case),
            arguments=(Argument("selected", selector),),
        )
        condition = scope.effective.guards.get(record, record.when)
        case_guard = self.guarded(
            source, case_guard, condition, case_key + ".$guard", owner=case_key
        )
        child = self.place(
            scope,
            f"{name}.{case}",
            record,
            case_guard,
            source_scope=source,  # the declaring body: its bindings stay readable
            writer=writer,
            keys=(record,),
        )
        choice.cases.append((case, child))
    self.shared_members(scope, choice)


@contextmanager
def engine_spike() -> Iterator[None]:
    """Narrowing and pinning by key, for declarations made and compiled inside the block."""
    with (
        mock.patch.object(_nodes, "_choice_supplier", _choice_supplier),
        mock.patch.object(_linker._Linker, "choice", _choice),
    ):
        yield


__all__ = ["KeySelection", "engine_spike"]
