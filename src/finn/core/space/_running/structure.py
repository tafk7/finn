# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-model record: what the running type derives from a compiled model once, held
weakly by the model's structure, so that it lives exactly as long as the model.

- each formal bound by reference, with its scope and the node it is bound to (a formal
  bound to a formal is followed hop by hop, each hop checked against the reader:
  ``source``);
- each scope with its ancestors;
- the flow order's ranks;
- the facts layers made for the model (``value.FactsLayer``), interned by their facts'
  hash keys and held weakly by value, so that a layer lives as long as a design space
  holds it.

The record is keyed by the model's structure (``Model.linked``), made with the model and
never shared by another: the compiled model itself is a slotted frozen dataclass, which
takes no weak reference. Nothing here is keyed by ``id()``, and the record holds nothing
of a design space: a dead model's id may be reused by a new one, which then starts with a
record of its own.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..ir import NodeId

if TYPE_CHECKING:
    from .._configuration import Space
    from .._structure import Structure
    from ..compiler import Model
    from .value import FactsLayer


class Record:
    """One model's derived tables (see the module docstring). Each is filled as it is
    read; nothing in it refers to the model or its structure, which key it weakly."""

    __slots__ = ("_bound", "_within", "_ranks", "layers")

    def __init__(self) -> None:
        # by node: ``None`` where it is no formal bound by reference, else (its scope, the
        # node it is bound to)
        self._bound: dict[NodeId, tuple[int, NodeId] | None] = {}
        # by scope: the scope and every scope it is placed below
        self._within: dict[int, frozenset[int]] = {}
        self._ranks: Mapping[NodeId, int] | None = None
        self.layers: weakref.WeakValueDictionary[object, FactsLayer] = weakref.WeakValueDictionary()

    def bound(self, linked: Structure, index: NodeId) -> tuple[int, NodeId] | None:
        """Where ``index`` is a formal bound by reference (to a member or a named shared
        Decision: an alias whose template node is a formal of a placed Space), its scope
        and the node it is bound to; else ``None``."""
        if index in self._bound:
            return self._bound[index]
        found: tuple[int, NodeId] | None = None
        node = linked.raw_nodes[index]
        if node.kind == "alias" and node.output is not None:
            record, local = linked.canonical(index)
            if record.parent is not None and record.template.nodes[local].kind == "param":
                found = (node.scope, node.output)
        self._bound[index] = found
        return found

    def within(self, linked: Structure, scope: int) -> frozenset[int]:
        """``scope`` and every scope it is placed below, to the root."""
        found = self._within.get(scope)
        if found is None:
            parent = linked.scopes[scope].parent
            above = frozenset() if parent is None else self.within(linked, parent)
            found = self._within[scope] = above | {scope}
        return found

    def source(self, linked: Structure, index: NodeId, scope: int) -> NodeId:
        """What a reader in ``scope`` reads when it reads ``index``: a formal bound by
        reference is read at the node it is bound to where ``scope`` is the formal's
        scope or one placed below it; a formal bound to a formal is followed while each
        hop holds for the same reader. Anything else is read as itself."""
        within: frozenset[int] | None = None
        while True:
            bound = self.bound(linked, index)
            if bound is None:
                return index
            if within is None:
                within = self.within(linked, scope)
            formal_scope, target = bound
            if formal_scope not in within:
                return index
            index = target

    def ranks(self, model: Model[Space]) -> Mapping[NodeId, int]:
        """Each choice's position in the flow order (today's, from the templates)."""
        if self._ranks is None:
            self._ranks = {index: rank for rank, index in enumerate(model.evaluator.flow())}
        return self._ranks


_RECORDS: weakref.WeakKeyDictionary[object, Record] = weakref.WeakKeyDictionary()
"""Each live model's record, by its structure."""


def record(model: Model[Space]) -> Record:
    """``model``'s record, made on first use."""
    found = _RECORDS.get(model.linked)
    if found is None:
        found = _RECORDS[model.linked] = Record()
    return found


def records() -> int:
    """How many models' records are alive (for the tests of their lifetime)."""
    return len(_RECORDS)


__all__ = ["Record", "record", "records"]
