# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The per-model record: what the running type derives from a compiled model once, held
by the model's scope records (``Model.scopes.record``), so that it lives exactly as long as
the model.

- each formal bound by reference, and the address it is bound to (a formal bound to a
  formal is followed hop by hop, each hop checked against the reader: ``source``); a
  formal taking fewer values than what it is bound to is read as itself (``narrows``);
- the flow order's ranks;
- the facts layers made for the model (``value.FactsLayer``), interned by their facts'
  hash keys and held weakly by value, so that a layer lives as long as a design space
  holds it.

A scope's ancestry is its path's prefixes (``ScopeRecord.within``). The record is held by
the model's scope records, made with the model and never shared by another: what it holds
(addresses) reaches the model through its classes, so it is not keyed weakly by the model
in a table of its own, which could never drop. Nothing here is keyed by ``id()``, and the
record holds nothing of a design space: a dead model's id may be reused by a new one, which
then starts with a record of its own.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from ..semantics import OptionalSemantics

if TYPE_CHECKING:
    from .._configuration import Space
    from .._scopes import Address, ScopeRecord
    from ..compiler import Model
    from .value import FactsLayer


class Record:
    """One model's derived tables (see the module docstring). Each is filled as it is
    read."""

    __slots__ = ("_bound", "_ranks", "layers", "__weakref__")

    def __init__(self) -> None:
        # by address: ``None`` where it is no formal bound by reference, else the address
        # it is bound to
        self._bound: dict[Address, Address | None] = {}
        self._ranks: Mapping[Address, int] | None = None
        self.layers: weakref.WeakValueDictionary[object, FactsLayer] = weakref.WeakValueDictionary()

    def bound(self, index: Address) -> Address | None:
        """Where ``index`` is a formal bound by reference (a formal of a placed scope whose
        placement binds it to something: an alias there), the address it is bound to; else
        ``None``. A formal that ``narrows`` is read as itself."""
        if index in self._bound:
            return self._bound[index]
        found: Address | None = None
        scope, member = index
        if scope.parent is not None and member in scope.compiled.formal_names:
            if index.kind == "alias" and not narrows(index):
                found = index.output
        self._bound[index] = found
        return found

    def source(self, index: Address, scope: ScopeRecord) -> Address:
        """What a reader in ``scope`` reads when it reads ``index``: a formal bound by
        reference is read at the address it is bound to where ``scope`` is the formal's
        scope or one placed below it; a formal bound to a formal is followed while each
        hop holds for the same reader. Anything else is read as itself."""
        while True:
            target = self.bound(index)
            if target is None or not scope.within(index[0]):
                return index
            index = target

    def ranks(self, model: Model[Space]) -> Mapping[Address, int]:
        """Each choice's position in the flow order (today's, from the compiled forms)."""
        if self._ranks is None:
            self._ranks = {index: rank for rank, index in enumerate(model.per_class.flow())}
        return self._ranks


def narrows(index: Address) -> bool:
    """Whether ``index``, an alias, takes fewer values than what it reads: a ``T`` formal
    bound to a ``T | None`` value (the two types are compatible, so the binding is a
    definition). Such a formal is no reference: it is read as itself, and recognizes the
    value it reads by its own type check (``evaluate._forward``)."""
    output = index.output
    if index.kind != "alias" or output is None:
        return False
    return isinstance(output.semantics, OptionalSemantics) and not isinstance(
        index.semantics, OptionalSemantics
    )


_RECORDS: weakref.WeakSet[Record] = weakref.WeakSet()
"""Each live model's record (for the tests of their lifetime)."""


def record(model: Model[Space]) -> Record:
    """``model``'s record, made on first use."""
    scopes = model.scopes
    found = cast("Record | None", scopes.record)
    if found is None:
        found = scopes.record = Record()
        _RECORDS.add(found)
    return found


def records() -> int:
    """How many models' records are alive (for the tests of their lifetime)."""
    return len(_RECORDS)


__all__ = ["Record", "narrows", "record", "records"]
