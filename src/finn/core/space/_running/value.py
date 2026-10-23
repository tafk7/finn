# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A design space as a plain immutable value: its facts layer (the compiled Space class
and its facts) and its stored choices, dormant ones included. Two design spaces with the
same layer and stored choices are equal, whatever led to them, and answer alike.

- ``FactsLayer(model, facts)``: one per model and facts, interned in the model's record
  (``structure.Record.layers``). Every fact is known: a root formal not given takes its
  default (``None`` for an optional fact), so a fact left out and one given its default
  are one layer. It holds what reads facts alone, for every design space
  with those facts: the membership readings (a choice's membership read as if the choice
  existed, an evaluation of the layer with nothing stored) and the domains' outcomes. Both
  are the layer's memo, part of the value, never a cache.
- ``DesignSpace(layer, stored)``: equal by value (each value of one type and equal by ``==``),
  hashed by value, its hash computed once. What an evaluation of it answered hangs off it,
  by session, and is never part of the value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .answers import _same_value
from .structure import record

if TYPE_CHECKING:
    from .._configuration import Space
    from .._scopes import Address, Scopes
    from ..compiler import Model
    from .answers import Called


def _frozen(mapping: Mapping[Address, object]) -> Mapping[Address, object]:
    return MappingProxyType(dict(mapping))


def hashkey(value: object) -> object:
    """``value`` as a dictionary key that tells types apart (``True`` is not ``1``, even
    inside a tuple); ``None`` where it cannot be hashed."""
    if type(value) is tuple:
        parts = tuple(hashkey(item) for item in value)
        return None if any(part is None for part in parts) else (tuple, parts)
    try:
        hash(value)
    except TypeError:
        return None
    return (type(value), value)


_EXACT = (int, str, bool, bytes, type(None))


def _exact(value: object) -> object:
    """``value``'s part of a design space's hash: the value itself where its type's
    equality is exact (a plain int, string, bool, bytes or ``None``, or a tuple or frozen
    set of them), else nothing. The same value (``answers._same_value``) then always hashes
    alike."""
    kind = type(value)
    if kind in _EXACT:
        return (kind, value)
    if kind is tuple or kind is frozenset:
        parts = [_exact(item) for item in value]  # type: ignore[attr-defined]
        if all(part is not None for part in parts):
            return (kind, tuple(parts) if kind is tuple else frozenset(parts))
    return None


def _part(values: Mapping[Address, object]) -> frozenset[tuple[Address, object]]:
    return frozenset((key, _exact(value)) for key, value in values.items())


def _same(mine: Mapping[Address, object], theirs: Mapping[Address, object]) -> bool:
    """The same keys, each value the same (``answers._same_value``: of one type and equal by
    ``==``)."""
    if mine.keys() != theirs.keys():
        return False
    return all(_same_value(key, mine[key], theirs[key]) for key in mine)


class FactsLayer:
    """A compiled Space class with its facts (root formals given, by address), and what reads
    them alone: the membership readings (``evaluate.Evaluation.membership``, kept on the
    layer's design space with nothing stored, ``bare``) and the domains' outcomes
    (``outcomes``, by address, purpose and the answers of their inputs). Equal by value: the
    same model, each fact the same value. Made by ``of``, interned per model."""

    __slots__ = ("model", "facts", "outcomes", "_bare", "_hash", "__weakref__")

    def __init__(self, model: Model[Space], facts: Mapping[Address, object]) -> None:
        self.model = model
        self.facts = _frozen(facts)
        self.outcomes: dict[object, Called] = {}
        self._bare: DesignSpace | None = None
        self._hash = hash((model, _part(self.facts)))

    @staticmethod
    def of(model: Model[Space], facts: Mapping[Address, object]) -> FactsLayer:
        """The layer of ``model`` with ``facts``, each root formal not given at its default
        (a formal's member record holds it): the one the model's record holds for facts
        with the same hash keys and equal by value, else a new one, interned there
        (unhashable facts give a new layer each time, equal by value all the same)."""
        facts = {
            **{
                index: index.value
                for index in model.scopes.parameters
                if not index.required and index not in facts
            },
            **facts,
        }
        layers = record(model).layers
        key = frozenset((index, hashkey(value)) for index, value in facts.items())
        hashable = all(part[1] is not None for part in key)
        if hashable:
            found = layers.get(key)
            if found is not None and _same(found.facts, facts):
                return found
        made = FactsLayer(model, facts)
        if hashable and key not in layers:
            layers[key] = made
        return made

    @property
    def scopes(self) -> Scopes:
        return self.model.scopes

    @property
    def bare(self) -> DesignSpace:
        """The design space of this layer with nothing stored, where the membership
        readings are made."""
        if self._bare is None:
            self._bare = DesignSpace(self, {})
        return self._bare

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactsLayer):
            return NotImplemented
        if other is self:
            return True
        return (
            other.model == self.model
            and other._hash == self._hash
            and _same(self.facts, other.facts)
        )

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"FactsLayer({self.model.space_type.__qualname__}, {len(self.facts)} facts)"


@dataclass(frozen=True, eq=False)
class DesignSpace:
    """A design space: its facts layer and its stored choices (by Decision, dormant
    ones included). Equality is by value: an equal layer and the same stored keys, each
    value the same."""

    layer: FactsLayer
    stored: Mapping[Address, object]
    # What an evaluation of it answered (``evaluate.Evaluation``), by session and the
    # guards it takes as holding: never part of the value.
    evaluations: dict[object, Any] = field(default_factory=dict, repr=False, compare=False)
    _hash: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stored", _frozen(self.stored))
        object.__setattr__(self, "_hash", hash((self.layer, _part(self.stored))))

    @property
    def model(self) -> Model[Space]:
        return self.layer.model

    @property
    def facts(self) -> Mapping[Address, object]:
        return self.layer.facts

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DesignSpace):
            return NotImplemented
        if other is self:
            return True
        return (
            other._hash == self._hash
            and other.layer == self.layer
            and _same(self.stored, other.stored)
        )

    def __hash__(self) -> int:
        return self._hash

    def with_stored(self, stored: Mapping[Address, object]) -> DesignSpace:
        return DesignSpace(self.layer, stored)

    # -- addresses: (scope path, member) as today's keys spell them

    def node(self, key: str) -> Address:
        """The address a key names (``"mm1.pe"``: scope path, member)."""
        return self.model.scopes.find(key)

    def key(self, node: Address) -> str:
        return node.key


__all__ = ["DesignSpace", "FactsLayer", "hashkey"]
