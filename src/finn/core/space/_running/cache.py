# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The node cache: the one reuse of answers across design spaces, behind one contract.

**The contract.** An answer of a design space is a function of the values of the choices
and facts its evaluation read there, directly or through anything it read (purity:
authored functions answer from what they read alone). The cache keeps, per member of a
compiled Space class, the sets of choices and facts the member was found to read, and for
each set the answer by those values. A design space asks the cache before it evaluates a
member: for each kept set it takes its own values of that set and looks them up by hash. A
hit is the answer; a miss is an evaluation, kept after. Values match by type and ``==``,
and only where their choice's or fact's value semantics compare by ``==``. The cache
belongs to no design space, only grows, and may drop any entry at any time: a miss changes
nothing but the work. So a design space answers the same with it as without it.

**What it keeps.** Its tables are per model, held weakly by the model's structure (the
object the model's record is keyed by, ``structure.record``), so a model's tables go with
the model, and a new model never finds an old one's, whatever its ``id()``. A table is
keyed by member, (the guards taken as holding, node), ``None`` for a design space's own
evaluation: a membership reading, made with its choice's existence guards taken as
holding (none for a choice with no guard), is a member of its own. Each member keeps its
read sets (the choices and facts its evaluation read, by node, sorted), and each read set
the answers (``evaluate.Ans``, with their trace and their read set) by the values read:
each value by ``key`` (type and ``==`` under semantics it trusts; a tuple or frozen set
item by item, a value class field by field), a choice or fact not given as such.

**Modes**, fixed when the cache is made (there is no global):

- on: no bound (``NodeCache()``);
- bounded: at most ``bound`` read sets kept per member, the oldest dropped for a new one
  (``NodeCache(bound=n)``);
- dropping: a seeded random eviction at the rate ``drop``, at lookup (a hit is dropped,
  and answers as a miss) and at keep (the answer is not kept)
  (``NodeCache(drop=rate, seed=s)``).

With no cache, a session evaluates every member (``evaluate.Session(cache=None)``). Under
the purity check (the session's, ``evaluate.Session.verifying``), any of these verifies: a
hit is evaluated anyway and compared with what was kept, by the member's value semantics.

**Values it keys** (``key``): a value of a choice or fact whose value semantics it trusts
(``trusted``): the default ones (recognition by exact type, equality by ``==``, a default
snapshot), or semantics whose equality is ``==`` itself, as their owner declares by
``equal=operator.eq``. What is keyed is the value stored, after its semantics' snapshot.
The trust widens which semantics are keyed, not which types: under them the cache keys, by
type and ``==``, the plain types (``PLAIN``) and the value types (``is_value_type``: a class
registered by its owner, ``finn.core.space.value_type``, whose ``==`` and hash are its
semantics, checked by the owner's test); an enum member, by its type and its name
(``enum_key``), whatever its class mixes in or overrides; tuples and frozen sets of keyed
values, item by item; and an instance of a value class (``value_class``: a frozen dataclass
whose ``==`` and hash are the ones ``dataclasses`` generates, and none of whose compared
fields is annotated with value semantics the cache does not trust), field by field, each
field's value keyed in turn (a value type held as a field included). Item by item and
field by field, the key tells types apart (``True`` is not ``1``, and a value type whose
``==`` holds it equal to a string is not that string): it is never coarser than ``==``, so
a hit is never a value ``==`` tells apart, nor, under semantics that compare by ``==``, a
value the semantics tell apart. An enum member's key is finer still: an equal key is the
same member, so an ``==`` or a hash of its class's own or a mixin's (``class E(int, Enum)``
compares as an ``int``) can only make a hit a miss. A value of a choice or fact with
semantics whose equality is another function (``ValueSemantics(...)``,
``immutable_nominal``), which may tell apart values ``==`` holds equal, is never hashed,
whatever its type; nor is a value of any other type: it has no key, and nothing is kept
for an answer whose read set holds one, so a lookup there is a miss. An unhashable value
has no key either.

The check is made where the semantics are: a value's semantics are its choice's or fact's
(``Param(semantics=...)``, an annotation's), not its type's, and one class may be held by
one member under the default semantics and by another under its own. So ``key`` is given
the semantics of the choice or fact read (the node's, as ``_world._same_value`` reads it),
and ``value_class`` checks what a class itself declares: its fields' annotations, its
``==`` and hash. The facts layer keys the domains' outcomes by the same
``key`` (``evaluate.Evaluation.domain_outcome``).

**Counters**, on the session's (``evaluate.Counters``): lookups, hits, values hashed (at
lookup and at keep), the members holding a read set, the read sets kept, and the most read
sets one member held at once.

Single-threaded, as the evaluator is: nothing here takes a lock.
"""

from __future__ import annotations

import dataclasses
import operator
import random
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Any, cast, get_args, get_type_hints

from .._value_types import is_value_type
from ..ir import NodeId
from ..semantics import ValueSemantics, default_semantics
from .value import hashkey

if TYPE_CHECKING:
    from .evaluate import Ans, Evaluation

PLAIN: tuple[type, ...] = (type(None), bool, int, float, complex, str, bytes)
"""The plain types the cache keys, each checked by the value-type test."""

_NOT_GIVEN = ("not given",)

Member = tuple[frozenset[NodeId] | None, NodeId]
ReadSet = tuple[NodeId, ...]


@dataclasses.dataclass(frozen=True)
class _Generated:
    """A frozen dataclass: where ``dataclasses`` compiles the methods it generates."""


_GENERATED = _Generated.__eq__.__code__.co_filename


def _generated(method: object) -> bool:
    code = getattr(method, "__code__", None)
    return code is not None and code.co_filename == _GENERATED


_DEFAULT = default_semantics(object)
_DEFAULT_SNAPSHOTS = tuple(default_semantics(kind).snapshot for kind in (int, tuple, object))


def _code(function: object) -> object:
    return getattr(function, "__code__", None)


def trusted(semantics: ValueSemantics[Any] | None) -> bool:
    """Whether the cache keys values under ``semantics``: the default ones
    (``semantics.default_semantics``: recognition by exact type, equality by ``==`` and a
    snapshot that keeps it), or semantics whose equality is ``operator.eq``, by which their
    owner declares that two stored values are equal exactly when ``==`` holds them equal.
    Other semantics of a value's own (``ValueSemantics(...)`` with another ``equal``,
    ``immutable_nominal``) or none (equal by identity alone) are not, whatever their
    functions compute: the cache cannot read a function's equality, so it keys none of
    their values."""
    if semantics is None:
        return False
    if semantics.equal is operator.eq:
        return True
    return (
        _code(semantics.recognizes) is _code(_DEFAULT.recognizes)
        and _code(semantics.equal) is _code(_DEFAULT.equal)
        and any(semantics.snapshot is snapshot for snapshot in _DEFAULT_SNAPSHOTS)
    )


def _own_semantics(annotation: object) -> bool:
    """Whether ``annotation`` names value semantics the cache does not trust (``trusted``), at
    any depth (``Annotated[tuple[int, int], PAIR]``, as a projection reads it)."""
    if isinstance(annotation, ValueSemantics):
        return not trusted(annotation)
    return any(_own_semantics(part) for part in get_args(annotation))


_VALUE_CLASSES: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()
"""``value_class``'s answer by class, made once."""


def value_class(kind: type) -> bool:
    """Whether the cache keys ``kind``'s instances field by field: a frozen dataclass
    (declared so itself, not by a base) whose ``__eq__`` and ``__hash__`` are the ones
    ``dataclasses`` generates, comparing the fields' values with ``==``, and none of whose
    compared fields is annotated with value semantics the cache does not trust
    (``_own_semantics``). A hand-written ``__eq__`` or hash, an ``eq=False`` dataclass
    (equal by identity), a mutable one, a field with such semantics or annotations that do
    not resolve is not."""
    found = _VALUE_CLASSES.get(kind)
    if found is None:
        found = _VALUE_CLASSES[kind] = _value_class(kind)
    return found


def _value_class(kind: type) -> bool:
    params = kind.__dict__.get("__dataclass_params__")
    if params is None or not (params.frozen and params.eq):
        return False
    if not (_generated(kind.__dict__.get("__eq__")) and _generated(kind.__dict__.get("__hash__"))):
        return False
    try:
        hints = get_type_hints(kind, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolvable annotations: not keyed, a miss
        return False
    return not any(
        _own_semantics(hints.get(item.name))
        for item in dataclasses.fields(cast(Any, kind))
        if item.compare
    )


def enum_key(member: Enum) -> object:
    """``member`` as the cache keys it: by its type and its name, whatever its class mixes
    in or overrides; ``None`` where its name does not find it among its class's members (a
    combination of flags).

    Why it is safe: a type and a name found among its members are one member, so an equal
    key means the same member, which equals itself under any reflexive ``==`` (and
    ``_world._same_value`` holds a value the same as itself). An ``==`` of the class's own or
    a mixin's (``class E(int, Enum)`` compares as an ``int``) can only make different
    members equal, and these keep different keys: a miss, never a wrong hit. The key hashes
    the name, a ``str``: it does not trust an overridden ``__hash__``."""
    kind, name = type(member), member.name
    if type(name) is not str or kind.__members__.get(name) is not member:
        return None
    return (kind, name)


def key(value: object, semantics: ValueSemantics[Any] | None) -> object:
    """``value``, the value of a choice or fact with ``semantics``, as the cache keys it,
    by type and ``==`` (see the module docstring); ``None`` where the cache does not trust
    the semantics (``trusted``), or the value is unhashable or holds a type not keyed."""
    return _key(value) if trusted(semantics) else None


def _key(value: object) -> object:
    """``value`` by type and ``==``: a plain type or a value type itself, an enum member by
    type and name, a tuple or a frozen set item by item, a value class field by field;
    ``None`` where it is unhashable or holds a type not keyed."""
    kind = type(value)
    if kind in PLAIN:
        return hashkey(value)
    if isinstance(value, Enum):
        return enum_key(value)
    if kind is tuple or kind is frozenset:
        parts = [_key(item) for item in value]  # type: ignore[attr-defined]
        if any(part is None for part in parts):
            return None
        return (kind, tuple(parts) if kind is tuple else frozenset(parts))
    if value_class(kind):
        fields = [
            _key(getattr(value, item.name))
            for item in dataclasses.fields(cast(Any, kind))
            if item.compare
        ]
        if any(part is None for part in fields):
            return None
        return (kind, tuple(fields))
    if is_value_type(kind):
        return hashkey(value)
    return None


class NodeCache:
    """A node cache in one mode (see the module docstring): ``lookup`` before a member is
    evaluated, ``keep`` after."""

    __slots__ = ("bound", "drop", "seed", "_random", "_tables")

    def __init__(self, *, bound: int | None = None, drop: float = 0.0, seed: int = 0) -> None:
        if bound is not None and bound < 1:
            raise ValueError(f"a bound keeps at least one read set per member, not {bound}")
        if not 0.0 <= drop <= 1.0:
            raise ValueError(f"a drop rate is between 0 and 1, not {drop}")
        self.bound = bound
        self.drop = drop
        self.seed = seed
        self._random = random.Random(seed)
        # by model structure (weakly), member, read set, the values read
        self._tables: weakref.WeakKeyDictionary[
            object, dict[Member, dict[ReadSet, dict[object, Ans]]]
        ] = weakref.WeakKeyDictionary()

    @property
    def mode(self) -> str:
        """``on``, ``bounded N``, ``dropping R (seed S)``, or both of the last two."""
        parts = [] if self.bound is None else [f"bounded {self.bound}"]
        if self.drop:
            parts.append(f"dropping {self.drop} (seed {self.seed})")
        return ", ".join(parts) or "on"

    def models(self) -> int:
        """How many models' tables are alive (for the tests of their lifetime)."""
        return len(self._tables)

    def holds(self, ev: Evaluation) -> bool:
        """Whether the cache holds a table for ``ev``'s model."""
        return ev.linked in self._tables

    def answers(self) -> int:
        """How many answers are kept, every model's."""
        return sum(
            len(answers)
            for table in self._tables.values()
            for sets in table.values()
            for answers in sets.values()
        )

    def _drops(self) -> bool:
        return bool(self.drop) and self._random.random() < self.drop

    def _values(self, ev: Evaluation, read: ReadSet) -> tuple[object, ...] | None:
        """``ev``'s values of ``read``, each keyed (``key``) by its choice's or fact's
        semantics; ``None`` where one has no key."""
        facts, stored, nodes = ev.ds.facts, ev.ds.stored, ev.raw
        ev.counters.hashed += len(read)
        parts: list[object] = []
        for index in read:
            value = facts.get(index, _NOT_GIVEN)
            if value is _NOT_GIVEN:
                value = stored.get(index, _NOT_GIVEN)
            if value is _NOT_GIVEN:
                parts.append(_NOT_GIVEN)
                continue
            part = key(value, nodes[index].semantics)
            if part is None:
                return None
            parts.append(part)
        return tuple(parts)

    def lookup(self, ev: Evaluation, index: NodeId) -> Ans | None:
        """The answer kept for ``index`` in ``ev`` under one of its member's read sets,
        by ``ev``'s values of that set; ``None`` (a miss) where none is."""
        ev.counters.lookups += 1
        table = self._tables.get(ev.linked)
        if table is None:
            return None
        sets = table.get((ev.assumed, index))
        if not sets:
            return None
        for read, answers in sets.items():
            values = self._values(ev, read)
            if values is None:
                continue
            found = answers.get(values)
            if found is None:
                continue
            if self._drops():
                del answers[values]
                if not answers:
                    del sets[read]
                return None
            ev.counters.hits += 1
            return found
        return None

    def keep(self, ev: Evaluation, index: NodeId, found: Ans) -> None:
        """``found``, ``index``'s answer in ``ev``, kept by ``ev``'s values of its read
        set (nothing kept where one has no key)."""
        read = tuple(sorted(found.reads))
        values = self._values(ev, read)
        if values is None or self._drops():
            return
        table = self._tables.get(ev.linked)
        if table is None:
            table = self._tables[ev.linked] = {}
        member = (ev.assumed, index)
        sets = table.get(member)
        if sets is None:
            sets = table[member] = {}
            ev.counters.members += 1
        answers = sets.get(read)
        if answers is None:
            if self.bound is not None and len(sets) >= self.bound:
                del sets[next(iter(sets))]
            answers = sets[read] = {}
            ev.counters.sets += 1
            ev.counters.most_sets = max(ev.counters.most_sets, len(sets))
        answers[values] = found


__all__ = ["PLAIN", "NodeCache", "enum_key", "key", "trusted", "value_class"]
