# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A value's type check, its sameness, and its canonical form where it enters a design
space.

Values are immutable, and ``==`` with ``hash`` is their semantics. Two values of a member
are the same exactly when they are of one type and ``==`` holds them equal (``same``); the
engine compares, interns and keys values by that alone, and copies none of them.

**The type check** is the member's annotation (``annotation_type``): a class recognizes its
instances (``isinstance``, a ``bool`` not taken for an ``int``), a parametrized generic its
origin's (``tuple[int, ...]`` is a ``tuple``; its items are walked by the canonical form,
not typed), ``T | None`` an optional value (``optional``: ``None`` first, by identity), and
a union of several classes any of its members' (``union``). A protocol, a type variable or
a special form is no value type, and a dataclass leaving a field out of ``==``
(``compare=False``) is refused as one. ``ValueSemantics`` holds a check: its type token (what
compatibility compares), its name and its recognizer. Views still state theirs
(``view(semantics=...)``, ``ViewKey``), until they are read out of the space.

**The canonical form** (``canonical``) is what a value becomes where it enters a design
space: a fact, a choice, a formal's default, a ``Const``, a literal bound at placement, a
domain's listed value. It walks the value item by item whatever the annotation (a tuple's
items, a frozen set's, a frozen dataclass's fields): ``-0.0`` becomes ``0.0``; NaN is
refused (``==`` does not hold it equal to itself); a mutable value is refused (a list, a
dict, a set, a bytearray, any mutable sequence, mapping or set, a dataclass that is not
frozen, a class without a hash), and so is a dataclass leaving a field out of ``==``. A
registered value type (``value_type``) and an enum member are taken as they are: their
owners state them immutable. A tuple subclass is rebuilt by its ``_make`` (a named tuple's);
one without is refused where an item changes. A frozen dataclass is rebuilt where a field
changes, a field its ``__init__`` does not take (``init=False``) set on the rebuilt value.
"""

from __future__ import annotations

import dataclasses
import math
import types
import weakref
from collections.abc import Callable, MutableMapping, MutableSequence, MutableSet
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Generic,
    NoReturn,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ._value_types import is_value_type
from .errors import DefinitionError, EvaluationError

T = TypeVar("T")


class NoTruthValue:
    """Semantic results require an explicit inspection of their variant."""

    __slots__ = ()

    def __bool__(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} has no truth value; inspect its fields")


@dataclass(frozen=True, slots=True)
class ValueSemantics(Generic[T]):
    """A value type's check: its ``type_token`` (two members are compatible when their
    tokens are equal), its ``name``, and what it ``recognizes``. A member's annotation states
    it (``annotation_type``); a view may state its own."""

    type_token: object
    name: str
    recognizes: Callable[[object], bool]

    def accepts(self, value: object) -> bool:
        from ._failure import transformation  # noqa: PLC0415 - semantics/evaluation

        with transformation("value recognition"):
            return bool(self.recognizes(value))

    def is_compatible_with(self, other: ValueSemantics[object]) -> bool:
        return self.type_token is other.type_token or self.type_token == other.type_token


def default_semantics(value_type: type[T]) -> ValueSemantics[T]:
    """The check of one class: its instances (``isinstance``), a ``bool`` not taken for an
    ``int``."""

    def recognizes(value: object) -> bool:
        return isinstance(value, value_type) and (
            type(value) is not bool or value_type in (bool, object)
        )

    return ValueSemantics(
        type_token=value_type, name=value_type.__qualname__, recognizes=recognizes
    )


@dataclass(frozen=True, slots=True)
class OptionalSemantics(ValueSemantics[T]):
    """The check of an optional value, ``T | None`` (``optional``): ``None``, the absent
    value, checked first and by identity, then ``inner``'s, ``T``'s. Its type token is
    ``inner``'s, so a binding of a ``T`` to it (or of it to a ``T``) is compatible."""

    inner: ValueSemantics[T]


def optional(inner: ValueSemantics[T]) -> OptionalSemantics[T]:
    """``inner``'s check over ``inner``'s values or ``None``, which it sees first, by
    identity: ``inner``'s recognizer never sees it."""

    def recognizes(value: object) -> bool:
        return value is None or inner.recognizes(value)

    return OptionalSemantics(
        type_token=inner.type_token,
        name=f"{inner.name} | None",
        recognizes=recognizes,
        inner=inner,
    )


def union(members: tuple[ValueSemantics[Any], ...]) -> ValueSemantics[Any]:
    """The check of a union of several value types: a value any member recognizes. Its
    token is the set of theirs."""
    tokens = frozenset(member.type_token for member in members)

    def recognizes(value: object) -> bool:
        return any(member.recognizes(value) for member in members)

    return ValueSemantics(
        type_token=tokens,
        name=" | ".join(member.name for member in members),
        recognizes=recognizes,
    )


BOOL = cast(ValueSemantics[object], default_semantics(bool))
STRING = cast(ValueSemantics[object], default_semantics(str))


def annotation_type(annotation: object, label: str, *, owner: str) -> ValueSemantics[Any]:
    """The type check an annotation states (see the module docstring); a ``DefinitionError``
    of ``owner``'s, naming ``label``, where it states none."""
    if annotation is None or annotation is type(None):
        return default_semantics(type(None))
    origin = get_origin(annotation)
    if origin is Annotated:
        return annotation_type(get_args(annotation)[0], label, owner=owner)
    if origin in (Union, types.UnionType):
        members = get_args(annotation)
        present = tuple(member for member in members if member is not type(None))
        checks = tuple(annotation_type(member, label, owner=owner) for member in present)
        found = checks[0] if len(checks) == 1 else union(checks)
        return optional(found) if len(present) < len(members) else found
    nominal = origin if origin is not None else annotation
    if not isinstance(nominal, type) or nominal is Any or getattr(nominal, "_is_protocol", False):
        raise DefinitionError.broken(
            "annotation-not-value-type",
            owner,
            f"{label}: the annotation {annotation!r} is no value type: annotate a class, a "
            "union of classes, or T | None",
            declaration=label,
        )
    uncompared = uncompared_field(nominal)
    if uncompared is not None:
        raise DefinitionError.broken(
            "value-field-uncompared",
            owner,
            f"{label}: {uncompared} is left out of == (compare=False): a value's == and hash "
            "are its semantics, so every field is compared",
            declaration=label,
        )
    return default_semantics(nominal)


_UNCOMPARED: weakref.WeakKeyDictionary[type, str | None] = weakref.WeakKeyDictionary()
"""``uncompared_field``'s answer by class, made once."""


def uncompared_field(kind: type) -> str | None:
    """``Class.field`` of a field a dataclass leaves out of ``==`` (``compare=False``), its
    own or one of a dataclass a field of it is annotated with; ``None`` where there is
    none."""
    if kind not in _UNCOMPARED:
        _UNCOMPARED[kind] = _uncompared(kind, set())
    return _UNCOMPARED[kind]


def _uncompared(kind: type, seen: set[type]) -> str | None:
    if kind in seen or not dataclasses.is_dataclass(kind):
        return None
    seen.add(kind)
    for item in dataclasses.fields(kind):
        if not item.compare:
            return f"{kind.__qualname__}.{item.name}"
    try:
        hints = get_type_hints(kind)
    except Exception:  # noqa: BLE001 - annotations that do not resolve: the value is walked
        return None
    for hint in hints.values():
        for part in (hint, *get_args(hint)):
            found = _uncompared(part, seen) if isinstance(part, type) else None
            if found is not None:
                return found
    return None


def unrecognized(semantics: ValueSemantics[T]) -> str:
    """The one wording for a value ``semantics`` does not recognize."""
    return f"expected value of nominal type {semantics.name}"


def recognize(semantics: ValueSemantics[T], value: object, *, owner: str, role: str) -> bool:
    """Whether ``semantics`` recognizes ``value``; a recognizer that raises fails
    ``owner``."""
    return _owned(lambda: semantics.accepts(value), owner, role)


def same(left: object, right: object, *, owner: str, role: str) -> bool:
    """Whether two values are the same value: the same object, or of one type and equal by
    ``==``; an ``==`` that raises fails ``owner``."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    return _owned(lambda: bool(left == right), owner, role)


R = TypeVar("R")


def _owned(call: Callable[[], R], owner: str, role: str) -> R:
    """``call``'s result; a failure of the function it runs is ``owner``'s, in ``role``.

    An ``EvaluationError`` passes unchanged: the function read a configuration
    (``_failure.transformation``), and that error already names who and where.
    """
    try:
        return call()
    except EvaluationError:
        raise
    except Exception as cause:
        raise EvaluationError(owner, role, str(cause)) from cause


# -- the canonical form ---------------------------------------------------------------------


class NotAValue(ValueError):
    """A value refused where it enters a design space; the message says why, and where in
    the value."""


_ATOMS: frozenset[type] = frozenset((bool, int, str, bytes, type(None), type(Ellipsis), range))
_MUTABLE = (list, dict, set, bytearray, MutableSequence, MutableMapping, MutableSet)

_CLASSES: weakref.WeakKeyDictionary[type, str | None] = weakref.WeakKeyDictionary()
"""``_refused_class``'s answer by dataclass, made once."""


def canonical(value: T) -> T:
    """``value`` in its canonical form (see the module docstring): the value itself where
    nothing in it changes; ``NotAValue`` where it is no value."""
    kind = type(value)
    if kind in _ATOMS:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise NotAValue("NaN is no value: == does not hold it equal to itself")
        negative_zero = value == 0.0 and math.copysign(1.0, value) < 0
        return cast(T, abs(value)) if negative_zero else value
    if isinstance(value, complex):
        real, imag = value.real, value.imag
        parts = (canonical(real), canonical(imag))
        if parts[0] is real and parts[1] is imag:
            return value
        return cast(T, complex(*parts))
    if isinstance(value, Enum) or is_value_type(kind):
        return value
    if isinstance(value, _MUTABLE):
        raise NotAValue(
            f"a {kind.__qualname__} is mutable: a value is immutable (a tuple, a frozen set, "
            "a frozen dataclass)"
        )
    if isinstance(value, tuple):
        return cast(T, _items(cast(tuple[object, ...], value)))
    if isinstance(value, frozenset):
        return cast(T, _members(cast(frozenset[object], value)))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return cast(T, _fields(value))
    if getattr(kind, "__hash__", None) is None:
        raise NotAValue(
            f"a {kind.__qualname__} has no hash: a value's == and hash are its semantics"
        )
    return value


def _items(value: tuple[object, ...]) -> tuple[object, ...]:
    items = []
    for position, item in enumerate(value):
        try:
            items.append(canonical(item))
        except NotAValue as refusal:
            raise NotAValue(f"item {position}: {refusal}") from None
    if all(new is old for new, old in zip(items, value)):
        return value
    kind = type(value)
    if kind is tuple:
        return tuple(items)
    make = getattr(kind, "_make", None)
    if make is None:
        raise NotAValue(
            f"a {kind.__qualname__} is a tuple subclass with no _make: its items' canonical "
            "form cannot be rebuilt"
        )
    return cast(tuple[object, ...], make(items))


def _members(value: frozenset[object]) -> frozenset[object]:
    pairs = []
    for item in value:
        try:
            pairs.append((item, canonical(item)))
        except NotAValue as refusal:
            raise NotAValue(f"a member: {refusal}") from None
    if all(new is old for old, new in pairs):
        return value
    return frozenset(new for _, new in pairs)


def _refused_class(kind: type) -> str | None:
    """Why instances of the dataclass ``kind`` are no values (not frozen, a field left out
    of ``==``); ``None`` where they may be."""
    if kind not in _CLASSES:
        params = getattr(kind, "__dataclass_params__", None)
        found: str | None = None
        if params is None or not params.frozen:
            found = f"a {kind.__qualname__} is a dataclass that is not frozen: a value is immutable"
        else:
            uncompared = next(
                (item.name for item in dataclasses.fields(kind) if not item.compare), None
            )
            if uncompared is not None:
                found = (
                    f"{kind.__qualname__}.{uncompared} is left out of == (compare=False): a "
                    "value's == and hash are its semantics"
                )
        _CLASSES[kind] = found
    return _CLASSES[kind]


def _fields(value: object) -> object:
    """A frozen dataclass's fields in their canonical form: rebuilt (``replace``) where a
    field its ``__init__`` takes changes; a field it does not take (``init=False``) is read
    again on what was rebuilt and set in place on the new value where it changes."""
    kind = type(value)
    refused = _refused_class(kind)
    if refused is not None:
        raise NotAValue(refused)
    fields = dataclasses.fields(cast(Any, value))
    given: dict[str, object] = {}
    for item in fields:
        found = _field(value, item.name)
        if item.init and found is not getattr(value, item.name):
            given[item.name] = found
    rebuilt = dataclasses.replace(cast(Any, value), **given) if given else value
    rest: dict[str, object] = {}
    for item in fields:
        if not item.init:
            found = _field(rebuilt, item.name)
            if found is not getattr(rebuilt, item.name):
                rest[item.name] = found
    if not rest:
        return rebuilt
    rebuilt = copy(rebuilt) if rebuilt is value else rebuilt
    for name, found in rest.items():
        object.__setattr__(rebuilt, name, found)
    return rebuilt


def _field(value: object, name: str) -> object:
    """The field ``name`` of ``value`` in its canonical form, a refusal naming it."""
    try:
        return canonical(getattr(value, name))
    except NotAValue as refusal:
        raise NotAValue(f"{type(value).__qualname__}.{name}: {refusal}") from None


_FIELD_NAMES: weakref.WeakKeyDictionary[type, tuple[str, ...]] = weakref.WeakKeyDictionary()
"""A dataclass's field names, made once (``holds_mutable``)."""


def holds_mutable(value: object) -> bool:
    """Whether ``value`` holds a mutable value anywhere the canonical form walks (a tuple's
    items, a tuple subclass's included, a frozen set's members, a dataclass's fields): a
    list, a dict, a set, a bytearray, any mutable sequence, mapping or set, or a dataclass
    that is not frozen. An answer that does is handed out as a copy
    (``answers.handed_out``). A registered value type and an enum member hold none."""
    kind = type(value)
    if kind in _ATOMS or kind is float:
        return False
    if isinstance(value, _MUTABLE):
        return True
    if isinstance(value, Enum) or is_value_type(kind):
        return False
    if isinstance(value, (tuple, frozenset)):
        return any(holds_mutable(item) for item in cast("tuple[object, ...]", value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        params = getattr(kind, "__dataclass_params__", None)
        if params is None or not params.frozen:
            return True
        names = _FIELD_NAMES.get(kind)
        if names is None:
            names = _FIELD_NAMES[kind] = tuple(
                item.name for item in dataclasses.fields(cast(Any, value))
            )
        return any(holds_mutable(getattr(value, name)) for name in names)
    return False


def semantics_for(value_type: type[T] | ValueSemantics[T]) -> ValueSemantics[T]:
    return value_type if isinstance(value_type, ValueSemantics) else default_semantics(value_type)


__all__ = [
    "NoTruthValue",
    "NotAValue",
    "OptionalSemantics",
    "ValueSemantics",
    "annotation_type",
    "canonical",
    "default_semantics",
    "holds_mutable",
    "optional",
    "same",
    "semantics_for",
    "union",
]
