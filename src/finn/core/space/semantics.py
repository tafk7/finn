# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A value's type check, its sameness, and its canonical form where it enters a design
space.

Values are immutable, and ``==`` with ``hash`` is their semantics. Two values of a member
are the same exactly when they are of one type and ``==`` holds them equal, item by item
where that ``==`` is the engine's to see through (``same``): a tuple's items, a frozen
set's members and a value class's fields (``value_class``: a frozen dataclass whose ``==``
and hash ``dataclasses`` generates) are each of one type and the same in turn, so
``Box(3)`` is not ``Box(3.0)``, nor ``(True,)`` ``(1,)``. A registered value type, an enum
member and a class with an ``==`` of its own are compared by that ``==``. ``identity`` is a
value as a key equal exactly where values are the same. The engine compares, interns and
keys values by that alone, and copies none of them.

**The type check** is the member's annotation (``annotation_type``): a class recognizes its
instances (``isinstance``, a ``bool`` not taken for an ``int``), a parametrized generic its
origin's (``tuple[int, ...]`` is a ``tuple``; its items are walked by the canonical form,
not typed), ``T | None`` an optional value (``optional``: ``None`` first, by identity), and
a union of several classes any of its members' (``union``). A protocol, a type variable or
a special form is no value type, and a dataclass leaving a field out of ``==``
(``compare=False``) is refused as one. ``ValueSemantics`` holds a check: its type token (what
compatibility compares), its name and its recognizer. It is the engine's record of a
member's check, not a spelling an author writes: a view's check is its annotation (or, for
``view(source)``, its source's).

**The canonical form** (``canonical``) is what a value becomes where it enters a design
space: a fact, a choice, a formal's default, a literal bound at placement, a
domain's listed value. It walks the value item by item whatever the annotation (a tuple's
items, a frozen set's, a frozen dataclass's fields): ``-0.0`` becomes ``0.0``; NaN is
refused (``==`` does not hold it equal to itself); a mutable value is refused (a list, a
dict, a set, a bytearray, any mutable sequence, mapping or set, a dataclass that is not
frozen, a class without a hash), and so is a dataclass leaving a field out of ``==``. A
registered value type (``value_type``) and an enum member are taken as they are: their
owners state them immutable. A tuple subclass is rebuilt by its ``_make`` (a named tuple's);
one without is refused where an item changes, and so is a frozen set subclass where a member
does. A tuple or frozen set subclass changing how it is read (its iteration, length, items,
membership, copy, type or attribute access: ``changed_reads``) is refused: what it says it
holds is not what it stores. A frozen dataclass is rebuilt where a field changes, a field its
``__init__`` does not take (``init=False``) set on the rebuilt value.

Anything else is refused: an object of a class the engine cannot see is immutable (a
service, a native library's handle, a class with an ``==`` of its own its owner has not
registered) is no value, whatever its hash. Two kinds are taken as they are: an instance
of a class derived from ``int``, ``str`` or ``bytes`` that holds nothing beside its value
(no ``__dict__``), and a function, whose identity is its ``==`` and hash.

A numpy number (an instance of ``numpy.number`` or ``numpy.bool_``: ``numpy.int64``,
``numpy.float32``, ``numpy.bool_``) is taken as it is, immutable and hashed by its value, as
a plain number is: a floating one's NaN refused and ``-0.0`` made ``0.0`` of its own type,
a complex one's parts so. The engine imports no numpy: a value is a numpy number only where
a process has loaded numpy (``sys.modules``).
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import math
import sys
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
    it (``annotation_type``)."""

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
        annotated, *metadata = get_args(annotation)
        if any(isinstance(item, ValueSemantics) for item in metadata):
            raise DefinitionError.broken(
                "annotation-semantics",
                owner,
                f"{label}: semantics= is gone: the annotation's type is the value's check, and "
                "a value is compared by ==; leave the ValueSemantics out of Annotated[...]",
                declaration=label,
            )
        return annotation_type(annotated, label, owner=owner)
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
    ``==``, item by item where the engine sees through that ``==`` (a tuple's items, a
    frozen set's members, a value class's fields); an ``==`` that raises fails ``owner``."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    return _owned(lambda: _same(left, right), owner, role)


def _same(left: object, right: object) -> bool:
    """``same`` for two values of one type."""
    walked = _walked(type(left))
    if walked is tuple:
        items, others = cast("tuple[object, ...]", left), cast("tuple[object, ...]", right)
        return len(items) == len(others) and all(map(_same_item, items, others))
    if walked is frozenset:
        return _same_members(cast("frozenset[object]", left), cast("frozenset[object]", right))
    if walked is not None:
        return all(
            _same_item(getattr(left, name), getattr(right, name))
            for name in cast("tuple[str, ...]", walked)
        )
    return bool(left == right)


def _same_item(item: object, other: object) -> bool:
    return item is other or (type(item) is type(other) and _same(item, other))


def _same_members(members: frozenset[object], others: frozenset[object]) -> bool:
    """Two frozen sets equal by ``==``, their members the same one to one: each member against
    the other set's of its hash (a frozen set holds no two members ``==`` holds equal, and
    the same members are equal, so hash alike: a member has one candidate at most). Their
    identities need not hash (a field a value class's hash leaves out may hold an unhashable
    value)."""
    if len(members) != len(others) or members != others:
        return False
    by_hash: dict[int, list[object]] = {}
    for other in others:
        by_hash.setdefault(hash(other), []).append(other)
    return all(
        any(_same_item(member, other) for other in by_hash.get(hash(member), ()))
        for member in members
    )


def identity(value: object) -> object:
    """``value`` as a key equal exactly where two values are the same (``same``): its type
    with its items, members or fields, each so in turn, where the engine sees through its
    ``==``; else its type with itself. It hashes where every part it holds does (a field a
    value class's hash leaves out may hold an unhashable value: ``TypeError``)."""
    kind = type(value)
    walked = _walked(kind)
    if walked is tuple:
        return (kind, tuple(map(identity, cast("tuple[object, ...]", value))))
    if walked is frozenset:
        return (kind, frozenset(map(identity, cast("frozenset[object]", value))))
    if walked is not None:
        names = cast("tuple[str, ...]", walked)
        return (kind, tuple(identity(getattr(value, name)) for name in names))
    return (kind, value)


_WALKED: weakref.WeakKeyDictionary[type, object] = weakref.WeakKeyDictionary()
"""``_walk``'s answer by class, made once."""


def _walked(kind: type) -> object:
    """How ``same`` sees through ``kind``'s ``==``: ``tuple`` (item by item: a tuple, or a
    tuple subclass compared and read as one, a named tuple's), ``frozenset`` (member by
    member, likewise), a value class's compared fields' names (field by field:
    ``value_class``), else ``None`` (by its ``==``: a plain value, a registered value type,
    an enum member, a class with an ``==`` of its own, or one changing how it is read). A
    class registered as a value type later is so from then on."""
    if kind is tuple or kind is frozenset:
        return kind
    if kind in _PLAIN or is_value_type(kind):
        return None
    found = _WALKED.get(kind, _MISSING)
    if found is _MISSING:
        found = _WALKED[kind] = _walk(kind)
    return found


_MISSING = object()
_PLAIN: frozenset[type] = frozenset((bool, int, float, complex, str, bytes, type(None)))


def _walk(kind: type) -> object:
    if issubclass(kind, Enum):
        return None
    for base in (tuple, frozenset):
        if issubclass(kind, base):
            compared = kind.__eq__ is base.__eq__ and kind.__hash__ is base.__hash__
            return base if compared and not changed_reads(kind) else None
    return compared_fields(kind)


_READ = (
    "__iter__",
    "__len__",
    "__getitem__",
    "__contains__",
    "__reduce__",
    "__reduce_ex__",
    "__getnewargs__",
    "__copy__",
    "__deepcopy__",
    "__class__",
    "__getattribute__",
)
"""What a tuple or a frozen set is read through: by the engine's walks, ``==``, its hash, its
type, and a copy."""

_IDENTITY = ("__class__", "__getattribute__")
"""What an object's type is read through (``__class__``), and every attribute
(``__getattribute__``): the engine's ``type`` stands for a generated ``==``'s
``__class__`` only where both are ``object``'s."""

_CHANGED: weakref.WeakKeyDictionary[type, tuple[str, ...]] = weakref.WeakKeyDictionary()
"""``changed_reads``'s answer by class, made once."""


def changed_reads(kind: type) -> tuple[str, ...]:
    """What a tuple or frozen set subclass ``kind`` changes of how its base is read (``_READ``:
    its iteration, length, items, membership, copy, type or attribute access; a named
    tuple's ``__getnewargs__`` reads what it stores), by name; the engine walks one that
    changes none, by what it stores. ``()`` for any other class."""
    found = _CHANGED.get(kind)
    if found is None:
        base = (
            tuple if issubclass(kind, tuple) else frozenset if issubclass(kind, frozenset) else None
        )
        found = _CHANGED[kind] = (
            ()
            if base is None
            else tuple(name for name in _READ if not _reads_as_base(kind, base, name))
        )
    return found


_NAMED = collections.namedtuple("_NAMED", "item")
"""A named tuple: every one's ``__getnewargs__`` is this one's code, reading what the tuple
stores."""


def _reads_as_base(kind: type, base: type, name: str) -> bool:
    found = _resolved(kind, name)
    if found is _resolved(base, name):
        return True
    named = _resolved(_NAMED, "__getnewargs__")
    return (
        name == "__getnewargs__"
        and getattr(found, "__code__", None) is getattr(named, "__code__", _MISSING)
        and _cells(found) == _cells(named)
    )


def _cells(function: object) -> list[object]:
    return [cell.cell_contents for cell in getattr(function, "__closure__", None) or ()]


def _resolved(kind: type, name: str) -> object:
    """The attribute ``name`` of ``kind``'s instances as their class gives it: the first
    class along its method resolution order declaring it; ``None`` where none does."""
    for base in kind.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return None


def value_class(kind: type) -> bool:
    """Whether ``kind`` is a value class, compared (``same``) and keyed (the node cache's
    ``key``) field by field (``compared_fields``)."""
    return compared_fields(kind) is not None


_COMPARED: weakref.WeakKeyDictionary[type, tuple[str, ...] | None] = weakref.WeakKeyDictionary()
"""``compared_fields``'s answer by class, made once."""


def compared_fields(kind: type) -> tuple[str, ...] | None:
    """The fields a value class's ``==`` compares, by name and in order; ``None`` where
    ``kind`` is no value class. A value class is a frozen dataclass whose ``__eq__`` and
    ``__hash__`` are the ones ``dataclasses`` generates, declared by it or inherited (a
    subclass not decorated, or decorated ``eq=False``, compares the fields of the dataclass
    that generated its ``==``), and whose type and attributes are read as ``object``'s
    (``__class__``, ``__getattribute__``: the generated ``==`` compares ``__class__``, which
    the engine reads as the type). Generated means the very code ``dataclasses`` makes for
    those fields (``_generated``), not where a method was compiled: a method that is not,
    written by hand or uncertain, is its author's ``==`` (or hash), and the class no value
    class. An ``eq=False`` dataclass inheriting no generated ``==`` (equal by identity) and a
    mutable one are none."""
    if kind not in _COMPARED:
        _COMPARED[kind] = _compared_fields(kind)
    return _COMPARED[kind]


def _compared_fields(kind: type) -> tuple[str, ...] | None:
    params = getattr(kind, "__dataclass_params__", None)
    if params is None or not params.frozen:
        return None
    if any(_resolved(kind, name) is not _resolved(object, name) for name in _IDENTITY):
        return None
    owner = next((base for base in kind.__mro__ if "__eq__" in base.__dict__), None)
    if owner is None or "__dataclass_fields__" not in owner.__dict__:
        return None
    fields = dataclasses.fields(cast(Any, owner))
    shape = tuple((item.name, item.compare, item.hash) for item in fields)
    generated = _generated(shape)
    if None in generated or (_code(kind.__eq__), _code(kind.__hash__)) != generated:
        return None
    return tuple(item.name for item in fields if item.compare)


@functools.lru_cache(maxsize=256)
def _generated(shape: tuple[tuple[str, bool, bool | None], ...]) -> tuple[object, object]:
    """The code ``dataclasses`` generates for a frozen dataclass's ``__eq__`` and
    ``__hash__`` with these fields (name, ``compare``, ``hash``), in order (``_code``), read
    off a dataclass made alike and dropped; kept for the latest shapes alone."""
    reference = dataclasses.make_dataclass(
        "_Reference",
        [
            (name, object, dataclasses.field(default=None, compare=compare, hash=hashed))
            for name, compare, hashed in shape
        ],
        frozen=True,
    )
    return _code(reference.__eq__), _code(reference.__hash__)


def _code(method: object) -> object:
    """A function's code as compared with the generated one: its instructions, names,
    constants and arguments; ``None`` where it is no function, or has defaults or a closure
    (no generated method has)."""
    code = getattr(method, "__code__", None)
    if code is None or any(
        getattr(method, name, None) is not None
        for name in ("__defaults__", "__kwdefaults__", "__closure__")
    ):
        return None
    return (code.co_code, code.co_names, code.co_consts, code.co_varnames, code.co_argcount)


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
_SUBCLASSED = (int, str, bytes)
"""The plain values a class may derive from: an instance holding nothing beside its value (no
``__dict__``) is as immutable as the plain value."""
_FUNCTIONS = (types.FunctionType, types.BuiltinFunctionType)
"""A function is taken as itself: its identity is its ``==`` and hash."""

_CLASSES: weakref.WeakKeyDictionary[type, str | None] = weakref.WeakKeyDictionary()
"""``_refused_class``'s answer by dataclass, made once."""


def canonical(value: T) -> T:
    """``value`` in its canonical form (see the module docstring): the value itself where
    nothing in it changes; ``NotAValue`` where it is no value, an object of a class the
    engine cannot see is immutable included."""
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
    numpy = _numpy_number(kind)
    if numpy is not None:
        return cast(T, _numpy_canonical(value, numpy))
    if isinstance(value, _MUTABLE):
        raise NotAValue(
            f"a {kind.__qualname__} is mutable: a value is immutable (a tuple, a frozen set, "
            "a frozen dataclass)"
        )
    if isinstance(value, (tuple, frozenset)):
        changed = changed_reads(kind)
        if changed:
            raise NotAValue(
                f"a {kind.__qualname__} changes how a "
                f"{'tuple' if isinstance(value, tuple) else 'frozen set'} is read "
                f"({', '.join(changed)}): the engine cannot see what it holds"
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
    if isinstance(value, _FUNCTIONS) or (
        isinstance(value, _SUBCLASSED) and not hasattr(value, "__dict__")
    ):
        return value
    raise NotAValue(
        f"a {kind.__qualname__} is no value the engine can see is immutable: a value is a "
        "plain value, an enum member, a registered value type (value_type), or a tuple, a "
        "frozen set or a frozen dataclass of values"
    )


def _numpy_number(kind: type) -> Any:
    """numpy, where ``kind`` is one of its numbers (``numpy.number``, ``numpy.bool_``);
    None otherwise, and wherever the process has not loaded numpy."""
    numpy = sys.modules.get("numpy")
    if numpy is None or not issubclass(kind, (numpy.number, numpy.bool_)):
        return None
    return numpy


def _numpy_canonical(value: Any, numpy: Any) -> object:
    """A numpy number in its canonical form: itself, but a floating one's (or a complex
    one's part's) NaN refused and ``-0.0`` made ``0.0``, of its own type."""
    kind = type(value)
    if isinstance(value, numpy.floating):
        part = float(value)
        if math.isnan(part):
            raise NotAValue("NaN is no value: == does not hold it equal to itself")
        return kind(0.0) if part == 0.0 and math.copysign(1.0, part) < 0 else value
    if isinstance(value, numpy.complexfloating):
        whole = complex(value)
        parts = (whole.real, whole.imag)
        if any(math.isnan(part) for part in parts):
            raise NotAValue("NaN is no value: == does not hold it equal to itself")
        if any(part == 0.0 and math.copysign(1.0, part) < 0 for part in parts):
            return kind(complex(*(abs(part) if part == 0.0 else part for part in parts)))
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
    kind = type(value)
    if kind is not frozenset:
        raise NotAValue(
            f"a {kind.__qualname__} is a frozen set subclass: its members' canonical form "
            "cannot be rebuilt"
        )
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
    (``answers.handed_out``). A registered value type and an enum member hold none, and a
    tuple or frozen set subclass changing how it is read (``changed_reads``) is not walked:
    the canonical form refuses it, and a copy would read it as it says, not as it stores."""
    kind = type(value)
    if kind in _ATOMS or kind is float:
        return False
    if isinstance(value, _MUTABLE):
        return True
    if isinstance(value, Enum) or is_value_type(kind):
        return False
    if isinstance(value, (tuple, frozenset)):
        if changed_reads(kind):
            return False  # no value, refused where one enters; not walked
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


__all__ = [
    "NoTruthValue",
    "NotAValue",
    "OptionalSemantics",
    "ValueSemantics",
    "annotation_type",
    "canonical",
    "changed_reads",
    "compared_fields",
    "default_semantics",
    "holds_mutable",
    "identity",
    "optional",
    "same",
    "union",
    "value_class",
]
