# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed declarations for the Space language.

A family's class body declares members (formals, decisions, computations,
views) and nodes. Calling a family, ``Room(area=12)``, declares a node: a
template with bindings, compiled only by ``configure``. Attribute access on a
node declaration, ``kitchen.finish``, is a symbolic reference to that node's
member. It is typed as the member's value (option A); at runtime it refuses
every value-like use with ``ReferenceUseError``.
"""

# Runtime dispatch is deliberately lazy to keep declaration collection independent.
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeAlias, TypeVar, cast, overload

from typing_extensions import Self

from .domains import Domain, finite
from .errors import DefinitionError, ReferenceUseError
from .graph import LOCATED, Located
from .results import QueryResult
from .semantics import ValueSemantics, default_semantics, semantics_for

if TYPE_CHECKING:
    from ._configuration import BoundView, Space
    from .expressions import Expr

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
N = TypeVar("N", bound="Space")

_STRING = default_semantics(str)


def local_name(value: str, role: str) -> str:
    """Keep authored identity segments disjoint from generated node names."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise DefinitionError(f"{role} must be one nonempty ASCII name segment")
    return value


def class_namespace(space_type: type[Space]) -> dict[str, object]:
    namespace: dict[str, object] = {}
    for base in reversed(space_type.__mro__):
        namespace.update(vars(base))
    namespace[space_type.__name__] = space_type
    return namespace


_PACKAGE = __name__.rsplit(".", 1)[0]


def source_origin() -> str | None:
    """``file:line`` of the nearest caller outside this package, for diagnostics."""
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != _PACKAGE and not module.startswith(_PACKAGE + "."):
            return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
        frame = frame.f_back  # type: ignore[assignment]
    return None


def at(origin: str | None) -> str:
    return "" if origin is None else f" (declared at {origin})"


class _Unsupplied:
    """Default of an optional formal: nobody need supply it."""

    _instance: _Unsupplied | None = None

    def __new__(cls) -> _Unsupplied:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSUPPLIED"


class _Missing:
    def __repr__(self) -> str:
        return "<required>"


UNSUPPLIED = _Unsupplied()
MISSING = _Missing()


class Declaration:
    """Identity-bearing source declaration; names are diagnostic hints only."""

    name: str | None = None
    owner: type[object] | None = None
    when: ValueRef[bool] | None = None
    origin: str | None = None
    _record_origin = True

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        instance = super().__new__(cls)
        if cls._record_origin:
            object.__setattr__(instance, "origin", source_origin())
        return instance

    def __set_name__(self, owner: type[object], name: str) -> None:
        if self.owner is not None and (self.owner is not owner or self.name != name):
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: declaration already belongs to "
                f"{self.owner.__qualname__}.{self.name}; create a fresh declaration"
            )
        self.owner, self.name = owner, name

    def __bool__(self) -> bool:
        raise TypeError("a declaration has no truth value")


class ValueRef(Declaration, Generic[T_co]):
    """A typed value handle: a member, a symbolic reference, or an expression."""

    semantics: ValueSemantics[T_co] | None

    @property
    def value_semantics(self) -> ValueSemantics[T_co] | None:
        return self.semantics

    def __add__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("add", self, other)

    def __radd__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("add", other, self)

    def __sub__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("sub", self, other)

    def __rsub__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("sub", other, self)

    def __mul__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mul", self, other)

    def __rmul__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mul", other, self)

    def __floordiv__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("floordiv", self, other)

    def __rfloordiv__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("floordiv", other, self)

    def __mod__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mod", self, other)

    def __rmod__(self: ValueRef[int], other: int | ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("mod", other, self)

    def __neg__(self: ValueRef[int]) -> Expr:
        from .expressions import Expr

        return Expr("neg", self)


def declared_path(instance: object) -> tuple[Declaration, ...] | None:
    """The node path of a declaration-mode Space object, or None for a configuration."""
    return cast("tuple[Declaration, ...] | None", vars(instance).get("_space_path"))


class ValueDecl(ValueRef[T_co], Generic[T_co]):
    """A class member descriptor.

    On a configuration it reads the value; on a node declaration it returns a
    symbolic ``MemberRef`` (statically typed as the value, see DESIGN.md).
    """

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> T_co: ...

    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | T_co:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return cast(T_co, MemberRef(path, self))
        from .occurrence import read_value

        return read_value(instance, self)


class Param(ValueDecl[T], Generic[T]):
    """A formal input of a family, supplied where a node of the family is declared.

    Supply it at the call (``Room(area=12)``) or by assignment before the
    family is prepared (``hall.area = kitchen.area``). ``Param(int)`` is
    required: preparing a family in which nothing supplies it is a definition
    error. ``default=`` makes it optional, and ``default=UNSUPPLIED`` leaves it
    unsupplied when nobody binds it. ``Param(Located)`` locates a plain
    reference automatically, and ``Param(Family)`` is a reference input: the
    caller supplies a node of that family, which is placed there if it is fresh
    and referenced if it is placed elsewhere. Annotate formals
    (``area: Param[int] = Param(int)``, ``output: Param[Stream] = Param(Stream)``)
    so family calls and assignments are typed.
    """

    default: object
    required: bool

    @overload
    def __new__(cls, value_type: type[N], *, default: _Unsupplied = ...) -> Param[N]: ...

    @overload
    def __new__(
        cls,
        value_type: type[Located[Any]] | ValueSemantics[Located[Any]],
        *,
        default: Located[Any] | _Unsupplied = ...,
    ) -> LocatedParam[Any]: ...

    @overload
    def __new__(
        cls,
        value_type: type[T] | ValueSemantics[T],
        *,
        default: T | _Unsupplied = ...,
        semantics: ValueSemantics[T] | None = None,
    ) -> Param[T]: ...

    def __new__(
        cls, value_type: object, *, default: object = MISSING, semantics: object = None
    ) -> Any:
        from ._configuration import Space

        if isinstance(value_type, type) and issubclass(value_type, Space):
            if default is not MISSING and default is not UNSUPPLIED:
                raise DefinitionError(
                    "a reference input has no value default; default=UNSUPPLIED makes it optional"
                )
            from ._nodes import family_formal

            return family_formal(value_type, required=default is MISSING)
        chosen = cast(
            ValueSemantics[object],
            semantics if semantics is not None else semantics_for(cast(Any, value_type)),
        )
        if not isinstance(chosen, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        kind = LocatedParam if chosen.type_token is Located and cls is Param else cls
        instance = cast(Param[object], super().__new__(kind))
        instance.semantics = chosen
        instance.required = default is MISSING
        if default is MISSING or default is UNSUPPLIED:
            instance.default = default
        else:
            try:
                instance.default = chosen.freeze(default)
            except (TypeError, ValueError) as error:
                raise DefinitionError(f"invalid formal default: {error}") from error
        return instance

    def __set__(self, instance: object, value: T | ValueRef[T] | View[T] | BoundView[T]) -> None:
        # The value type types both the family's constructor keyword (through
        # dataclass_transform) and assignment to a declaration's formal. At
        # runtime Space.__setattr__ handles assignment before this is reached.
        from ._nodes import assign

        assign(instance, cast(str, self.name), value)


class LocatedParam(Param[Located[T]], Generic[T]):
    """A formal holding a ``Located`` value.

    Binding a plain reference supplies its located form: node name, member
    name and value. Declare it with ``Param(Located)``.
    """

    def __set__(
        self,
        instance: object,
        value: T
        | Located[T]
        | ValueRef[T]
        | ValueRef[Located[T]]
        | View[T]
        | View[Located[T]]
        | BoundView[T]
        | BoundView[Located[T]],
    ) -> None:
        from ._nodes import assign

        assign(instance, cast(str, self.name), value)


class Const(ValueDecl[T], Generic[T]):
    """A definition-owned frozen value."""

    def __init__(self, value: T, *, semantics: ValueSemantics[T] | None = None) -> None:
        self.semantics = semantics if semantics is not None else semantics_for(type(value))
        self.value = self.semantics.freeze(value)


Guard: TypeAlias = "ValueRef[bool] | bool | None"


class Decision(ValueDecl[T], Generic[T]):
    """An independently editable choice. Its type parameter is invariant.

    ``Decision(int, values=(1, 2))`` chooses a value. ``Decision(values={"a":
    A(), "b": B(), "none": None})`` chooses a node: the persisted value is the
    key; each candidate is a node named ``<decision>.<key>`` whose presence
    derives from the decision; ``None`` places nothing. Such a decision is typed
    as its candidates, so ``decision.member`` reads the selected candidate's member.

    A Decision that is not a class attribute may supply exactly one formal
    (``Fifo(depth=Decision(int, values=(4, 8)))``), and is keyed by that
    formal's path. One that supplies several formals is shared and must be
    named: a class attribute, or ``Decision(..., name="depth")``, which is
    owned by the lowest scope containing every node it supplies.
    """

    domain: Domain[T]
    # Where this Decision supplies a formal (for the shared-decision rule).
    sites: list[str]

    @overload
    def __new__(  # type: ignore[misc]
        cls, *, values: Mapping[str, T], when: Guard = None
    ) -> T: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls, *, values: Mapping[str, T | None], when: Guard = None
    ) -> T | None: ...

    @overload
    def __new__(
        cls,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: Domain[T] | None = None,
        values: Iterable[T] | None = None,
        semantics: ValueSemantics[T] | None = None,
        when: Guard = None,
        name: str | None = None,
    ) -> Decision[T]: ...

    def __new__(
        cls,
        value_type: object = None,
        *,
        domain: object = None,
        values: object = None,
        semantics: object = None,
        when: object = None,
        name: object = None,
    ) -> Any:
        if value_type is None and domain is None and isinstance(values, Mapping):
            from ._nodes import node_choice

            return node_choice(values, when=_guard(when))
        if value_type is None:
            raise DefinitionError("a Decision needs a value type, or values= mapping keys to nodes")
        if (domain is None) == (values is None):
            raise DefinitionError("a Decision needs exactly one of domain= or values=")
        instance = cast(Decision[object], super().__new__(cls))
        chosen = cast(
            ValueSemantics[object],
            semantics if semantics is not None else semantics_for(cast(Any, value_type)),
        )
        instance.semantics = chosen
        instance.when = _guard(when)
        instance.domain = (
            cast(Domain[object], domain).with_semantics(chosen)
            if domain is not None
            else finite(cast(Iterable[object], values), chosen)
        )
        instance.sites = []
        if name is not None:
            instance.name = local_name(cast(str, name), "decision name")
        return instance

    def __set_name__(self, owner: type[object], name: str) -> None:
        if self.owner is None and self.name is not None and self.name != name:
            raise DefinitionError(
                f"{owner.__qualname__}.{name}: Decision{at(self.origin)} is named "
                f"{self.name!r}; a class attribute takes its attribute name"
            )
        super().__set_name__(owner, name)


def _guard(when: object) -> ValueRef[bool] | None:
    if when is None:
        return None
    if not isinstance(when, ValueRef):
        raise DefinitionError("when= requires a Boolean reference")
    return cast(ValueRef[bool], when)


class Derived(ValueDecl[T], Generic[T]):
    """A function whose dependencies are bound after effective collection."""

    def __init__(
        self,
        function: Callable[..., T | QueryResult[T]],
        *,
        semantics: ValueSemantics[T] | None = None,
        aliases: Mapping[str, object] | None = None,
        when: Guard = None,
    ) -> None:
        self.function = function
        self.semantics = semantics
        self.aliases = MappingProxyType(dict(aliases or {}))
        self.when = _guard(when)


class _DerivedDecorator:
    def __init__(self, aliases: Mapping[str, object], when: Guard) -> None:
        self.aliases, self.when = aliases, when

    def __call__(self, function: Callable[..., T]) -> Derived[T]:
        return Derived(function, aliases=self.aliases, when=self.when)


class _SemanticDerivedDecorator(Generic[T]):
    def __init__(self, semantics: ValueSemantics[T], aliases: Mapping[str, object], when: Guard):
        self.semantics, self.aliases, self.when = semantics, aliases, when

    def __call__(self, function: Callable[..., T | QueryResult[T]]) -> Derived[T]:
        return Derived(function, semantics=self.semantics, aliases=self.aliases, when=self.when)


@overload
def derived(function: Callable[..., T], /) -> Derived[T]: ...


@overload
def derived(
    *, semantics: ValueSemantics[T], when: ValueRef[bool] | bool | None = None, **aliases: object
) -> _SemanticDerivedDecorator[T]: ...


@overload
def derived(
    *, semantics: None = None, when: ValueRef[bool] | bool | None = None, **aliases: object
) -> _DerivedDecorator: ...


def derived(
    function: Callable[..., object] | None = None,
    /,
    *,
    semantics: object = None,
    when: Guard = None,
    **aliases: object,
) -> object:
    if function is not None:
        return Derived(function, aliases=aliases, when=when)
    if semantics is not None:
        if not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        return _SemanticDerivedDecorator(semantics, aliases, when)
    return _DerivedDecorator(aliases, when)


class _MemberOfNode:
    """A non-value member: read through a node declaration it is a reference."""

    def __get__(self, instance: object, owner: type[object] | None = None) -> Self:
        if instance is not None:
            path = declared_path(instance)
            if path is not None:
                return cast(Self, MemberRef(path, cast(Declaration, self)))
        return self


class Constraint(_MemberOfNode, Declaration):
    def __init__(
        self,
        function: Callable[..., bool | QueryResult[bool]],
        *,
        aliases: Mapping[str, object] | None = None,
        when: Guard = None,
    ) -> None:
        self.function = function
        self.aliases = MappingProxyType(dict(aliases or {}))
        self.semantics = semantics_for(bool)
        self.when = _guard(when)


class _ConstraintDecorator:
    def __init__(self, aliases: Mapping[str, object], when: Guard) -> None:
        self.aliases, self.when = aliases, when

    def __call__(self, function: Callable[..., bool | QueryResult[bool]]) -> Constraint:
        return Constraint(function, aliases=self.aliases, when=self.when)


@overload
def constraint(function: Callable[..., bool | QueryResult[bool]], /) -> Constraint: ...


@overload
def constraint(
    *, when: ValueRef[bool] | bool | None = None, **aliases: object
) -> _ConstraintDecorator: ...


def constraint(
    function: Callable[..., bool | QueryResult[bool]] | None = None,
    /,
    *,
    when: Guard = None,
    **aliases: object,
) -> Constraint | _ConstraintDecorator:
    if function is not None:
        return Constraint(function, aliases=aliases, when=when)
    return _ConstraintDecorator(aliases, when)


class ConstraintGroup(_MemberOfNode, Declaration):
    def __init__(self, *constraints: Constraint) -> None:
        self.constraints = constraints


# What a view may require: constraints, groups, other views (their acceptance),
# references to views, and member families (each member's acceptance).
Obligation: TypeAlias = (
    "Constraint | ConstraintGroup | View[Any] | BoundView[Any] | Members[Any] | ValueRef[Any]"
)


class View(Declaration, Generic[T]):
    """One assessment declaration for either a value or an authored function."""

    @overload
    def __init__(
        self,
        source: ValueRef[T] | View[T] | BoundView[T],
        *,
        requires: Sequence[Obligation] = (),
        when: Guard = None,
    ) -> None: ...

    @overload
    def __init__(
        self, source: T, *, requires: Sequence[Obligation] = (), when: Guard = None
    ) -> None: ...

    def __init__(
        self,
        source: object,
        *,
        requires: Sequence[Obligation] = (),
        when: Guard = None,
    ) -> None:
        self.source: object | None = source
        self.function: Callable[..., T | QueryResult[T]] | None = None
        self.aliases: Mapping[str, object] = MappingProxyType({})
        self.semantics: ValueSemantics[T] | None = cast(
            "ValueSemantics[T] | None", getattr(source, "semantics", None)
        )
        self.requires = tuple(requires)
        self.when = _guard(when)

    @classmethod
    def from_function(
        cls,
        function: Callable[..., T | QueryResult[T]],
        *,
        semantics: ValueSemantics[T] | None,
        aliases: Mapping[str, object],
        requires: Sequence[Obligation],
        when: Guard = None,
    ) -> View[T]:
        result = cls.__new__(cls)
        result.source = None
        result.function = function
        result.aliases = MappingProxyType(dict(aliases))
        result.semantics = semantics
        result.requires = tuple(requires)
        result.when = _guard(when)
        return result

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> BoundView[T]: ...

    def __get__(
        self, instance: Space | None, owner: type[object] | None = None
    ) -> Self | BoundView[T]:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return cast("BoundView[T]", MemberRef(path, self))
        from ._configuration import BoundView

        return BoundView(instance, self)


class _ViewDecorator:
    def __init__(
        self, aliases: Mapping[str, object], requires: Sequence[Obligation], when: Guard
    ) -> None:
        self.aliases, self.requires, self.when = aliases, requires, when

    def __call__(self, function: Callable[..., T]) -> View[T]:
        return View.from_function(
            function,
            semantics=None,
            aliases=self.aliases,
            requires=self.requires,
            when=self.when,
        )


class _SemanticViewDecorator(Generic[T]):
    def __init__(self, semantics: ValueSemantics[T], decorator: _ViewDecorator) -> None:
        self.semantics, self.decorator = semantics, decorator

    def __call__(self, function: Callable[..., T | QueryResult[T]]) -> View[T]:
        return View.from_function(
            function,
            semantics=self.semantics,
            aliases=self.decorator.aliases,
            requires=self.decorator.requires,
            when=self.decorator.when,
        )


@overload
def view(function: Callable[..., T], /) -> View[T]: ...


@overload
def view(
    *,
    semantics: ValueSemantics[T],
    when: ValueRef[bool] | bool | None = None,
    requires: Sequence[Obligation] = (),
    **aliases: object,
) -> _SemanticViewDecorator[T]: ...


@overload
def view(
    *,
    semantics: None = None,
    when: ValueRef[bool] | bool | None = None,
    requires: Sequence[Obligation] = (),
    **aliases: object,
) -> _ViewDecorator: ...


def view(
    function: Callable[..., object] | None = None,
    /,
    *,
    semantics: object = None,
    when: Guard = None,
    requires: Sequence[Obligation] = (),
    **aliases: object,
) -> object:
    decorator = _ViewDecorator(aliases, requires, when)
    if function is not None:
        return decorator(function)
    if semantics is not None:
        if not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        return _SemanticViewDecorator(semantics, decorator)
    return decorator


class ViewKey(Generic[T_co]):
    """A typed export contract independent of a concrete family: see ``Members``."""

    def __init__(self, name: str, value_type: type[T_co] | ValueSemantics[T_co]) -> None:
        self.name, self.semantics = local_name(name, "export name"), semantics_for(value_type)

    def __repr__(self) -> str:
        return f"ViewKey({self.name!r})"


# -- Symbolic references ---------------------------------------------------------------


def _path_text(path: Sequence[Declaration]) -> str:
    parts: list[str] = []
    for record in path:
        name = record.name
        if name is None:
            describe = getattr(record, "describe", None)
            name = f"<{describe()}>" if callable(describe) else f"<{type(record).__name__}>"
        parts.append(name)
    return ".".join(parts)


class _Symbolic:
    """Refuses value-like use of a declaration reference, loudly and specifically.

    A reference is typed as the value it stands for, so mypy cannot catch
    ``if kitchen.area > 3`` in a class body. At runtime these operations raise
    ``ReferenceUseError``; equality with another reference and hashing are
    structural, so references work as mapping keys.
    """

    def _describe(self) -> str:
        raise NotImplementedError

    def _key(self) -> tuple[object, ...]:
        raise NotImplementedError

    def _misuse(self, what: str) -> NoReturn:
        raise ReferenceUseError(
            f"{self._describe()} is a declaration reference, not a value: {what} is not "
            "available while a Space is declared. Compute with it in a @derived or @view "
            "method instead (or pass the reference where the value is needed)."
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Symbolic):
            return type(self) is type(other) and self._key() == other._key()
        from ._configuration import Space
        from ._nodes import NodeChoice

        if isinstance(other, (Declaration, ViewKey, Space, NodeChoice)):
            return False
        self._misuse("equality comparison with a value")

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((type(self), self._key()))

    def __lt__(self, other: object) -> bool:
        self._misuse("ordering comparison")

    def __le__(self, other: object) -> bool:
        self._misuse("ordering comparison")

    def __gt__(self, other: object) -> bool:
        self._misuse("ordering comparison")

    def __ge__(self, other: object) -> bool:
        self._misuse("ordering comparison")

    def __bool__(self) -> bool:
        self._misuse("a truth value")

    def __len__(self) -> int:
        self._misuse("a length")

    def __iter__(self) -> NoReturn:
        self._misuse("iteration")

    def __contains__(self, item: object) -> bool:
        self._misuse("membership testing")

    def __getitem__(self, item: object) -> NoReturn:
        self._misuse("indexing")

    def __call__(self, *args: object, **kwargs: object) -> NoReturn:
        self._misuse("calling")

    def __str__(self) -> str:
        self._misuse("str()")

    def __format__(self, spec: str) -> str:
        self._misuse("formatting")

    def __int__(self) -> int:
        self._misuse("int()")

    def __float__(self) -> float:
        self._misuse("float()")

    def __complex__(self) -> complex:
        self._misuse("complex()")

    def __index__(self) -> int:
        self._misuse("an index")

    def __round__(self, ndigits: int | None = None) -> int:
        self._misuse("round()")

    def __trunc__(self) -> int:
        self._misuse("truncation")

    def __floor__(self) -> int:
        self._misuse("floor()")

    def __ceil__(self) -> int:
        self._misuse("ceil()")

    def __abs__(self) -> NoReturn:
        self._misuse("abs()")

    # Integer +, -, *, //, % and unary - build expressions (see ValueRef);
    # every other operator is value-like use.
    def __truediv__(self, other: object) -> NoReturn:
        self._misuse("true division (integer references support //)")

    def __rtruediv__(self, other: object) -> NoReturn:
        self._misuse("true division (integer references support //)")

    def __pow__(self, other: object) -> NoReturn:
        self._misuse("exponentiation")

    def __rpow__(self, other: object) -> NoReturn:
        self._misuse("exponentiation")

    def __and__(self, other: object) -> NoReturn:
        self._misuse("a bitwise or Boolean operator")

    __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = __and__

    def __lshift__(self, other: object) -> NoReturn:
        self._misuse("a shift")

    __rlshift__ = __rshift__ = __rrshift__ = __lshift__

    def __invert__(self) -> NoReturn:
        self._misuse("inversion")

    def __matmul__(self, other: object) -> NoReturn:
        self._misuse("matrix multiplication")

    __rmatmul__ = __matmul__

    def __repr__(self) -> str:
        return f"<reference {self._describe()}>"


class MemberRef(_Symbolic, ValueRef[T], Generic[T]):
    """``node.member``: one member of the node at ``path``.

    ``path`` holds node declarations, outermost first; each one after the
    first is a node of the previous one's family. ``member`` is a declaration
    of the last node's family (or an inherited one it overrides).
    """

    def __init__(self, path: tuple[Declaration, ...], member: Declaration) -> None:
        self.path, self.member = path, member
        self.semantics = cast("ValueSemantics[T] | None", getattr(member, "semantics", None))

    def _describe(self) -> str:
        return f"{_path_text(self.path)}.{self.member.name}{at(self.origin)}"

    def _key(self) -> tuple[object, ...]:
        return (*(id(item) for item in self.path), id(self.member))


class ChoiceMemberRef(_Symbolic, ValueRef[Any]):
    """``decision.member``: the selected candidate's member, by name.

    Resolves to the selected candidate's ``member``; it is unresolved until the
    decision is made, and inapplicable when the selected candidate lacks it.
    """

    def __init__(self, path: tuple[Declaration, ...], member: str) -> None:
        self.path, self.member = path, member
        self.semantics = None

    def _describe(self) -> str:
        return f"{_path_text(self.path)}.{self.member}{at(self.origin)}"

    def _key(self) -> tuple[object, ...]:
        return (*(id(item) for item in self.path), self.member)


class CaseRef(ValueDecl[str]):
    """The key a structural decision selects, read-only (see ``selected``)."""

    def __init__(self, path: tuple[Declaration, ...]) -> None:
        self.path = path
        self.semantics = _STRING


def selected(decision: object) -> str:
    """The selected key of a Decision over nodes, as a read-only value.

    Typed as ``str`` like every reference; inside a method it reads the key.
    """
    from ._nodes import NodeChoice

    if not isinstance(decision, NodeChoice):
        raise DefinitionError("selected() requires a Decision over nodes")
    return cast(str, CaseRef(decision._space_path))


class Present(ValueDecl[T], Generic[T]):
    """The value of whichever one of ``sources`` is present (applicable).

    Unresolved while any source is unresolved; refused if two are present;
    unsupplied (unresolved) if none is.
    """

    @overload
    def __init__(
        self,
        *sources: ValueRef[T] | View[T] | BoundView[T],
        semantics: ValueSemantics[T] | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *sources: T, semantics: ValueSemantics[T] | None = None) -> None: ...

    def __init__(self, *sources: object, semantics: ValueSemantics[T] | None = None) -> None:
        if not sources or any(not isinstance(source, (ValueRef, View)) for source in sources):
            raise DefinitionError("Present requires one or more value references")
        self.sources = cast(tuple[ValueRef[T], ...], sources)
        self.semantics = (
            semantics
            if semantics is not None
            else cast("ValueSemantics[T] | None", getattr(sources[0], "semantics", None))
        )


class Members(ValueDecl[tuple[Located[T], ...]], Generic[T]):
    """Every present child node exporting ``key``, as ``Located`` values in
    declaration order; as an obligation, each member's acceptance counts.

    Each entry is ``Located(node=<child name>, member=<key name>, value=<export>)``.
    """

    def __init__(self, key: ViewKey[T]) -> None:
        if not isinstance(key, ViewKey):
            raise DefinitionError(f"{type(self).__name__} requires a ViewKey")
        self.key = key
        self.semantics = cast("ValueSemantics[tuple[Located[T], ...]]", semantics_for(tuple))


class Users(Members[T], Generic[T]):
    """The mirror of ``Members``: every present node whose input references this one.

    In declaration order, each entry is ``Located(node=<user's name beside this
    node>, member=<the user's input name>, value=<the user's export of key>)``.
    A user reached through a Decision candidate or a guard is present only when
    that candidate is selected or that guard holds; a user that does not export
    ``key`` is omitted. As an obligation, each user's acceptance counts.
    """


__all__ = [
    "CaseRef",
    "ChoiceMemberRef",
    "Const",
    "Constraint",
    "ConstraintGroup",
    "Decision",
    "Declaration",
    "Derived",
    "LOCATED",
    "LocatedParam",
    "Members",
    "MemberRef",
    "Param",
    "Present",
    "UNSUPPLIED",
    "Users",
    "ValueDecl",
    "ValueRef",
    "View",
    "ViewKey",
    "constraint",
    "derived",
    "selected",
    "view",
]
