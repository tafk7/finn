# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed declarations for the Space language.

A family's class body declares members (formals, decisions, computations,
views) and nodes. Calling a family, ``Room(area=12)``, declares a node: a
template with bindings, compiled only by ``design_space``. Attribute access on a
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
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    NoReturn,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .domains import Domain, Requirement, finite, requiring
from .errors import DefinitionError, ReferenceUseError
from .graph import LOCATED, Located
from .results import NonValue, QueryResult, marked_value_type
from .semantics import (
    ValueSemantics,
    default_semantics,
    recognize,
    semantics_for,
    snapshot,
    unrecognized,
)

if TYPE_CHECKING:
    from ._configuration import Space
    from .expressions import Expr

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

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


def _class_body() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """The globals, namespace and enclosing locals of the class body declaring a member.

    Annotations are strings under ``from __future__ import annotations``; a
    family declared inside a function names that function's local classes, so
    the enclosing frame's locals are kept to resolve them.
    """
    frame = sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != _PACKAGE and not module.startswith(_PACKAGE + "."):
            break
        frame = frame.f_back  # type: ignore[assignment]
    if frame is None:
        return None
    namespace = frame.f_locals
    if "__qualname__" in namespace and "__module__" in namespace:
        outer = frame.f_back
        enclosing = (
            outer.f_locals if outer is not None and outer.f_locals is not frame.f_globals else {}
        )
        return frame.f_globals, namespace, enclosing
    return None


def _describe_formal(declaration: Declaration, kind: str) -> str:
    owner = declaration.owner
    where = f"{owner.__qualname__}.{declaration.name}" if owner is not None else kind
    return f"{where}{at(declaration.origin)}"


def declared_annotation(declaration: Declaration, kind: str) -> object:
    """The evaluated annotation of a class attribute: the single source of its value type.

    After class creation it is read from the owner's ``__annotations__``; in the
    class body itself (``output.spec`` before the class exists) from the body's
    namespace. A missing or unresolvable annotation is a definition error.
    """
    body = cast(
        "tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None",
        vars(declaration).get("_body"),
    )
    name = declaration.name
    annotations: Mapping[str, object]
    enclosing: Mapping[str, object] = {} if body is None else body[2]
    if declaration.owner is not None and name is not None:
        owner = declaration.owner
        annotations = owner.__dict__.get("__annotations__", {})
        module = sys.modules.get(owner.__module__)
        globals_ = vars(module) if module is not None else {}
        locals_: Mapping[str, object] = {
            **enclosing,
            **class_namespace(cast("type[Space]", owner)),
        }
    elif body is not None:
        globals_, namespace, _ = body
        locals_ = {**enclosing, **namespace}
        annotations = cast(Mapping[str, object], namespace.get("__annotations__", {}))
        name = next((key for key, value in namespace.items() if value is declaration), None)
    else:
        annotations, name, globals_, locals_ = {}, None, {}, {}
    if name is None or name not in annotations:
        return MISSING
    annotation = annotations[name]
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, dict(globals_), dict(locals_))  # noqa: S307 - authored annotation
    except Exception as cause:
        raise PendingAnnotation(
            f"{_describe_formal(declaration, kind)}: cannot resolve the annotation "
            f"{annotation!r}: {cause}"
        ) from cause


class PendingAnnotation(DefinitionError):
    """An annotation that does not resolve yet: it names a family still being defined
    (across an import cycle), or a name that does not exist, which collection reports."""


def _unannotated(declaration: Declaration, kind: str) -> DefinitionError:
    attribute = declaration.name or kind.lower()
    return DefinitionError(
        f"{_describe_formal(declaration, kind)}: annotate the {kind.lower()} with its value "
        f"type, as in `{attribute}: int = {kind}()`"
    )


def _annotation_semantics(
    annotation: object, explicit: ValueSemantics[Any] | None, label: str
) -> ValueSemantics[Any]:
    """Value semantics for an annotated value type; ``semantics=`` overrides the default."""
    if annotation is MISSING:
        # Declared outside a class body (a family built as data) with no
        # annotation: its explicit semantics carry the value type.
        assert explicit is not None
        return explicit
    if annotation is None:
        annotation = type(None)
    origin = get_origin(annotation)
    nominal = origin if origin is not None else annotation
    union = origin in (Union, types.UnionType)
    protocol = bool(getattr(nominal, "_is_protocol", False))
    if explicit is not None:
        token = explicit.type_token
        token = get_origin(token) or token
        if (
            not union
            and not protocol
            and isinstance(nominal, type)
            and isinstance(token, type)
            and nominal is not token
        ):
            raise DefinitionError(
                f"{label}: annotation {nominal.__qualname__} is incompatible with "
                f"{explicit.name} semantics"
            )
        return explicit
    if union or protocol or not isinstance(nominal, type):
        raise DefinitionError(
            f"{label}: the annotation {annotation!r} needs explicit semantics= (a union, "
            "protocol or special form has no default value semantics)"
        )
    return default_semantics(nominal)


def _space_family(annotation: object) -> type[Space] | None:
    from ._configuration import Space

    if isinstance(annotation, type) and issubclass(annotation, Space):
        return annotation
    return None


class Param(ValueDecl[T], Generic[T]):
    """A formal input of a family, supplied where a node of the family is declared.

    Annotate it with its value type: ``area: int = Param()``. The annotation is
    the single source of the value type; ``semantics=`` gives custom value
    semantics for it. Supply a formal at the call (``Room(area=12)``) or by
    assignment (``hall.area = kitchen.area``); an enclosing body may override
    what an inner one supplied, and the outermost assignment wins.

    ``Param()`` is required: preparing a family in which nothing supplies it is
    a definition error. ``default=`` makes it optional, and ``required=False``
    leaves it unsupplied when nobody binds it. A formal annotated with a family
    (``output: Stream = Param()``) is a reference input: the caller supplies a
    node, placed there if it is fresh and referenced if it is placed elsewhere.
    """

    default: object
    required: bool
    explicit: ValueSemantics[T] | None
    family: type[Space] | None
    resolved: bool

    @overload
    def __new__(  # type: ignore[misc]
        cls, *, default: T, semantics: ValueSemantics[T] | None = None
    ) -> T: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls, *, required: bool = True, semantics: ValueSemantics[T] | None = None
    ) -> T: ...

    def __new__(
        cls, *, default: object = MISSING, required: object = None, semantics: object = None
    ) -> Any:
        if default is not MISSING and required is not None:
            raise DefinitionError(
                "Param(default=...) is already optional; required= applies without a default"
            )
        if semantics is not None and not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        instance = cast(Param[object], super().__new__(cls))
        object.__setattr__(instance, "_body", _class_body())
        instance.explicit = cast("ValueSemantics[object] | None", semantics)
        instance.semantics = instance.explicit
        instance.family = None
        instance.resolved = False
        instance.required = default is MISSING and required is not False
        instance.default = default if default is not MISSING or instance.required else UNSUPPLIED
        return instance

    def __set_name__(self, owner: type[object], name: str) -> None:
        super().__set_name__(owner, name)
        try:
            self.resolve()
        except DefinitionError:
            pass  # a forward reference: resolved again when the family is collected

    def resolve(self) -> Param[T]:
        """Read the annotation: a value type (with its semantics) or a family (a reference)."""
        if self.resolved:
            return self
        annotation = declared_annotation(self, "Param")
        if annotation is MISSING and self.explicit is None:
            raise _unannotated(self, "Param")
        label = _describe_formal(self, "Param")
        family = _space_family(annotation)
        if family is not None:
            if self.explicit is not None or self.default not in (MISSING, UNSUPPLIED):
                raise DefinitionError(
                    f"{label}: a reference input has no value default or semantics; "
                    "required=False makes it optional"
                )
            self.family, self.semantics = family, None
        else:
            semantics = _annotation_semantics(annotation, self.explicit, label)
            if self.default not in (MISSING, UNSUPPLIED):
                if not recognize(semantics, self.default, owner=label, role="default recognition"):
                    raise DefinitionError(
                        f"{label}: invalid formal default: {unrecognized(semantics)}"
                    )
                self.default = snapshot(
                    semantics, self.default, owner=label, role="default snapshot"
                )
            self.semantics = semantics
        self.resolved = True
        return self

    def reference_family(self) -> type[Space] | None:
        """The family of a reference input, or None for a value formal."""
        return self.resolve().family

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object] | None = None) -> T: ...

    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Any:
        if instance is None:
            return self
        family = self.reference_family()
        if family is None:
            return ValueDecl.__get__(self, instance, owner)
        path = declared_path(instance)
        if path is not None:
            from ._nodes import path_proxy

            return path_proxy(family, (*path, self))
        from .occurrence import child

        return child(instance, self)

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            # ``output.spec`` in the class body that declares ``output: Stream``:
            # a member of the node the reference input will name. A value formal
            # projects an attribute of its value (``platform.uram``), as a derived
            # value does.
            if name.startswith("_"):
                raise AttributeError(name)
            try:
                family = self.reference_family()
            except PendingAnnotation:
                raise AttributeError(
                    f"{name}: the reference's annotation does not resolve yet (its family is "
                    "still being defined); read the member in a method"
                ) from None
            except DefinitionError:
                raise AttributeError(name) from None
            if family is None:
                return project(self, name)
            if not _has_member(family, name):
                raise AttributeError(name)
            from ._nodes import path_proxy

            return getattr(path_proxy(family, (self,)), name)


def _has_member(family: type[Space], name: str) -> bool:
    from ._nodes import slot_declaration

    return any(slot_declaration(vars(base).get(name)) is not None for base in family.__mro__)


class LocatedParam(Param[Located[T]], Generic[T]):
    """A formal holding a ``Located`` value: ``a: LocatedParam[int] = LocatedParam()``.

    Binding a plain reference supplies its located form: node name, member
    name and value. Its suppliers are typed as ``T`` and its value as
    ``Located[T]``, so the annotation names this descriptor rather than the
    value type (the one exception to annotating a formal with its value type).
    """

    def __new__(cls, *, required: bool = True) -> LocatedParam[T]:
        instance = Declaration.__new__(cls)
        object.__setattr__(instance, "_body", None)
        instance.explicit = cast("ValueSemantics[Located[T]]", LOCATED)
        instance.semantics = instance.explicit
        instance.family = None
        instance.resolved = True
        instance.required = required
        instance.default = MISSING if required else UNSUPPLIED
        return instance

    def __set__(
        self,
        instance: object,
        value: T | Located[T] | ValueRef[T] | ValueRef[Located[T]] | View[T] | View[Located[T]],
    ) -> None:
        from ._nodes import assign

        assign(instance, cast(str, self.name), value)


class Const(ValueDecl[T], Generic[T]):
    """A definition-owned frozen value."""

    def __init__(self, value: T, *, semantics: ValueSemantics[T] | None = None) -> None:
        self.semantics = semantics if semantics is not None else semantics_for(type(value))
        label = f"Const{at(self.origin)}"
        if not recognize(self.semantics, value, owner=label, role="constant recognition"):
            raise DefinitionError(f"{label}: {unrecognized(self.semantics)}")
        self.value = snapshot(self.semantics, value, owner=label, role="constant snapshot")


Guard: TypeAlias = "ValueRef[bool] | bool | None"


class Decision(ValueDecl[T], Generic[T]):
    """An independently editable choice, annotated with its value type.

    ``finish: int = Decision(values=(1, 2, 3))`` chooses a value; the
    annotation is its value type, ``semantics=`` gives custom value semantics.
    ``heating: Boiler | HeatPump = Decision({"boiler": Boiler, "pump": HeatPump(cop=4)},
    area=area)`` chooses a node: the persisted value is the key; each candidate
    is a node named ``<decision>.<key>`` whose presence derives from the
    decision. An entry is a family (a fresh node) or a call on one carrying the
    bindings only that candidate takes; the keyword arguments are shared
    bindings, supplied to every candidate, each of which must declare them.
    ``optional=True`` adds a ``None`` candidate keyed ``"none"``, which places
    nothing. ``requires=`` states what a value Decision's cases need
    (``requires(platform.uram, "uram-absent: ...", cases=("ultra",))``): a case
    whose fact does not hold is refused with that finding, at a commitment and
    when forcing reads its viability.

    ``heating.area`` reads a member every candidate declares;
    ``heating["pump"].cop`` reads one candidate's member, inapplicable while
    another is selected. An enclosing body pins the choice with a key
    (``house.heating = "pump"``) or narrows it with a Decision over keys
    (``Decision(values=("boiler",))``), keeping the declared candidates.

    A Decision that is not a class attribute takes its value type from the
    formal it supplies. It may supply exactly one formal
    (``Fifo(depth=Decision(values=(4, 8)))``), and is keyed by that formal's
    path. One that supplies several formals is shared and must be named: a
    class attribute, or ``Decision(..., name="depth")``, which is owned by the
    lowest scope containing every node it supplies.

    An enclosing body may override a Decision of a node it contains: a value
    pins the coordinate (its key disappears), another Decision replaces it
    under the same key. Either is checked against this Decision's domain.
    """

    domain: Domain[T]
    explicit: ValueSemantics[T] | None
    resolved: bool
    # Where this Decision supplies a formal (for the shared-decision rule).
    sites: list[str]

    @overload
    def __new__(
        cls,
        entries: Mapping[str, type[Space] | Space],
        /,
        *,
        optional: bool = False,
        when: Guard = None,
        **shared: object,
    ) -> Any: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls,
        /,
        *,
        values: Iterable[T],
        semantics: ValueSemantics[T] | None = None,
        requires: Iterable[Requirement] = (),
        when: Guard = None,
        name: str | None = None,
    ) -> T: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls,
        /,
        *,
        domain: Domain[T],
        semantics: ValueSemantics[T] | None = None,
        requires: Iterable[Requirement] = (),
        when: Guard = None,
        name: str | None = None,
    ) -> T: ...

    def __new__(
        cls,
        entries: object = None,
        /,
        *,
        domain: object = None,
        values: object = None,
        semantics: object = None,
        requires: object = None,
        when: object = None,
        name: object = None,
        optional: object = False,
        **shared: object,
    ) -> Any:
        if entries is not None:
            reserved = {
                key: value
                for key, value in (
                    ("domain", domain),
                    ("values", values),
                    ("semantics", semantics),
                    ("requires", requires),
                    ("name", name),
                )
                if value is not None
            }
            if reserved:
                raise DefinitionError(
                    f"Decision{at(source_origin())}: shared bindings {sorted(reserved)} are "
                    "named like the Decision's own arguments; rename the Param (values -> "
                    "contents), or write the binding on each entry that takes it"
                )
            from ._nodes import entry_choice

            return entry_choice(entries, shared, optional=optional, when=_guard(when))
        if shared or optional is not False:
            raise DefinitionError(
                "optional= and shared bindings apply to a Decision over candidate entries: "
                'Decision({"key": Family, ...}, ...)'
            )
        if domain is None and isinstance(values, Mapping):
            raise DefinitionError(
                f"Decision{at(source_origin())}: a Decision over nodes lists its candidates as "
                'entries, Decision({"key": Family, "other": Family(...)}); optional=True adds '
                'the None candidate "none"'
            )
        if (domain is None) == (values is None):
            raise DefinitionError("a Decision needs exactly one of domain= or values=")
        if semantics is not None and not isinstance(semantics, ValueSemantics):
            raise DefinitionError("semantics= must be a ValueSemantics")
        if domain is not None and not isinstance(domain, Domain):
            raise DefinitionError("domain= must be a Domain")
        instance = cast(Decision[object], super().__new__(cls))
        object.__setattr__(instance, "_body", _class_body())
        explicit = cast("ValueSemantics[object] | None", semantics)
        instance.explicit = instance.semantics = explicit
        instance.resolved = False
        instance.when = _guard(when)
        # The domain is bound to the value semantics when the model is linked.
        instance.domain = (
            cast(Domain[object], domain)
            if domain is not None
            else finite(cast(Iterable[object], values), explicit)
        )
        if requires is not None:
            stated = tuple(cast(Iterable[object], requires))
            if not all(isinstance(item, Requirement) for item in stated):
                raise DefinitionError("requires= takes requirements, as built by requires()")
            instance.domain = requiring(instance.domain, *cast(tuple[Requirement, ...], stated))
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
        try:
            self.resolve()
        except DefinitionError:
            pass  # a forward reference: resolved again when the family is collected

    def resolve(self) -> Decision[T]:
        """A class attribute reads its annotation; an inline one waits for its formal."""
        if self.resolved:
            return self
        annotation = declared_annotation(self, "Decision")
        if annotation is MISSING and self.owner is None:
            return self  # inline: typed by the formal it supplies
        if annotation is MISSING and self.explicit is None:
            raise _unannotated(self, "Decision")
        self.semantics = _annotation_semantics(
            annotation, self.explicit, _describe_formal(self, "Decision")
        )
        self.resolved = True
        return self


class Required:
    """A member every subclass must define: ``schedule = required(Schedule)``.

    It is not a member itself: collection ignores it. A subclass defines the
    member with any attribute (a Param, a derived value, a view, a method). A
    family whose effective attribute is still this marker is unfinished: it
    cannot be placed, and naming it as a candidate is a definition error.
    """

    __slots__ = ("kind", "name", "owner")

    def __init__(self, kind: type[object]) -> None:
        self.kind = kind
        self.name: str | None = None
        self.owner: type[object] | None = None

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.owner, self.name = owner, name

    def __repr__(self) -> str:
        where = f"{self.owner.__qualname__}." if self.owner is not None else ""
        return f"required({where}{self.name}: {self.kind.__qualname__})"


def required(kind: type[T]) -> T:
    """Declare a member of type ``kind`` that every subclass must define.

    It is typed as its value, like a derived member, so the family's own
    methods read ``self.schedule`` as a ``Schedule``; it is not a formal, so it
    is never a keyword of the family call.
    """
    if not isinstance(kind, type):
        raise DefinitionError("required() takes the member's value type, as in required(int)")
    return cast(T, Required(kind))


def unmet_required(family: type[object]) -> tuple[str, ...]:
    """``family.X`` for every member whose effective attribute is still ``required()``."""
    unmet: list[str] = []
    seen: set[str] = set()
    for base in family.__mro__:
        for name, value in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(value, Required):
                unmet.append(f"{base.__qualname__}.{name}")
    return tuple(unmet)


def unfinished(family: type[object], where: str) -> DefinitionError | None:
    """The error for placing ``family`` while it leaves required members unmet."""
    unmet = unmet_required(family)
    if not unmet:
        return None
    return DefinitionError(
        f"{family.__qualname__}{where} leaves required members unmet: {', '.join(unmet)}; a "
        "family with an unmet required() member cannot be placed (define them in a subclass)"
    )


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

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Projection[Any]:
            # ``facts.word_bits`` in a class body: an attribute of this derived value.
            return project(self, name)


class _DerivedDecorator:
    def __init__(self, aliases: Mapping[str, object], when: Guard) -> None:
        self.aliases, self.when = aliases, when

    def __call__(self, function: Callable[..., T | NonValue]) -> T:
        # Typed as its value, like a formal: it may supply a formal in the class body.
        # ``T | Rejected`` is typed ``T``, as its value semantics are ``T``'s.
        return cast(T, Derived(function, aliases=self.aliases, when=self.when))


class _SemanticDerivedDecorator(Generic[T]):
    def __init__(self, semantics: ValueSemantics[T], aliases: Mapping[str, object], when: Guard):
        self.semantics, self.aliases, self.when = semantics, aliases, when

    def __call__(self, function: Callable[..., T | QueryResult[T]]) -> T:
        return cast(
            T, Derived(function, semantics=self.semantics, aliases=self.aliases, when=self.when)
        )


@overload
def derived(function: Callable[..., T | NonValue], /) -> T: ...


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
# references to views or constraints through a node, and member families (each
# member's acceptance). A reference through a node (``kitchen.cost``) is typed
# as the value it stands for, so statically an obligation is any object; the
# linker refuses anything else with a DefinitionError.
Obligation: TypeAlias = object


class View(Declaration, Generic[T]):
    """One assessment declaration for either a value or an authored function.

    A view reads as its **accepted** value, like every member typed as its
    value: on a configuration ``point.total`` is the accepted value (a view
    that is not accepted raises ``ValueUnavailableError`` carrying its result,
    and inside a method the read blocks like any other); on a node declaration
    ``kitchen.cost`` is a reference to that accepted value, typed ``T``, which
    supplies a formal, feeds ``Present`` or joins ``requires=`` (where it
    contributes only its acceptance). Class access, ``House.total``, is the
    declaration itself: pass it to ``point.inspect`` for the assessment and to
    ``point.query`` for the accepted result.
    """

    @overload
    def __init__(
        self,
        source: ValueRef[T] | View[T],
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
    def __get__(self, instance: Space, owner: type[object] | None = None) -> T: ...

    def __get__(self, instance: Space | None, owner: type[object] | None = None) -> Self | T:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return cast(T, MemberRef(path, self))
        from .occurrence import read_value

        return read_value(instance, self)


class _ViewDecorator:
    def __init__(
        self, aliases: Mapping[str, object], requires: Sequence[Obligation], when: Guard
    ) -> None:
        self.aliases, self.requires, self.when = aliases, requires, when

    def __call__(self, function: Callable[..., T | NonValue]) -> View[T]:
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
def view(function: Callable[..., T | NonValue], /) -> View[T]: ...


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

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Projection[Any]:
            return project(self, name)


def _attribute_type(owner: object, name: str) -> object:
    """The annotated type of a value type's attribute: a property's return, or a field."""
    import inspect

    if not isinstance(owner, type):
        return MISSING
    # A dataclass field without a default is annotated but no class attribute.
    attribute = inspect.getattr_static(owner, name, None)
    if isinstance(attribute, property) and attribute.fget is not None:
        return _hints(attribute.fget, name, include_extras=True).get("return", MISSING)
    return _hints(owner, name, include_extras=True).get(name, MISSING)


def _returned_type(function: Callable[..., object], name: str) -> object:
    """``T`` of a ``T`` or ``T | Rejected`` return annotation."""
    annotation = _hints(function, name, include_extras=False).get("return")
    marked = marked_value_type(annotation)
    return marked if marked is not None else annotation


def _hints(annotated: object, name: str, *, include_extras: bool) -> dict[str, object]:
    """``annotated``'s resolved annotations; unresolvable ones leave ``name`` unprojectable.

    The failure stays an ``AttributeError`` (introspection is unaffected) that
    names the annotations and why they do not resolve.
    """
    try:
        return get_type_hints(annotated, include_extras=include_extras)
    except (NameError, SyntaxError, TypeError) as cause:
        label = getattr(annotated, "__qualname__", repr(annotated))
        raise AttributeError(
            f"{name}: cannot resolve the annotations of {label}: {cause}"
        ) from cause


def project(source: ValueRef[Any], name: str) -> Projection[Any]:
    """``output.spec.payload_bits``: an attribute of a referenced value, typed by its class.

    Only an attribute the value's type annotates projects; anything else is
    the ordinary ``AttributeError``, so introspection stays unaffected.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    semantics = source.semantics
    token = None if semantics is None else semantics.type_token
    if token is None and isinstance(source, Derived) and source.function is not None:
        # No semantics= yet: the value type the return annotation names (``T | Rejected``).
        token = _returned_type(source.function, name)
    token = get_origin(token) or token
    if not isinstance(token, type):
        raise AttributeError(name)
    annotation = _attribute_type(token, name)
    if annotation is MISSING:
        raise AttributeError(name)
    return Projection(source, name, annotation)


class Projection(_Symbolic, ValueRef[T], Generic[T]):
    """An attribute of a referenced value, read in the class body (``spec.payload_bits``).

    Its semantics are the attribute's class's, or the ``ValueSemantics`` its
    annotation names (``dims: Annotated[tuple[int, ...], INTEGER_VECTOR]``).
    """

    def __init__(self, source: ValueRef[Any], attribute: str, annotation: object) -> None:
        self.source, self.attribute = source, attribute
        if get_origin(annotation) is Annotated:
            named = [item for item in get_args(annotation) if isinstance(item, ValueSemantics)]
            if named:
                self.semantics = cast("ValueSemantics[T]", named[0])
                return
            annotation = get_args(annotation)[0]
        origin = get_origin(annotation)
        nominal = origin if origin is not None else annotation
        self.semantics = (
            cast("ValueSemantics[T]", default_semantics(nominal))
            if isinstance(nominal, type)
            else None
        )

    def _describe(self) -> str:
        inner = self.source._describe() if isinstance(self.source, _Symbolic) else "value"
        return f"{inner}.{self.attribute}"

    def _key(self) -> tuple[object, ...]:
        inner = self.source._key() if isinstance(self.source, _Symbolic) else (id(self.source),)
        return (*inner, self.attribute)

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Projection[Any]:
            return project(self, name)


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


class Supplied(ValueDecl[bool]):
    """Whether a value input of this family is supplied (see ``supplied``)."""

    def __init__(self, formal: Param[object]) -> None:
        self.formal = formal
        self.semantics = cast("ValueSemantics[bool]", default_semantics(bool))


def supplied(formal: object) -> bool:
    """Whether the value input ``formal`` is supplied, as a declaration: what
    ``present(formal)`` answers in a method (whether a source applies; its value
    is not evaluated).

    As a guard (``when=supplied(contents)``) the compiler reads it before
    evaluation: where the declaration never supplies the input, a Decision over
    nodes it guards can never apply, and its candidates are not compiled.
    """
    if not isinstance(formal, Param) or formal.reference_family() is not None:
        raise DefinitionError("supplied() takes a value input (a Param) of this family")
    return cast(bool, Supplied(cast("Param[object]", formal)))


class Present(ValueDecl[T], Generic[T]):
    """The value of whichever one of ``sources`` is present (applicable).

    Unresolved while any source is unresolved; refused if two are present;
    unsupplied (unresolved) if none is. Typed as its value, so it may supply a
    formal (``sink.width = Present(a.out, b.out)``).
    """

    @overload
    def __new__(  # type: ignore[misc]
        cls,
        *sources: ValueRef[T] | View[T],
        semantics: ValueSemantics[T] | None = None,
    ) -> T: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls, *sources: T, semantics: ValueSemantics[T] | None = None
    ) -> T: ...

    def __new__(cls, *sources: object, semantics: object = None) -> Any:
        if not sources or any(not isinstance(source, (ValueRef, View)) for source in sources):
            raise DefinitionError("Present requires one or more value references")
        instance = cast("Present[object]", super().__new__(cls))
        instance.sources = cast(tuple[ValueRef[object], ...], sources)
        instance.semantics = cast(
            "ValueSemantics[object] | None",
            semantics if semantics is not None else getattr(sources[0], "semantics", None),
        )
        return instance

    sources: tuple[ValueRef[T], ...]


class Members(ValueDecl[tuple[Located[T], ...]], Generic[T]):
    """Every present child node exporting ``key``, as ``Located`` values in
    declaration order; as an obligation, each member's acceptance counts.

    Each entry is ``Located(node=<child name>, member=<key name>, value=<export>)``.
    A child exporting ``key`` per input (``exports = {key: {input: view}}``)
    contributes one entry per input, with ``member=<input name>``.
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

    A user may export ``key`` per input, ``exports = {key: {input: view, ...}}``:
    each node it references then sees only the view presented through the
    input that references it, and an input without an entry is omitted.
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
    "Projection",
    "Required",
    "Supplied",
    "UNSUPPLIED",
    "Users",
    "ValueDecl",
    "ValueRef",
    "View",
    "ViewKey",
    "constraint",
    "derived",
    "required",
    "selected",
    "supplied",
    "unfinished",
    "unmet_required",
    "view",
]
