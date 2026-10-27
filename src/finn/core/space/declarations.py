# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed declarations for the Space language.

A Space class's body declares members (formals, decisions, computations,
views) and nodes. Calling a Space class, ``Room(area=12)``, declares a node: a
template with bindings, compiled only by ``design_space``. Attribute access on a
node declaration, ``kitchen.finish``, is a symbolic reference to that node's
member. It is typed as the member's value; at runtime it refuses
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
    Any,
    Generic,
    NoReturn,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    get_origin,
    get_type_hints,
    overload,
)

from .domains import Domain, finite
from .errors import DefinitionError, ReferenceUseError
from .graph import Located
from .results import NonValue, QueryResult, marked_value_type
from .semantics import (
    BOOL,
    STRING,
    NotAValue,
    ValueSemantics,
    annotation_type,
    canonical,
    default_semantics,
    recognize,
    unrecognized,
)

if TYPE_CHECKING:
    from ._configuration import _Namespace

_OWNER = "space.declarations"

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def local_name(value: str, role: str) -> str:
    """Keep authored identity segments disjoint from generated node names."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise DefinitionError.broken(
            "name-invalid", _OWNER, f"{role} must be one nonempty ASCII name segment"
        )
    return value


def class_namespace(space_type: type[_Namespace]) -> dict[str, object]:
    namespace: dict[str, object] = {}
    for base in reversed(space_type.__mro__):
        namespace.update(vars(base))
    namespace[space_type.__name__] = space_type
    return namespace


_PACKAGE = __name__.rsplit(".", 1)[0]


def _outside_frame(depth: int) -> types.FrameType | None:
    """The nearest frame outside this package, starting ``depth`` frames above the caller."""
    frame: types.FrameType | None = sys._getframe(depth + 1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != _PACKAGE and not module.startswith(_PACKAGE + "."):
            return frame
        frame = frame.f_back
    return None


def source_origin() -> str | None:
    """``file:line`` of the nearest caller outside this package, for diagnostics."""
    frame = _outside_frame(1)
    if frame is None:
        return None
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"


def at(origin: str | None) -> str:
    return "" if origin is None else f" (declared at {origin})"


class _Missing:
    def __repr__(self) -> str:
        return "<required>"


MISSING = _Missing()


class Declaration:
    """Identity-bearing source declaration; names are diagnostic hints only.

    A declaration is written while it is open: in the class body that declares it, or,
    for a node, at its call and by assignment there. It is frozen when the Space class
    holding it is created, or when ``design_space`` takes it as a root (``frozen``, the
    reason); from then on writing a field of it is refused (``declaration-frozen``). What
    the engine works out or records about it (its annotation resolved, its placement) it
    writes with ``_settle``.
    """

    name: str | None = None
    owner: type[object] | None = None
    when: ValueRef[bool] | None = None
    origin: str | None = None
    frozen: str | None = None
    _record_origin = True

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        instance = super().__new__(cls)
        if cls._record_origin:
            object.__setattr__(instance, "origin", source_origin())
        return instance

    def __set_name__(self, owner: type[object], name: str) -> None:
        if self.owner is not None and (self.owner is not owner or self.name != name):
            raise DefinitionError.broken(
                "declaration-reused",
                _OWNER,
                f"{owner.__qualname__}.{name}: declaration already belongs to "
                f"{self.owner.__qualname__}.{self.name}; create a fresh declaration",
                declaration=f"{owner.__qualname__}.{name}",
            )
        self._settle(owner=owner, name=name)

    def freeze(self, reason: str) -> None:
        """Close the declaration (the first reason is kept)."""
        if self.frozen is None:
            self._settle(frozen=reason)

    def _settle(self, **fields: object) -> None:
        """The engine's own writes, which a frozen declaration takes: what it works out or
        records about the declaration, never what its author wrote."""
        for field, value in fields.items():
            object.__setattr__(self, field, value)

    if not TYPE_CHECKING:
        # Hidden from type checkers: a declared __setattr__ would make mypy accept
        # assignment to any attribute.

        def __setattr__(self, field: str, value: object) -> None:
            if self.frozen is not None:
                _refuse_frozen(self, field)
            object.__setattr__(self, field, value)

    def __bool__(self) -> bool:
        raise TypeError("a declaration has no truth value")


def _refuse_frozen(declaration: Declaration, field: str) -> NoReturn:
    owner, name = declaration.owner, declaration.name
    what = (
        f"{owner.__qualname__}.{name}"
        if owner is not None and name is not None
        else type(declaration).__name__
    )
    label = f"{what}.{field}{at(declaration.origin)}"
    raise DefinitionError.broken(
        "declaration-frozen",
        _OWNER,
        f"{label}: the declaration is frozen ({declaration.frozen}); a declaration is written "
        "in the class body that declares it, and a subclass redeclares it",
        declaration=label,
    )


class ValueRef(Declaration, Generic[T_co]):
    """A typed value handle: a member or a symbolic reference."""

    semantics: ValueSemantics[T_co] | None

    @property
    def value_semantics(self) -> ValueSemantics[T_co] | None:
        return self.semantics

    if not TYPE_CHECKING:
        # A reference is not an integer to compute with: arithmetic on one is refused,
        # for a derived value to state.

        def __add__(self, other: object) -> NoReturn:
            _expression_refused(self, "+")

        def __sub__(self, other: object) -> NoReturn:
            _expression_refused(self, "-")

        def __mul__(self, other: object) -> NoReturn:
            _expression_refused(self, "*")

        def __floordiv__(self, other: object) -> NoReturn:
            _expression_refused(self, "//")

        def __mod__(self, other: object) -> NoReturn:
            _expression_refused(self, "%")

        __radd__, __rsub__, __rmul__ = __add__, __sub__, __mul__
        __rfloordiv__, __rmod__ = __floordiv__, __mod__

        def __neg__(self) -> NoReturn:
            _expression_refused(self, "unary -")


def reference_label(reference: ValueRef[Any]) -> str:
    """How a refusal names a reference: ``Class.name`` once its class exists, the name it
    has in the class body being written (found in the nearest frame outside the engine),
    else its kind, each with where it was declared."""
    if isinstance(reference, _Symbolic):
        return reference._describe()
    if reference.name is not None and reference.owner is not None:
        return describe_formal(reference, type(reference).__name__)
    frame = _outside_frame(0)
    if frame is not None:
        for name, value in frame.f_locals.items():
            if value is reference:
                return f"{name}{at(reference.origin)}"
    return describe_formal(reference, type(reference).__name__)


def _expression_refused(reference: ValueRef[Any], operator: str) -> NoReturn:
    label = reference_label(reference)
    raise DefinitionError.broken(
        "reference-expression",
        _OWNER,
        f"{label}: arithmetic ({operator}) on a reference is not available in a class body; "
        "declare a derived value that computes it, and bind that member",
        declaration=label,
    )


def declared_path(instance: object) -> tuple[Declaration, ...] | None:
    """The node path of a declaration-mode Space object, or None for a configuration."""
    return cast("tuple[Declaration, ...] | None", vars(instance).get("_space_path"))


class ValueDecl(ValueRef[T_co], Generic[T_co]):
    """A class member descriptor.

    On a configuration it reads the value; on a node declaration it returns a
    symbolic ``MemberRef``, statically typed as the value, which refuses value-like
    use at runtime (``ReferenceUseError``).
    """

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: _Namespace, owner: type[object] | None = None) -> T_co: ...

    def __get__(
        self, instance: _Namespace | None, owner: type[object] | None = None
    ) -> Self | T_co:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return cast(T_co, MemberRef(path, self))
        from .occurrence import read_value

        return read_value(instance, self)


def class_body() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """The globals, namespace and enclosing locals of the class body declaring a member.

    Annotations are strings under ``from __future__ import annotations``; a
    Space class declared inside a function names that function's local classes, so
    the enclosing frame's locals are kept to resolve them.
    """
    frame = _outside_frame(2)
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


def describe_formal(declaration: Declaration, kind: str) -> str:
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
            **class_namespace(cast("type[_Namespace]", owner)),
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
        raise PendingAnnotation.broken(
            "annotation-unresolved",
            _OWNER,
            f"{describe_formal(declaration, kind)}: cannot resolve the annotation "
            f"{annotation!r}: {cause}",
            declaration=describe_formal(declaration, kind),
        ) from cause


class PendingAnnotation(DefinitionError):
    """An annotation that does not resolve yet: it names a Space class still being defined
    (across an import cycle), or a name that does not exist, which collection reports."""


def _unannotated(declaration: Declaration, kind: str) -> DefinitionError:
    attribute = declaration.name or kind.lower()
    return DefinitionError.broken(
        "annotation-missing",
        _OWNER,
        f"{describe_formal(declaration, kind)}: annotate the {kind.lower()} with its value "
        f"type, as in `{attribute}: int = {kind}()`",
        declaration=describe_formal(declaration, kind),
    )


def _annotation_semantics(annotation: object, label: str) -> ValueSemantics[Any]:
    """The type check a declared annotation states (``semantics.annotation_type``)."""
    return annotation_type(annotation, label, owner=_OWNER)


def literal(
    semantics: ValueSemantics[Any],
    value: object,
    label: str,
    *,
    role: str,
    owner: str,
    what: str = "",
) -> object:
    """A value written in a definition (a default, a constant, a literal bound at
    placement), type-checked by ``semantics`` and in its canonical form
    (``semantics.canonical``); a ``DefinitionError`` naming ``label`` where it is neither."""
    if not recognize(semantics, value, owner=owner, role=f"{role} recognition"):
        raise DefinitionError.broken(
            "literal-unrecognized",
            _OWNER,
            f"{label}: {what}{unrecognized(semantics)}",
            declaration=label,
        )
    try:
        return canonical(value)
    except NotAValue as refusal:
        raise DefinitionError.broken(
            "literal-not-value", _OWNER, f"{label}: {what}{refusal}", declaration=label
        ) from None


def _annotated_space_type(annotation: object) -> type[_Namespace] | None:
    from ._configuration import _Namespace

    if isinstance(annotation, type) and issubclass(annotation, _Namespace):
        return annotation
    return None


class Param(ValueDecl[T], Generic[T]):
    """A formal input of a Space class, supplied where a node of the class is declared.

    Annotate it with its value type: ``area: int = Param()``. The annotation is
    the single source of the value type, and its check (``semantics.annotation_type``:
    a class, a union of classes, ``T | None``); a value is compared by ``==``. Supply a
    formal at the call (``Room(area=12)``) or by assignment (``hall.area = kitchen.area``),
    once, in the body that declares the node.

    ``Param()`` is required: preparing a Space class in which nothing supplies it is
    a definition error. ``default=`` gives the value it takes where nobody supplies it.
    An optional fact is ``x: T | None = Param(default=None)``: left out it is ``None``,
    a known value, the same fact as one given ``None``; its readers test it with
    ``self.x is None``. A formal takes a value, never a node: a formal annotated with a
    Space class is refused (a node is placed in a class body, and reads what it needs as
    values).
    """

    default: object
    required: bool
    resolved: bool

    @overload
    def __new__(cls, *, default: T) -> T: ...  # type: ignore[misc]

    @overload
    def __new__(cls) -> T: ...  # type: ignore[misc]

    def __new__(cls, *, default: object = MISSING) -> Any:
        instance = cast(Param[object], super().__new__(cls))
        object.__setattr__(instance, "_body", class_body())
        instance.semantics = None
        instance.resolved = False
        instance.required = default is MISSING
        instance.default = default
        return instance

    def __set_name__(self, owner: type[object], name: str) -> None:
        super().__set_name__(owner, name)
        try:
            self.resolve()
        except PendingAnnotation:
            pass  # a forward reference: resolved again when the Space class is collected

    def resolve(self) -> Param[T]:
        """Read the annotation: a value type, and its check."""
        if self.resolved:
            return self
        annotation = declared_annotation(self, "Param")
        if annotation is MISSING:
            raise _unannotated(self, "Param")
        label = describe_formal(self, "Param")
        space_type = _annotated_space_type(annotation)
        if space_type is not None:
            raise DefinitionError.broken(
                "formal-takes-node",
                _OWNER,
                f"{label}: a formal takes a value, not a {space_type.__qualname__} node; "
                "place the node in a class body, and give the formal the values it reads",
                declaration=label,
            )
        semantics = _annotation_semantics(annotation, label)
        if self.default is not MISSING:
            # Its canonical form, the engine's: a frozen formal takes it too.
            self._settle(
                default=literal(
                    semantics,
                    self.default,
                    label,
                    role="default",
                    owner=label,
                    what="invalid formal default: ",
                )
            )
        self._settle(semantics=semantics, resolved=True)
        return self

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: _Namespace, owner: type[object] | None = None) -> T: ...

    def __get__(self, instance: _Namespace | None, owner: type[object] | None = None) -> Any:
        if instance is None:
            return self
        return ValueDecl.__get__(self, instance, owner)

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> NoReturn:
            # An attribute of a formal's value (``platform.uram``) is refused, for a
            # derived value to read.
            if name.startswith("_"):
                raise AttributeError(name)
            try:
                self.resolve()
            except DefinitionError:
                raise AttributeError(name) from None
            _projection_refused(self, name)


Guard: TypeAlias = "ValueRef[bool] | bool | None"

SEMANTICS_GONE = (
    "semantics= is gone: the annotation is the value's type and its check, and a value is "
    "compared by =="
)
"""The refusal of ``semantics=`` where a value's annotation states its type."""

_RETIRED: Mapping[str, tuple[str, str]] = {
    "optional": (
        "decision-optional",
        "a case that places nothing is a None case of a selector: write Select(none=None, ...)",
    ),
    "required": (
        "decision-required",
        "a Decision states no policy; a choice no default may set is the compiler's to "
        "state: write FINN's Required() policy on the Decision, in its KernelOp or the cut",
    ),
    "name": (
        "decision-named",
        "a Decision belongs to the Space class that declares it: write it as a class "
        "attribute, and bind that attribute to each formal it supplies",
    ),
    "semantics": (
        "decision-semantics",
        SEMANTICS_GONE,
    ),
}


def _refuse_retired(entries: tuple[object, ...], retired: Mapping[str, object]) -> None:
    """The retired spellings of a Decision, each refused with its replacement."""
    where = at(source_origin())
    if entries:
        raise DefinitionError.broken(
            "decision-over-nodes",
            _OWNER,
            f"Decision{where}: a Decision keeps value domains (values= or domain=); a choice "
            "among Space classes is a selector: write Select(name=Class, other=Class(...), "
            "shared=...), a case that places nothing as None, Select(none=None, ...)",
        )
    for keyword in retired:
        if keyword in _RETIRED:
            code, replacement = _RETIRED[keyword]
            raise DefinitionError.broken(code, _OWNER, f"Decision{where}: {replacement}")
    if retired:
        raise TypeError(
            f"Decision() got unexpected keyword arguments {sorted(retired)}; it takes values= "
            "or domain=, when= and ordered="
        )


class Decision(ValueDecl[T], Generic[T]):
    """An independently editable choice, annotated with its value type.

    ``finish: int = Decision(values=(1, 2, 3))`` chooses a value; the
    annotation is its value type and its check, and a value is compared by ``==``.
    A Decision keeps value domains only: a choice among Space classes is a selector,
    ``Select`` (``heating = Select(boiler=Boiler, pump=HeatPump(cop=4), area=area)``).
    A Decision's values are its domain, whose membership reads facts only:
    a rule linking a value to anything else (``ultra`` needs UltraRAM) is a
    ``@constraint`` of the Space that owns the choice, and a change that breaks it
    is refused, naming it. ``ordered=True`` states that listed values
    are in an order (a domain states its own, ``divisors_of``); otherwise they
    are unordered. The engine sets no choice itself, by default or otherwise, and holds
    no policy: whether a policy may set a choice, and how, is the compiler's.

    A Decision belongs to the Space class that declares it, as a class attribute,
    and is supplied to a formal by binding that attribute
    (``depth: int = Decision(...)``, then ``buffer = Fifo(depth=depth)``). No body
    supplies it a value, a key or another Decision. A choice is set on a design space
    (``with_choices``); a narrower one is a subclass's.

    Retired spellings, each refused with its replacement: entries
    (``Decision({...})``: ``Select``), ``optional=`` (a ``None`` case of a ``Select``),
    ``required=`` (a policy, the compiler's), ``name=`` and a Decision given at a call
    (declare it on the class, and bind it), ``semantics=`` (the annotation).
    """

    domain: Domain[T]
    resolved: bool

    @overload
    def __new__(  # type: ignore[misc]
        cls,
        /,
        *,
        values: Iterable[T],
        when: Guard = None,
        ordered: bool = False,
    ) -> T: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls,
        /,
        *,
        domain: Domain[T],
        when: Guard = None,
    ) -> T: ...

    def __new__(
        cls,
        *entries: object,
        domain: object = None,
        values: object = None,
        when: object = None,
        ordered: object = None,
        **retired: object,
    ) -> Any:
        _refuse_retired(entries, retired)
        if domain is None and isinstance(values, Mapping):
            raise DefinitionError.broken(
                "node-decision-not-entries",
                _OWNER,
                f"Decision{at(source_origin())}: a choice among Space classes is a selector: "
                "write Select(key=Room, other=Room(...)); a case that places nothing is None",
            )
        if (domain is None) == (values is None):
            raise DefinitionError.broken(
                "decision-domain-ambiguous",
                _OWNER,
                "a Decision needs exactly one of domain= or values=",
            )
        if domain is not None and not isinstance(domain, Domain):
            raise DefinitionError.broken(
                "domain-argument-invalid", _OWNER, "domain= must be a Domain"
            )
        if ordered is not None and (domain is not None or type(ordered) is not bool):
            raise DefinitionError.broken(
                "ordered-invalid",
                _OWNER,
                "ordered= is a bool, for listed values= (a domain= states its own order)",
            )
        instance = cast(Decision[object], super().__new__(cls))
        object.__setattr__(instance, "_body", class_body())
        instance.semantics = None
        instance.resolved = False
        instance.when = guard(when)
        # The domain is bound to the value's type check when the model is linked.
        instance.domain = (
            cast(Domain[object], domain)
            if domain is not None
            else finite(cast(Iterable[object], values), ordered=ordered is True)
        )
        return instance

    def __set_name__(self, owner: type[object], name: str) -> None:
        super().__set_name__(owner, name)
        try:
            self.resolve()
        except PendingAnnotation:
            pass  # a forward reference: resolved again when the Space class is collected

    def resolve(self) -> Decision[T]:
        """A class attribute reads its annotation (in its class body, before the class is
        made, as its sibling's binding reads it)."""
        if self.resolved:
            return self
        annotation = declared_annotation(self, "Decision")
        if annotation is MISSING and self.owner is None:
            return self  # still in its class body: resolved when the class is made
        if annotation is MISSING:
            raise _unannotated(self, "Decision")
        self._settle(
            semantics=_annotation_semantics(annotation, describe_formal(self, "Decision")),
            resolved=True,
        )
        return self


def Select(  # noqa: N802 - a declaration, spelled as one
    *,
    by: object = None,
    when: object = None,
    **members: object,
) -> Any:
    """A selector: ``heating = Select(boiler=Boiler(kw=24), heat_pump=HeatPump, kw=kw)``.

    Each keyword whose value is a Space class, a call on one or ``None`` is a **case**
    (``None`` places nothing); each other keyword is a **shared binding**, supplied to
    every case that places something, which must declare it; a class object as a shared
    binding is refused (``select-binding-class``). Picked by a choice, its stored value is
    the case's name (``heating``'s key). Picked by a derived value, ``Select(by=chain,
    ...)``, it has no key and is never a choice: the case is the one the value names, and
    a value naming no case (``None`` included: a ``None`` case is named by its key)
    refuses, ``select-no-case``. While the value waits its cases wait; while it refuses
    they are not in play, refusing the Space once, in the selector's name.

    **Reading it.** A case's path is ``<selector>.<case>``: ``heating["boiler"].kw``
    reads one case's member, and ``heating.kw`` a member every case declares. A binding
    to the selector itself (``derived(case=heating)(f)``, a view's, a rule's or a
    guard's argument) reads its value, the case's name: a ``None`` case's included, its
    key (``"none"`` for ``none=None``). In a method body, ``self.heating`` is the
    selected case's design space, ``None`` for a ``None`` case.

    ``optional=`` and ``required=`` are refused (``select-optional``,
    ``select-required``): a ``None`` case, and a policy (the compiler's).
    """
    from ._nodes import select_choice

    return select_choice(members, by=by, when=guard(when))


class Required:
    """A member every subclass must define: ``schedule = required(Schedule)``.

    It is not a member itself: collection ignores it. A subclass defines the
    member with any attribute (a Param, a derived value, a view, a method). A
    Space class whose effective attribute is still this marker is unfinished: it
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

    It is typed as its value, like a derived member, so the Space class's own
    methods read ``self.schedule`` as a ``Schedule``; it is not a formal, so it
    is never a keyword of the class's call.
    """
    if not isinstance(kind, type):
        raise DefinitionError.broken(
            "required-type-invalid",
            _OWNER,
            "required() takes the member's value type, as in required(int)",
        )
    return cast(T, Required(kind))


def unmet_required(space_type: type[object]) -> tuple[str, ...]:
    """``space_type.X`` for every member whose effective attribute is still ``required()``."""
    unmet: list[str] = []
    seen: set[str] = set()
    for base in space_type.__mro__:
        for name, value in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            if isinstance(value, Required):
                unmet.append(f"{base.__qualname__}.{name}")
    return tuple(unmet)


def unfinished(space_type: type[object], where: str) -> DefinitionError | None:
    """The error for placing ``space_type`` while it leaves required members unmet."""
    unmet = unmet_required(space_type)
    if not unmet:
        return None
    return DefinitionError.broken(
        "required-unmet",
        _OWNER,
        f"{space_type.__qualname__}{where} leaves required members unmet: {', '.join(unmet)}; a "
        "Space class with an unmet required() member cannot be placed (define them in a subclass)",
        declaration=space_type.__qualname__,
    )


def guard(when: object) -> ValueRef[bool] | None:
    if when is None:
        return None
    if not isinstance(when, ValueRef):
        raise DefinitionError.broken(
            "guard-not-boolean", _OWNER, "when= requires a Boolean reference"
        )
    return cast(ValueRef[bool], when)


def _refuse_computation_when(kind: str, aliases: Mapping[str, object]) -> None:
    """``when=`` on a derived value, constraint or view: a computation says it does not
    apply by returning ``Inapplicable()``, its guard read first (``when=`` is a placement's
    and a Decision's presence)."""
    if "when" in aliases:
        raise DefinitionError.broken(
            "computation-when",
            _OWNER,
            f"{kind}{at(source_origin())}: {kind} takes no when=: return Inapplicable() from "
            "its body, reading the guard first (when= is for a placement or a Decision)",
        )


class Derived(ValueDecl[T], Generic[T]):
    """A function whose dependencies are bound after effective collection."""

    def __init__(
        self,
        function: Callable[..., T | QueryResult[T]],
        *,
        aliases: Mapping[str, object] | None = None,
    ) -> None:
        if aliases is not None and "semantics" in aliases:
            raise DefinitionError.broken(
                "derived-semantics",
                _OWNER,
                f"derived{at(source_origin())}: {SEMANTICS_GONE} (the return annotation)",
            )
        self.function = function
        self.semantics: ValueSemantics[T] | None = None
        self.aliases = MappingProxyType(dict(aliases or {}))

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> NoReturn:
            # ``facts.word_bits`` in a class body: an attribute of this derived value.
            _projection_refused(self, name)


class _DerivedDecorator:
    def __init__(self, aliases: Mapping[str, object]) -> None:
        self.aliases = aliases

    @overload
    def __call__(self, function: Callable[..., QueryResult[T]]) -> T: ...

    @overload
    def __call__(self, function: Callable[..., T | NonValue]) -> T: ...

    def __call__(self, function: Callable[..., object]) -> object:
        # Typed as its value, like a formal: it may supply a formal in the class body.
        # ``T | Rejected`` and ``QueryResult[T]`` are typed ``T``: its value's type.
        return Derived(function, aliases=self.aliases)


@overload
def derived(function: Callable[..., QueryResult[T]], /) -> T: ...


@overload
def derived(function: Callable[..., T | NonValue], /) -> T: ...


@overload
def derived(**aliases: object) -> _DerivedDecorator: ...


def derived(function: Callable[..., object] | None = None, /, **aliases: object) -> object:
    """A derived value: ``@derived`` on a method, or ``@derived(**aliases)``. Its return
    annotation is its value's type and check (``T``, or ``T | Rejected``). It does not
    apply by returning ``Inapplicable()``, its guard read first."""
    _refuse_computation_when("derived", aliases)
    if function is not None:
        return Derived(function, aliases=aliases)
    if "semantics" in aliases:
        raise DefinitionError.broken(
            "derived-semantics",
            _OWNER,
            f"derived{at(source_origin())}: {SEMANTICS_GONE} (the return annotation)",
        )
    return _DerivedDecorator(aliases)


class _MemberOfNode:
    """A non-value member: read through a node declaration it is a reference."""

    def __get__(self, instance: object, owner: type[object] | None = None) -> Self:
        if instance is not None:
            path = declared_path(instance)
            if path is not None:
                return cast(Self, MemberRef(path, cast(Declaration, self)))
        return self


class Constraint(_MemberOfNode, Declaration):
    """A rule of its Space: true, false (refused, with its findings) or unknown. Every
    constraint of a present Space counts toward the Space's validity
    (``inspection.validity``); a constraint that returns ``Inapplicable()`` counts as
    true. A constraint reads values (formals, choices, derived values, a child's
    derived value), never a view of its own Space or of an ancestor: that view's
    answer reads this constraint, a dependency cycle."""

    def __init__(
        self,
        function: Callable[..., bool | QueryResult[bool]],
        *,
        aliases: Mapping[str, object] | None = None,
    ) -> None:
        self.function = function
        self.aliases = MappingProxyType(dict(aliases or {}))
        self.semantics = cast("ValueSemantics[bool]", BOOL)


class _ConstraintDecorator:
    def __init__(self, aliases: Mapping[str, object]) -> None:
        self.aliases = aliases

    def __call__(self, function: Callable[..., bool | QueryResult[bool]]) -> Constraint:
        return Constraint(function, aliases=self.aliases)


@overload
def constraint(function: Callable[..., bool | QueryResult[bool]], /) -> Constraint: ...


@overload
def constraint(**aliases: object) -> _ConstraintDecorator: ...


def constraint(
    function: Callable[..., bool | QueryResult[bool]] | None = None,
    /,
    **aliases: object,
) -> Constraint | _ConstraintDecorator:
    _refuse_computation_when("constraint", aliases)
    if function is not None:
        return Constraint(function, aliases=aliases)
    return _ConstraintDecorator(aliases)


class View(Declaration, Generic[T]):
    """A local readout of its Space, from a value or authored function.

    On a configuration it reads its value; unavailable inputs, local refusal and
    ordinary presence determine its answer after its real owning Space is checked.
    Owning refusal blocks output entry; open validity permits local computation.
    On the class it is a declaration, and on a placement a typed symbolic reference.
    Views compose views but cannot feed model bindings, domains or rules.
    ``point.inspect`` records applicability, the owning gate and attempted output.
    An operation needing a valid, complete design still checks that at its boundary.
    """

    @overload
    def __init__(self, source: ValueRef[T] | View[T]) -> None: ...

    @overload
    def __init__(self, source: T) -> None: ...

    def __init__(self, source: object) -> None:
        self.source: object | None = source
        self.function: Callable[..., T | QueryResult[T]] | None = None
        self.aliases: Mapping[str, object] = MappingProxyType({})
        self.semantics: ValueSemantics[T] | None = cast(
            "ValueSemantics[T] | None", getattr(source, "semantics", None)
        )

    @classmethod
    def from_function(
        cls,
        function: Callable[..., T | QueryResult[T]],
        *,
        aliases: Mapping[str, object],
    ) -> View[T]:
        result = cls.__new__(cls)
        result.source = None
        result.function = function
        result.aliases = MappingProxyType(dict(aliases))
        result.semantics = None
        return result

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: _Namespace, owner: type[object] | None = None) -> T: ...

    def __get__(self, instance: _Namespace | None, owner: type[object] | None = None) -> Self | T:
        if instance is None:
            return self
        path = declared_path(instance)
        if path is not None:
            return cast(T, MemberRef(path, self))
        from .occurrence import read_value

        return read_value(instance, self)


class _ViewDecorator:
    def __init__(self, aliases: Mapping[str, object]) -> None:
        self.aliases = aliases

    @overload
    def __call__(self, function: Callable[..., QueryResult[T]]) -> View[T]: ...

    @overload
    def __call__(self, function: Callable[..., T | NonValue]) -> View[T]: ...

    def __call__(self, function: Callable[..., object]) -> View[Any]:
        return View.from_function(function, aliases=self.aliases)


@overload
def view(source: ValueRef[T] | View[T], /) -> View[T]: ...


@overload
def view(function: Callable[..., QueryResult[T]], /) -> View[T]: ...


@overload
def view(function: Callable[..., T | NonValue], /) -> View[T]: ...


@overload
def view(source: T, /) -> View[T]: ...


@overload
def view(**aliases: object) -> _ViewDecorator: ...


def view(function: object | None = None, /, **aliases: object) -> object:
    """A view: ``view(reference)``, the product of a value (``view(width)``,
    ``view(adapter.stages)``); anything else that is not a function is refused
    (``view-source-invalid``). Or ``@view`` on a method, or ``@view(**aliases)``.
    A function's return annotation is its value's type and check (``T``, or ``T |
    Rejected``); a reference's is its source's. A function view does not apply by
    returning ``Inapplicable()``, its guard read first."""
    _refuse_computation_when("view", aliases)
    if "semantics" in aliases:
        raise DefinitionError.broken(
            "view-semantics",
            _OWNER,
            f"view{at(source_origin())}: {SEMANTICS_GONE} (the return annotation, or the "
            "source of a view)",
        )
    if function is None:
        return _ViewDecorator(aliases)
    if isinstance(function, (ValueRef, View)):
        # A reference: a symbolic reference answers every call by refusing it, so it is
        # told apart from a function before it is called.
        if aliases:
            raise DefinitionError.broken(
                "view-source-aliases",
                _OWNER,
                f"view{at(source_origin())}: a view of a reference takes no aliases "
                f"({', '.join(sorted(aliases))}): they name the reads of a function",
            )
        return View(function)
    if not callable(function) or isinstance(function, type):
        what = (
            f"the class {function.__qualname__}"
            if isinstance(function, type)
            else f"{function!r} ({type(function).__name__})"
        )
        raise DefinitionError.broken(
            "view-source-invalid",
            _OWNER,
            f"view{at(source_origin())}: a view's source is a reference (view(width)) or a "
            f"function (@view), not {what}",
        )
    return _ViewDecorator(aliases)(function)


# -- Symbolic references ---------------------------------------------------------------


def path_text(path: Sequence[Declaration]) -> str:
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
        from ._configuration import _Namespace
        from ._nodes import NodeChoice

        if isinstance(other, (Declaration, _Namespace, NodeChoice)):
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

    # Integer +, -, *, //, % and unary - are refused by name (see ValueRef); every other
    # operator is value-like use.
    def __truediv__(self, other: object) -> NoReturn:
        self._misuse("true division")

    def __rtruediv__(self, other: object) -> NoReturn:
        self._misuse("true division")

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
    first is a node of the previous one's Space class. ``member`` is a declaration
    of the last node's Space class (or an inherited one it overrides).
    """

    def __init__(self, path: tuple[Declaration, ...], member: Declaration) -> None:
        self.path, self.member = path, member
        self.semantics = cast("ValueSemantics[T] | None", getattr(member, "semantics", None))

    def _describe(self) -> str:
        return f"{path_text(self.path)}.{self.member.name}{at(self.origin)}"

    def _key(self) -> tuple[object, ...]:
        return (*(id(item) for item in self.path), id(self.member))

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> NoReturn:
            _projection_refused(self, name)


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


def _projection_refused(source: ValueRef[Any], name: str) -> NoReturn:
    """``output.spec.payload_bits`` in a class body: an attribute of a referenced value.

    Only an attribute the value's type annotates is refused, ``reference-projection``;
    anything else is the ordinary ``AttributeError``, so introspection stays unaffected.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    semantics = source.semantics
    token = None if semantics is None else semantics.type_token
    # A derived value, or a node's derived value, not collected yet: the value type its
    # return annotation names (``T | Rejected``).
    named = source.member if isinstance(source, MemberRef) else source
    if token is None and isinstance(named, Derived) and named.function is not None:
        token = _returned_type(named.function, name)
    token = get_origin(token) or token
    if not isinstance(token, type):
        raise AttributeError(name)
    if _attribute_type(token, name) is MISSING:
        raise AttributeError(name)
    label = reference_label(source)
    raise DefinitionError.broken(
        "reference-projection",
        _OWNER,
        f"{label}: reading .{name} of a referenced value is not available in a class body; "
        "declare a derived value that returns it, and bind that member",
        declaration=label,
    )


class ChoiceMemberRef(_Symbolic, ValueRef[Any]):
    """``decision.member``: the selected candidate's member, by name.

    Resolves to the selected candidate's ``member``; it is unresolved until the
    decision is made, and inapplicable when the selected candidate lacks it.
    """

    def __init__(self, path: tuple[Declaration, ...], member: str) -> None:
        self.path, self.member = path, member
        self.semantics = None

    def _describe(self) -> str:
        return f"{path_text(self.path)}.{self.member}{at(self.origin)}"

    def _key(self) -> tuple[object, ...]:
        return (*(id(item) for item in self.path), self.member)


class CaseRef(ValueDecl[str]):
    """A selector's value, the case's name, read-only: what a binding to the selector
    reads."""

    def __init__(self, path: tuple[Declaration, ...]) -> None:
        self.path = path
        self.semantics = cast("ValueSemantics[str]", STRING)


class Members(ValueDecl[tuple[Located[T], ...]], Generic[T]):
    """Every present child node reading ``Base.member``, as ``Located`` values in
    declaration order.

    ``Members(Base.member)`` ranges over the children whose class is ``Base`` or derives
    from it, each read at its member of that name (so a subclass's override is the one
    read): ``Located(node=<child name>, member=<member name>, value=<member>)``. The
    member is a view or a derived value of ``Base``, spelled ``Base.member`` from
    outside it, or ``Members(member)`` in ``Base``'s own body after the member is
    declared. ``Base`` and the name are those of the class that declares the member,
    resolved when the class holding the ``Members`` is compiled.

    A view answers its accepted value (refused with its Space); a derived value answers
    its value. An entry that is inapplicable is left out. One refused entry refuses the
    whole, else one unresolved leaves it unresolved.
    """

    @overload
    def __init__(self, member: View[T]) -> None: ...

    @overload
    def __init__(self, member: T) -> None: ...  # a derived value: a class reads it as its value

    def __init__(self, member: object) -> None:
        if not isinstance(member, (View, Derived)):
            raise DefinitionError.broken(
                "members-member-invalid",
                _OWNER,
                f"{type(self).__name__} requires a view or a derived value of a Space "
                f"class, not {type(member).__name__}",
            )
        self.member: View[T] | ValueRef[T] = cast("View[T] | ValueRef[T]", member)
        self.semantics = cast("ValueSemantics[tuple[Located[T], ...]]", default_semantics(tuple))


__all__ = [
    "CaseRef",
    "ChoiceMemberRef",
    "Constraint",
    "Decision",
    "Declaration",
    "Derived",
    "Members",
    "MemberRef",
    "Param",
    "Required",
    "ValueDecl",
    "ValueRef",
    "View",
    "constraint",
    "derived",
    "required",
    "unfinished",
    "unmet_required",
    "view",
]
