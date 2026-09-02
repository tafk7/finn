# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable class declarations for the dataflow design-space frontend.

These values are source-language templates.  They carry relative references
between class members and are lowered by :mod:`finn.dataflow.model.compiler`
into the ordinary immutable records consumed by ``finn.dataflow._engine``.
They never acquire a namespace or a bound engine handle in place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Generic, TypeVar, Union, cast

from finn.dataflow._engine import (
    AbsenceMode,
    FindingKind,
    QualifiedPath,
    ValueSemantics,
    as_object_semantics,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")
E = TypeVar("E", bound=Enum)


class AuthoringError(ValueError):
    """A declarative Space is malformed before engine validation."""


def enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    """Identity semantics for one Enum type."""

    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


def semantics_for(value_type: type[T] | ValueSemantics[T]) -> ValueSemantics[object]:
    """Return explicit semantics or nominal semantics for a Python type."""

    if isinstance(value_type, ValueSemantics):
        return as_object_semantics(value_type)
    if issubclass(value_type, Enum):
        return enum_semantics(value_type)
    return as_object_semantics(ValueSemantics.immutable_nominal(value_type))


@dataclass(frozen=True, slots=True)
class PendingFinding:
    """A finding whose owning declaration path is assigned during lowering."""

    kind: FindingKind
    code: str
    message: str
    values: tuple[tuple[str, object], ...] = ()
    trace: tuple[ValueSource[object] | QualifiedPath, ...] = ()


@dataclass(frozen=True, slots=True)
class Rejected:
    """A constraint or property explicitly does not apply at this point."""

    finding: PendingFinding


@dataclass(frozen=True, slots=True)
class Unresolvable:
    """A declaration cannot resolve because required information is missing."""

    finding: PendingFinding


def reject(
    code: str,
    message: str,
    *,
    values: Mapping[str, object] | None = None,
    trace: Sequence[ValueSource[object] | QualifiedPath] = (),
) -> Rejected:
    return Rejected(
        PendingFinding(
            FindingKind.REJECTION,
            code,
            message,
            tuple((name, value) for name, value in (values or {}).items()),
            tuple(trace),
        )
    )


def unresolved(
    code: str,
    message: str,
    *,
    values: Mapping[str, object] | None = None,
    trace: Sequence[ValueSource[object] | QualifiedPath] = (),
) -> Unresolvable:
    return Unresolvable(
        PendingFinding(
            FindingKind.LIMITATION,
            code,
            message,
            tuple((name, value) for name, value in (values or {}).items()),
            tuple(trace),
        )
    )


class Space:
    """Base class for a declarative, reusable design-space specification."""

    exports: tuple[ValueSource[object], ...] = ()
    _implicit_exports: tuple[str, ...] = ()

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        """Private specialization hook; generic Spaces leave the result unchanged."""

        return compiled


@dataclass(frozen=True, slots=True, eq=False)
class ValueSource(Generic[T_co]):
    """Base for immutable declarations that produce a typed engine value."""

    value_semantics: ValueSemantics[object]
    stable_name: str | None = None

    def __get__(self, instance: object | None, owner: type[object]) -> object:
        if instance is None:
            return self
        resolver = getattr(instance, "_space_value", None)
        if resolver is None:
            raise AttributeError("declarative values exist only on configured instances")
        return resolver(self)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Problem(ValueSource[T_co]):
    """A root-owned problem field."""

    required: bool = True
    validate: Callable[[object], bool] | None = None
    description: str = ""

    def __init__(
        self,
        value_type: type[T_co] | ValueSemantics[T_co],
        *,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
        name: str | None = None,
    ) -> None:
        object.__setattr__(self, "value_semantics", semantics_for(value_type))
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "validate", validate)
        object.__setattr__(self, "description", description)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Input(ValueSource[T_co]):
    """A typed hole bound to a value declaration owned by an enclosing Space."""

    absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE

    def __init__(
        self,
        value_type: type[T_co] | ValueSemantics[T_co],
        *,
        allow_absent: bool = False,
        name: str | None = None,
    ) -> None:
        object.__setattr__(self, "value_semantics", semantics_for(value_type))
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(
            self,
            "absence",
            AbsenceMode.ALLOWS_ABSENT if allow_absent else AbsenceMode.REQUIRES_APPLICABLE,
        )


@dataclass(frozen=True, slots=True)
class Domain:
    """A relative decision domain over other declaration templates."""

    dependencies: tuple[tuple[str, ValueSource[object]], ...]
    accepts: Callable[..., object]
    candidates: Callable[..., object] | None = None


def finite(values: Iterable[object]) -> Domain:
    """A deterministic finite domain."""

    ordered = tuple(values)
    allowed = frozenset(ordered)

    def accepts(*, candidate: object) -> bool:
        return candidate in allowed

    def candidates() -> tuple[object, ...]:
        return ordered

    return Domain((), accepts, candidates)


def domain(
    *,
    accepts: Callable[..., object],
    candidates: Callable[..., object] | None = None,
    **dependencies: ValueSource[object],
) -> Domain:
    """A dependency-driven decision domain."""

    return Domain(tuple(dependencies.items()), accepts, candidates)


def divisors_of(extent: ValueSource[int]) -> Domain:
    """Every positive divisor of one declared extent."""

    def accepts(*, candidate: object, extent: int) -> bool:
        return type(candidate) is int and candidate > 0 and extent % candidate == 0

    def candidates(*, extent: int) -> tuple[object, ...]:
        return tuple(value for value in range(1, extent + 1) if extent % value == 0)

    return domain(
        extent=cast("ValueSource[object]", extent),
        accepts=accepts,
        candidates=candidates,
    )


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class Decision(ValueSource[T_co]):
    """A value selected within this Space."""

    domain: Domain

    def __init__(
        self,
        value_type: type[T_co] | ValueSemantics[T_co],
        *,
        domain: Domain | None = None,
        values: Iterable[object] | None = None,
        name: str | None = None,
    ) -> None:
        if (domain is None) == (values is None):
            raise AuthoringError("a Decision needs exactly one of domain= or values=")
        object.__setattr__(self, "value_semantics", semantics_for(value_type))
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(self, "domain", domain if domain is not None else finite(values or ()))


@dataclass(frozen=True, slots=True, eq=False)
class Derived(ValueSource[T_co]):
    """A relative derived-property declaration."""

    dependencies: tuple[tuple[str, ValueSource[object]], ...] = ()
    evaluate: Callable[..., object] = field(default=lambda: None, compare=False, repr=False)


def derived(
    value_type: type[T] | ValueSemantics[T],
    /,
    *,
    name: str | None = None,
    **dependencies: ValueSource[object],
) -> Callable[[Callable[..., object]], Derived[T]]:
    """Declare a derived property from explicit class-member dependencies."""

    semantics = semantics_for(value_type)

    def decorate(evaluate: Callable[..., object]) -> Derived[T]:
        return Derived(semantics, name, tuple(dependencies.items()), evaluate)

    return decorate


@dataclass(frozen=True, slots=True, eq=False)
class Constraint:
    """A relative constraint declaration."""

    dependencies: tuple[tuple[str, ValueSource[object]], ...]
    evaluate: Callable[..., object] = field(compare=False, repr=False)
    stable_name: str | None = None


def constraint(
    *,
    name: str | None = None,
    **dependencies: ValueSource[object],
) -> Callable[[Callable[..., object]], Constraint]:
    """Declare a constraint from explicit class-member dependencies."""

    def decorate(evaluate: Callable[..., object]) -> Constraint:
        return Constraint(tuple(dependencies.items()), evaluate, name)

    return decorate


@dataclass(frozen=True, slots=True, eq=False)
class ConstraintGroup:
    """A named collection of Constraint members."""

    constraints: tuple[Constraint, ...]
    stable_name: str | None = None

    def __init__(self, *constraints: Constraint, name: str | None = None) -> None:
        object.__setattr__(self, "constraints", tuple(constraints))
        object.__setattr__(self, "stable_name", name)


@dataclass(frozen=True, slots=True, eq=False)
class Readiness:
    """A named readiness profile over declarations in this Space."""

    decisions: tuple[Decision[object], ...] = ()
    properties: tuple[Derived[object], ...] = ()
    constraints: tuple[Constraint, ...] = ()
    stable_name: str | None = None

    def __init__(
        self,
        *,
        decisions: Sequence[Decision[object]] = (),
        properties: Sequence[Derived[object]] = (),
        constraints: Sequence[Constraint] | ConstraintGroup = (),
        name: str | None = None,
    ) -> None:
        grouped = (
            constraints.constraints if isinstance(constraints, ConstraintGroup) else constraints
        )
        object.__setattr__(self, "decisions", tuple(decisions))
        object.__setattr__(self, "properties", tuple(properties))
        object.__setattr__(self, "constraints", tuple(grouped))
        object.__setattr__(self, "stable_name", name)


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class ChildValue(ValueSource[T_co]):
    """One exported value of a recursively used child Space."""

    use: Use[Space]
    member_name: str

    def __init__(
        self,
        value_semantics: ValueSemantics[object],
        use: Use[Space],
        member_name: str,
    ) -> None:
        object.__setattr__(self, "value_semantics", value_semantics)
        object.__setattr__(self, "stable_name", None)
        object.__setattr__(self, "use", use)
        object.__setattr__(self, "member_name", member_name)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Use(Generic[S]):
    """Compile-time placement of one reusable child Space."""

    space_type: type[S]
    bindings: tuple[tuple[str, ValueSource[object]], ...]
    when: ValueSource[bool] | None
    stable_name: str | None

    def __init__(
        self,
        space_type: type[S],
        /,
        *,
        when: ValueSource[bool] | None = None,
        name: str | None = None,
        **bindings: ValueSource[object],
    ) -> None:
        if not issubclass(space_type, Space):
            raise AuthoringError("Use requires a Space subclass")
        object.__setattr__(self, "space_type", space_type)
        object.__setattr__(self, "bindings", tuple(bindings.items()))
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "stable_name", name)

    def __getattr__(self, member_name: str) -> ChildValue[object]:
        exported = exported_members(self.space_type)
        try:
            declaration = exported[member_name]
        except KeyError:
            raise AttributeError(
                f"{self.space_type.__name__} does not export {member_name!r}"
            ) from None
        return ChildValue(declaration.value_semantics, cast("Use[Space]", self), member_name)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Case:
    """One named alternative child Space inside an exclusive branch.

    A ``Case`` is a declaration record, not a runtime level.  It owns the exact
    Input bindings of its own child Space, so alternatives with unrelated Input
    vocabularies stay legible without a parallel case-to-binding map.

    Deliberately not generic in its Space type: a branch's whole point is to
    hold unrelated classes side by side, and an invariant ``Case[S]`` makes
    ``OneOf(Case(A, ...), Case(B, ...))`` uninferrable at every call site.  A
    specialization such as ``Kernels`` states its own requirement as a compile
    check with a message, which is what an author needs anyway.
    """

    space_type: type[Space]
    bindings: tuple[tuple[str, ValueSource[object]], ...]
    stable_name: str | None

    def __init__(
        self,
        space_type: type[Space],
        /,
        *,
        name: str | None = None,
        **bindings: ValueSource[object],
    ) -> None:
        if not isinstance(space_type, type) or not issubclass(space_type, Space):
            raise AuthoringError("Case requires a Space subclass")
        if name is not None and not name:
            raise AuthoringError("a Case name must be non-empty")
        object.__setattr__(self, "space_type", space_type)
        object.__setattr__(self, "bindings", tuple(bindings.items()))
        object.__setattr__(self, "stable_name", name)


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class BranchOutput(ValueSource[T_co]):
    """One selected output of an exclusive branch, forwarded from the live case."""

    branch: OneOf
    output_name: str

    def __init__(
        self,
        value_semantics: ValueSemantics[object],
        branch: OneOf,
        output_name: str,
    ) -> None:
        object.__setattr__(self, "value_semantics", value_semantics)
        object.__setattr__(self, "stable_name", None)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "output_name", output_name)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OneOf:
    """Compile-time placement of exactly one of several child Spaces.

    ``OneOf`` is declaration, never policy.  It says which cases exist, lowers
    them beneath stable disjoint namespaces, and -- when there is more than one
    -- adds one ordinary selector ``Decision``.  It stores no search callback:
    an external specialization algorithm discovers the selector through the
    compiled branch catalog and commits it like any other decision.
    """

    cases: tuple[Case, ...]
    outputs: tuple[str, ...]
    when: ValueSource[bool] | None
    stable_name: str | None

    #: Local name of the generated selector; specializations may rename it.
    selector_name: ClassVar[str] = "case"

    def __init__(
        self,
        *cases: Case,
        outputs: Sequence[str] = (),
        when: ValueSource[bool] | None = None,
        name: str | None = None,
    ) -> None:
        self._initialize(cases, outputs, when, name)

    def _initialize(
        self,
        cases: Sequence[Case],
        outputs: Sequence[str],
        when: ValueSource[bool] | None,
        name: str | None,
    ) -> None:
        if not cases:
            raise AuthoringError("a branch needs at least one Case")
        if any(not isinstance(case, Case) for case in cases):
            raise AuthoringError("a branch takes Case declarations as positional arguments")
        ordered = tuple(outputs)
        if len(set(ordered)) != len(ordered):
            raise AuthoringError("a branch names one selected output twice")
        object.__setattr__(self, "cases", tuple(cases))
        object.__setattr__(self, "outputs", ordered)
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "stable_name", name)

    def case_id(self, case: Case) -> str | None:
        """The stable case id, or ``None`` when the author must supply one."""

        return case.stable_name

    def __getattr__(self, member_name: str) -> BranchOutput[object]:
        if member_name.startswith("_"):
            raise AttributeError(member_name)
        if member_name not in self.outputs:
            raise AttributeError(f"this branch does not select an output named {member_name!r}")
        semantics = self._output_semantics(member_name)
        return BranchOutput(semantics, self, member_name)

    def _output_semantics(self, output_name: str) -> ValueSemantics[object]:
        first: ValueSemantics[object] | None = None
        for case in self.cases:
            exported = exported_members(case.space_type)
            declaration = exported.get(output_name)
            if declaration is None:
                raise AuthoringError(f"{case.space_type.__name__} does not export {output_name!r}")
            if first is None:
                first = declaration.value_semantics
            elif not first.is_compatible_with(declaration.value_semantics):
                raise AuthoringError(
                    f"branch output {output_name!r} changes value semantics from "
                    f"{first.name} to {declaration.value_semantics.name}"
                )
        assert first is not None
        return first


Declaration = Union[
    Problem[object],
    Input[object],
    Decision[object],
    Derived[object],
    Constraint,
    ConstraintGroup,
    Readiness,
    Use[Space],
    OneOf,
]

#: Every class-body value the declarative compiler recognizes as a declaration.
DECLARATION_TYPES: tuple[type, ...] = (
    Problem,
    Input,
    Decision,
    Derived,
    Constraint,
    ConstraintGroup,
    Readiness,
    Use,
    OneOf,
)


def declared_members(space_type: type[Space]) -> tuple[tuple[str, Declaration], ...]:
    """Collect effective declarations in deterministic inherited order."""

    ordered: dict[str, Declaration] = {}
    declaration_types = DECLARATION_TYPES
    for base in reversed(space_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if not isinstance(value, declaration_types):
                if name in ordered:
                    raise AuthoringError(
                        f"{base.__name__}.{name} replaces a declaration with {type(value).__name__}"
                    )
                continue
            if name in ordered and type(value) is not type(ordered[name]):
                raise AuthoringError(
                    f"{base.__name__}.{name} changes declaration category from "
                    f"{type(ordered[name]).__name__} to {type(value).__name__}"
                )
            if name in ordered and isinstance(value, ValueSource):
                previous = cast(ValueSource[object], ordered[name])
                if not value.value_semantics.is_compatible_with(previous.value_semantics):
                    raise AuthoringError(
                        f"{base.__name__}.{name} changes value semantics from "
                        f"{previous.value_semantics.name} to {value.value_semantics.name}"
                    )
            ordered[name] = cast(Declaration, value)
    return tuple(ordered.items())


def exported_members(space_type: type[Space]) -> Mapping[str, ValueSource[object]]:
    """Resolve the class's declared export objects to effective member names."""

    members = dict(declared_members(space_type))
    by_identity: dict[int, str] = {}
    for base in reversed(space_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, ValueSource):
                by_identity[id(value)] = name
    exported: dict[str, ValueSource[object]] = {}
    for name in space_type._implicit_exports:
        declaration = members.get(name)
        if not isinstance(declaration, ValueSource):
            raise AuthoringError(
                f"{space_type.__name__} implicitly exports {name!r}, which is not a value member"
            )
        exported[name] = declaration
    for value in space_type.exports:
        export_name = by_identity.get(id(value))
        if export_name is None:
            raise AuthoringError(
                f"{space_type.__name__} exports a value that is not an effective class member"
            )
        member = members.get(export_name)
        # A ``ChildValue`` or ``BranchOutput`` is a derived handle on another
        # declaration rather than a declaration of its own, so it never appears
        # in ``declared_members``.  Re-exporting one is still legal: the
        # compiler resolves it through the child or branch that owns it.
        exported[export_name] = member if isinstance(member, ValueSource) else value
    return exported


__all__ = [
    "DECLARATION_TYPES",
    "AuthoringError",
    "BranchOutput",
    "Case",
    "ChildValue",
    "Constraint",
    "ConstraintGroup",
    "Decision",
    "Derived",
    "Domain",
    "Input",
    "OneOf",
    "PendingFinding",
    "Problem",
    "Readiness",
    "Rejected",
    "Space",
    "Unresolvable",
    "Use",
    "ValueSource",
    "constraint",
    "declared_members",
    "derived",
    "divisors_of",
    "domain",
    "enum_semantics",
    "exported_members",
    "finite",
    "reject",
    "semantics_for",
    "unresolved",
]
