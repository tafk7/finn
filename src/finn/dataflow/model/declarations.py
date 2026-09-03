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
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, Union, cast, overload

from typing_extensions import Self

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

if TYPE_CHECKING:
    from finn.dataflow._engine import Answer, ConstraintAssessment, ReadinessAssessment
    from finn.dataflow.model.occurrence import (
        BranchView,
        OccurrenceDiagnostic,
        ProblemSource,
        ProjectionAssessment,
    )


class AuthoringError(ValueError):
    """A declarative Space is malformed before engine validation."""


#: Class-member names an authored Space may not use for a declaration, because
#: each one is an occurrence lifecycle operation every authored class inherits.
#: The check is on the *Python member name*, never on a declaration's stable
#: compiled name: ``choice = OneOf(..., name="branch")`` keeps the engine path
#: ``<ns>.branch`` without shadowing :meth:`Space.branch`.
RESERVED_LIFECYCLE_NAMES: frozenset[str] = frozenset(
    {
        "start",
        "assign",
        "answer",
        "assess",
        "project",
        "diagnostics",
        "branch",
        "child",
        "root",
        "problem_snapshot",
        "problem_fingerprint",
        "is_stale",
        "reconstruct",
    }
)


def check_reserved_names(
    space_type: type[object],
    members: Iterable[tuple[str, object]],
) -> None:
    """Refuse a declaration that shadows an occurrence lifecycle operation.

    Hiding the lifecycle behind a ``.occurrence`` accessor would recreate the
    very wrapper the class-centered design removes, so the names are reserved
    instead -- and reserved loudly, because the alternative failure is a
    declaration silently winning and the method vanishing with an unrelated
    error message at the call site.
    """

    for member_name, declaration in members:
        if member_name in RESERVED_LIFECYCLE_NAMES:
            raise AuthoringError(
                f"{space_type.__name__}.{member_name} declares a "
                f"{type(declaration).__name__} under a reserved name; "
                f"{member_name!r} is the occurrence operation Space.{member_name}. "
                f"Rename the class member and pass name={member_name!r} to keep the "
                "compiled path unchanged"
            )


_OCCURRENCE_API: ModuleType | None = None


def _occurrence_api() -> Any:
    """The occurrence runtime, imported once and cached.

    ``occurrence`` is built on the compiler, which is built on these
    declarations, so the import cannot be stated at module scope.  Resolving it
    once into a module global -- rather than calling ``import_module`` inside
    every lifecycle method -- keeps the dispatch a single, statically
    reviewable seam instead of a dynamic lookup repeated on every query.
    """

    global _OCCURRENCE_API
    if _OCCURRENCE_API is None:
        from finn.dataflow.model import occurrence  # noqa: PLC0415 - see docstring

        _OCCURRENCE_API = occurrence
    return _OCCURRENCE_API


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


@dataclass(frozen=True, slots=True)
class OccurrenceContext:
    """What an authored class is told while one of its occurrences is allocated.

    The construction seam, and deliberately nothing more.  A class whose normal
    constructor needs context -- a future ``DataflowOp`` around a ``NodeProto``,
    say -- overrides :meth:`Space._new_occurrence` and reads this record.  What
    it does *not* contain is the Engine, the point, a ``_Ref``, or any compiled
    record: an authored constructor is given identity and its root, never the
    runtime.

    ``root`` is ``None`` for a root occurrence and the already-constructed root
    for a child view.  A child therefore reaches its root's context by asking
    the root, rather than by having the root's state copied into it.
    """

    space_type: type[Space]
    namespace: str
    scope: tuple[str, ...]
    root: Space | None

    @property
    def is_root(self) -> bool:
        return self.root is None


class Space:
    """Base class for a declarative, reusable design-space specification."""

    exports: tuple[ValueSource[object], ...] = ()
    _implicit_exports: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        check_reserved_names(
            cls,
            tuple(
                (name, value)
                for name, value in cls.__dict__.items()
                if isinstance(value, DECLARATION_TYPES)
            ),
        )

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        """Private specialization hook; generic Spaces leave the result unchanged."""

        return compiled

    @classmethod
    def _new_occurrence(cls, context: OccurrenceContext) -> Space:
        """Allocate one occurrence instance of this authored class.

        The default deliberately does *not* call ``__init__``: an ordinary
        declaration-only Space has no constructor worth running, and running one
        would mean every authored class had to accept whatever arguments the
        occurrence layer happened to pass.  That behaviour is documented here
        rather than being an accident of ``object.__new__`` at the call site.

        A subclass whose instances genuinely need context overrides this,
        allocates however it likes, and initializes from ``context``.  The
        occurrence layer then attaches its private runtime to whatever this
        returns, so an override must return an instance of ``cls``.
        """

        return object.__new__(cls)

    @classmethod
    def start(
        cls: type[S],
        problem: ProblemSource,
        *,
        namespace: str = "root",
        expected_problem_fingerprint: str | None = None,
    ) -> S:
        """Start one class-centered root occurrence over a frozen problem."""

        api = _occurrence_api()
        return cast(
            "S",
            api.start_occurrence(
                cls,
                problem,
                namespace=namespace,
                expected_problem_fingerprint=expected_problem_fingerprint,
            ),
        )

    def assign(self, declaration: Decision[T], value: T) -> Self:
        """Return the same authored occurrence class over a successor point."""

        api = _occurrence_api()
        return cast("Self", api.occurrence_assign(self, declaration, value))

    def answer(self, declaration: ValueSource[T]) -> Answer[T]:
        """Query one declaration through this occurrence's bound scope."""

        api = _occurrence_api()
        return cast("Answer[T]", api.occurrence_answer(self, declaration))

    @overload
    def assess(self, declaration: Readiness) -> ReadinessAssessment: ...

    @overload
    def assess(self, declaration: ConstraintGroup | Constraint) -> ConstraintAssessment: ...

    def assess(
        self, declaration: Readiness | ConstraintGroup | Constraint
    ) -> ReadinessAssessment | ConstraintAssessment:
        """Assess one readiness or constraint declaration in this scope."""

        api = _occurrence_api()
        return cast(
            "ReadinessAssessment | ConstraintAssessment",
            api.occurrence_assess(self, declaration),
        )

    def project(self, declaration: Projection[T]) -> ProjectionAssessment[T]:
        """Evaluate one validated projection at this occurrence's point."""

        api = _occurrence_api()
        return cast("ProjectionAssessment[T]", api.occurrence_project(self, declaration))

    def diagnostics(
        self,
        subject: object,
        *,
        projection: Projection[object] | None = None,
    ) -> tuple[OccurrenceDiagnostic, ...]:
        """Interpret findings in this root's occurrence vocabulary."""

        api = _occurrence_api()
        return cast(
            "tuple[OccurrenceDiagnostic, ...]",
            api.occurrence_diagnostics(self, subject, projection=projection),
        )

    def branch(self, declaration: OneOf) -> BranchView:
        """Return a capability-limited view of one branch in this scope."""

        api = _occurrence_api()
        return cast("BranchView", api.occurrence_branch(self, declaration))

    @overload
    def child(self, declaration: Use[S]) -> S: ...

    @overload
    def child(self, declaration: type[S]) -> S: ...

    @overload
    def child(self, declaration: Case) -> Space: ...

    def child(self, declaration: Use[S] | Case | type[S]) -> Space:
        """Return one exact direct child occurrence of this scope."""

        api = _occurrence_api()
        return cast("Space", api.occurrence_child(self, declaration))

    @property
    def root(self) -> Space:
        """The root facade at this occurrence's immutable point."""

        api = _occurrence_api()
        return cast("Space", api.occurrence_root(self))

    @property
    def problem_snapshot(self) -> Mapping[Problem[object], object]:
        """The immutable Problem values captured when the root was started."""

        api = _occurrence_api()
        return cast(
            "Mapping[Problem[object], object]",
            api.occurrence_problem_snapshot(self),
        )

    @property
    def problem_fingerprint(self) -> str:
        """Stable identity of the root's declared Problem snapshot."""

        api = _occurrence_api()
        return cast(str, api.occurrence_problem_fingerprint(self))

    def is_stale(self, problem: ProblemSource | None = None) -> bool:
        """Explicitly compare current declared Problem facts with the snapshot."""

        api = _occurrence_api()
        return cast(bool, api.occurrence_is_stale(self, problem))

    def reconstruct(
        self,
        problem: ProblemSource | None = None,
        *,
        expected_problem_fingerprint: str | None = None,
    ) -> Self:
        """Create a fresh strict lineage, retaining no assignments."""

        api = _occurrence_api()
        return cast(
            "Self",
            api.occurrence_reconstruct(
                self,
                problem,
                expected_problem_fingerprint=expected_problem_fingerprint,
            ),
        )


def resolve_declared_value(instance: object, declaration: ValueSource[object]) -> object:
    """The one dispatcher every value descriptor goes through.

    Two protocols reach the same attribute access and must not be allowed to
    decide by inheritance order.  An *attached occurrence* resolves through the
    occurrence runtime; a *legacy configured instance* -- what
    ``configure_kernel`` and ``configure_design`` still return -- resolves
    through the ``_space_value`` hook those classes already implement.  Asking
    "is this attached?" first, explicitly, is what keeps a configured Kernel's
    behaviour identical while an attached Kernel occurrence gets the runtime,
    even though both are instances of the same class.

    Neither present is an ``AttributeError``, because a bare declaration-only
    instance has no values at all.
    """

    api = _occurrence_api()
    if api.is_attached_occurrence(instance):
        return api.occurrence_value(instance, declaration)
    resolver = getattr(instance, "_space_value", None)
    if resolver is None:
        raise AttributeError("declarative values exist only on configured instances")
    return resolver(declaration)


@dataclass(frozen=True, slots=True, eq=False)
class ValueSource(Generic[T_co]):
    """Base for immutable declarations that produce a typed engine value."""

    value_semantics: ValueSemantics[object]
    stable_name: str | None = None

    @overload
    def __get__(self, instance: None, owner: type[object]) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> T_co: ...

    def __get__(self, instance: object | None, owner: type[object]) -> Self | T_co:
        if instance is None:
            return self
        return cast("T_co", resolve_declared_value(instance, cast("ValueSource[object]", self)))


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


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Projection(Generic[T_co]):
    """One output plus the readiness and constraint groups that validate it."""

    output: ValueSource[T_co]
    readiness: Readiness
    constraints: tuple[ConstraintGroup, ...]
    absence_policy: Literal["propagate"]
    snapshot_policy: Literal["declared"]
    stable_name: str | None

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        readiness: Readiness,
        constraints: Sequence[ConstraintGroup] = (),
        absence_policy: Literal["propagate"] = "propagate",
        snapshot_policy: Literal["declared"] = "declared",
        name: str | None = None,
    ) -> None:
        if not isinstance(output, ValueSource):
            raise AuthoringError("a Projection output must be a declared value")
        if not isinstance(readiness, Readiness):
            raise AuthoringError("a Projection readiness must be a Readiness declaration")
        if any(not isinstance(group, ConstraintGroup) for group in constraints):
            raise AuthoringError("Projection constraints must be ConstraintGroup declarations")
        if absence_policy != "propagate":
            raise AuthoringError("the only U1 Projection absence policy is 'propagate'")
        if snapshot_policy != "declared":
            raise AuthoringError("the only U1 Projection snapshot policy is 'declared'")
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "readiness", readiness)
        object.__setattr__(self, "constraints", tuple(constraints))
        object.__setattr__(self, "absence_policy", absence_policy)
        object.__setattr__(self, "snapshot_policy", snapshot_policy)
        object.__setattr__(self, "stable_name", name)

    @overload
    def __get__(self, instance: None, owner: type[Space]) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[Space]) -> ProjectionAssessment[T_co]: ...

    def __get__(
        self, instance: Space | None, owner: type[Space]
    ) -> Self | ProjectionAssessment[T_co]:
        if instance is None:
            return self
        return instance.project(self)


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

    def check_case(self, owner_name: str, member_name: str, case: Case) -> None:
        """A specialization's own admission rule for one case; generic branches have none."""

        del owner_name, member_name, case

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
    Projection[object],
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
    Projection,
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
        # A missing implicit export is not diagnosed here.  The specialization
        # that declared it -- ``Kernel`` and its ``region`` -- owns the rule and
        # says so in its own vocabulary; raising first would replace that with
        # an export-mechanism complaint the author cannot act on.
        if isinstance(declaration, ValueSource):
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
    "RESERVED_LIFECYCLE_NAMES",
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
    "OccurrenceContext",
    "OneOf",
    "PendingFinding",
    "Problem",
    "Projection",
    "Readiness",
    "Rejected",
    "Space",
    "Unresolvable",
    "Use",
    "ValueSource",
    "check_reserved_names",
    "constraint",
    "declared_members",
    "derived",
    "divisors_of",
    "domain",
    "enum_semantics",
    "exported_members",
    "finite",
    "reject",
    "resolve_declared_value",
    "semantics_for",
    "unresolved",
]
