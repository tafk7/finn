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
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from json import dumps
from math import isfinite
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, Union, cast, overload

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
T_contra = TypeVar("T_contra", contravariant=True)
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


#: The JSON-shaped result every canonical encoding produces.  Deliberately not
#: "any object with a repr": a fingerprint is only as trustworthy as this type.
CanonicalValue = Union[None, bool, int, str, list[object], dict[str, object]]


@dataclass(frozen=True, slots=True)
class CanonicalValueCodec(Generic[T_contra]):
    """How one declaration's values are encoded into a persistent fingerprint.

    Owned by the ``Problem`` declaration rather than by a process-global
    registry or a magic method on the value's class, because this defines
    *persisted problem identity* -- not engine equality and not evaluation
    semantics.  A QONNX ``DataType`` is a class this project does not own and
    must not monkey-patch; the declaration that admits one says how it is
    encoded, and the ``identity``/``version`` pair goes into the digest so a
    changed encoding cannot be mistaken for a changed value.
    """

    identity: str
    version: int
    encode: Callable[[T_contra], CanonicalValue]


def _type_token(kind: type[object]) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


def _structural(value: object) -> CanonicalValue:
    """The default strict encoding: built-ins, containers, enums, dataclasses.

    Everything else is refused rather than guessed at.  Folding
    ``object.__repr__``'s address into a digest produces a fingerprint that is
    worse than none, because it would be trusted: two identical problems would
    compare unequal in one process, and two different ones could compare equal
    across a reload.  The author of the declaration is the only one who can say
    what an external type's canonical form is, so they are asked.
    """

    if value is None or type(value) in (bool, int, str):
        return cast(CanonicalValue, value)
    if type(value) is float:
        if not isfinite(value):
            raise AuthoringError("a Problem fingerprint requires finite float values")
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, QualifiedPath):
        return {"qualified_path": value.value}
    if isinstance(value, Enum):
        return {"enum": _type_token(type(value)), "value": _structural(value.value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": _type_token(type(value)),
            "fields": [
                [item.name, _structural(getattr(value, item.name))] for item in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        pairs = [[_structural(key), _structural(item)] for key, item in value.items()]
        return {"mapping": sorted(pairs, key=lambda pair: dumps(pair[0], sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return {"sequence": [_structural(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        members = [_structural(item) for item in value]
        return {"set": sorted(members, key=lambda item: dumps(item, sort_keys=True))}
    raise AuthoringError(
        f"a Problem value of type {_type_token(type(value))} has no canonical encoding; "
        "give its Problem declaration a canonical=CanonicalValueCodec(...)"
    )


#: The codec a ``Problem`` uses when its author declares none.
STRUCTURAL_CODEC: CanonicalValueCodec[object] = CanonicalValueCodec(
    "dataflow.structural", 1, _structural
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
    def child(self, declaration: Case) -> Space: ...

    def child(self, declaration: Use[S] | Case) -> Space:
        """Return one exact direct child occurrence named by a use site.

        A Python class is not accepted.  One class may be placed at several
        roles, and inferring "the only one" is what breaks the day a second
        placement appears.
        """

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
    #: How this field's values are canonically encoded for the problem
    #: fingerprint.  Declared here because it defines persisted identity, which
    #: is this declaration's business and not the value class's.
    canonical: CanonicalValueCodec[Any] = STRUCTURAL_CODEC

    def __init__(
        self,
        value_type: type[T_co] | ValueSemantics[T_co],
        *,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
        name: str | None = None,
        canonical: CanonicalValueCodec[Any] | None = None,
    ) -> None:
        if canonical is not None and not isinstance(canonical, CanonicalValueCodec):
            raise AuthoringError("a Problem canonical= is one CanonicalValueCodec")
        object.__setattr__(self, "value_semantics", semantics_for(value_type))
        object.__setattr__(self, "stable_name", name)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "validate", validate)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "canonical", canonical or STRUCTURAL_CODEC)


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
    """A named readiness profile over declarations in this Space.

    ``properties`` accepts any value declaration, not only a locally declared
    ``Derived``.  A profile's job is to name the values that must be final
    before a question can be asked, and the interesting one is routinely a
    child's export or a branch's selected output -- handles on a declaration
    elsewhere rather than declarations of this class.  The compiler resolves all
    three the same way, so narrowing this would only force an author to launder
    the value through a forwarding property.
    """

    decisions: tuple[Decision[object], ...] = ()
    properties: tuple[ValueSource[object], ...] = ()
    constraints: tuple[Constraint, ...] = ()
    stable_name: str | None = None

    def __init__(
        self,
        *,
        decisions: Sequence[Decision[object]] = (),
        properties: Sequence[ValueSource[object]] = (),
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
    """One named question a Space promises to answer about its own point.

    A projection binds three things a caller would otherwise re-supply at every
    call site -- *which* value is the answer, *when* the point is final enough
    to be inspected, and *which* constraint groups must accept before that value
    may be exposed.  Keeping them in one declaration is what lets readiness,
    validity and availability stay three separate questions at the answer
    boundary instead of collapsing into one Boolean.

    ``readiness`` is required.  A projection with no further obligation declares
    an empty profile; expressing "no readiness" by omitting the concept is how
    two questions silently become one.

    ``constraints`` is a tuple because membership is many-to-many in both
    directions: one projection may own several groups, and one group may serve
    several projections when each legitimately depends on it.  A group is named,
    never inferred from what its constraints happen to read -- a physical
    feasibility constraint can depend on exactly the same folding decisions as a
    model constraint and still say something entirely different.  A bare
    ``Constraint`` is refused for the same reason: only a named group can be
    shared and pointed at in a diagnostic.

    There are no absence- or snapshot-policy arguments.  Both policies are real
    and both are recorded in the compiled metadata, but each has exactly one
    legal value in U1 -- propagate final inapplicability, and snapshot through
    the output declaration's own ``ValueSemantics`` -- so advertising them as
    constructor knobs would offer a choice that does not exist.
    """

    output: ValueSource[T_co]
    readiness: Readiness
    constraints: tuple[ConstraintGroup, ...]
    stable_name: str | None

    def __init__(
        self,
        output: ValueSource[T_co],
        *,
        readiness: Readiness,
        constraints: ConstraintGroup | Sequence[ConstraintGroup] = (),
        name: str | None = None,
    ) -> None:
        if not isinstance(output, ValueSource):
            raise AuthoringError("a Projection names one value declaration as its output")
        if not isinstance(readiness, Readiness):
            raise AuthoringError(
                "a Projection's readiness= is one Readiness declaration; declare an empty "
                "profile rather than omitting the obligation"
            )
        groups = (constraints,) if isinstance(constraints, ConstraintGroup) else tuple(constraints)
        if any(not isinstance(group, ConstraintGroup) for group in groups):
            raise AuthoringError(
                "a Projection's constraints= are ConstraintGroup declarations; a bare "
                "Constraint belongs to a group, so that the group can be named and shared"
            )
        if name is not None and not name:
            raise AuthoringError("a Projection name must be non-empty")
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "readiness", readiness)
        object.__setattr__(self, "constraints", groups)
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
    "CanonicalValue",
    "CanonicalValueCodec",
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
    "STRUCTURAL_CODEC",
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
