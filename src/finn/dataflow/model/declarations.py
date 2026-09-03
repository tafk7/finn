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
        OccurrenceDiagnostic,
        ProblemSource,
        ProjectionAssessment,
        VariantView,
    )


class AuthoringError(ValueError):
    """A declarative Space is malformed before engine validation."""


#: Class-member names an authored Space may not use for a declaration, because
#: each one is an occurrence lifecycle operation every authored class inherits.
#: The check is on the *Python member name*, never on a declaration's stable
#: compiled name: ``choice = Variant(..., name="branch")`` keeps the engine path
#: ``<ns>.branch`` while leaving the Python member free.
#:
#: Eleven names, not thirteen: navigation is descriptor-based, so there is no
#: ``Space.child`` or ``Space.branch`` for a declaration to shadow.  Any future
#: public lifecycle method spends one more member name and needs an explicit
#: authoring-compatibility review before it is added here.
RESERVED_LIFECYCLE_NAMES: frozenset[str] = frozenset(
    {
        "start",
        "assign",
        "answer",
        "assess",
        "project",
        "diagnostics",
        "root",
        "problem_snapshot",
        "problem_fingerprint",
        "is_stale",
        "reconstruct",
    }
)

#: Names that are not public API but are how the layers below talk to an
#: authored class.  A declaration bound to one of these does not merely shadow a
#: convenience: it replaces the construction hook, the specialization hook, the
#: attached-state slot, the configured-value protocol, or the export metadata,
#: and the resulting failure surfaces far from the class body that caused it.
#:
#: This prohibits a *declaration* under the name.  Overriding the method or the
#: metadata itself -- which is what ``_finalize_compilation``, ``_new_occurrence``
#: and ``exports`` exist for -- stays entirely legal.
RESERVED_PROTOCOL_NAMES: frozenset[str] = frozenset(
    {
        "_new_occurrence",
        "_finalize_compilation",
        "_occurrence_state",
        "_space_value",
        "_implicit_exports",
        "exports",
    }
)


def check_reserved_names(
    space_type: type[object],
    members: Iterable[tuple[str, object]],
) -> None:
    """Refuse a declaration that shadows a lifecycle or private protocol name.

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
        if member_name in RESERVED_PROTOCOL_NAMES:
            raise AuthoringError(
                f"{space_type.__name__}.{member_name} declares a "
                f"{type(declaration).__name__} under a reserved name; "
                f"{member_name!r} is a private Space protocol member. "
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

    def __post_init__(self) -> None:
        # Checked where the codec is written, not where a fingerprint is first
        # persisted.  An empty identity or a bumped-but-not-integer version is a
        # silent collision between two encodings, and the place it would
        # otherwise surface is a stale-state comparison months later.
        if not isinstance(self.identity, str) or not self.identity:
            raise AuthoringError("a CanonicalValueCodec identity is a non-empty stable string")
        if type(self.version) is not int or self.version < 1:
            raise AuthoringError(
                f"CanonicalValueCodec {self.identity!r} needs a positive integer version"
            )
        if not callable(self.encode):
            raise AuthoringError(f"CanonicalValueCodec {self.identity!r} needs a callable encode")


def check_canonical(value: object, what: str) -> CanonicalValue:
    """Validate and normalize one codec's output, or say which codec is wrong.

    A codec is contributor code, so its result is checked rather than trusted.
    Letting an unencodable object through would surface as a bare ``TypeError``
    from ``json.dumps`` naming neither the Problem nor the codec, and a
    non-finite float would serialize as ``NaN`` -- valid for Python's encoder,
    not valid JSON, and never equal to itself on the way back.
    """

    if value is None or type(value) in (bool, int, str):
        return cast(CanonicalValue, value)
    if type(value) is float:
        if not isfinite(value):
            raise AuthoringError(f"{what} encoded a non-finite float")
        return cast(CanonicalValue, value)
    if isinstance(value, (tuple, list)):
        return [check_canonical(item, what) for item in value]
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise AuthoringError(f"{what} encoded a mapping with a non-string key")
        return {key: check_canonical(item, what) for key, item in value.items()}
    raise AuthoringError(
        f"{what} encoded a {_type_token(type(value))}, which is not a canonical value; "
        "a codec produces None, bool, int, str, finite float, list, or str-keyed dict"
    )


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
        root_factory: Callable[[OccurrenceContext], S] | None = None,
    ) -> S:
        """Start one class-centered root occurrence over a frozen problem.

        ``root_factory`` allocates the root occurrence -- and every successor
        root of the same lineage -- for a caller holding context the design
        space has no declaration for.  It is never used for a child, so nothing
        it captures is reachable from a nested Design or Kernel.
        """

        api = _occurrence_api()
        return cast(
            "S",
            api.start_occurrence(
                cls,
                problem,
                namespace=namespace,
                expected_problem_fingerprint=expected_problem_fingerprint,
                root_factory=root_factory,
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
    def assess(self, declaration: ConstraintGroup) -> ConstraintAssessment: ...

    def assess(
        self, declaration: Readiness | ConstraintGroup
    ) -> ReadinessAssessment | ConstraintAssessment:
        """Assess one Readiness profile or one named ConstraintGroup in this scope.

        A bare ``Constraint`` is not accepted.  An individual constraint is an
        engine and compiler unit; the *named group* is the authoring unit that
        can be shared between projections, pointed at in a diagnostic, and
        renamed without every call site following.  Admitting both would widen
        the public surface to say the same thing twice.
        """

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

    There is now one protocol.  U2 and U3 retired the detached *configured*
    Kernel and Design objects that used to answer through a ``_space_value``
    hook, so an instance either carries an occurrence runtime or has no values
    at all -- and a bare declaration-only instance saying ``AttributeError`` is
    the right answer for the latter.  ``_space_value`` stays a reserved
    protocol name because a future layer may reintroduce a second protocol, and
    the failure of two protocols deciding by inheritance order is exactly what
    the reservation prevents.
    """

    api = _occurrence_api()
    if not api.is_attached_occurrence(instance):
        raise AttributeError(
            "declarative values exist only on an attached Space occurrence; "
            f"start one with {type(instance).__name__}.start(...)"
        )
    return api.occurrence_value(instance, declaration)


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


@dataclass(frozen=True, slots=True, eq=False, init=False)
class _AbsenceTolerant(ValueSource[T_co]):
    """One *dependency* marked absence-tolerant at its use site.

    Not a declaration and never a class member: it wraps a declaration where a
    ``Derived`` or ``Constraint`` names it, and the compiler lowers it to the
    same ``AbsenceMode.ALLOWS_ABSENT`` that ``Input(allow_absent=True)``
    already produces.  The distinction matters because absence tolerance is a
    property of *this reader*, not of the value: a Design's network constraint
    tolerates an inactive segment contributing no Region, while the same Region
    remains required by everything inside the segment that is active.
    """

    # Defaulted only because ``ValueSource`` defaults ``stable_name`` above it;
    # ``init=False`` means the real value always arrives through ``__init__``.
    source: ValueSource[Any] = cast("ValueSource[Any]", None)

    def __init__(self, source: ValueSource[T_co]) -> None:
        if not isinstance(source, ValueSource):
            raise AuthoringError("allow_absent() takes one value declaration")
        if isinstance(source, _AbsenceTolerant):
            raise AuthoringError("allow_absent() is already applied to this dependency")
        object.__setattr__(self, "value_semantics", source.value_semantics)
        object.__setattr__(self, "stable_name", source.stable_name)
        object.__setattr__(self, "source", source)

    def __set_name__(self, owner: type[object], name: str) -> None:
        raise AuthoringError(
            f"{owner.__name__}.{name} is an allow_absent() marker in a class body; it marks "
            "one dependency where a Derived or Constraint names it, and declares nothing"
        )


def allow_absent(source: ValueSource[T]) -> ValueSource[T]:
    """Mark one dependency of a ``Derived`` or ``Constraint`` as absence-tolerant.

    Without it a reader of a conditionally-absent value is unwritable: the
    dependency resolves to a final ``Absent`` and the reader propagates it,
    which is right for a Kernel reading its own Region and wrong for a Design
    constraint whose whole job is to say "an inactive role contributes no node".
    """

    return cast("ValueSource[T]", _AbsenceTolerant(source))


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
    """One exported value of an embedded child Space."""

    subspace: Subspace[Space]
    member_name: str

    def __init__(
        self,
        value_semantics: ValueSemantics[object],
        subspace: Subspace[Space],
        member_name: str,
    ) -> None:
        object.__setattr__(self, "value_semantics", value_semantics)
        object.__setattr__(self, "stable_name", None)
        object.__setattr__(self, "subspace", subspace)
        object.__setattr__(self, "member_name", member_name)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Subspace(Generic[S]):
    """One embedded child Space, and the descriptor that reaches its occurrence.

    A ``Subspace`` is the *only* nested-space declaration.  Used directly as a
    class member it denotes one fixed child occurrence; used as a value inside a
    :class:`Variant` it denotes one alternative, and the mapping key -- not a
    ``name=`` -- is that alternative's stable id.  There is no second
    ``Case``-shaped spelling for the second position, because the exclusivity,
    the selector and the selected outputs all belong to the container.

    Class access returns the declaration; instance access on an attached
    occurrence returns the exact bound child, so navigation is an ordinary
    attribute and never a class-keyed lookup::

        pipeline.fixed          # -> FixedImplementation occurrence
        Pipeline.fixed          # -> this Subspace declaration
    """

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
        if not isinstance(space_type, type) or not issubclass(space_type, Space):
            raise AuthoringError("Subspace requires a Space subclass")
        if name is not None and not name:
            raise AuthoringError("a Subspace name must be non-empty")
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
        return ChildValue(declaration.value_semantics, cast("Subspace[Space]", self), member_name)

    @overload
    def __get__(self, instance: None, owner: type[object]) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object]) -> S: ...

    def __get__(self, instance: Space | None, owner: type[object]) -> Self | S:
        if instance is None:
            return self
        api = _occurrence_api()
        if not api.is_attached_occurrence(instance):
            raise AttributeError(
                "a child occurrence exists only on an attached Space occurrence; "
                f"start one with {type(instance).__name__}.start(...)"
            )
        return cast("S", api.occurrence_child(instance, self))


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class BranchOutput(ValueSource[T_co]):
    """One selected output of a Variant, forwarded from the live alternative."""

    variant: Variant
    output_name: str

    def __init__(
        self,
        value_semantics: ValueSemantics[object],
        variant: Variant,
        output_name: str,
    ) -> None:
        object.__setattr__(self, "value_semantics", value_semantics)
        object.__setattr__(self, "stable_name", None)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "output_name", output_name)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Variant:
    """One structural choice: exactly one of several named alternative Subspaces.

    ``Variant`` is declaration, never policy.  It says which alternatives exist,
    lowers them beneath stable disjoint namespaces, and -- when there is more
    than one -- adds one ordinary selector ``Decision``.  It stores no search
    callback: an external specialization algorithm discovers the selector
    through the compiled branch catalog and commits it like any other decision.

    The name states what the parent owns.  A parent does not own "a one-of"; it
    owns a structural variation point, selected from alternatives it names::

        implementation = Variant(
            {
                "fast": Subspace(FastImplementation, size=size),
                "small": Subspace(SmallImplementation, size=size),
            },
            outputs=("result",),
        )

    The mapping key is the alternative's stable id, so nothing repeats it.  A
    per-alternative ``when=`` is refused: the Variant owns the outer condition
    and the selection, and candidate-specific applicability is a separate
    question that has not yet been forced.  Instance access returns the bound
    :class:`VariantView`, never the declaration.

    **Ordering.**  The mapping is consumed exactly once, at construction, in its
    own iteration order, and frozen into ``alternatives``.  That order is then
    the order of everything downstream: the selector's finite domain, the
    compiled case namespaces, ``BranchInfo.cases``, and
    ``VariantView.alternatives``.  An ordinary ``dict`` literal therefore says
    what it looks like it says, and a caller who wants a different order writes
    a different literal or passes an ``OrderedDict``.  Nothing here introduces a
    second collection type to express that: the mapping a Python author already
    has is enough, and reading it once means a later mutation of the caller's
    dict cannot change a compiled space.
    """

    alternatives: tuple[tuple[str, Subspace[Space]], ...]
    outputs: tuple[str, ...]
    when: ValueSource[bool] | None
    stable_name: str | None

    #: Local name of the generated selector; specializations may rename it.
    selector_name: ClassVar[str] = "case"

    def __init__(
        self,
        alternatives: Mapping[str, Subspace[Space]],
        *,
        outputs: Sequence[str] = (),
        when: ValueSource[bool] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(alternatives, Mapping):
            raise AuthoringError("a Variant takes an ordered mapping of alternative id to Subspace")
        ordered: list[tuple[str, Subspace[Space]]] = []
        for alternative_id, subspace in alternatives.items():
            if not isinstance(alternative_id, str) or not alternative_id:
                raise AuthoringError("a Variant alternative id is a non-empty string")
            if not isinstance(subspace, Subspace):
                raise AuthoringError(
                    f"Variant alternative {alternative_id!r} is a "
                    f"{type(subspace).__name__}, not a Subspace"
                )
            if subspace.stable_name is not None:
                raise AuthoringError(
                    f"Variant alternative {alternative_id!r} also carries name="
                    f"{subspace.stable_name!r}; the mapping key is the alternative id"
                )
            ordered.append((alternative_id, subspace))
        self._initialize(tuple(ordered), outputs, when, name)

    def _initialize(
        self,
        alternatives: Sequence[tuple[str, Subspace[Space]]],
        outputs: Sequence[str],
        when: ValueSource[bool] | None,
        name: str | None,
    ) -> None:
        if not alternatives:
            raise AuthoringError("a Variant needs at least one alternative Subspace")
        for alternative_id, subspace in alternatives:
            if subspace.when is not None:
                raise AuthoringError(
                    f"Variant alternative {alternative_id!r} declares when=; a Variant owns "
                    "the outer condition and the selection between its alternatives"
                )
        ordered = tuple(outputs)
        if len(set(ordered)) != len(ordered):
            raise AuthoringError("a Variant names one selected output twice")
        object.__setattr__(self, "alternatives", tuple(alternatives))
        object.__setattr__(self, "outputs", ordered)
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "stable_name", name)

    def check_alternative(
        self, owner_name: str, member_name: str, subspace: Subspace[Space]
    ) -> None:
        """A specialization's own admission rule; a generic Variant has none."""

        del owner_name, member_name, subspace

    def __getattr__(self, member_name: str) -> BranchOutput[object]:
        if member_name.startswith("_"):
            raise AttributeError(member_name)
        if member_name not in self.outputs:
            raise AttributeError(f"this Variant does not select an output named {member_name!r}")
        semantics = self._output_semantics(member_name)
        return BranchOutput(semantics, self, member_name)

    @overload
    def __get__(self, instance: None, owner: type[object]) -> Self: ...

    @overload
    def __get__(self, instance: Space, owner: type[object]) -> VariantView: ...

    def __get__(self, instance: Space | None, owner: type[object]) -> Self | VariantView:
        if instance is None:
            return self
        api = _occurrence_api()
        if not api.is_attached_occurrence(instance):
            raise AttributeError(
                "a Variant view exists only on an attached Space occurrence; "
                f"start one with {type(instance).__name__}.start(...)"
            )
        return cast("VariantView", api.occurrence_variant(instance, self))

    def _output_semantics(self, output_name: str) -> ValueSemantics[object]:
        first: ValueSemantics[object] | None = None
        for _alternative_id, subspace in self.alternatives:
            exported = exported_members(subspace.space_type)
            declaration = exported.get(output_name)
            if declaration is None:
                raise AuthoringError(
                    f"{subspace.space_type.__name__} does not export {output_name!r}"
                )
            if first is None:
                first = declaration.value_semantics
            elif not first.is_compatible_with(declaration.value_semantics):
                raise AuthoringError(
                    f"Variant output {output_name!r} changes value semantics from "
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
    Subspace[Space],
    Variant,
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
    Subspace,
    Variant,
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
    "RESERVED_PROTOCOL_NAMES",
    "AuthoringError",
    "BranchOutput",
    "CanonicalValue",
    "CanonicalValueCodec",
    "ChildValue",
    "Constraint",
    "ConstraintGroup",
    "Decision",
    "Derived",
    "Domain",
    "Input",
    "OccurrenceContext",
    "PendingFinding",
    "Problem",
    "Projection",
    "Readiness",
    "Rejected",
    "STRUCTURAL_CODEC",
    "Space",
    "Subspace",
    "Unresolvable",
    "ValueSource",
    "Variant",
    "allow_absent",
    "check_canonical",
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
