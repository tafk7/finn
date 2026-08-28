# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped engine authoring: declaration handles over ordinary engine objects.

Every method here constructs a real ``ProblemField``, ``Decision``,
``DerivedProperty``, ``Constraint``, ``ConstraintSet``, or ``ReadinessProfile``
and returns a handle that retains it.  There is no parallel declaration model
and no separate compilation step: ``Engine.validate()`` remains the only
compiler from specification to accepted design space.

A handle carries its own path, dependency kind, value semantics, and absence
mode, so a declared thing is stated once and referenced by identity thereafter.
Dependencies are always an explicit mapping from evaluator parameter name to
handle -- never a name search across namespaces, never attribute tracing.

A ``Scope`` owns one path namespace, so the same authoring function may be
placed more than once without path collisions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from inspect import signature
from typing import Generic, TypeVar, cast

from finn.dataflow.design import (
    AbsenceMode,
    Absent,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
E = TypeVar("E", bound=Enum)

#: Reserved semantic-property and constraint path prefixes, matching the
#: existing FINN convention so scoped authoring produces the same paths raw
#: authoring does.
SEMANTIC_PREFIX = "semantic"
CONSTRAINT_PREFIX = "constraint"


# -- value semantics ---------------------------------------------------------


def enum_semantics(enum_type: type[E]) -> ValueSemantics[object]:
    """Identity-compared semantics for one enumeration type."""

    semantics: ValueSemantics[E] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


def semantics_for(value_type: type[T] | ValueSemantics[T]) -> ValueSemantics[object]:
    """Derive value semantics from an ordinary Python type.

    An explicit ``ValueSemantics`` passes through unchanged, so pre-built
    semantics such as ``DATAFLOW_REGION_SEMANTICS`` remain usable wherever a
    type is accepted.
    """

    if isinstance(value_type, ValueSemantics):
        return as_object_semantics(value_type)
    if issubclass(value_type, Enum):
        return enum_semantics(value_type)
    return as_object_semantics(ValueSemantics.immutable_nominal(value_type))


# -- deferred findings -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PendingFinding:
    """A finding an evaluator raises without knowing its own path.

    The adapter completes it with the owning declaration's path, which removes
    the ``owner``-threading evaluator factories that path knowledge otherwise
    forces on every author.
    """

    kind: FindingKind
    code: str
    message: str
    values: tuple[tuple[str, object], ...] = ()
    trace: tuple[QualifiedPath, ...] = ()

    def at(self, path: QualifiedPath) -> Finding:
        return Finding(self.kind, self.code, path, self.message, self.values, self.trace)


@dataclass(frozen=True, slots=True)
class Rejected:
    """Not applicable here: adapted to ``Absent`` at the owning declaration."""

    finding: PendingFinding


@dataclass(frozen=True, slots=True)
class Unresolvable:
    """Missing information: adapted to ``Unresolved`` at the owning declaration."""

    finding: PendingFinding


def _pending(
    kind: FindingKind,
    code: str,
    message: str,
    values: Mapping[str, object] | None,
    trace: Sequence[Ref[object] | QualifiedPath],
) -> PendingFinding:
    return PendingFinding(
        kind,
        code,
        message,
        tuple((key, value) for key, value in (values or {}).items()),
        tuple(item.path if isinstance(item, Ref) else item for item in trace),
    )


def reject(
    code: str,
    message: str,
    *,
    values: Mapping[str, object] | None = None,
    trace: Sequence[Ref[object] | QualifiedPath] = (),
) -> Rejected:
    """Report that this declaration does not apply to the current point."""

    return Rejected(_pending(FindingKind.REJECTION, code, message, values, trace))


def unresolved(
    code: str,
    message: str,
    *,
    values: Mapping[str, object] | None = None,
    trace: Sequence[Ref[object] | QualifiedPath] = (),
) -> Unresolvable:
    """Report that required information is missing from the problem."""

    return Unresolvable(_pending(FindingKind.LIMITATION, code, message, values, trace))


# -- declaration handles -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ref(Generic[T_co]):
    """A declared problem field, decision, or derived property.

    The handle is the single source of the path, dependency kind, and value
    semantics wherever the declaration is referenced, so none of the three is
    restated at a use site.
    """

    path: QualifiedPath
    kind: DependencyKind
    semantics: ValueSemantics[object]
    absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE

    def allow_absent(self) -> Ref[T_co]:
        """Read this reference as optional at one use site.

        Requiredness of a problem field and absence handling at a dependency
        are distinct: one evaluator may accept an absent optional field while
        another requires the same field whenever it applies.
        """

        return replace(self, absence=AbsenceMode.ALLOWS_ABSENT)

    def required(self) -> Ref[T_co]:
        """Read this reference as required at one use site."""

        return replace(self, absence=AbsenceMode.REQUIRES_APPLICABLE)

    def dependency(self, name: str) -> DependencyRef:
        return DependencyRef(name, self.path, self.kind, self.semantics, self.absence)


@dataclass(frozen=True, slots=True)
class ConstraintRef:
    """A declared constraint.  Constraints are asked about, never read."""

    path: QualifiedPath


Dependencies = Mapping[str, "Ref[object]"]


class AuthoringError(ValueError):
    """One scoped-authoring mistake, reported at declaration time."""


# -- evaluator adaptation ----------------------------------------------------


def _check_signature(dependencies: Dependencies, evaluate: Callable[..., object]) -> None:
    parameters = signature(evaluate).parameters
    if any(
        parameter.kind
        in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD, parameter.POSITIONAL_ONLY)
        for parameter in parameters.values()
    ):
        raise AuthoringError(
            f"{getattr(evaluate, '__qualname__', evaluate)} must take only named parameters"
        )
    declared, accepted = set(dependencies), set(parameters)
    if declared != accepted:
        missing = sorted(declared - accepted)
        extra = sorted(accepted - declared)
        raise AuthoringError(
            f"{getattr(evaluate, '__qualname__', evaluate)} signature does not match its "
            f"dependency mapping; unused dependencies {missing}, unbound parameters {extra}"
        )


def _adapt(result: object, owner: QualifiedPath) -> Answer[object]:
    if isinstance(result, (Decided, Absent, Unresolved)):
        return cast(Answer[object], result)
    if isinstance(result, Rejected):
        return Absent((result.finding.at(owner),))
    if isinstance(result, Unresolvable):
        return Unresolved((result.finding.at(owner),))
    return Decided(result)


def evaluator(
    owner: QualifiedPath,
    dependencies: Dependencies,
    evaluate: Callable[..., object],
) -> EvaluatorSpec[Answer[object]]:
    """Adapt a typed callable and an explicit dependency mapping.

    The callable receives its dependencies as keyword arguments under the
    mapping's names.  An ordinary returned value becomes ``Decided``; an
    explicit ``Answer`` passes through; ``reject``/``unresolved`` are completed
    with ``owner``.
    """

    _check_signature(dependencies, evaluate)
    refs = tuple(ref.dependency(name) for name, ref in sorted(dependencies.items()))
    names = tuple(sorted(dependencies))

    def call(view: DependencyView) -> Answer[object]:
        return _adapt(evaluate(**{name: view[name] for name in names}), owner)

    return EvaluatorSpec(refs, call)


def predicate(
    owner: QualifiedPath,
    dependencies: Dependencies,
    evaluate: Callable[..., object],
) -> EvaluatorSpec[Answer[bool]]:
    """Build a reusable applicability gate over the same adapter."""

    return cast(EvaluatorSpec[Answer[bool]], evaluator(owner, dependencies, evaluate))


# -- decision domains --------------------------------------------------------


DomainFactory = Callable[[QualifiedPath], DecisionDomain]


def finite(values: Iterable[object]) -> DomainFactory:
    """A fixed candidate set, independent of the problem."""

    ordered = tuple(values)
    allowed = frozenset(ordered)

    def build(_owner: QualifiedPath) -> DecisionDomain:
        def accepts(value: object, _view: DependencyView) -> Answer[bool]:
            return Decided(value in allowed)

        def candidates(_view: DependencyView) -> Answer[tuple[object, ...]]:
            return Decided(ordered)

        return DecisionDomain((), accepts, EvaluatorSpec((), candidates))

    return build


def divisors_of(extent: Ref[int]) -> DomainFactory:
    """Every positive divisor of one declared extent."""

    dependency = extent.dependency("extent")

    def build(_owner: QualifiedPath) -> DecisionDomain:
        def accepts(value: object, view: DependencyView) -> Answer[bool]:
            limit = cast(int, view["extent"])
            return Decided(type(value) is int and value > 0 and limit % value == 0)

        def candidates(view: DependencyView) -> Answer[tuple[object, ...]]:
            limit = cast(int, view["extent"])
            return Decided(tuple(value for value in range(1, limit + 1) if limit % value == 0))

        return DecisionDomain((dependency,), accepts, EvaluatorSpec((dependency,), candidates))

    return build


def domain(
    dependencies: Dependencies,
    *,
    accepts: Callable[..., object],
    candidates: Callable[..., object] | None = None,
) -> DomainFactory:
    """An arbitrary domain over an explicit dependency mapping.

    ``accepts`` takes a ``candidate`` parameter alongside its dependencies and
    may return a bool, an ``Answer``, or ``reject(...)``.  ``candidates`` takes
    the dependencies alone and returns a tuple of values.
    """

    refs = tuple(ref.dependency(name) for name, ref in sorted(dependencies.items()))
    names = tuple(sorted(dependencies))

    def build(owner: QualifiedPath) -> DecisionDomain:
        _check_signature({**dependencies, "candidate": _CANDIDATE}, accepts)

        def accept(value: object, view: DependencyView) -> Answer[bool]:
            arguments = {name: view[name] for name in names}
            return cast(Answer[bool], _adapt(accepts(candidate=value, **arguments), owner))

        enumerate_spec: EvaluatorSpec[Answer[tuple[object, ...]]] | None = None
        if candidates is not None:
            enumerate_spec = cast(
                EvaluatorSpec[Answer[tuple[object, ...]]],
                evaluator(owner, dependencies, candidates),
            )
        return DecisionDomain(refs, accept, enumerate_spec)

    return build


#: Placeholder handle so ``domain`` can validate the reserved ``candidate``
#: parameter with the same signature check every other adapter uses.
_CANDIDATE: Ref[object] = Ref(
    QualifiedPath("candidate"), DependencyKind.PROBLEM, semantics_for(object)
)


# -- the scope ---------------------------------------------------------------


class Scope:
    """One authoring namespace producing ordinary engine declarations.

    Placing the same authoring function under two namespaces produces two
    non-colliding declaration sets, which is what makes a reusable scope
    reusable.
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._problem: list[ProblemField] = []
        self._decisions: list[Decision] = []
        self._properties: list[DerivedProperty] = []
        self._constraints: list[Constraint] = []
        self._sets: dict[str, list[QualifiedPath]] = {}
        self._profiles: list[ReadinessProfile] = []

    # -- path allocation ---------------------------------------------------

    def local_path(self, name: str) -> QualifiedPath:
        return QualifiedPath(f"{self.namespace}.{name}")

    def semantic_path(self, name: str) -> QualifiedPath:
        return QualifiedPath(f"{SEMANTIC_PREFIX}.{self.namespace}.{name}")

    def constraint_path(self, name: str) -> QualifiedPath:
        return QualifiedPath(f"{CONSTRAINT_PREFIX}.{self.namespace}.{name}")

    # -- declarations ------------------------------------------------------

    def problem_field(
        self,
        path: QualifiedPath | str,
        value_type: type[T] | ValueSemantics[T],
        *,
        required: bool = True,
        validate: Callable[[object], bool] | None = None,
        description: str = "",
    ) -> Ref[T]:
        """Declare one problem field.

        Problem paths are supplied whole because a problem field is owned by
        the operation's problem namespace rather than by this scope.
        """

        semantics = semantics_for(value_type)
        resolved = QualifiedPath.parse(path)
        self._problem.append(ProblemField(resolved, semantics, required, validate, description))
        handle: Ref[T] = Ref(resolved, DependencyKind.PROBLEM, semantics)
        return handle

    def decision(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: DomainFactory,
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[T]:
        semantics = semantics_for(value_type)
        path = self.local_path(name)
        self._decisions.append(Decision(path, semantics, domain(path), applies_if))
        handle: Ref[T] = Ref(path, DependencyKind.DECISION, semantics)
        return handle

    def derived(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[T]:
        semantics = semantics_for(value_type)
        path = self.semantic_path(name)
        self._properties.append(
            DerivedProperty(path, semantics, evaluator(path, dependencies, evaluate), applies_if)
        )
        handle: Ref[T] = Ref(path, DependencyKind.PROPERTY, semantics)
        return handle

    def constraint(
        self,
        name: str,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
        sets: Sequence[str] = (),
    ) -> ConstraintRef:
        """Declare one constraint and register its set membership here.

        Registering at the declaration site is what makes a constraint
        impossible to declare and then forget to include, which is the failure
        the separate path lists invited.
        """

        path = self.constraint_path(name)
        self._constraints.append(
            Constraint(path, predicate(path, dependencies, evaluate), applies_if)
        )
        for group in sets:
            self._sets.setdefault(group, []).append(path)
        return ConstraintRef(path)

    def include_in(self, group: str, *constraints: ConstraintRef) -> None:
        """Add already-declared constraints to one more set."""

        for item in constraints:
            paths = self._sets.setdefault(group, [])
            if item.path not in paths:
                paths.append(item.path)

    def readiness_profile(
        self,
        name: str,
        *,
        decisions: Sequence[Ref[object]] = (),
        properties: Sequence[Ref[object]] = (),
        constraints: Sequence[ConstraintRef] = (),
    ) -> None:
        self._profiles.append(
            ReadinessProfile(
                name,
                tuple(item.path for item in decisions),
                tuple(item.path for item in properties),
                tuple(item.path for item in constraints),
            )
        )

    # -- output ------------------------------------------------------------

    def constraints_in(self, group: str) -> tuple[ConstraintRef, ...]:
        return tuple(ConstraintRef(path) for path in self._sets.get(group, ()))

    def spec(self) -> DesignSpaceSpec:
        """Return the ordinary engine specification this scope declared."""

        return DesignSpaceSpec(
            ProblemSchema(tuple(self._problem)),
            tuple(self._decisions),
            tuple(self._properties),
            tuple(self._constraints),
            tuple(ConstraintSet(name, tuple(paths)) for name, paths in sorted(self._sets.items())),
            tuple(self._profiles),
        )


__all__ = [
    "AuthoringError",
    "CONSTRAINT_PREFIX",
    "ConstraintRef",
    "Dependencies",
    "DomainFactory",
    "PendingFinding",
    "Ref",
    "Rejected",
    "SEMANTIC_PREFIX",
    "Scope",
    "Unresolvable",
    "divisors_of",
    "domain",
    "enum_semantics",
    "evaluator",
    "finite",
    "predicate",
    "reject",
    "semantics_for",
    "unresolved",
]
