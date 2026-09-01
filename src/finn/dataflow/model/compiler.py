# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower declarative :class:`Space` classes into ordinary engine records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from inspect import signature
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from finn.dataflow._engine import (
    AbsenceMode,
    Absent,
    Answer,
    Constraint as EngineConstraint,
    ConstraintSet,
    Decided,
    Decision as EngineDecision,
    DecisionDomain,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
)
from finn.dataflow.model.declarations import (
    AuthoringError,
    ChildValue,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Domain,
    Input,
    PendingFinding,
    Problem,
    Readiness,
    Rejected,
    Space,
    Unresolvable,
    Use,
    ValueSource,
    declared_members,
    exported_members,
)
from finn.dataflow.spec_algebra import assemble_specs, gate_spec

S = TypeVar("S", bound=Space)
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class _Ref(Generic[T_co]):
    """One typed bound handle used only by the declarative compiler."""

    path: QualifiedPath
    kind: DependencyKind
    semantics: ValueSemantics[object]
    absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE

    def dependency(self, name: str) -> DependencyRef:
        return DependencyRef(
            name,
            self.path,
            self.kind,
            self.semantics,
            self.absence,
        )


@dataclass(frozen=True)
class _CompiledSpace(Generic[S]):
    """One bound, namespaced Space declaration and its typed member handles."""

    owner: type[S]
    namespace: str
    spec: DesignSpaceSpec
    inputs: tuple[tuple[str, _Ref[object]], ...]
    members: tuple[tuple[str, _Ref[object]], ...]
    exports: tuple[tuple[str, _Ref[object]], ...]

    def member(self, name: str) -> _Ref[object]:
        try:
            return dict(self.members)[name]
        except KeyError:
            raise AuthoringError(
                f"{self.owner.__name__} has no compiled value member {name!r}"
            ) from None

    def exported(self, name: str) -> _Ref[object]:
        try:
            return dict(self.exports)[name]
        except KeyError:
            raise AuthoringError(f"{self.owner.__name__} does not export {name!r}") from None


def _path(prefix: str, name: str) -> QualifiedPath:
    return QualifiedPath(f"{prefix}.{name}")


def _local_name(member_name: str, declaration: object) -> str:
    stable = getattr(declaration, "stable_name", None)
    return cast(str, stable or member_name)


def _check_signature(
    dependencies: tuple[tuple[str, ValueSource[object]], ...],
    evaluate: Callable[..., object],
    *,
    candidate: bool = False,
) -> None:
    parameters = signature(evaluate).parameters
    if any(
        parameter.kind
        in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD, parameter.POSITIONAL_ONLY)
        for parameter in parameters.values()
    ):
        raise AuthoringError(
            f"{getattr(evaluate, '__qualname__', evaluate)} must take only named parameters"
        )
    declared = {name for name, _ in dependencies}
    if candidate:
        declared.add("candidate")
    accepted = set(parameters)
    if declared != accepted:
        missing = sorted(declared - accepted)
        extra = sorted(accepted - declared)
        raise AuthoringError(
            f"{getattr(evaluate, '__qualname__', evaluate)} signature does not match its "
            f"dependency mapping; unused dependencies {missing}, unbound parameters {extra}"
        )


class _Compilation:
    def __init__(
        self,
        space_type: type[Space],
        namespace: str,
        inputs: Mapping[str, _Ref[object]],
        *,
        problem_namespace: str | None,
        applies_if: EvaluatorSpec[Answer[bool]] | None,
        allow_problem: bool,
    ) -> None:
        if not namespace:
            raise AuthoringError("a Space compilation needs a namespace")
        self.space_type = space_type
        self.namespace = namespace
        self.supplied_inputs = dict(inputs)
        self.problem_namespace = problem_namespace
        self.applies_if = applies_if
        self.allow_problem = allow_problem
        self.declarations = declared_members(space_type)
        self.names = {id(value): name for name, value in self.declarations}
        self.refs: dict[int, _Ref[object]] = {}
        self.constraints: dict[int, QualifiedPath] = {}
        self.uses: dict[int, _CompiledSpace[Space]] = {}
        self.use_names = {
            id(value): name for name, value in self.declarations if isinstance(value, Use)
        }
        self.compiling_uses: set[int] = set()

    def compile(self) -> _CompiledSpace[Space]:
        self._validate_inputs()
        self._allocate_local_refs()
        for name, declaration in self.declarations:
            if isinstance(declaration, Use):
                self._compile_use(name, declaration)
        local = self._local_spec()
        children = tuple(
            self.uses[id(declaration)].spec
            for _name, declaration in self.declarations
            if isinstance(declaration, Use)
        )
        specification = assemble_specs((local, *children))
        if self.applies_if is not None:
            specification = gate_spec(specification, self.applies_if)
        exports = tuple(
            (name, self._source_ref(declaration))
            for name, declaration in exported_members(self.space_type).items()
        )
        members = tuple(
            (name, self.refs[id(declaration)])
            for name, declaration in self.declarations
            if isinstance(declaration, ValueSource) and not isinstance(declaration, ChildValue)
        )
        bound_inputs = tuple(
            (name, self.refs[id(declaration)])
            for name, declaration in self.declarations
            if isinstance(declaration, Input)
        )
        return _CompiledSpace(
            self.space_type,
            self.namespace,
            specification,
            bound_inputs,
            members,
            exports,
        )

    def _validate_inputs(self) -> None:
        expected = {
            name: declaration
            for name, declaration in self.declarations
            if isinstance(declaration, Input)
        }
        missing = sorted(set(expected) - set(self.supplied_inputs))
        extra = sorted(set(self.supplied_inputs) - set(expected))
        if missing or extra:
            raise AuthoringError(
                f"{self.space_type.__name__} Input binding is not exact; "
                f"missing {missing}, extra {extra}"
            )
        for name, declaration in expected.items():
            supplied = self.supplied_inputs[name]
            if not declaration.value_semantics.is_compatible_with(supplied.semantics):
                raise AuthoringError(
                    f"{self.space_type.__name__}.{name} expects "
                    f"{declaration.value_semantics.name}, got {supplied.semantics.name}"
                )

    def _allocate_local_refs(self) -> None:
        for member_name, declaration in self.declarations:
            name = _local_name(member_name, declaration)
            if isinstance(declaration, Problem):
                if not self.allow_problem or self.problem_namespace is None:
                    raise AuthoringError(
                        f"{self.space_type.__name__}.{member_name} declares a Problem "
                        "inside a reusable child Space"
                    )
                path = _path(self.problem_namespace, name)
                self.refs[id(declaration)] = _Ref(
                    path, DependencyKind.PROBLEM, declaration.value_semantics
                )
            elif isinstance(declaration, Input):
                supplied = self.supplied_inputs[member_name]
                self.refs[id(declaration)] = replace(supplied, absence=declaration.absence)
            elif isinstance(declaration, Decision):
                self.refs[id(declaration)] = _Ref(
                    _path(self.namespace, name),
                    DependencyKind.DECISION,
                    declaration.value_semantics,
                )
            elif isinstance(declaration, Derived):
                self.refs[id(declaration)] = _Ref(
                    _path(f"semantic.{self.namespace}", name),
                    DependencyKind.PROPERTY,
                    declaration.value_semantics,
                )
            elif isinstance(declaration, Constraint):
                self.constraints[id(declaration)] = _path(f"constraint.{self.namespace}", name)

    def _source_ref(self, source: ValueSource[object]) -> _Ref[object]:
        if isinstance(source, ChildValue):
            use = source.use
            use_name = self.use_names.get(id(use))
            if use_name is None:
                raise AuthoringError("a child export belongs to a Use outside this Space")
            return self._compile_use(use_name, use).exported(source.member_name)
        try:
            return self.refs[id(source)]
        except KeyError:
            raise AuthoringError(
                f"{self.space_type.__name__} references a value outside its declarations"
            ) from None

    def _dependency_refs(
        self, dependencies: tuple[tuple[str, ValueSource[object]], ...]
    ) -> tuple[DependencyRef, ...]:
        names = tuple(name for name, _ in dependencies)
        if len(names) != len(set(names)):
            raise AuthoringError("a declaration names one dependency twice")
        return tuple(
            self._source_ref(source).dependency(name)
            for name, source in sorted(dependencies, key=lambda item: item[0])
        )

    def _trace_path(self, item: ValueSource[object] | QualifiedPath) -> QualifiedPath:
        return item if isinstance(item, QualifiedPath) else self._source_ref(item).path

    def _adapt(self, result: object, owner: QualifiedPath) -> Answer[object]:
        if isinstance(result, (Decided, Absent, Unresolved)):
            return cast(Answer[object], result)
        if isinstance(result, (Rejected, Unresolvable)):
            pending: PendingFinding = result.finding
            finding = Finding(
                pending.kind,
                pending.code,
                owner,
                pending.message,
                pending.values,
                tuple(self._trace_path(item) for item in pending.trace),
            )
            return Absent((finding,)) if isinstance(result, Rejected) else Unresolved((finding,))
        return Decided(result)

    def _evaluator(
        self,
        owner: QualifiedPath,
        dependencies: tuple[tuple[str, ValueSource[object]], ...],
        evaluate: Callable[..., object],
    ) -> EvaluatorSpec[Answer[object]]:
        _check_signature(dependencies, evaluate)
        refs = self._dependency_refs(dependencies)
        names = tuple(item.name for item in refs)

        def call(view: DependencyView) -> Answer[object]:
            return self._adapt(evaluate(**{name: view[name] for name in names}), owner)

        return EvaluatorSpec(refs, call)

    def _domain(self, owner: QualifiedPath, declaration: Domain) -> DecisionDomain:
        _check_signature(declaration.dependencies, declaration.accepts, candidate=True)
        refs = self._dependency_refs(declaration.dependencies)
        names = tuple(item.name for item in refs)

        def accepts(candidate: object, view: DependencyView) -> Answer[bool]:
            result = declaration.accepts(
                candidate=candidate, **{name: view[name] for name in names}
            )
            return cast(Answer[bool], self._adapt(result, owner))

        candidates: EvaluatorSpec[Answer[tuple[object, ...]]] | None = None
        enumerator = declaration.candidates
        if enumerator is not None:
            _check_signature(declaration.dependencies, enumerator)

            def enumerate_values(view: DependencyView) -> Answer[tuple[object, ...]]:
                result = enumerator(**{name: view[name] for name in names})
                return cast(Answer[tuple[object, ...]], self._adapt(result, owner))

            candidates = EvaluatorSpec(refs, enumerate_values)
        return DecisionDomain(refs, accepts, candidates)

    def _constraint_path(self, declaration: Constraint) -> QualifiedPath:
        try:
            return self.constraints[id(declaration)]
        except KeyError:
            raise AuthoringError(
                f"{self.space_type.__name__} references a Constraint outside its declarations"
            ) from None

    def _local_spec(self) -> DesignSpaceSpec:
        fields: list[ProblemField] = []
        decisions: list[EngineDecision] = []
        properties: list[DerivedProperty] = []
        constraints: list[EngineConstraint] = []
        groups: list[ConstraintSet] = []
        readiness: list[ReadinessProfile] = []
        for member_name, declaration in self.declarations:
            if isinstance(declaration, Problem):
                handle = self.refs[id(declaration)]
                fields.append(
                    ProblemField(
                        handle.path,
                        declaration.value_semantics,
                        declaration.required,
                        declaration.validate,
                        declaration.description,
                    )
                )
            elif isinstance(declaration, Decision):
                handle = self.refs[id(declaration)]
                decisions.append(
                    EngineDecision(
                        handle.path,
                        declaration.value_semantics,
                        self._domain(handle.path, declaration.domain),
                    )
                )
            elif isinstance(declaration, Derived):
                handle = self.refs[id(declaration)]
                properties.append(
                    DerivedProperty(
                        handle.path,
                        declaration.value_semantics,
                        self._evaluator(
                            handle.path, declaration.dependencies, declaration.evaluate
                        ),
                    )
                )
            elif isinstance(declaration, Constraint):
                path = self._constraint_path(declaration)
                constraints.append(
                    EngineConstraint(
                        path,
                        cast(
                            EvaluatorSpec[Answer[bool]],
                            self._evaluator(path, declaration.dependencies, declaration.evaluate),
                        ),
                    )
                )
            elif isinstance(declaration, ConstraintGroup):
                groups.append(
                    ConstraintSet(
                        f"{self.namespace}.{_local_name(member_name, declaration)}",
                        tuple(self._constraint_path(item) for item in declaration.constraints),
                    )
                )
            elif isinstance(declaration, Readiness):
                readiness.append(
                    ReadinessProfile(
                        f"{self.namespace}.{_local_name(member_name, declaration)}",
                        tuple(self._source_ref(item).path for item in declaration.decisions),
                        tuple(self._source_ref(item).path for item in declaration.properties),
                        tuple(self._constraint_path(item) for item in declaration.constraints),
                    )
                )
        return DesignSpaceSpec(
            ProblemSchema(tuple(fields)),
            tuple(decisions),
            tuple(properties),
            tuple(constraints),
            tuple(groups),
            tuple(readiness),
        )

    def _compile_use(self, member_name: str, declaration: Use[Space]) -> _CompiledSpace[Space]:
        key = id(declaration)
        if key in self.uses:
            return self.uses[key]
        if key in self.compiling_uses:
            raise AuthoringError(f"{self.space_type.__name__}.{member_name} forms a Use cycle")
        self.compiling_uses.add(key)
        try:
            bound = {name: self._source_ref(source) for name, source in declaration.bindings}
            gate: EvaluatorSpec[Answer[bool]] | None = None
            if declaration.when is not None:
                condition = self._source_ref(cast("ValueSource[object]", declaration.when))
                if condition.semantics.type_token is not bool:
                    raise AuthoringError(
                        f"{self.space_type.__name__}.{member_name} when= is not Boolean"
                    )

                def applies(values: DependencyView) -> Answer[bool]:
                    return Decided(cast(bool, values["condition"]))

                gate = EvaluatorSpec((condition.dependency("condition"),), applies)
            child_namespace = f"{self.namespace}.{_local_name(member_name, declaration)}"
            child = _compile_space(
                declaration.space_type,
                child_namespace,
                bound,
                applies_if=gate,
                _allow_problem=False,
            )
            self.uses[key] = child
            return child
        finally:
            self.compiling_uses.discard(key)


def _compile_space(
    space_type: type[S],
    namespace: str,
    inputs: Mapping[str, _Ref[object]] | None = None,
    *,
    problem_namespace: str | None = None,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    _allow_problem: bool = True,
) -> _CompiledSpace[S]:
    """Compile one class declaration into an ordinary flat spec fragment."""

    if not issubclass(space_type, Space):
        raise AuthoringError("only a Space subclass can be compiled")
    compiled = _Compilation(
        space_type,
        namespace,
        inputs or MappingProxyType({}),
        problem_namespace=problem_namespace,
        applies_if=applies_if,
        allow_problem=_allow_problem,
    ).compile()
    return cast("_CompiledSpace[S]", compiled)


def compile_space(
    space_type: type[Space],
    namespace: str,
    *,
    problem_namespace: str | None = None,
) -> DesignSpaceSpec:
    """Compile a closed root Space into the ordinary engine specification."""

    return _compile_space(
        space_type,
        namespace,
        problem_namespace=problem_namespace,
    ).spec


__all__ = ["compile_space"]
