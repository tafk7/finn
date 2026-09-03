# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower declarative :class:`Space` classes into ordinary engine records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set as AbstractSet
from dataclasses import dataclass, field, replace
from inspect import signature
from threading import RLock
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

from finn.dataflow._engine import (
    ABSENT,
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
    DesignPoint,
    DesignSpace,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    RequestError,
    Unresolved,
    ValueSemantics,
)
from finn.dataflow.model.branching import (
    BranchCatalog,
    BranchInfo,
    BranchOutputInfo,
    CaseInfo,
)
from finn.dataflow.model.declarations import (
    DECLARATION_TYPES,
    AuthoringError,
    BranchOutput,
    Case,
    ChildValue,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Domain,
    Input,
    OneOf,
    PendingFinding,
    Problem,
    Projection,
    Readiness,
    Rejected,
    Space,
    Unresolvable,
    Use,
    ValueSource,
    check_reserved_names,
    declared_members,
    exported_members,
    finite,
)
from finn.dataflow.spec_algebra import assemble_specs, combine_applicability, gate_spec

S = TypeVar("S", bound=Space)
T_co = TypeVar("T_co", covariant=True)

#: Case ids are ordinary strings; the branch selector is an ordinary Decision.
_CASE_ID_SEMANTICS: ValueSemantics[object] = ValueSemantics.immutable_nominal(str, name="CaseId")


def _direct_branches(catalog: BranchCatalog) -> tuple[str, ...]:
    """The branches a case owns itself, excluding those a nested branch owns.

    The catalog is flat, so ownership is read back from the namespaces: a branch
    inside another branch's case is that case's child, not this one's.
    """

    owned = tuple(f"{case.namespace}." for branch in catalog.branches for case in branch.cases)
    return tuple(
        branch.namespace
        for branch in catalog.branches
        if not any(branch.namespace.startswith(prefix) for prefix in owned)
    )


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
class _CompiledCase:
    """One compiled alternative beneath a branch namespace."""

    case_id: str
    compiled: _CompiledSpace[Space]


@dataclass(frozen=True)
class _CompiledBranch:
    """The private compilation record behind one public :class:`BranchInfo`."""

    member_name: str
    namespace: str
    selector: _Ref[object] | None
    cases: tuple[_CompiledCase, ...]
    outputs: tuple[tuple[str, _Ref[object]], ...]
    info: BranchInfo
    #: The ``when=`` condition, if the branch is conditional at all.
    active: _Ref[object] | None = None

    def case(self, case_id: str) -> _CompiledCase:
        for candidate in self.cases:
            if candidate.case_id == case_id:
                return candidate
        raise AuthoringError(f"branch {self.namespace!r} has no case {case_id!r}")

    def output(self, name: str) -> _Ref[object]:
        try:
            return dict(self.outputs)[name]
        except KeyError:
            raise AuthoringError(
                f"branch {self.namespace!r} does not select an output named {name!r}"
            ) from None


@dataclass(frozen=True)
class _CompiledProjection(Generic[T_co]):
    """Private bound metadata for one public Projection declaration.

    ``readiness_profile`` and ``constraint_sets`` are the engine-facing *names*
    the generic lowering already produced, not copies of the declarations, so a
    Projection adds no engine declaration of its own: it is a stored question
    over paths that already exist.

    The two policies are recorded rather than passed.  They are real parts of
    the contract, and writing them down is what makes it checkable that U1 has
    exactly one behaviour for each; they are not constructor arguments because
    there is no second value to choose.
    """

    member_name: str
    name: str
    output: _Ref[T_co]
    readiness_profile: str
    constraint_sets: tuple[str, ...]
    #: Final inapplicability of the output propagates as the output's own Absent.
    absence_policy: str = "propagate"
    #: The snapshot is the output declaration's own ``ValueSemantics``.
    snapshot_policy: str = "declared"


@dataclass(frozen=True)
class _CompiledSpace(Generic[S]):
    """One bound, namespaced Space declaration and its typed member handles."""

    owner: type[S]
    namespace: str
    spec: DesignSpaceSpec
    inputs: tuple[tuple[str, _Ref[object]], ...]
    members: tuple[tuple[str, _Ref[object]], ...]
    exports: tuple[tuple[str, _Ref[object]], ...]
    children: tuple[tuple[str, _CompiledSpace[Space]], ...] = ()
    extension: object | None = None
    branches: tuple[tuple[str, _CompiledBranch], ...] = ()
    catalog: BranchCatalog = BranchCatalog()
    projections: tuple[tuple[str, _CompiledProjection[object]], ...] = ()
    #: Constraint members by class-member name.  Kept beside ``members``, which
    #: holds only value declarations, because a diagnostic that cannot name the
    #: constraint that refused is the one nobody can act on.
    constraint_members: tuple[tuple[str, QualifiedPath], ...] = ()
    #: Readiness and ConstraintGroup members mapped to the engine names the
    #: generic lowering gave them, so the occurrence layer names a declaration
    #: and never rebuilds a profile name from a convention.
    readiness_names: tuple[tuple[str, str], ...] = ()
    constraint_set_names: tuple[tuple[str, str], ...] = ()

    def engine_name(self, table: tuple[tuple[str, str], ...], name: str, what: str) -> str:
        try:
            return dict(table)[name]
        except KeyError:
            raise AuthoringError(f"{self.owner.__name__} has no {what} member {name!r}") from None

    def branch(self, name: str) -> _CompiledBranch:
        try:
            return dict(self.branches)[name]
        except KeyError:
            raise AuthoringError(f"{self.owner.__name__} has no branch {name!r}") from None

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

    def child(self, name: str) -> _CompiledSpace[Space]:
        try:
            return dict(self.children)[name]
        except KeyError:
            raise AuthoringError(f"{self.owner.__name__} has no child Use {name!r}") from None

    def projection(self, name: str) -> _CompiledProjection[object]:
        try:
            return dict(self.projections)[name]
        except KeyError:
            raise AuthoringError(f"{self.owner.__name__} has no Projection {name!r}") from None


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
        ancestors: tuple[type[Space], ...],
    ) -> None:
        if not namespace:
            raise AuthoringError("a Space compilation needs a namespace")
        self.space_type = space_type
        self.namespace = namespace
        self.supplied_inputs = dict(inputs)
        self.problem_namespace = problem_namespace
        self.applies_if = applies_if
        self.allow_problem = allow_problem
        self.ancestors = ancestors
        self.declarations = declared_members(space_type)
        # ``__init_subclass__`` catches the ordinary class-body spelling at the
        # point of the mistake.  This is the backstop for a declaration attached
        # to a class after it was created -- the recursive-Space idiom does
        # exactly that -- which no class-creation hook can see.
        check_reserved_names(space_type, self.declarations)
        self.names = {id(value): name for name, value in self.declarations}
        self.alias_names: dict[int, str] = {}
        for base in reversed(space_type.__mro__):
            if not issubclass(base, Space) or base is Space:
                continue
            for name, value in base.__dict__.items():
                if isinstance(value, DECLARATION_TYPES):
                    self.alias_names[id(value)] = name
        self.refs: dict[int, _Ref[object]] = {}
        self.constraints: dict[int, QualifiedPath] = {}
        self.uses: dict[int, _CompiledSpace[Space]] = {}
        self.use_names = {
            id(value): name for name, value in self.declarations if isinstance(value, Use)
        }
        self.compiling_uses: set[int] = set()
        self.branches: dict[int, _CompiledBranch] = {}
        self.branch_names = {
            id(value): name for name, value in self.declarations if isinstance(value, OneOf)
        }
        self.compiling_branches: set[int] = set()
        self.branch_decisions: dict[int, EngineDecision] = {}
        self.branch_properties: dict[int, tuple[DerivedProperty, ...]] = {}

    def compile(self) -> _CompiledSpace[Space]:
        self._validate_inputs()
        self._allocate_local_refs()
        for name, declaration in self.declarations:
            if isinstance(declaration, Use):
                self._compile_use(name, declaration)
            elif isinstance(declaration, OneOf):
                self._compile_branch(name, declaration)
        local = self._local_spec()
        children: list[DesignSpaceSpec] = []
        for _name, declaration in self.declarations:
            if isinstance(declaration, Use):
                children.append(self.uses[id(declaration)].spec)
            elif isinstance(declaration, OneOf):
                children.extend(case.compiled.spec for case in self.branches[id(declaration)].cases)
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
        branches = tuple(
            (name, self.branches[id(declaration)])
            for name, declaration in self.declarations
            if isinstance(declaration, OneOf)
        )
        projections = tuple(
            (name, self._compile_projection(name, declaration))
            for name, declaration in self.declarations
            if isinstance(declaration, Projection)
        )
        compiled = _CompiledSpace(
            self.space_type,
            self.namespace,
            specification,
            bound_inputs,
            members,
            exports,
            tuple(
                (name, self.uses[id(declaration)])
                for name, declaration in self.declarations
                if isinstance(declaration, Use)
            ),
            None,
            branches,
            self._catalog(),
            projections,
            tuple(
                (name, self.constraints[id(declaration)])
                for name, declaration in self.declarations
                if isinstance(declaration, Constraint)
            ),
            tuple(
                (name, self._group_name(name, declaration))
                for name, declaration in self.declarations
                if isinstance(declaration, Readiness)
            ),
            tuple(
                (name, self._group_name(name, declaration))
                for name, declaration in self.declarations
                if isinstance(declaration, ConstraintGroup)
            ),
        )
        finalized = self.space_type._finalize_compilation(compiled)
        if not isinstance(finalized, _CompiledSpace):
            raise AuthoringError(
                f"{self.space_type.__name__} returned a non-Space compilation result"
            )
        return finalized

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
        effective = dict(self.declarations)
        for identity, member_name in self.alias_names.items():
            declaration = effective[member_name]
            if isinstance(declaration, ValueSource):
                self.refs[identity] = self.refs[id(declaration)]
            elif isinstance(declaration, Constraint):
                self.constraints[identity] = self.constraints[id(declaration)]

    def _catalog(self) -> BranchCatalog:
        """Flatten this Space's branches outermost-first in declaration order."""

        collected: list[BranchInfo] = []
        for _name, declaration in self.declarations:
            if isinstance(declaration, Use):
                collected.extend(self.uses[id(declaration)].catalog.branches)
            elif isinstance(declaration, OneOf):
                record = self.branches[id(declaration)]
                collected.append(record.info)
                for case in record.cases:
                    collected.extend(case.compiled.catalog.branches)
        return BranchCatalog(tuple(collected))

    def _source_ref(self, source: ValueSource[object]) -> _Ref[object]:
        if isinstance(source, BranchOutput):
            branch = source.branch
            branch_name = self.branch_names.get(id(branch))
            if branch_name is None:
                raise AuthoringError("a selected output belongs to a branch outside this Space")
            return self._compile_branch(branch_name, branch).output(source.output_name)
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
                        self._group_name(member_name, declaration),
                        tuple(self._constraint_path(item) for item in declaration.constraints),
                    )
                )
            elif isinstance(declaration, OneOf):
                selector = self.branch_decisions.get(id(declaration))
                if selector is not None:
                    decisions.append(selector)
                properties.extend(self.branch_properties.get(id(declaration), ()))
            elif isinstance(declaration, Readiness):
                readiness.append(
                    ReadinessProfile(
                        self._group_name(member_name, declaration),
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

    def _group_name(self, member_name: str, declaration: object) -> str:
        """The one engine name a ConstraintGroup or Readiness member is given.

        Stated once and read by both the spec lowering and the projection
        lowering.  Two copies of this rule is precisely how a projection ends up
        naming a profile the engine does not have -- and how occurrence runtime
        code ends up rebuilding a path the compiler already knew.
        """

        return f"{self.namespace}.{_local_name(member_name, declaration)}"

    def _owned_declaration(
        self,
        declaration: object,
        expected: type,
        what: str,
        role: str,
    ) -> tuple[str, object]:
        """Resolve one declaration a Projection names, or say where it is not.

        Ownership is checked at compile time because the alternative is a
        projection that looks fine until the first query, then names an engine
        profile nobody declared.
        """

        member_name = self.alias_names.get(id(declaration))
        effective = dict(self.declarations).get(member_name or "")
        if member_name is None or not isinstance(effective, expected):
            raise AuthoringError(
                f"{self.space_type.__name__} {what} names a {role} that "
                f"{self.space_type.__name__} does not declare"
            )
        return member_name, effective

    def _compile_projection(
        self,
        member_name: str,
        declaration: Projection[object],
    ) -> _CompiledProjection[object]:
        what = f"Projection {_local_name(member_name, declaration)!r}"
        readiness_name, readiness = self._owned_declaration(
            declaration.readiness, Readiness, what, "Readiness declaration"
        )
        groups: list[str] = []
        for group in declaration.constraints:
            group_name, effective = self._owned_declaration(
                group, ConstraintGroup, what, "ConstraintGroup"
            )
            engine_name = self._group_name(group_name, effective)
            if engine_name in groups:
                raise AuthoringError(
                    f"{self.space_type.__name__} {what} names constraint group {group_name!r} twice"
                )
            groups.append(engine_name)
        try:
            output = self._source_ref(declaration.output)
        except AuthoringError:
            raise AuthoringError(
                f"{self.space_type.__name__} {what} names an output that "
                f"{self.space_type.__name__} cannot resolve"
            ) from None
        return _CompiledProjection(
            member_name,
            self._group_name(member_name, declaration),
            output,
            self._group_name(readiness_name, readiness),
            tuple(groups),
        )

    def _when_gate(
        self,
        member_name: str,
        when: ValueSource[bool] | None,
        scope: str,
    ) -> EvaluatorSpec[Answer[bool]] | None:
        """Lower one ``when=`` condition into an applicability evaluator.

        The dependency name is derived from the gated scope rather than being a
        fixed word, because ``combine_applicability`` puts an outer gate and an
        inner gate in one dependency namespace and rejects a collision.
        """

        if when is None:
            return None
        condition = self._source_ref(cast("ValueSource[object]", when))
        if condition.semantics.type_token is not bool:
            raise AuthoringError(f"{self.space_type.__name__}.{member_name} when= is not Boolean")
        name = f"when@{scope}"

        def applies(values: DependencyView) -> Answer[bool]:
            return Decided(cast(bool, values[name]))

        return EvaluatorSpec((condition.dependency(name),), applies)

    def _compile_branch(self, member_name: str, declaration: OneOf) -> _CompiledBranch:
        key = id(declaration)
        if key in self.branches:
            return self.branches[key]
        if key in self.compiling_branches:
            raise AuthoringError(f"{self.space_type.__name__}.{member_name} forms a branch cycle")
        self.compiling_branches.add(key)
        try:
            record = self._build_branch(member_name, declaration)
            self.branches[key] = record
            return record
        finally:
            self.compiling_branches.discard(key)

    def _case_ids(self, member_name: str, declaration: OneOf) -> tuple[str, ...]:
        ids: list[str] = []
        for case in declaration.cases:
            case_id = declaration.case_id(case)
            if case_id is None:
                raise AuthoringError(
                    f"{self.space_type.__name__}.{member_name} needs an explicit name= on the "
                    f"{case.space_type.__name__} case"
                )
            if not case_id:
                raise AuthoringError(
                    f"{self.space_type.__name__}.{member_name} has an empty case id"
                )
            if "." in case_id:
                raise AuthoringError(
                    f"{self.space_type.__name__}.{member_name} case id {case_id!r} contains a "
                    "dot; a case id is one path segment of its namespace"
                )
            declaration.check_case(self.space_type.__name__, member_name, case)
            if case_id in ids:
                raise AuthoringError(
                    f"{self.space_type.__name__}.{member_name} declares case id {case_id!r} twice"
                )
            ids.append(case_id)
        return tuple(ids)

    def _selected_output(
        self,
        path: QualifiedPath,
        selector: _Ref[object] | None,
        cases: tuple[tuple[str, _Ref[object]], ...],
        semantics: ValueSemantics[object],
    ) -> DerivedProperty:
        """Forward the exact value of whichever case is live at this point."""

        dependencies = tuple(
            replace(reference, absence=AbsenceMode.ALLOWS_ABSENT).dependency(f"case@{case_id}")
            for case_id, reference in cases
        )
        if selector is not None:
            dependencies = (
                replace(selector, absence=AbsenceMode.ALLOWS_ABSENT).dependency("selector"),
                *dependencies,
            )

        def forward(values: DependencyView) -> Answer[object]:
            if selector is not None:
                chosen = values["selector"]
                if chosen is ABSENT:
                    return Absent()
                value = values[f"case@{chosen}"]
                if value is ABSENT:
                    return Absent(
                        (
                            Finding(
                                FindingKind.LIMITATION,
                                "branch-selected-case-absent",
                                path,
                                "the selected case does not produce this output",
                                (("case", cast(str, chosen)),),
                            ),
                        )
                    )
                return Decided(value)
            live = tuple(
                values[f"case@{case_id}"]
                for case_id, _reference in cases
                if values[f"case@{case_id}"] is not ABSENT
            )
            if not live:
                return Absent()
            return Decided(live[0])

        return DerivedProperty(path, semantics, EvaluatorSpec(dependencies, forward))

    def _build_branch(self, member_name: str, declaration: OneOf) -> _CompiledBranch:
        branch_name = _local_name(member_name, declaration)
        namespace = f"{self.namespace}.{branch_name}"
        case_ids = self._case_ids(member_name, declaration)
        active = (
            None
            if declaration.when is None
            else self._source_ref(cast("ValueSource[object]", declaration.when))
        )
        branch_gate = self._when_gate(member_name, declaration.when, namespace)

        selector: _Ref[object] | None = None
        if len(case_ids) > 1:
            selector_path = _path(namespace, declaration.selector_name)
            selector = _Ref(selector_path, DependencyKind.DECISION, _CASE_ID_SEMANTICS)
            self.branch_decisions[id(declaration)] = EngineDecision(
                selector_path,
                _CASE_ID_SEMANTICS,
                self._domain(selector_path, finite(case_ids)),
                branch_gate,
            )

        compiled_cases: list[_CompiledCase] = []
        for case_id, case in zip(case_ids, declaration.cases):
            compiled_cases.append(
                _CompiledCase(
                    case_id,
                    self._compile_case(namespace, case_id, case, selector, branch_gate),
                )
            )

        outputs: list[tuple[str, _Ref[object]]] = []
        output_infos: list[BranchOutputInfo] = []
        generated: list[DerivedProperty] = []
        for output_name in declaration.outputs:
            per_case: list[tuple[str, _Ref[object]]] = []
            semantics: ValueSemantics[object] | None = None
            for compiled_case in compiled_cases:
                try:
                    reference = compiled_case.compiled.exported(output_name)
                except AuthoringError:
                    raise AuthoringError(
                        f"{self.space_type.__name__}.{member_name} selects {output_name!r}, "
                        f"which {compiled_case.compiled.owner.__name__} does not export"
                    ) from None
                if semantics is None:
                    semantics = reference.semantics
                elif not semantics.is_compatible_with(reference.semantics):
                    raise AuthoringError(
                        f"{self.space_type.__name__}.{member_name} output {output_name!r} "
                        f"changes value semantics from {semantics.name} to "
                        f"{reference.semantics.name}"
                    )
                per_case.append((compiled_case.case_id, reference))
            assert semantics is not None
            path = _path(f"semantic.{namespace}", output_name)
            generated.append(
                replace(
                    self._selected_output(path, selector, tuple(per_case), semantics),
                    applies_if=branch_gate,
                )
            )
            reference = _Ref(path, DependencyKind.PROPERTY, semantics)
            outputs.append((output_name, reference))
            output_infos.append(BranchOutputInfo(output_name, path, semantics))
        self.branch_properties[id(declaration)] = tuple(generated)

        info = BranchInfo(
            namespace,
            selector.path if selector is not None else None,
            tuple(
                CaseInfo(
                    compiled_case.case_id,
                    compiled_case.compiled.namespace,
                    tuple(item.path for item in compiled_case.compiled.spec.decisions),
                    tuple(item.path for item in compiled_case.compiled.spec.properties),
                    tuple(item.path for item in compiled_case.compiled.spec.constraints),
                    tuple(item.name for item in compiled_case.compiled.spec.readiness_profiles),
                    _direct_branches(compiled_case.compiled.catalog),
                )
                for compiled_case in compiled_cases
            ),
            tuple(output_infos),
        )
        return _CompiledBranch(
            member_name,
            namespace,
            selector,
            tuple(compiled_cases),
            tuple(outputs),
            info,
            active,
        )

    def _compile_case(
        self,
        namespace: str,
        case_id: str,
        case: Case,
        selector: _Ref[object] | None,
        branch_gate: EvaluatorSpec[Answer[bool]] | None,
    ) -> _CompiledSpace[Space]:
        bound = {name: self._source_ref(source) for name, source in case.bindings}
        case_namespace = f"{namespace}.{case_id}"
        gate = branch_gate
        if selector is not None:
            name = f"selector@{case_namespace}"

            def chosen(values: DependencyView) -> Answer[bool]:
                return Decided(values[name] == case_id)

            selection = EvaluatorSpec((selector.dependency(name),), chosen)
            gate = (
                selection if branch_gate is None else combine_applicability(branch_gate, selection)
            )
        return _compile_space(
            case.space_type,
            case_namespace,
            bound,
            applies_if=gate,
            _allow_problem=False,
            _ancestors=(*self.ancestors, self.space_type),
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
            child_namespace = f"{self.namespace}.{_local_name(member_name, declaration)}"
            gate = self._when_gate(member_name, declaration.when, child_namespace)
            child = _compile_space(
                declaration.space_type,
                child_namespace,
                bound,
                applies_if=gate,
                _allow_problem=False,
                _ancestors=(*self.ancestors, self.space_type),
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
    _ancestors: tuple[type[Space], ...] = (),
) -> _CompiledSpace[S]:
    """Compile one class declaration into an ordinary flat spec fragment."""

    if not issubclass(space_type, Space):
        raise AuthoringError("only a Space subclass can be compiled")
    if space_type in _ancestors:
        cycle = " -> ".join(item.__name__ for item in (*_ancestors, space_type))
        raise AuthoringError(f"Use cycle: {cycle}")
    compiled = _Compilation(
        space_type,
        namespace,
        inputs or MappingProxyType({}),
        problem_namespace=problem_namespace,
        applies_if=applies_if,
        allow_problem=_allow_problem,
        ancestors=_ancestors,
    ).compile()
    return cast("_CompiledSpace[S]", compiled)


def answer_for(
    engine: Engine,
    point: DesignPoint,
    reference: _Ref[object],
) -> Answer[object]:
    """Read one bound handle at a point, whatever kind of declaration it is."""

    if reference.kind is DependencyKind.PROBLEM:
        if reference.path not in point.problem:
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "kernel-input-problem-absent",
                        reference.path,
                        "a required Kernel input is absent from the problem",
                    ),
                )
            )
        return Decided(point.problem[reference.path])
    if reference.kind is DependencyKind.DECISION:
        if reference.path not in point.assignments:
            return Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "kernel-decision-unassigned",
                        reference.path,
                        "a required Kernel decision is not committed",
                    ),
                )
            )
        return Decided(point.assignments[reference.path])
    try:
        return engine.query_property(point, reference.path)
    except RequestError as error:
        return Unresolved(error.findings)


def _members_of(space_type: type[Space], kinds: tuple[type, ...]) -> dict[int, str]:
    return {
        id(value): name
        for base in reversed(space_type.__mro__)
        if issubclass(base, Space) and base is not Space
        for name, value in base.__dict__.items()
        if isinstance(value, kinds)
    }


def resolve_value_source(
    compiled: _CompiledSpace[S],
    source: ValueSource[object],
    what: str,
) -> _Ref[object]:
    """Resolve any class-body value declaration to its compiled handle.

    A specialization -- a Kernel ``Parameter``, a Design ``when=`` or position
    map -- reads values the generic compiler has already bound, so it resolves
    them here rather than reimplementing the walk.  Each partial copy of this
    was a place where one kind of source silently stopped composing: both
    handled ``ChildValue`` and neither handled ``BranchOutput``, so a value
    selected by a ``OneOf`` could not reach a Parameter or a topology condition
    even though the generic compiler understood it perfectly well.
    """

    owner = compiled.owner
    if isinstance(source, BranchOutput):
        branches = _members_of(owner, (OneOf,))
        name = branches.get(id(source.branch))
        if name is None:
            raise AuthoringError(f"{owner.__name__} {what} names a branch outside the class")
        return compiled.branch(name).output(source.output_name)
    if isinstance(source, ChildValue):
        children = _members_of(owner, (Use,))
        name = children.get(id(source.use))
        if name is None:
            raise AuthoringError(f"{owner.__name__} {what} names a child outside the class")
        return compiled.child(name).exported(source.member_name)
    name = _members_of(owner, (ValueSource,)).get(id(source))
    if name is None:
        raise AuthoringError(f"{owner.__name__} {what} names a value outside the class")
    return compiled.member(name)


def imported_decisions(
    point: DesignPoint,
    spec: DesignSpaceSpec,
    inputs: tuple[tuple[str, _Ref[object]], ...],
    owned: AbstractSet[QualifiedPath],
) -> tuple[QualifiedPath, ...]:
    """Every committed decision this fragment reads but does not own.

    Provenance, not ownership: a configured Kernel or Design keeps the paths of
    the outside choices it was configured against, so a later reader can tell
    which external commitments its values depend on.
    """

    pending: list[DependencyRef] = []
    for declaration in (*spec.decisions, *spec.properties, *spec.constraints):
        evaluator = getattr(declaration, "evaluator", None)
        if evaluator is not None:
            pending.extend(evaluator.dependencies)
        domain = getattr(declaration, "domain", None)
        if domain is not None:
            pending.extend(domain.dependencies)
    pending.extend(reference.dependency(name) for name, reference in inputs)
    found: list[QualifiedPath] = []
    visited: set[tuple[QualifiedPath, DependencyKind]] = set()
    while pending:
        dependency = pending.pop()
        key = (dependency.path, dependency.kind)
        if key in visited:
            continue
        visited.add(key)
        if dependency.kind is DependencyKind.DECISION:
            if dependency.path not in owned and dependency.path in point.assignments:
                found.append(dependency.path)
            continue
        if dependency.kind is not DependencyKind.PROPERTY:
            continue
        declared = point.design_space.properties.get(dependency.path)
        if declared is not None:
            pending.extend(declared.evaluator.dependencies)
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True, slots=True)
class SpaceModel:
    """One compiled Space: the ordinary flat spec plus its branch catalog.

    The two public fields are deliberately separate values.  ``specification``
    is everything the engine sees; ``branches`` is the class-to-flat-spec
    relationship the engine does not model and a specialization algorithm
    needs.  Neither exposes a compiled declaration, a ``_Ref``, or an evaluator.

    ``start`` adds the third thing a caller needs and neither public field can
    give: a bound occurrence of the authored class.  The compiled declaration
    tree it needs is held privately, because that is exactly the record which
    would let a contributor reconstruct paths.

    What the model owns is *immutable and reusable*: the tree, the validated
    ``DesignSpace``, the branch catalog and the projection metadata.  What it
    does **not** own is the Engine, the point or the lock -- every ``start``
    mints those fresh, so two roots from one model never share an evaluation
    cache and never serialize on each other.
    """

    specification: DesignSpaceSpec
    branches: BranchCatalog
    _tree: _CompiledSpace[Space] | None = field(default=None, repr=False, compare=False)
    _shared: _SharedCompilation = field(
        default_factory=lambda: _SharedCompilation(), repr=False, compare=False
    )

    @property
    def space_type(self) -> type[Space]:
        """The authored root class this model was compiled from."""

        return self._compiled_tree().owner

    def start(
        self,
        problem: object,
        *,
        expected_problem_fingerprint: str | None = None,
    ) -> Space:
        """Freeze one problem into a new root occurrence of the authored class.

        The compiler-service entry point.  A pipeline that processes many
        occurrences of one family holds the model and calls this repeatedly;
        ``Space.start`` is the ordinary authoring spelling of the same thing and
        goes through here.
        """

        started = _occurrence_api().start_from_model(
            self,
            problem,
            expected_problem_fingerprint=expected_problem_fingerprint,
        )
        return cast("Space", started)

    def _compiled_tree(self) -> _CompiledSpace[Space]:
        if self._tree is None:
            raise AuthoringError(
                "this SpaceModel was built without its compiled declarations and cannot "
                "start an occurrence; use compile_space_model()"
            )
        return self._tree

    def _design_space(self) -> DesignSpace:
        """The validated space, computed once and shared by every root."""

        return self._shared.design_space(self.specification)


class _SharedCompilation:
    """The immutable-by-contract results one compiled model memoizes.

    Separate from ``SpaceModel`` so the model itself stays a frozen, comparable
    value.  Validation is memoized rather than eager because compiling a Space
    and validating one are different questions and existing callers compile
    specifications they never intend to run.  The lock guards only this memo --
    it is a *compilation* lock, not a lineage lock, and no engine call is made
    under it.
    """

    __slots__ = ("_lock", "_space")

    def __init__(self) -> None:
        self._lock = RLock()
        self._space: DesignSpace | None = None

    def design_space(self, specification: DesignSpaceSpec) -> DesignSpace:
        with self._lock:
            if self._space is None:
                # ``Engine.validate`` is a pure function of the specification, so
                # the transient engine here validates without becoming the one
                # the lineages evaluate through.  Reaching into ``_engine`` for
                # the underlying function would be the only alternative, and U1
                # is not allowed to widen that surface.
                self._space = Engine().validate(specification)
            return self._space


def _occurrence_api() -> Any:
    """The occurrence runtime, resolved once; see ``declarations._occurrence_api``."""

    from finn.dataflow.model.declarations import _occurrence_api as resolve  # noqa: PLC0415

    return resolve()


def _structural_signature(
    space_type: type[Space],
    seen: frozenset[type[Space]] = frozenset(),
) -> tuple[object, ...]:
    """A hashable witness of everything compilation reads from a class tree.

    Caching a compiled model on class identity alone is wrong: a class body is
    ordinary mutable Python, and a test or a plugin that rebinds one member
    would silently keep answering with stale compiled metadata.  The signature
    therefore carries the declaration *objects* themselves -- they hash by
    identity and, being held by the cache key, cannot have their ids recycled --
    together with the member names and the classes referenced by every ``Use``
    and ``Case``.  Rebinding, adding, removing or reordering a declaration
    anywhere in the tree changes the key, so the next request recompiles instead
    of reusing.
    """

    if space_type in seen:
        return (space_type, "<recursive>")
    members = declared_members(space_type)
    parts: list[object] = [space_type, tuple((name, declaration) for name, declaration in members)]
    nested = seen | {space_type}
    for _name, declaration in members:
        if isinstance(declaration, Use):
            parts.append(_structural_signature(declaration.space_type, nested))
        elif isinstance(declaration, OneOf):
            for case in declaration.cases:
                parts.append(_structural_signature(case.space_type, nested))
    return tuple(parts)


def compiled_model_for(
    space_type: type[Space],
    namespace: str,
    *,
    problem_namespace: str | None = None,
) -> SpaceModel:
    """One compiled model per (class, namespaces, declaration structure).

    The cache lives on the authored class rather than in a module-level table so
    that it dies with the class -- a process that builds Spaces dynamically must
    not accumulate compiled models forever.  ``__dict__`` rather than attribute
    lookup, so a subclass never reuses its parent's entries.
    """

    key = (namespace, problem_namespace, _structural_signature(space_type))
    cache = space_type.__dict__.get("_space_compiled_models")
    if not isinstance(cache, dict):
        cache = {}
        setattr(space_type, "_space_compiled_models", cache)
    cached = cache.get(key)
    if cached is not None:
        return cast(SpaceModel, cached)
    compiled = _compile_space(
        space_type,
        namespace,
        problem_namespace=problem_namespace,
    )
    model = SpaceModel(compiled.spec, compiled.catalog, compiled)
    cache[key] = model
    return model


def compile_space_model(
    space_type: type[Space],
    namespace: str,
    *,
    problem_namespace: str | None = None,
) -> SpaceModel:
    """Compile a closed root Space into its spec and its branch catalog."""

    return compiled_model_for(
        space_type,
        namespace,
        problem_namespace=problem_namespace,
    )


def compile_space(
    space_type: type[Space],
    namespace: str,
    *,
    problem_namespace: str | None = None,
) -> DesignSpaceSpec:
    """Compile a closed root Space into the ordinary engine specification."""

    return compile_space_model(
        space_type,
        namespace,
        problem_namespace=problem_namespace,
    ).specification


__all__ = [
    "SpaceModel",
    "answer_for",
    "resolve_value_source",
    "compile_space",
    "compile_space_model",
    "compiled_model_for",
    "imported_decisions",
]
