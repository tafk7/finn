# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Class-centered occurrence views over the private design-space runtime.

An authored :class:`~finn.dataflow.model.Space` subclass is both the family
declaration and the public type of each occurrence.  The objects constructed
here deliberately contain no public Engine, point, reference, or path access.
One root owns those capabilities privately; children are exact namespace-bound
views that can only name declarations in their own authored scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from types import MappingProxyType
from typing import TypeVar, cast

from finn.dataflow._engine import (
    Absent,
    Answer,
    ConstraintAssessment,
    Decided,
    DesignPoint,
    Engine,
    Finding,
    ItemOutcome,
    QualifiedPath,
    ReadinessAssessment,
    Unresolved,
)
from finn.dataflow.model.compiler import (
    _CompiledBranch,
    _CompiledSpace,
    _Ref,
    _compile_space,
    answer_for,
    resolve_value_source,
)
from finn.dataflow.model.declarations import (
    Case,
    Constraint,
    ConstraintGroup,
    Decision,
    OneOf,
    Problem,
    Readiness,
    Space,
    Use,
    ValueSource,
    declared_members,
)

S = TypeVar("S", bound=Space)
T = TypeVar("T")

ProblemSource = Mapping[object, object] | Callable[[], Mapping[object, object]]


class OccurrenceError(ValueError):
    """A declaration cannot be used through this occurrence capability."""

    def __init__(self, message: str, findings: tuple[Finding, ...] = ()) -> None:
        super().__init__(message)
        self.findings = findings


class _RootRuntime:
    """The only owner of compiled records, Engine, point, and synchronization."""

    def __init__(
        self,
        compiled: _CompiledSpace[Space],
        engine: Engine,
        point: DesignPoint,
        problem_source: ProblemSource,
        problem_snapshot: Mapping[Problem[object], object],
    ) -> None:
        self.compiled = compiled
        self.engine = engine
        self.point = point
        self.problem_source = problem_source
        self.problem_snapshot = MappingProxyType(dict(problem_snapshot))
        self.lock = RLock()

    def successor(self, point: DesignPoint) -> _RootRuntime:
        successor = _RootRuntime.__new__(_RootRuntime)
        successor.compiled = self.compiled
        successor.engine = self.engine
        successor.point = point
        successor.problem_source = self.problem_source
        successor.problem_snapshot = self.problem_snapshot
        successor.lock = self.lock
        return successor


def _occurrence_parts(instance: Space) -> tuple[_RootRuntime, _CompiledSpace[Space], Space]:
    try:
        runtime = cast(_RootRuntime, object.__getattribute__(instance, "_occurrence_runtime"))
        compiled = cast(
            "_CompiledSpace[Space]", object.__getattribute__(instance, "_occurrence_compiled")
        )
        root = cast(Space, object.__getattribute__(instance, "_occurrence_root"))
    except AttributeError:
        raise OccurrenceError(
            f"{type(instance).__name__} is not an attached Space occurrence"
        ) from None
    return runtime, compiled, root


def _make_occurrence(
    runtime: _RootRuntime,
    compiled: _CompiledSpace[S],
    root: Space | None = None,
) -> S:
    instance = object.__new__(compiled.owner)
    object.__setattr__(instance, "_occurrence_runtime", runtime)
    object.__setattr__(instance, "_occurrence_compiled", compiled)
    object.__setattr__(instance, "_occurrence_root", instance if root is None else root)
    return instance


def _problem_members(space_type: type[Space]) -> tuple[tuple[str, Problem[object]], ...]:
    return tuple(
        (name, declaration)
        for name, declaration in declared_members(space_type)
        if isinstance(declaration, Problem)
    )


def _read_problem_source(source: ProblemSource) -> Mapping[object, object]:
    values = source() if callable(source) else source
    if not isinstance(values, Mapping):
        raise OccurrenceError("a Space problem source must produce a mapping")
    return values


def _prepare_problem(
    compiled: _CompiledSpace[Space], source: ProblemSource
) -> tuple[dict[QualifiedPath, object], dict[Problem[object], object]]:
    raw = _read_problem_source(source)
    members = _problem_members(compiled.owner)
    expected = {declaration for _name, declaration in members}
    unknown = tuple(key for key in raw if key not in expected)
    if unknown:
        raise OccurrenceError(
            "a Space problem mapping must use only Problem declarations from the root class"
        )

    by_path: dict[QualifiedPath, object] = {}
    by_declaration: dict[Problem[object], object] = {}
    for name, declaration in members:
        if declaration not in raw:
            if declaration.required:
                raise OccurrenceError(
                    f"required Problem {compiled.owner.__name__}.{name} is absent"
                )
            continue
        reference = compiled.member(name)
        by_path[reference.path] = raw[declaration]
        by_declaration[declaration] = raw[declaration]
    return by_path, by_declaration


def start_occurrence(
    space_type: type[S],
    problem: ProblemSource,
    *,
    namespace: str = "root",
) -> S:
    """Compile and start one root occurrence of the authored class."""

    compiled = _compile_space(
        space_type,
        namespace,
        problem_namespace=f"problem.{namespace}",
    )
    engine = Engine()
    design_space = engine.validate(compiled.spec)
    problem_paths, _problem_declarations = _prepare_problem(
        cast("_CompiledSpace[Space]", compiled), problem
    )
    point = engine.start(design_space, problem_paths)
    frozen = {
        declaration: point.problem[compiled.member(name).path]
        for name, declaration in _problem_members(space_type)
        if compiled.member(name).path in point.problem
    }
    runtime = _RootRuntime(cast("_CompiledSpace[Space]", compiled), engine, point, problem, frozen)
    return _make_occurrence(runtime, compiled)


def occurrence_root(instance: Space) -> Space:
    """Return the root facade for this exact immutable occurrence state."""

    _runtime, _compiled, root = _occurrence_parts(instance)
    return root


def _effective_member_name(
    space_type: type[Space], declaration: object, expected: tuple[type, ...]
) -> str | None:
    found: str | None = None
    for base in reversed(space_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, expected) and value is declaration:
                found = name
    return found


def _decision_reference(
    compiled: _CompiledSpace[Space], declaration: Decision[object]
) -> _Ref[object]:
    name = _effective_member_name(compiled.owner, declaration, (Decision,))
    if name is None:
        raise OccurrenceError(
            f"{compiled.owner.__name__} does not own that Decision; "
            "assign it through the exact child occurrence"
        )
    return compiled.member(name)


def _raise_failed_assignment(outcomes: tuple[ItemOutcome, ...]) -> None:
    failures = tuple(
        item for item in outcomes if item.disposition not in {"committed", "unchanged"}
    )
    if not failures:
        return
    findings = tuple(finding for item in failures for finding in item.findings)
    disposition = ", ".join(item.disposition for item in failures)
    raise OccurrenceError(f"assignment was not accepted ({disposition})", findings)


def _successor_root(runtime: _RootRuntime) -> Space:
    return _make_occurrence(runtime, runtime.compiled)


def occurrence_assign(instance: S, declaration: Decision[T], value: T) -> S:
    """Assign one Decision owned by this exact view and return its successor view."""

    runtime, compiled, _root = _occurrence_parts(instance)
    reference = _decision_reference(compiled, cast("Decision[object]", declaration))
    with runtime.lock:
        committed = runtime.engine.commit_assignments(runtime.point, {reference.path: value})
    _raise_failed_assignment(committed.outcomes)
    successor_runtime = runtime.successor(committed.point)
    successor_root = _successor_root(successor_runtime)
    if compiled is runtime.compiled:
        return cast(S, successor_root)
    return _make_occurrence(successor_runtime, cast("_CompiledSpace[S]", compiled), successor_root)


def occurrence_answer(instance: Space, declaration: ValueSource[T]) -> Answer[T]:
    """Answer one declaration that this view is allowed to name."""

    runtime, compiled, _root = _occurrence_parts(instance)
    try:
        reference = resolve_value_source(
            compiled,
            cast("ValueSource[object]", declaration),
            "occurrence query",
        )
    except ValueError as error:
        raise OccurrenceError(str(error)) from error
    with runtime.lock:
        answer = answer_for(runtime.engine, runtime.point, reference)
    return cast("Answer[T]", answer)


def occurrence_value(instance: Space, declaration: ValueSource[T]) -> T:
    """Descriptor support for a declaration whose answer is already decided."""

    answer = occurrence_answer(instance, declaration)
    if isinstance(answer, Decided):
        return answer.value
    if isinstance(answer, Absent):
        raise OccurrenceError("the declaration is not applicable", answer.findings)
    raise OccurrenceError("the declaration is unresolved", answer.findings)


def occurrence_assess(
    instance: Space,
    declaration: Readiness | ConstraintGroup | Constraint,
) -> ReadinessAssessment | ConstraintAssessment:
    """Assess one readiness or constraint declaration owned by this view."""

    runtime, compiled, _root = _occurrence_parts(instance)
    if isinstance(declaration, Readiness):
        name = _effective_member_name(compiled.owner, declaration, (Readiness,))
        if name is None:
            raise OccurrenceError(
                f"{compiled.owner.__name__} does not own that Readiness declaration"
            )
        profile = f"{compiled.namespace}.{declaration.stable_name or name}"
        with runtime.lock:
            return runtime.engine.check_readiness(runtime.point, profile)
    if isinstance(declaration, ConstraintGroup):
        name = _effective_member_name(compiled.owner, declaration, (ConstraintGroup,))
        if name is None:
            raise OccurrenceError(f"{compiled.owner.__name__} does not own that ConstraintGroup")
        group = f"{compiled.namespace}.{declaration.stable_name or name}"
        with runtime.lock:
            return runtime.engine.evaluate_constraint_set(runtime.point, group)
    if isinstance(declaration, Constraint):
        name = _effective_member_name(compiled.owner, declaration, (Constraint,))
        if name is None:
            raise OccurrenceError(f"{compiled.owner.__name__} does not own that Constraint")
        path = f"constraint.{compiled.namespace}.{declaration.stable_name or name}"
        with runtime.lock:
            return runtime.engine.evaluate_constraints(runtime.point, (path,))
    raise TypeError("assess() requires a Readiness, ConstraintGroup, or Constraint declaration")


def _direct_children(compiled: _CompiledSpace[Space]) -> tuple[_CompiledSpace[Space], ...]:
    return (
        *(child for _name, child in compiled.children),
        *(case.compiled for _name, branch in compiled.branches for case in branch.cases),
    )


def _child_for_declaration(
    compiled: _CompiledSpace[Space], declaration: Use[Space] | Case
) -> _CompiledSpace[Space]:
    if isinstance(declaration, Use):
        name = _effective_member_name(compiled.owner, declaration, (Use,))
        if name is None:
            raise OccurrenceError(f"{compiled.owner.__name__} does not own that Use")
        return compiled.child(name)

    found = tuple(
        candidate.compiled
        for authored_branch_name, authored_branch in declared_members(compiled.owner)
        if isinstance(authored_branch, OneOf)
        for authored_case, candidate in zip(
            authored_branch.cases,
            compiled.branch(authored_branch_name).cases,
        )
        if authored_case is declaration
    )
    if len(found) != 1:
        raise OccurrenceError(f"{compiled.owner.__name__} does not own that Case")
    return found[0]


def occurrence_child(
    instance: Space,
    declaration: Use[S] | Case | type[S],
) -> S:
    """Return the exact direct child occurrence selected by a use site or case."""

    runtime, compiled, root = _occurrence_parts(instance)
    if isinstance(declaration, type) and issubclass(declaration, Space):
        matches = tuple(child for child in _direct_children(compiled) if child.owner is declaration)
        if not matches:
            raise OccurrenceError(
                f"{compiled.owner.__name__} has no direct child of class {declaration.__name__}"
            )
        if len(matches) != 1:
            raise OccurrenceError(
                f"{compiled.owner.__name__} has {len(matches)} direct occurrences of "
                f"{declaration.__name__}; name the exact Use or Case"
            )
        child = matches[0]
    elif isinstance(declaration, (Use, Case)):
        child = _child_for_declaration(compiled, cast("Use[Space] | Case", declaration))
    else:
        raise TypeError("child() requires a Use, Case, or Space subclass")
    return _make_occurrence(runtime, cast("_CompiledSpace[S]", child), root)


def _branch_for_declaration(compiled: _CompiledSpace[Space], declaration: OneOf) -> _CompiledBranch:
    name = _effective_member_name(compiled.owner, declaration, (OneOf,))
    if name is None:
        raise OccurrenceError(f"{compiled.owner.__name__} does not own that OneOf")
    return compiled.branch(name)


class BranchView:
    """Capability-limited inspection and selection for one exact ``OneOf``."""

    __slots__ = ("__branch", "__owner")

    def __init__(self, owner: Space, branch: _CompiledBranch) -> None:
        self.__owner = owner
        self.__branch = branch

    @property
    def name(self) -> str:
        return self.__branch.member_name

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self.__branch.cases)

    @property
    def root(self) -> Space:
        return occurrence_root(self.__owner)

    def selected(self) -> Answer[str]:
        runtime, _compiled, _root = _occurrence_parts(self.__owner)
        active = self.__branch.active
        if active is not None:
            with runtime.lock:
                active_answer = answer_for(runtime.engine, runtime.point, active)
            if isinstance(active_answer, Unresolved):
                return active_answer
            if isinstance(active_answer, Absent) or not cast(bool, active_answer.value):
                return Absent()
        if self.__branch.selector is None:
            return Decided(self.__branch.cases[0].case_id)
        with runtime.lock:
            answer = answer_for(runtime.engine, runtime.point, self.__branch.selector)
        return cast("Answer[str]", answer)

    def case(self, case_id: str) -> Space:
        try:
            child = self.__branch.case(case_id).compiled
        except ValueError as error:
            raise OccurrenceError(str(error)) from error
        runtime, _compiled, root = _occurrence_parts(self.__owner)
        return _make_occurrence(runtime, child, root)

    def select(self, case_id: str) -> BranchView:
        if case_id not in self.cases:
            raise OccurrenceError(
                f"branch {self.name!r} has no case {case_id!r}; expected one of {self.cases}"
            )
        if self.__branch.selector is None:
            return self
        runtime, compiled, _root = _occurrence_parts(self.__owner)
        with runtime.lock:
            committed = runtime.engine.commit_assignments(
                runtime.point, {self.__branch.selector.path: case_id}
            )
        _raise_failed_assignment(committed.outcomes)
        successor_runtime = runtime.successor(committed.point)
        successor_root = _successor_root(successor_runtime)
        successor_owner = (
            successor_root
            if compiled is runtime.compiled
            else _make_occurrence(successor_runtime, compiled, successor_root)
        )
        return BranchView(successor_owner, self.__branch)


def occurrence_branch(instance: Space, declaration: OneOf) -> BranchView:
    """Bind one authored branch to this exact occurrence namespace."""

    _runtime, compiled, _root = _occurrence_parts(instance)
    return BranchView(instance, _branch_for_declaration(compiled, declaration))


__all__ = ["BranchView", "OccurrenceError", "ProblemSource", "start_occurrence"]
