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

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from threading import RLock
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

from finn.dataflow._engine import (
    Absent,
    Answer,
    ConstraintAssessment,
    Decided,
    DesignPoint,
    Engine,
    Finding,
    FindingKind,
    ItemOutcome,
    QualifiedPath,
    ReadinessAssessment,
    RequestError,
    Unresolved,
)
from finn.dataflow._engine.results import ordered_findings
from finn.dataflow.model.compiler import (
    _CompiledBranch,
    _CompiledProjection,
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
    Projection,
    Readiness,
    Space,
    Use,
    ValueSource,
    declared_members,
)

S = TypeVar("S", bound=Space)
T = TypeVar("T")

ProblemSource = Mapping[Any, object] | Callable[[], Mapping[Any, object]]


class OccurrenceError(ValueError):
    """A declaration cannot be used through this occurrence capability."""

    def __init__(self, message: str, findings: tuple[Finding, ...] = ()) -> None:
        super().__init__(message)
        self.findings = findings


@dataclass(frozen=True, slots=True)
class ProjectionAssessment(Generic[T]):
    """Readiness, validation, raw output, and accepted projection answer."""

    readiness: ReadinessAssessment
    constraints: tuple[ConstraintAssessment, ...]
    output: Answer[T]
    accepted_answer: Answer[T]


@dataclass(frozen=True, slots=True)
class OccurrenceDiagnostic:
    """One engine finding interpreted in declaration and occurrence vocabulary."""

    finding: Finding
    scope: tuple[str, ...]
    declaration: str
    projection: str | None = None

    def render(self) -> str:
        location = " / ".join((*self.scope, self.declaration))
        requested = "" if self.projection is None else f" [{self.projection}]"
        return f"{location}{requested}: {self.finding.message} ({self.finding.code})"


class _RootRuntime:
    """The only owner of compiled records, Engine, point, and synchronization."""

    def __init__(
        self,
        compiled: _CompiledSpace[Space],
        engine: Engine,
        point: DesignPoint,
        problem_source: ProblemSource,
        problem_snapshot: Mapping[Problem[object], object],
        problem_fingerprint: str,
    ) -> None:
        self.compiled = compiled
        self.engine = engine
        self.point = point
        self.problem_source = problem_source
        self.problem_snapshot = MappingProxyType(dict(problem_snapshot))
        self.problem_fingerprint = problem_fingerprint
        self.lock = RLock()

    def successor(self, point: DesignPoint) -> _RootRuntime:
        successor = _RootRuntime.__new__(_RootRuntime)
        successor.compiled = self.compiled
        successor.engine = self.engine
        successor.point = point
        successor.problem_source = self.problem_source
        successor.problem_snapshot = self.problem_snapshot
        successor.problem_fingerprint = self.problem_fingerprint
        successor.lock = self.lock
        return successor


def _occurrence_parts(instance: Space) -> tuple[_RootRuntime, _CompiledSpace[Space], Space]:
    try:
        compiled = cast(
            "_CompiledSpace[Space]", object.__getattribute__(instance, "_occurrence_compiled")
        )
        root = cast(Space, object.__getattribute__(instance, "_occurrence_root"))
        runtime = cast(_RootRuntime, object.__getattribute__(root, "_occurrence_runtime"))
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
    object.__setattr__(instance, "_occurrence_compiled", compiled)
    object.__setattr__(instance, "_occurrence_root", instance if root is None else root)
    if root is None:
        object.__setattr__(instance, "_occurrence_runtime", runtime)
    return instance


def _problem_members(space_type: type[Space]) -> tuple[tuple[str, Problem[object]], ...]:
    return tuple(
        (name, declaration)
        for name, declaration in declared_members(space_type)
        if isinstance(declaration, Problem)
    )


def _read_problem_source(source: ProblemSource) -> Mapping[Any, object]:
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


def _type_token(kind: type[object]) -> str:
    token = getattr(kind, "__dataflow_identity_token__", f"{kind.__module__}.{kind.__qualname__}")
    if not isinstance(token, str) or not token:
        raise OccurrenceError(f"{kind.__name__} has no stable dataflow identity token")
    return token


def _canonical_problem_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise OccurrenceError("problem fingerprints require finite floats")
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, QualifiedPath):
        return {"qualified_path": value.value}
    if isinstance(value, Enum):
        return {
            "enum_type": _type_token(type(value)),
            "value": _canonical_problem_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass_type": _type_token(type(value)),
            "fields": [
                [field.name, _canonical_problem_value(getattr(value, field.name))]
                for field in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        pairs = [
            [_canonical_problem_value(key), _canonical_problem_value(item)]
            for key, item in value.items()
        ]
        return {"mapping": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return {"sequence": [_canonical_problem_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        members = [_canonical_problem_value(item) for item in value]
        return {"set": sorted(members, key=lambda item: json.dumps(item, sort_keys=True))}
    raise OccurrenceError(
        f"unsupported problem fingerprint value {type(value).__module__}.{type(value).__qualname__}"
    )


def _problem_fingerprint(
    space_type: type[Space], snapshot: Mapping[Problem[object], object]
) -> str:
    payload = {
        "space": _type_token(space_type),
        "problem": [
            {
                "name": declaration.stable_name or name,
                "semantics": declaration.value_semantics.name,
                "value": (
                    _canonical_problem_value(snapshot[declaration])
                    if declaration in snapshot
                    else {"absent": True}
                ),
            }
            for name, declaration in _problem_members(space_type)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def start_occurrence(
    space_type: type[S],
    problem: ProblemSource,
    *,
    namespace: str = "root",
    expected_problem_fingerprint: str | None = None,
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
    try:
        point = engine.start(design_space, problem_paths)
    except RequestError as error:
        raise OccurrenceError("the Space problem was rejected", error.findings) from error
    frozen = {
        declaration: point.problem[compiled.member(name).path]
        for name, declaration in _problem_members(space_type)
        if compiled.member(name).path in point.problem
    }
    fingerprint = _problem_fingerprint(space_type, frozen)
    if expected_problem_fingerprint is not None and fingerprint != expected_problem_fingerprint:
        raise OccurrenceError(
            "persisted assignments were recorded for an incompatible problem fingerprint"
        )
    runtime = _RootRuntime(
        cast("_CompiledSpace[Space]", compiled),
        engine,
        point,
        problem,
        frozen,
        fingerprint,
    )
    return _make_occurrence(runtime, compiled)


def occurrence_root(instance: Space) -> Space:
    """Return the root facade for this exact immutable occurrence state."""

    _runtime, _compiled, root = _occurrence_parts(instance)
    return root


def occurrence_problem_snapshot(instance: Space) -> Mapping[Problem[object], object]:
    """Return the immutable declaration-keyed Problem snapshot."""

    runtime, _compiled, _root = _occurrence_parts(instance)
    return runtime.problem_snapshot


def occurrence_problem_fingerprint(instance: Space) -> str:
    """Return the stable identity of this root's authored Problem facts."""

    runtime, _compiled, _root = _occurrence_parts(instance)
    return runtime.problem_fingerprint


def _snapshot_problem(
    runtime: _RootRuntime, source: ProblemSource
) -> tuple[Mapping[Problem[object], object], str]:
    problem_paths, _raw = _prepare_problem(runtime.compiled, source)
    with runtime.lock:
        try:
            point = runtime.engine.start(runtime.point.design_space, problem_paths)
        except RequestError as error:
            raise OccurrenceError("the Space problem was rejected", error.findings) from error
    frozen = {
        declaration: point.problem[runtime.compiled.member(name).path]
        for name, declaration in _problem_members(runtime.compiled.owner)
        if runtime.compiled.member(name).path in point.problem
    }
    return MappingProxyType(frozen), _problem_fingerprint(runtime.compiled.owner, frozen)


def occurrence_is_stale(instance: Space, problem: ProblemSource | None = None) -> bool:
    """Explicitly compare current declared Problem facts with the frozen snapshot."""

    runtime, _compiled, _root = _occurrence_parts(instance)
    source = runtime.problem_source if problem is None else problem
    _snapshot, fingerprint = _snapshot_problem(runtime, source)
    return fingerprint != runtime.problem_fingerprint


def _find_compiled(compiled: _CompiledSpace[Space], namespace: str) -> _CompiledSpace[Space] | None:
    if compiled.namespace == namespace:
        return compiled
    for child in _direct_children(compiled):
        found = _find_compiled(child, namespace)
        if found is not None:
            return found
    return None


def occurrence_reconstruct(
    instance: S,
    problem: ProblemSource | None = None,
    *,
    expected_problem_fingerprint: str | None = None,
) -> S:
    """Strictly construct a fresh root lineage, retaining no old assignments."""

    runtime, compiled, _root = _occurrence_parts(instance)
    source = runtime.problem_source if problem is None else problem
    fresh_root = start_occurrence(
        runtime.compiled.owner,
        source,
        namespace=runtime.compiled.namespace,
        expected_problem_fingerprint=expected_problem_fingerprint,
    )
    fresh_runtime, fresh_compiled, _fresh_root = _occurrence_parts(fresh_root)
    if compiled.namespace == runtime.compiled.namespace:
        return cast(S, fresh_root)
    rebound = _find_compiled(fresh_compiled, compiled.namespace)
    if rebound is None:
        raise OccurrenceError(
            f"fresh compilation no longer contains occurrence {compiled.namespace!r}"
        )
    return _make_occurrence(fresh_runtime, cast("_CompiledSpace[S]", rebound), fresh_root)


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
        try:
            committed = runtime.engine.commit_assignments(runtime.point, {reference.path: value})
        except RequestError as error:
            raise OccurrenceError("the assignment request was rejected", error.findings) from error
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


def _projection_for_declaration(
    compiled: _CompiledSpace[Space], declaration: Projection[object]
) -> _CompiledProjection[object]:
    name = _effective_member_name(compiled.owner, declaration, (Projection,))
    if name is None:
        raise OccurrenceError(f"{compiled.owner.__name__} does not own that Projection")
    return compiled.projection(name)


def _assessment_findings(
    readiness: ReadinessAssessment,
    constraints: tuple[ConstraintAssessment, ...],
    output: Answer[object],
) -> tuple[Finding, ...]:
    answers = (
        *readiness.answers.values(),
        *(answer for assessment in constraints for answer in assessment.answers.values()),
        output,
    )
    unique = {
        finding: None
        for answer in answers
        if isinstance(answer, (Absent, Unresolved))
        for finding in answer.findings
    }
    return ordered_findings(list(unique))


def _reduce_projection(
    compiled: _CompiledProjection[object],
    readiness: ReadinessAssessment,
    constraints: tuple[ConstraintAssessment, ...],
    output: Answer[object],
) -> Answer[object]:
    findings = _assessment_findings(readiness, constraints, output)
    any_unresolved = (
        readiness.ready is None
        or isinstance(output, Unresolved)
        or any(assessment.verdict is None for assessment in constraints)
    )
    if any_unresolved:
        return Unresolved(
            findings
            or (
                Finding(
                    FindingKind.BLOCKER,
                    "projection-not-ready",
                    compiled.output.path,
                    f"projection {compiled.name!r} is not ready",
                ),
            )
        )

    refused: list[Finding] = [
        finding for finding in findings if finding.kind is FindingKind.REJECTION
    ]
    for assessment in constraints:
        for path, answer in assessment.answers.items():
            if isinstance(answer, Decided) and answer.value is False:
                refused.append(
                    Finding(
                        FindingKind.REJECTION,
                        "projection-constraint-rejected",
                        path,
                        f"constraint rejected projection {compiled.name!r}",
                    )
                )
    if any(assessment.verdict is False for assessment in constraints):
        return Absent(ordered_findings(refused))
    if isinstance(output, Absent):
        return Absent(findings)
    assert isinstance(output, Decided)
    return Decided(compiled.output.semantics.freeze(output.value))


def occurrence_project(instance: Space, declaration: Projection[T]) -> ProjectionAssessment[T]:
    """Evaluate a Projection without exposing its bound runtime handles."""

    runtime, compiled_space, _root = _occurrence_parts(instance)
    compiled = _projection_for_declaration(compiled_space, cast("Projection[object]", declaration))
    with runtime.lock:
        readiness = runtime.engine.check_readiness(runtime.point, compiled.readiness_profile)
        output = answer_for(runtime.engine, runtime.point, compiled.output)
        constraints = tuple(
            runtime.engine.evaluate_constraint_set(runtime.point, name)
            for name in compiled.constraint_sets
        )
        accepted = _reduce_projection(compiled, readiness, constraints, output)
    return ProjectionAssessment(
        readiness,
        constraints,
        cast("Answer[T]", output),
        cast("Answer[T]", accepted),
    )


def _scope_entries(
    compiled: _CompiledSpace[Space],
    chain: tuple[str, ...] | None = None,
) -> tuple[tuple[_CompiledSpace[Space], tuple[str, ...]], ...]:
    here = chain or (f"{compiled.owner.__name__} {compiled.namespace}",)
    entries: list[tuple[_CompiledSpace[Space], tuple[str, ...]]] = [(compiled, here)]
    for member_name, child in compiled.children:
        entries.extend(_scope_entries(child, (*here, f"{member_name}: {child.owner.__name__}")))
    for _member_name, branch in compiled.branches:
        for case in branch.cases:
            entries.extend(
                _scope_entries(
                    case.compiled,
                    (
                        *here,
                        f"{branch.member_name}[{case.case_id}]: {case.compiled.owner.__name__}",
                    ),
                )
            )
    return tuple(entries)


def _semantic_path(path: QualifiedPath) -> str:
    for prefix in ("semantic.", "constraint.", "problem."):
        if path.value.startswith(prefix):
            return path.value[len(prefix) :]
    return path.value


def _scope_for_finding(
    root: _CompiledSpace[Space], finding: Finding
) -> tuple[_CompiledSpace[Space], tuple[str, ...]]:
    semantic = _semantic_path(finding.path)
    candidates = tuple(
        item
        for item in _scope_entries(root)
        if semantic == item[0].namespace or semantic.startswith(f"{item[0].namespace}.")
    )
    return (
        max(candidates, key=lambda item: len(item[0].namespace))
        if candidates
        else (
            root,
            (f"{root.owner.__name__} {root.namespace}",),
        )
    )


def _declaration_for_finding(compiled: _CompiledSpace[Space], finding: Finding) -> str:
    for name, declaration in declared_members(compiled.owner):
        if isinstance(declaration, ValueSource):
            try:
                if resolve_value_source(compiled, declaration, "diagnostic").path == finding.path:
                    return f"{compiled.owner.__name__}.{name}"
            except ValueError:
                continue
        if isinstance(declaration, Constraint):
            local = declaration.stable_name or name
            if finding.path.value == f"constraint.{compiled.namespace}.{local}":
                return f"{compiled.owner.__name__}.{name}"
        if isinstance(declaration, OneOf):
            branch = compiled.branch(name)
            if branch.selector is not None and branch.selector.path == finding.path:
                return f"{compiled.owner.__name__}.{name}"
            for output_name, output in branch.outputs:
                if output.path == finding.path:
                    return f"{compiled.owner.__name__}.{name}.{output_name}"
    return f"{compiled.owner.__name__}.{finding.path.value.rsplit('.', 1)[-1]}"


def _subject_findings(subject: object) -> tuple[Finding, ...]:
    if isinstance(subject, OccurrenceError):
        return ordered_findings(list(subject.findings))
    if isinstance(subject, (Absent, Unresolved)):
        return subject.findings
    if isinstance(subject, Decided):
        return ()
    answers: list[object] = []
    if isinstance(subject, ReadinessAssessment):
        answers.extend(subject.answers.values())
    elif isinstance(subject, ConstraintAssessment):
        answers.extend(subject.answers.values())
    elif isinstance(subject, ProjectionAssessment):
        answers.extend(subject.readiness.answers.values())
        for assessment in subject.constraints:
            answers.extend(assessment.answers.values())
        answers.extend((subject.output, subject.accepted_answer))
    else:
        raise TypeError("diagnostics() requires an Answer, assessment, or OccurrenceError")
    unique = {
        finding: None
        for answer in answers
        if isinstance(answer, (Absent, Unresolved))
        for finding in answer.findings
    }
    return ordered_findings(list(unique))


def occurrence_diagnostics(
    instance: Space,
    subject: object,
    *,
    projection: Projection[object] | None = None,
) -> tuple[OccurrenceDiagnostic, ...]:
    """Interpret findings without losing their original paths or causal traces."""

    runtime, compiled, _root = _occurrence_parts(instance)
    projection_name: str | None = None
    if projection is not None:
        projection_name = _projection_for_declaration(compiled, projection).member_name
    interpreted: list[OccurrenceDiagnostic] = []
    for finding in _subject_findings(subject):
        owner, chain = _scope_for_finding(runtime.compiled, finding)
        interpreted.append(
            OccurrenceDiagnostic(
                finding,
                chain,
                _declaration_for_finding(owner, finding),
                projection_name,
            )
        )
    return tuple(interpreted)


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
            try:
                committed = runtime.engine.commit_assignments(
                    runtime.point, {self.__branch.selector.path: case_id}
                )
            except RequestError as error:
                raise OccurrenceError(
                    "the branch selection request was rejected", error.findings
                ) from error
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


__all__ = [
    "BranchView",
    "OccurrenceDiagnostic",
    "OccurrenceError",
    "ProblemSource",
    "ProjectionAssessment",
    "start_occurrence",
]
