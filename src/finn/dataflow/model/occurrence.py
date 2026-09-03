# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Class-centered occurrence views over the private design-space runtime.

An authored :class:`~finn.dataflow.model.Space` subclass is both the family
declaration and the public type of each occurrence.  The objects constructed
here deliberately contain no public Engine, point, reference, or path access.
One root owns those capabilities privately; children are exact namespace-bound
views that can only name declarations in their own authored scope.

**Two error kinds, one rule.**  A mistake about *declarations* -- naming
something outside the view's scope, a malformed Projection, a Problem value with
no canonical encoding -- is an :class:`AuthoringError`, because the caller's
source is wrong.  A mistake about *state* -- refused Problem data, a rejected
assignment, an unselected branch, a fingerprint belonging to another problem --
is the engine's existing ``RequestError`` carrying findings, because the
caller's source is fine and the point is not where they thought.  There is no
third public spelling of those two ideas.

**The capability guarantee is the public surface, not Python reflection.**  No
public method or property returns an ``Engine``, a ``DesignPoint``, a ``_Ref``,
a compiled record, or an unrestricted path lookup, and contributor callbacks
receive resolved declared values only.  Underscore-private attributes remain
inspectable by deliberately hostile code; this is a normal Python privacy
boundary, not a sandbox, and is not claimed to be one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
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
from finn.dataflow._engine.requests import request_finding
from finn.dataflow._engine.results import ordered_findings
from finn.dataflow.model.compiler import (
    SpaceModel,
    _CompiledBranch,
    _CompiledProjection,
    _CompiledSpace,
    _members_of,
    _Ref,
    answer_for,
    compiled_model_for,
    resolve_value_source,
)
from finn.dataflow.model.declarations import (
    DECLARATION_TYPES,
    AuthoringError,
    Case,
    Constraint,
    ConstraintGroup,
    Decision,
    OccurrenceContext,
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


class _NotAttached(RuntimeError):
    """A lifecycle operation reached an instance with no occurrence runtime.

    Deliberately private and deliberately neither of the two public categories:
    it is not a malformed declaration and it is not a refused request, it is a
    caller holding an object that was never started.  Keeping it unexported
    stops it becoming a third public spelling of authoring-versus-request.
    """


def _request_error(code: str, message: str, path: QualifiedPath) -> RequestError:
    return RequestError((request_finding(code, message, path),))


@dataclass(frozen=True, slots=True)
class ProjectionAssessment(Generic[T]):
    """Readiness, constraint acceptance, and output availability, kept apart.

    ``accepted_answer`` is the reduction a caller normally wants.  The other
    three are why it must not be the only thing exposed: an occurrence can be
    perfectly *ready* -- every obligation final, nothing left to decide -- and
    still be refused, because a constraint answered a final ``False``.
    Collapsing them into one Boolean is exactly what lets a rejected point be
    handed on as though it were merely incomplete.

    ``output`` is the raw answer before any constraint had a say, so a
    diagnostic can distinguish "the Region resolved and the width constraint
    refused it" from "the Region did not resolve".

    ``projection`` is the compiled authored name.  It is diagnostic metadata,
    not a path capability: it exists so an assessment can say which projection
    it is instead of forcing every caller to thread that string alongside it.
    """

    projection: str
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


@dataclass(frozen=True, slots=True)
class _Lineage:
    """Everything one root and all its successors privately share.

    The split from :class:`SpaceModel` is the point.  The *model* holds what is
    immutable and worth reusing -- the compiled tree, the validated
    ``DesignSpace``, the branch catalog and the projection metadata -- and is
    shared by every root of that family.  The *lineage* holds what must not be
    shared: one ``Engine``, whose evaluation caches belong to this root alone,
    the frozen problem it was started from, and the one lock that serializes
    engine calls for this root and its successors.  Two independent roots
    therefore reuse compilation without ever serializing on each other or
    reading each other's caches.
    """

    model: SpaceModel
    tree: _CompiledSpace[Space]
    engine: Engine
    problem_source: ProblemSource
    problem_snapshot: Mapping[Problem[object], object]
    problem_fingerprint: str
    lock: RLock


@dataclass(frozen=True, slots=True)
class _Runtime:
    """One immutable point inside one lineage.  A successor replaces the point."""

    lineage: _Lineage
    point: DesignPoint

    def successor(self, point: DesignPoint) -> _Runtime:
        return _Runtime(self.lineage, point)


@dataclass(frozen=True, slots=True)
class _State:
    """The single private attribute an attached occurrence carries."""

    runtime: _Runtime
    compiled: _CompiledSpace[Space]
    scope: tuple[str, ...]
    root: Space


#: The one attribute name that means "this instance is an attached occurrence".
#: One name, checked in one place, so descriptor resolution never has to guess
#: which of two protocols an instance is speaking.
_STATE_ATTRIBUTE = "_occurrence_state"


def is_attached_occurrence(instance: object) -> bool:
    """Whether this instance carries an attached occurrence runtime."""

    return isinstance(getattr(instance, _STATE_ATTRIBUTE, None), _State)


def _occurrence_state(instance: Space) -> _State:
    state = getattr(instance, _STATE_ATTRIBUTE, None)
    if not isinstance(state, _State):
        raise _NotAttached(
            f"{type(instance).__name__} is not an attached Space occurrence; start one with "
            f"{type(instance).__name__}.start(...)"
        )
    return state


def _make_occurrence(
    runtime: _Runtime,
    compiled: _CompiledSpace[S],
    scope: tuple[str, ...],
    root: Space | None = None,
) -> S:
    """Allocate one occurrence through the authored class's construction hook.

    The hook exists because ``object.__new__`` as a convention is a promise
    nobody made: it silently skips whatever initialization a subclass needs, and
    the subclass that will need it -- a ``DataflowOp`` around a ``NodeProto`` --
    is the whole reason this layer is being built.  The default still allocates
    without calling ``__init__``, but now that is a documented contract with a
    seam, and the runtime is attached *after* the class has had its say so no
    constructor ever sees the Engine or the point.
    """

    owner = compiled.owner
    context = OccurrenceContext(owner, compiled.namespace, scope, root)
    instance = owner._new_occurrence(context)
    if not isinstance(instance, owner):
        raise AuthoringError(
            f"{owner.__name__}._new_occurrence returned "
            f"{type(instance).__name__}, which is not an instance of {owner.__name__}"
        )
    typed = instance
    object.__setattr__(
        typed,
        _STATE_ATTRIBUTE,
        _State(runtime, cast("_CompiledSpace[Space]", compiled), scope, root or typed),
    )
    return typed


def _problem_members(space_type: type[Space]) -> tuple[tuple[str, Problem[object]], ...]:
    return tuple(
        (name, declaration)
        for name, declaration in declared_members(space_type)
        if isinstance(declaration, Problem)
    )


def _read_problem_source(source: ProblemSource) -> Mapping[Any, object]:
    values = source() if callable(source) else source
    if not isinstance(values, Mapping):
        raise AuthoringError("a Space problem source must produce a mapping")
    return values


def _prepare_problem(
    compiled: _CompiledSpace[Space], source: ProblemSource
) -> dict[QualifiedPath, object]:
    """Key the caller's mapping by declaration and lower it onto problem paths.

    Declaration keys, never path strings.  A contributor writes the very objects
    they declared in the class body; reconstructing a ``QualifiedPath`` is not
    something the authoring surface asks anyone to do.
    """

    raw = _read_problem_source(source)
    members = _problem_members(compiled.owner)
    expected = {declaration for _name, declaration in members}
    unknown = tuple(key for key in raw if key not in expected)
    if unknown:
        raise AuthoringError(
            f"a {compiled.owner.__name__} problem mapping is keyed by that class's own "
            f"Problem declarations; {len(unknown)} key(s) belong to no such declaration"
        )

    by_path: dict[QualifiedPath, object] = {}
    for name, declaration in members:
        if declaration not in raw:
            if declaration.required:
                raise _request_error(
                    "occurrence-problem-incomplete",
                    f"required Problem {compiled.owner.__name__}.{name} is absent",
                    compiled.member(name).path,
                )
            continue
        by_path[compiled.member(name).path] = raw[declaration]
    return by_path


def _problem_fingerprint(
    space_type: type[Space], snapshot: Mapping[Problem[object], object]
) -> str:
    """Digest the frozen problem together with the schema authority reading it.

    Every part of the payload earns its place.  The Space token stops two
    families that happen to share a problem shape from colliding.  The ordered
    member names and value semantics make a renamed or retyped field a different
    problem.  Explicit absence distinguishes "not supplied" from "supplied as
    something that encodes like nothing".  The codec identity and version travel
    with the value, so changing how a type is encoded can never be mistaken for
    the value having changed.
    """

    payload = {
        "space": f"{space_type.__module__}.{space_type.__qualname__}",
        "problem": [
            {
                "name": declaration.stable_name or name,
                "semantics": declaration.value_semantics.name,
                "codec": f"{declaration.canonical.identity}@{declaration.canonical.version}",
                "value": (
                    {"present": declaration.canonical.encode(snapshot[declaration])}
                    if declaration in snapshot
                    else {"absent": True}
                ),
            }
            for name, declaration in _problem_members(space_type)
        ],
    }
    encoded = dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def start_from_model(
    model: SpaceModel,
    problem: ProblemSource,
    *,
    expected_problem_fingerprint: str | None = None,
) -> Space:
    """Freeze one problem into a new root occurrence over a reusable model.

    The compiled model is shared; everything minted here is not.  A fresh
    ``Engine``, a fresh lock and a fresh frozen problem mean this root's
    evaluation caches are its own, which is the whole reason compilation reuse
    can be a cache and concurrency still be per-lineage.
    """

    tree = model._compiled_tree()
    engine = Engine()
    problem_paths = _prepare_problem(tree, problem)
    # A refused problem is already a RequestError with findings; re-labelling it
    # would only lose the engine's own account of why.
    point = engine.start(model._design_space(), problem_paths)
    frozen = _freeze_problem(tree, point)
    fingerprint = _problem_fingerprint(tree.owner, frozen)
    if expected_problem_fingerprint is not None and fingerprint != expected_problem_fingerprint:
        raise RequestError(
            (
                request_finding(
                    "occurrence-problem-fingerprint-mismatch",
                    "recorded state belongs to a different problem and must be reconstructed; "
                    f"expected {expected_problem_fingerprint}, got {fingerprint}",
                    QualifiedPath(tree.namespace),
                ),
            )
        )
    lineage = _Lineage(model, tree, engine, problem, frozen, fingerprint, RLock())
    return _make_occurrence(_Runtime(lineage, point), tree, (tree.namespace,))


def _freeze_problem(
    tree: _CompiledSpace[Space], point: DesignPoint
) -> Mapping[Problem[object], object]:
    return MappingProxyType(
        {
            declaration: point.problem[tree.member(name).path]
            for name, declaration in _problem_members(tree.owner)
            if tree.member(name).path in point.problem
        }
    )


def start_occurrence(
    space_type: type[S],
    problem: ProblemSource,
    *,
    namespace: str = "root",
    expected_problem_fingerprint: str | None = None,
) -> S:
    """Start one root occurrence of the authored class through the model service."""

    model = compiled_model_for(
        space_type,
        namespace,
        problem_namespace=f"problem.{namespace}",
    )
    return cast(
        S,
        start_from_model(
            model,
            problem,
            expected_problem_fingerprint=expected_problem_fingerprint,
        ),
    )


def occurrence_root(instance: Space) -> Space:
    """Return the root facade for this exact immutable occurrence state."""

    return _occurrence_state(instance).root


def occurrence_problem_snapshot(instance: Space) -> Mapping[Problem[object], object]:
    """Return the immutable declaration-keyed Problem snapshot."""

    return _occurrence_state(instance).runtime.lineage.problem_snapshot


def occurrence_problem_fingerprint(instance: Space) -> str:
    """Return the stable identity of this root's authored Problem facts."""

    return _occurrence_state(instance).runtime.lineage.problem_fingerprint


def _project_problem(lineage: _Lineage, source: ProblemSource) -> str:
    """Read the source again and fingerprint it, without disturbing this point."""

    problem_paths = _prepare_problem(lineage.tree, source)
    with lineage.lock:
        point = lineage.engine.start(lineage.model._design_space(), problem_paths)
    return _problem_fingerprint(lineage.tree.owner, _freeze_problem(lineage.tree, point))


def occurrence_is_stale(instance: Space, problem: ProblemSource | None = None) -> bool:
    """Explicitly compare current declared Problem facts with the frozen snapshot."""

    lineage = _occurrence_state(instance).runtime.lineage
    source = lineage.problem_source if problem is None else problem
    return _project_problem(lineage, source) != lineage.problem_fingerprint


def occurrence_reconstruct(
    instance: S,
    problem: ProblemSource | None = None,
    *,
    expected_problem_fingerprint: str | None = None,
) -> S:
    """Strictly construct a fresh root lineage, retaining no old assignments."""

    state = _occurrence_state(instance)
    lineage = state.runtime.lineage
    fresh_root = start_from_model(
        lineage.model,
        lineage.problem_source if problem is None else problem,
        expected_problem_fingerprint=expected_problem_fingerprint,
    )
    if state.compiled is lineage.tree:
        return cast(S, fresh_root)
    fresh = _occurrence_state(fresh_root)
    # The compiled model is shared, so the fresh lineage carries the identical
    # compiled child record; rebinding is a change of runtime, never of identity.
    return _make_occurrence(
        fresh.runtime,
        cast("_CompiledSpace[S]", state.compiled),
        state.scope,
        fresh_root,
    )


def _member_name(
    space_type: type[Space], declaration: object, expected: tuple[type, ...]
) -> str | None:
    """The class-member name this declaration was authored under, if any.

    One index, the compiler's, so the occurrence layer and the lowering can
    never disagree about which member a declaration is.
    """

    return _members_of(space_type, expected).get(id(declaration))


def _occurrences_of(
    compiled: _CompiledSpace[Space], declaration: object
) -> tuple[tuple[str, str], ...]:
    """Every (class, namespace) in the tree whose class declares this object."""

    found: list[tuple[str, str]] = []
    if id(declaration) in _members_of(compiled.owner, DECLARATION_TYPES):
        found.append((compiled.owner.__name__, compiled.namespace))
    for _name, child in _direct_children(compiled):
        found.extend(_occurrences_of(child, declaration))
    return tuple(found)


def _out_of_scope(state: _State, declaration: object, what: str) -> AuthoringError:
    """Refuse a declaration this view does not own, and say where it does live.

    The message is the point.  ``root.assign(SomeKernel.pumping, 2)`` is a
    natural thing to write and a wrong thing to mean the moment that Kernel is
    placed twice, so the refusal names every placement and sends the caller
    through the view for the one they meant.  An occurrence is never inferred
    from a Python class.
    """

    here = f"{state.compiled.owner.__name__} at {state.compiled.namespace}"
    placements = _occurrences_of(state.runtime.lineage.tree, declaration)
    if not placements:
        return AuthoringError(
            f"this {what} declaration is declared by no Space in this model; "
            f"the occurrence asked was {here}"
        )
    where = ", ".join(f"{space} at {namespace}" for space, namespace in placements)
    return AuthoringError(
        f"this {what} declaration is not owned by {here}; it is declared by {where}. "
        "Reach that occurrence with child() and use its own view"
    )


def _owned_member(state: _State, declaration: object, kinds: tuple[type, ...], what: str) -> str:
    name = _member_name(state.compiled.owner, declaration, kinds)
    if name is None:
        raise _out_of_scope(state, declaration, what)
    return name


def _decision_reference(state: _State, declaration: Decision[object]) -> _Ref[object]:
    return state.compiled.member(_owned_member(state, declaration, (Decision,), "Decision"))


def _raise_failed_assignment(outcomes: tuple[ItemOutcome, ...]) -> None:
    failures = tuple(
        item for item in outcomes if item.disposition not in {"committed", "unchanged"}
    )
    if not failures:
        return
    findings = [finding for item in failures for finding in item.findings]
    for item in failures:
        if not item.findings:
            findings.append(
                request_finding(
                    "occurrence-assignment-refused",
                    f"this value was {item.disposition} by the {item.source} check",
                    item.path,
                )
            )
    raise RequestError(findings)


def _successor_root(runtime: _Runtime) -> Space:
    tree = runtime.lineage.tree
    return _make_occurrence(runtime, tree, (tree.namespace,))


def occurrence_assign(instance: S, declaration: Decision[T], value: T) -> S:
    """Assign one Decision owned by this exact view and return its successor view."""

    state = _occurrence_state(instance)
    runtime = state.runtime
    lineage = runtime.lineage
    reference = _decision_reference(state, cast("Decision[object]", declaration))
    with lineage.lock:
        committed = lineage.engine.commit_assignments(runtime.point, {reference.path: value})
    _raise_failed_assignment(committed.outcomes)
    successor = runtime.successor(committed.point)
    successor_root = _successor_root(successor)
    if state.compiled is lineage.tree:
        return cast(S, successor_root)
    return _make_occurrence(
        successor,
        cast("_CompiledSpace[S]", state.compiled),
        state.scope,
        successor_root,
    )


def occurrence_answer(instance: Space, declaration: ValueSource[T]) -> Answer[T]:
    """Answer one declaration that this view is allowed to name."""

    state = _occurrence_state(instance)
    try:
        reference = resolve_value_source(
            state.compiled,
            cast("ValueSource[object]", declaration),
            "occurrence query",
        )
    except AuthoringError:
        raise _out_of_scope(state, declaration, "query") from None
    lineage = state.runtime.lineage
    with lineage.lock:
        answer = answer_for(lineage.engine, state.runtime.point, reference)
    return cast("Answer[T]", answer)


def occurrence_value(instance: Space, declaration: ValueSource[T]) -> T:
    """Descriptor support for a declaration whose answer is already decided."""

    answer = occurrence_answer(instance, declaration)
    if isinstance(answer, Decided):
        return answer.value
    raise RequestError(
        answer.findings
        or (
            request_finding(
                "occurrence-value-unavailable",
                "this declaration has no decided value at this point",
            ),
        )
    )


def occurrence_assess(
    instance: Space,
    declaration: Readiness | ConstraintGroup | Constraint,
) -> ReadinessAssessment | ConstraintAssessment:
    """Assess one readiness or constraint declaration owned by this view."""

    state = _occurrence_state(instance)
    compiled = state.compiled
    lineage = state.runtime.lineage
    # Every name below comes from the compiler's own tables.  Rebuilding
    # ``f"{namespace}.{local}"`` here was a second copy of a convention only the
    # compiler is entitled to state, and a second copy is how a query ends up
    # naming a profile the engine does not have.
    if isinstance(declaration, Readiness):
        name = _owned_member(state, declaration, (Readiness,), "Readiness declaration")
        profile = compiled.engine_name(compiled.readiness_names, name, "Readiness")
        with lineage.lock:
            return lineage.engine.check_readiness(state.runtime.point, profile)
    if isinstance(declaration, ConstraintGroup):
        name = _owned_member(state, declaration, (ConstraintGroup,), "ConstraintGroup")
        group = compiled.engine_name(compiled.constraint_set_names, name, "ConstraintGroup")
        with lineage.lock:
            return lineage.engine.evaluate_constraint_set(state.runtime.point, group)
    if isinstance(declaration, Constraint):
        name = _owned_member(state, declaration, (Constraint,), "Constraint")
        path = dict(compiled.constraint_members)[name]
        with lineage.lock:
            return lineage.engine.evaluate_constraints(state.runtime.point, (path,))
    raise AuthoringError("assess() takes a Readiness, ConstraintGroup, or Constraint declaration")


def _projection_for_declaration(
    state: _State, declaration: Projection[object]
) -> _CompiledProjection[object]:
    name = _owned_member(state, declaration, (Projection,), "Projection")
    return state.compiled.projection(name)


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
    """The normative projection reduction.

    The order is the contract, not an implementation detail.  *Unresolved*
    dominates, because an obligation not yet met is evidence of nothing.  A
    final *inapplicability* comes next and is returned as the output's own
    ``Absent``, so a projection that legitimately does not arise keeps saying so
    in the engine's own vocabulary with the output's own findings -- not with
    every finding the readiness profile happened to collect.  Only then does a
    constraint refusal turn an available value into a rejecting ``Absent``.
    ``Decided`` is exposed last, and only when every obligation is final and
    every constraint accepted.

    Putting refusal before absence would report "something refused this" for a
    point where the value simply does not exist, which is a different and
    misleading sentence.
    """

    owner = QualifiedPath(compiled.name)
    if (
        readiness.ready is None
        or isinstance(output, Unresolved)
        or any(assessment.verdict is None for assessment in constraints)
    ):
        blocked = _assessment_findings(readiness, constraints, output)
        return Unresolved(
            blocked
            or (
                Finding(
                    FindingKind.BLOCKER,
                    "projection-not-ready",
                    owner,
                    f"projection {compiled.name!r} is not ready at this point",
                ),
            )
        )
    if isinstance(output, Absent):
        # The absence policy, applied: propagate final inapplicability exactly
        # as the output declared it.
        return output
    refusals: list[Finding] = []
    for assessment in constraints:
        if assessment.verdict is not False:
            continue
        refusals.extend(_rejection_findings(assessment, owner))
    if refusals:
        return Absent(ordered_findings(refusals))
    assert isinstance(output, Decided)
    # The snapshot policy, applied: the output declaration's own value
    # semantics, which additionally re-check the nominal type on the way out.
    return Decided(compiled.output.semantics.freeze(output.value))


def _rejection_findings(
    assessment: ConstraintAssessment, owner: QualifiedPath
) -> tuple[Finding, ...]:
    """Every reason one constraint set said no, in both of its spellings.

    A ``reject(...)`` carries its own reason.  A bare ``Decided(False)`` carries
    none, so one is synthesized whose trace names the constraint path -- a flat
    refusal must still be attributable to the constraint that made it.
    """

    findings: list[Finding] = []
    for path in assessment.refused:
        answer = assessment.answers[path]
        if isinstance(answer, Absent) and answer.findings:
            findings.extend(answer.findings)
            continue
        findings.append(
            Finding(
                FindingKind.REJECTION,
                "projection-constraint-refused",
                owner,
                "a projection constraint refused this point",
                (("constraint", path),),
                (path,),
            )
        )
    return tuple(findings)


def occurrence_project(instance: Space, declaration: Projection[T]) -> ProjectionAssessment[T]:
    """Evaluate a Projection without exposing its bound runtime handles."""

    state = _occurrence_state(instance)
    lineage = state.runtime.lineage
    point = state.runtime.point
    compiled = _projection_for_declaration(state, cast("Projection[object]", declaration))
    with lineage.lock:
        readiness = lineage.engine.check_readiness(point, compiled.readiness_profile)
        output = answer_for(lineage.engine, point, compiled.output)
        constraints = tuple(
            lineage.engine.evaluate_constraint_set(point, name) for name in compiled.constraint_sets
        )
        accepted = _reduce_projection(compiled, readiness, constraints, output)
    return ProjectionAssessment(
        compiled.name,
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
    if isinstance(subject, RequestError):
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
        raise AuthoringError("diagnostics() takes an Answer, an assessment, or a RequestError")
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

    state = _occurrence_state(instance)
    projection_name: str | None = None
    if projection is not None:
        projection_name = _projection_for_declaration(state, projection).member_name
    interpreted: list[OccurrenceDiagnostic] = []
    for finding in _subject_findings(subject):
        owner, chain = _scope_for_finding(state.runtime.lineage.tree, finding)
        interpreted.append(
            OccurrenceDiagnostic(
                finding,
                chain,
                _declaration_for_finding(owner, finding),
                projection_name,
            )
        )
    return tuple(interpreted)


def _direct_children(
    compiled: _CompiledSpace[Space],
) -> tuple[tuple[str, _CompiledSpace[Space]], ...]:
    return (
        *compiled.children,
        *(
            (f"{name}.{case.case_id}", case.compiled)
            for name, branch in compiled.branches
            for case in branch.cases
        ),
    )


def _child_for_declaration(
    state: _State, declaration: Use[Space] | Case
) -> tuple[tuple[str, ...], _CompiledSpace[Space]]:
    """The compiled child a use site names, and the scope segment it adds."""

    if isinstance(declaration, Use):
        name = _owned_member(state, declaration, (Use,), "Use")
        return (name,), state.compiled.child(name)

    found = tuple(
        ((authored_branch_name, candidate.case_id), candidate.compiled)
        for authored_branch_name, authored_branch in declared_members(state.compiled.owner)
        if isinstance(authored_branch, OneOf)
        for authored_case, candidate in zip(
            authored_branch.cases,
            state.compiled.branch(authored_branch_name).cases,
        )
        if authored_case is declaration
    )
    if len(found) != 1:
        raise _out_of_scope(state, declaration, "Case")
    return found[0]


def occurrence_child(instance: Space, declaration: Use[S] | Case) -> S:
    """Return the exact direct child occurrence one use site names.

    A Python class is not an occurrence identity.  One Kernel class may be
    placed at several roles, and "you probably meant the only one" is precisely
    the behaviour that breaks the day a second placement appears -- silently,
    and in whichever call site happened to be written first.  So the key is
    always an exact declaration: a ``Use``, a ``Case``, or a case id through the
    branch view.
    """

    state = _occurrence_state(instance)
    if not isinstance(declaration, (Use, Case)):
        raise AuthoringError(
            "child() takes an exact Use or Case declaration; a Space class names a family, "
            "not one of its occurrences"
        )
    segment, child = _child_for_declaration(state, cast("Use[Space] | Case", declaration))
    return _make_occurrence(
        state.runtime,
        cast("_CompiledSpace[S]", child),
        (*state.scope, *segment),
        state.root,
    )


def _branch_for_declaration(state: _State, declaration: OneOf) -> _CompiledBranch:
    return state.compiled.branch(_owned_member(state, declaration, (OneOf,), "OneOf"))


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
        state = _occurrence_state(self.__owner)
        lineage = state.runtime.lineage
        point = state.runtime.point
        active = self.__branch.active
        if active is not None:
            with lineage.lock:
                active_answer = answer_for(lineage.engine, point, active)
            if isinstance(active_answer, Unresolved):
                return active_answer
            if isinstance(active_answer, Absent) or not cast(bool, active_answer.value):
                return Absent()
        if self.__branch.selector is None:
            return Decided(self.__branch.cases[0].case_id)
        with lineage.lock:
            answer = answer_for(lineage.engine, point, self.__branch.selector)
        return cast("Answer[str]", answer)

    def case(self, case_id: str) -> Space:
        child = self.__branch.case(case_id).compiled
        state = _occurrence_state(self.__owner)
        return _make_occurrence(
            state.runtime,
            child,
            (*state.scope, self.name, case_id),
            state.root,
        )

    def select(self, case_id: str) -> BranchView:
        if case_id not in self.cases:
            raise AuthoringError(
                f"branch {self.name!r} has no case {case_id!r}; expected one of {self.cases}"
            )
        if self.__branch.selector is None:
            return self
        state = _occurrence_state(self.__owner)
        lineage = state.runtime.lineage
        with lineage.lock:
            committed = lineage.engine.commit_assignments(
                state.runtime.point, {self.__branch.selector.path: case_id}
            )
        _raise_failed_assignment(committed.outcomes)
        successor = state.runtime.successor(committed.point)
        successor_root = _successor_root(successor)
        successor_owner = (
            successor_root
            if state.compiled is lineage.tree
            else _make_occurrence(successor, state.compiled, state.scope, successor_root)
        )
        return BranchView(successor_owner, self.__branch)


def occurrence_branch(instance: Space, declaration: OneOf) -> BranchView:
    """Bind one authored branch to this exact occurrence namespace."""

    state = _occurrence_state(instance)
    return BranchView(instance, _branch_for_declaration(state, declaration))


__all__ = [
    "BranchView",
    "OccurrenceDiagnostic",
    "ProblemSource",
    "ProjectionAssessment",
    "is_attached_occurrence",
    "start_from_model",
    "start_occurrence",
]
