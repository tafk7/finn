# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Demand-limited iterative evaluation over one immutable snapshot."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from . import _execution
from .errors import EvaluationError
from .ir import Argument, LinkedModel, Node, NodeKind
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Finding,
    FindingKind,
    Inapplicable,
    NonValue,
    QueryResult,
    ReadinessAssessment,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_constraints,
    assess_view,
    constraint_result,
    merged_findings,
    owned_result,
    reject,
)
from .semantics import snapshot as snapshot_value

if TYPE_CHECKING:
    from ._configuration import Space
    from .compiler import Model

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment


@dataclass(frozen=True, slots=True)
class Evaluation:
    result: QueryResult[object]
    dependencies: tuple[int, ...] = ()
    assessment: Assessment | None = None
    # Method reads that went straight to a forwarding alias's source: (alias, source).
    via: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class Snapshot:
    """One root binding and commitment set with a local evaluation cache.

    An open Decision with one viable case reads as that case (``forcing``), false
    only on the copies forcing evaluates on. ``found`` holds the snapshot's forced
    Decisions once found; ``verdicts`` are the ones its base found, which forcing
    reuses where the change did not reach (``finn.core.space._forcing``).
    """

    model: Model[Space]
    parameters: Mapping[int, object]
    assignments: Mapping[int, object] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock, repr=False)
    forcing: bool = True
    verdicts: Mapping[int, object] = field(default_factory=dict, repr=False)
    cache: dict[int, Evaluation] = field(default_factory=dict, init=False, repr=False)
    work: _execution.Work = field(default_factory=_execution.Work, init=False, repr=False)
    found: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, MappingProxyType):
            object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    @property
    def linked(self) -> LinkedModel:
        return self.model.linked


class _TrialSnapshot(Snapshot):
    """Private complete candidates, admitted on demand before value publication.

    Every candidate, including retained assignments, must pass admission before
    publication. The base contributes its model, frozen facts and lock, its
    verdicts, and the values it forces, which the trial reads for the Decisions it
    does not change. A Decision the base does not force reads as the configuration
    the trial would publish forces it (``successor``): found once, and that
    configuration is the one published.
    """

    __slots__ = ("_pending", "_published", "_candidates", "_base", "_successor")

    _pending: dict[int, object]
    _published: bool
    _candidates: Mapping[int, object]
    _base: Snapshot
    _successor: Snapshot | None

    def __init__(self, base: Snapshot, candidates: Mapping[int, object]) -> None:
        pending: dict[int, object] = {}
        super().__init__(base.model, base.parameters, {}, base.lock, base.forcing)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_successor", None)
        object.__setattr__(self, "assignments", MappingProxyType(pending))
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_candidates", MappingProxyType(dict(candidates)))
        object.__setattr__(self, "_published", False)

    def admit(self, node_index: int, value: object) -> None:
        if self._published:
            raise RuntimeError("a published admission trial is closed")
        if node_index in self._pending or node_index in self.cache:
            raise RuntimeError("admission dependency order admitted a previously resolved decision")
        self._pending[node_index] = value

    def successor(self) -> Snapshot:
        """The configuration this trial would publish: every candidate committed."""
        if self._successor is None:
            from ._forcing import inherited  # noqa: PLC0415 - runtime/forcing cycle

            successor = Snapshot(
                self.model,
                self.parameters,
                self._candidates,
                forcing=self.forcing,
                verdicts=inherited(self._base),
            )
            object.__setattr__(self, "_successor", successor)
        assert self._successor is not None
        return self._successor

    def publish(self) -> Snapshot:
        if self._published:
            raise RuntimeError("a admission trial can only be published once")
        object.__setattr__(self, "_published", True)
        # Every candidate was admitted: the successor's assignments are the pending ones.
        return self.successor()


def _forced(snapshot: Snapshot, index: int) -> QueryResult[object] | None:
    """An open Decision's forced value (its one viable case), its refusal (no viable
    case), or None (several, or a snapshot that does not force). A trial reads its
    base's, and where the base forces nothing, the configuration's it would publish."""
    from ._forcing import forced  # noqa: PLC0415 - runtime/forcing cycle

    if not snapshot.forcing:
        return None
    source = snapshot
    if isinstance(snapshot, _TrialSnapshot):
        base = forced(snapshot._base)
        if index in base.values:
            return Available(base.values[index])
        source = snapshot.successor()
    found = forced(source)
    if index in found.values:
        return Available(found.values[index])
    return found.refused.get(index)


def _blocked(answers: list[QueryResult[object]]) -> NonValue | None:
    """Required inputs retain all distinct findings of the highest-precedence nonvalue."""

    for variant in (Unresolved, Rejected, Inapplicable):
        matches = tuple(answer for answer in answers if isinstance(answer, variant))
        if matches:
            return variant(merged_findings(matches))
    return None


def _clone(node: Node, value: object, *, owner: str, role: str) -> object:
    if node.semantics is None:
        raise EvaluationError(owner, role, f"{node.key} has no value semantics")
    return snapshot_value(node.semantics, value, owner=owner, role=role)


def _arguments(
    linked: LinkedModel, arguments: tuple[Argument, ...], owner: str
) -> Generator[int, object, tuple[dict[str, object], NonValue | None]]:
    values: dict[str, object] = {}
    failures: list[QueryResult[object]] = []
    for argument in arguments:
        answer = cast(QueryResult[object], (yield argument.node))
        if isinstance(answer, Available):
            values[argument.name] = _clone(
                linked.nodes[argument.node], answer.value, owner=owner, role="dependency snapshot"
            )
        else:
            failures.append(answer)
    return values, _blocked(failures)


def _guard_result(node: Node, answer: QueryResult[object]) -> NonValue | None:
    if not isinstance(answer, Available):
        return answer
    if type(answer.value) is not bool:
        raise EvaluationError(node.owner, "applicability", "guard must return bool")
    return None if answer.value else Inapplicable()


def _guard_assessment(node: Node, answer: NonValue) -> Assessment | None:
    if node.kind == "view":
        return assess_view(answer, owner=node.key, applicability=answer)
    if node.kind == "constraint" or node.kind == "group":
        return ConstraintAssessment({node.key: answer}, answer)
    return None


def _constraint_members(
    snapshot: Snapshot, reference: int, answer: QueryResult[object]
) -> Mapping[str, QueryResult[bool]]:
    assessment = snapshot.cache[reference].assessment
    if isinstance(assessment, ConstraintAssessment) and assessment.results:
        return assessment.results
    return {snapshot.linked.nodes[reference].key: cast(QueryResult[bool], answer)}


def _obligation(
    snapshot: Snapshot, reference: int, answer: QueryResult[object]
) -> QueryResult[object]:
    """An obliged view contributes only its acceptance."""
    if snapshot.linked.nodes[reference].kind == "view" and isinstance(answer, Available):
        return Available(True)
    return answer


def _present(node: Node, answers: list[QueryResult[object]]) -> QueryResult[object]:
    """The single present source. While any source is unresolved the answer is
    unresolved too: a later commitment could still make a second one present."""
    active = [answer for answer in answers if not isinstance(answer, Inapplicable)]
    blocked = _blocked([answer for answer in active if not isinstance(answer, Available)])
    if blocked is not None:
        return blocked
    if len(active) > 1:
        return reject(
            "multiple-suppliers",
            f"{len(active)} sources are present; at most one may supply this value",
            owner=node.owner,
        )
    if not active:
        return Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "input-unsupplied",
                    node.owner,
                    "no present source supplies this value",
                ),
            )
        )
    return active[0]


def _graph_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    from .graph import Located  # noqa: PLC0415 - value type of the graph primitives

    if node.kind == "present":
        answers: list[QueryResult[object]] = []
        for _, target in node.alternatives:
            answers.append(cast(QueryResult[object], (yield target)))
        return Evaluation(_present(node, answers))
    if node.kind == "locate":
        assert node.output is not None
        answer = cast(QueryResult[object], (yield node.output))
        if not isinstance(answer, Available):
            return Evaluation(answer)
        where, member = cast(tuple[str | None, str], node.value)
        return Evaluation(Available(Located(where, member, answer.value)))
    assert node.kind == "members"
    # Members and Users: one (node name, export) alternative per entry, with
    # the member name of each entry (the key's name, or the user's input name).
    located: list[Located[object]] = []
    failures: list[QueryResult[object]] = []
    for (name, target), member in zip(node.alternatives, cast(tuple[str, ...], node.value)):
        answer = cast(QueryResult[object], (yield target))
        if isinstance(answer, Inapplicable):
            continue
        if isinstance(answer, Available):
            located.append(Located(name or None, member, answer.value))
        else:
            failures.append(answer)
    blocked = _blocked(failures)
    return Evaluation(blocked if blocked is not None else Available(tuple(located)))


def _frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """A node's evaluation: inapplicable while its guard does not hold, else its kind's."""
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive, assessment=_guard_assessment(node, inactive))
    return (yield from _FRAMES[node.kind](snapshot, node))


def _param_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    yield from ()  # answers at once
    if node.index in snapshot.parameters:
        return Evaluation(Available(snapshot.parameters[node.index]))
    return Evaluation(
        Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "input-missing",
                    node.owner,
                    "optional input was omitted at start",
                ),
            )
        )
    )


def _const_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.domain is not None:
        return (yield from _pinned(snapshot, node, Available(node.value)))
    return Evaluation(Available(node.value))


def _decision_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.index in snapshot.assignments:
        return Evaluation(Available(snapshot.assignments[node.index]))
    if isinstance(snapshot, _TrialSnapshot) and node.index in snapshot._candidates:
        candidate = snapshot._candidates[node.index]
        admission = yield from _membership_frame(snapshot, node, candidate, check_guard=False)
        if isinstance(admission.result, Available) and admission.result.value is True:
            snapshot.admit(node.index, candidate)
            return Evaluation(Available(snapshot.assignments[node.index]))
        return Evaluation(admission.result)
    found = _forced(snapshot, node.index)
    if found is not None:
        return Evaluation(found)
    return Evaluation(
        Unresolved(
            (
                Finding(
                    FindingKind.BLOCKER,
                    "decision-unassigned",
                    node.owner,
                    "decision requires a commitment",
                ),
            )
        )
    )


def _forward_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """An alias or a guard: its output's answer, a pinned one checked against its domain."""
    if node.output is None:
        raise EvaluationError(node.owner, node.kind, "missing output reference")
    answer = cast(QueryResult[object], (yield node.output))
    if node.domain is not None:
        return (yield from _pinned(snapshot, node, answer))
    return Evaluation(answer)


def _select_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.selector is None or node.selection_index is None:
        raise EvaluationError(node.owner, "selection", "missing selector")
    selector = cast(QueryResult[object], (yield node.selector))
    if not isinstance(selector, Available):
        return Evaluation(selector)
    case = selector.value
    if type(case) is not str:
        raise EvaluationError(node.owner, "selection", "selector is not a declared case")
    target = node.selection_index.get(case)
    if target is None:
        # The selected candidate is None, or has no such member: absent.
        return Evaluation(Inapplicable())
    return Evaluation(cast(QueryResult[object], (yield target)))


def _view_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.output is None:
        raise EvaluationError(node.owner, "view", "missing output reference")
    output = cast(QueryResult[object], (yield node.output))
    constraints: dict[str, QueryResult[bool]] = {}
    for reference in node.constraints:
        answer = _obligation(snapshot, reference, cast(QueryResult[object], (yield reference)))
        constraints.update(_constraint_members(snapshot, reference, answer))
    view = assess_view(output, owner=node.key, constraints=constraints)
    return Evaluation(view.accepted_result, assessment=view)


def _group_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    answers: dict[str, QueryResult[bool]] = {}
    for reference in node.constraints:
        answer = cast(QueryResult[object], (yield reference))
        answers.update(_constraint_members(snapshot, reference, answer))
    group = assess_constraints(answers)
    return Evaluation(cast(QueryResult[object], group.result), assessment=group)


def _callback_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    """A derived value or a constraint: its authored callback, called on its arguments.

    What the callback raises, or a result it returns that fails to snapshot, is
    the owner's failure; an engine invariant is raised as itself.
    """
    arguments, failure = yield from _arguments(snapshot.linked, node.arguments, node.owner)
    if failure is not None:
        return Evaluation(failure, assessment=_guard_assessment(node, failure))
    if node.function is None:
        raise EvaluationError(node.owner, node.kind, "missing callback")
    try:
        positional = (_self_point(snapshot, node.scope),) if node.call_style == "self" else ()
        called = yield _execution.Call(node.function, positional, arguments, node.kind)
        if isinstance(called, _execution._Halt):
            blocked = _blocked(list(called.results))
            assert blocked is not None
            return Evaluation(blocked, assessment=_guard_assessment(node, blocked))
        assert isinstance(called, _execution._Returned)
        result = called.value
        if node.kind == "constraint":
            normalized = constraint_result(cast(bool | QueryResult[bool], result), node.owner)
            assessment = ConstraintAssessment({node.key: normalized}, normalized)
            return Evaluation(cast(QueryResult[object], normalized), assessment=assessment)
        if isinstance(result, (Inapplicable, Rejected, Unresolved)):
            return Evaluation(owned_result(result, node.owner))
    except Exception as cause:
        raise EvaluationError(node.owner, node.kind, str(cause)) from cause
    if node.semantics is None:
        raise EvaluationError(node.owner, node.kind, "derived output has no value semantics")
    value = result.value if isinstance(result, Available) else result
    frozen = snapshot_value(node.semantics, value, owner=node.owner, role=node.kind)
    return Evaluation(Available(frozen))


_FRAMES: Mapping[NodeKind, Callable[[Snapshot, Node], _execution.Frame]] = {
    "param": _param_frame,
    "const": _const_frame,
    "present": _graph_frame,
    "locate": _graph_frame,
    "members": _graph_frame,
    "decision": _decision_frame,
    "alias": _forward_frame,
    "guard": _forward_frame,
    "select": _select_frame,
    "view": _view_frame,
    "group": _group_frame,
    "derived": _callback_frame,
    "constraint": _callback_frame,
}


def _refused(node: Node, value: object, answer: QueryResult[object]) -> QueryResult[object]:
    """A supplied value outside the declared domain, reported with who supplied it."""
    if not isinstance(answer, Rejected):
        return answer
    reason = "; ".join(finding.message for finding in answer.findings)
    supplied = node.note or f"{node.key} = {value!r}"
    return reject(
        "domain-membership",
        f"{supplied}: {value!r} is outside the declared domain ({reason})",
        owner=node.owner,
    )


def _pinned(snapshot: Snapshot, node: Node, answer: QueryResult[object]) -> _execution.Frame:
    """A pinned coordinate: the supplied value must be in the declared domain."""
    if not isinstance(answer, Available):
        return Evaluation(answer)
    admission = yield from _membership_frame(snapshot, node, answer.value, check_guard=False)
    if isinstance(admission.result, Available) and admission.result.value is True:
        return Evaluation(answer)
    return Evaluation(_refused(node, answer.value, admission.result))


def supplied_provenance(snapshot: Snapshot, evaluation: Evaluation, index: int) -> Evaluation:
    """A computation's own refusal names who set the overridden values it read.

    ``kitchen.area = 16 (set by House at house.py:42; declared 12 at room.py:10)``
    is appended to each finding the computation owns, for every value it read
    that an enclosing body set over another setting.
    """
    result = evaluation.result
    linked = snapshot.linked
    if not isinstance(result, Rejected) or not linked.provenance:
        return evaluation
    node = linked.nodes[index]
    if node.function is None:
        return evaluation
    read = (*evaluation.dependencies, *(alias for alias, _ in evaluation.via))
    texts = tuple(
        dict.fromkeys(
            linked.provenance[item].text()
            for item in read
            if item in linked.provenance and len(linked.provenance[item].layers) > 1
        )
    )
    if not texts:
        return evaluation
    findings = tuple(
        replace(
            finding,
            message=f"{finding.message}; {'; '.join(texts)}",
            details=(*finding.details, ("provenance", texts)),
        )
        if finding.owner == node.owner
        else finding
        for finding in result.findings
    )
    annotated = Rejected(findings)
    assessment = evaluation.assessment
    if isinstance(assessment, ConstraintAssessment):
        assessment = ConstraintAssessment(
            {
                key: annotated if answer is result else answer
                for key, answer in assessment.results.items()
            },
            annotated if assessment.result is result else assessment.result,
        )
    return Evaluation(annotated, evaluation.dependencies, assessment, evaluation.via)


def _self_point(snapshot: Snapshot, scope: int) -> Space:
    from .occurrence import _attach  # noqa: PLC0415 - scoped callback receiver

    return _attach(snapshot, scope)


def evaluate(snapshot: Snapshot, node_index: int) -> Evaluation:
    """Demand one result through the canonical native dispatcher."""
    with snapshot.lock:
        cached = snapshot.cache.get(node_index)
        return cached if cached is not None else _execution.run(snapshot, node_index)


def _membership_frame(
    snapshot: Snapshot, node: Node, value: object, *, check_guard: bool = True
) -> _execution.Frame:
    if check_guard and node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive)
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        return Evaluation(failure)
    if node.domain is None or node.semantics is None:
        raise EvaluationError(node.owner, "domain membership", "reference has no domain")
    candidate = _clone(node, value, owner=node.owner, role="domain candidate snapshot")
    called = yield _execution.Call(
        node.domain.membership,
        (candidate, arguments),
        {"semantics": node.semantics, "owner": node.owner},
        "domain membership",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return Evaluation(blocked)
    assert isinstance(called, _execution._Returned)
    result = cast(QueryResult[object], called.value)
    if node.contract is not None and isinstance(result, Available) and result.value is True:
        # A replaced Decision: the declared domain still checks the candidate.
        contract = yield from _contract_frame(snapshot, node, candidate)
        if not (isinstance(contract, Available) and contract.value is True):
            return Evaluation(_refused(node, candidate, contract))
    return Evaluation(result)


def _contract_frame(
    snapshot: Snapshot, node: Node, candidate: object
) -> Generator[int | _execution.Call, object, QueryResult[object]]:
    assert node.contract is not None and node.semantics is not None
    arguments, failure = yield from _arguments(snapshot.linked, node.contract_arguments, node.owner)
    if failure is not None:
        return failure
    called = yield _execution.Call(
        node.contract.membership,
        (candidate, arguments),
        {"semantics": node.semantics, "owner": node.owner},
        "domain membership",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return blocked
    assert isinstance(called, _execution._Returned)
    return cast(QueryResult[object], called.value)


def _enumeration_frame(snapshot: Snapshot, node: Node) -> _execution.Frame:
    if node.guard is not None:
        guard = cast(QueryResult[object], (yield node.guard))
        inactive = _guard_result(node, guard)
        if inactive is not None:
            return Evaluation(inactive)
    arguments, failure = yield from _arguments(snapshot.linked, node.domain_arguments, node.owner)
    if failure is not None:
        return Evaluation(failure)
    assert node.domain is not None and node.semantics is not None
    if node.domain.candidates is None:
        return Evaluation(Available(None))
    called = yield _execution.Call(
        node.domain.enumerate,
        (arguments,),
        {"semantics": node.semantics, "owner": node.owner},
        "domain enumeration",
    )
    if isinstance(called, _execution._Halt):
        blocked = _blocked(list(called.results))
        assert blocked is not None
        return Evaluation(blocked)
    assert isinstance(called, _execution._Returned)
    result = cast(QueryResult[object], called.value)
    if node.contract is None or not isinstance(result, Available):
        return Evaluation(result)
    # Advisory enumeration of a replaced Decision lists what its contract admits.
    admitted: list[object] = []
    for candidate in cast(tuple[object, ...], result.value):
        contract = yield from _contract_frame(snapshot, node, candidate)
        if isinstance(contract, Available) and contract.value is True:
            admitted.append(candidate)
        elif not isinstance(contract, Rejected):
            return Evaluation(contract)
    return Evaluation(Available(tuple(admitted)))


def _applicability(snapshot: Snapshot, node: Node) -> NonValue | None:
    if node.guard is None:
        return None
    return _guard_result(node, evaluate(snapshot, node.guard).result)


def decision_state(snapshot: Snapshot, node_index: int) -> QueryResult[DecisionState[object]]:
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind != "decision":
            raise EvaluationError(
                node.owner, "decision state", "reference is not an owning decision"
            )
        inactive = _applicability(snapshot, node)
        if inactive is not None:
            return inactive
        if node_index not in snapshot.assignments:
            return Available(DecisionState(node.owner))
        value = _clone(
            node, snapshot.assignments[node_index], owner=node.owner, role="decision state"
        )
        return Available(DecisionState(node.owner, "committed", value))


def enumeration(snapshot: Snapshot, node_index: int) -> Evaluation:
    """A Decision's enumerated candidates (``Available(None)``: not enumerable), with
    what the enumeration read."""
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.domain is None or node.semantics is None:
            raise EvaluationError(node.owner, "domain enumeration", "reference has no domain")
        return _execution.run(snapshot, node_index, _enumeration_frame(snapshot, node))


def candidate_values(snapshot: Snapshot, node_index: int) -> QueryResult[tuple[object, ...]] | None:
    result = enumeration(snapshot, node_index).result
    if isinstance(result, Available) and result.value is None:
        return None
    return cast(QueryResult[tuple[object, ...]], result)


def membership(snapshot: Snapshot, node_index: int, value: object) -> Evaluation:
    """Whether a Decision's domain admits ``value`` here (its requirements too), with
    the refusal's finding and what membership read."""
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        frame = _membership_frame(snapshot, node, value, check_guard=False)
        return _execution.run(snapshot, node_index, frame)


def copy_result(
    snapshot: Snapshot, node_index: int, answer: QueryResult[object]
) -> QueryResult[object]:
    """Detach a public answer from cached values using its declared semantics."""

    if not isinstance(answer, Available):
        return answer
    with snapshot.lock:
        node = snapshot.linked.nodes[node_index]
        if node.kind == "group" and type(answer.value) is bool:
            # Aggregate verdicts are intrinsic immutable booleans; these
            # declarations need no separate user-owned value semantics.
            return answer
        return Available(_clone(node, answer.value, owner=node.owner, role="public value snapshot"))


def _copy_readiness(
    snapshot: Snapshot, assessment: ReadinessAssessment, own: str
) -> ReadinessAssessment:
    def copy(key: str, answer: QueryResult[object]) -> QueryResult[object]:
        index = snapshot.linked.keys[key]
        if key != own and snapshot.linked.nodes[index].kind == "view":
            return answer  # an obliged view contributes only its Boolean acceptance
        return copy_result(snapshot, index, answer)

    return ReadinessAssessment(
        {key: copy(key, answer) for key, answer in assessment.results.items()},
        assessment.result,
    )


def copy_assessment(snapshot: Snapshot, node_index: int, assessment: Assessment) -> Assessment:
    """Detach every mutable value visible in an assessment from its cache."""

    with snapshot.lock:
        if isinstance(assessment, ConstraintAssessment):
            # Validated constraint values are exact bool, and findings are
            # immutable primitives. Their scalar answers can safely be shared.
            return assessment
        node = snapshot.linked.nodes[node_index]
        output_index = node.output if node.output is not None else node_index
        output = copy_result(snapshot, output_index, assessment.output_result)
        accepted = (
            output
            if assessment.accepted_result is assessment.output_result
            else copy_result(snapshot, node_index, assessment.accepted_result)
        )
        return ViewAssessment(
            output,
            _copy_readiness(snapshot, assessment.readiness, node.key),
            assessment.constraints,
            accepted,
        )


__all__ = [
    "Evaluation",
    "Snapshot",
    "candidate_values",
    "copy_result",
    "copy_assessment",
    "decision_state",
    "enumeration",
    "evaluate",
    "membership",
]
