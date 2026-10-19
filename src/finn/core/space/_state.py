# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A configuration's state: its facts, its stored choices and the world they read in.

A configuration is its facts (the root's formals) and its stored choices, dormant
ones included: the same two give the same answers, whatever the history. Its world
(``_world.World``) is where every read of it is evaluated. A change publishes a new
state whose world was made from its receiver's (``_assign``); nothing else is kept but
what inspection answered of it (``answers``), by question and scope.

A stored choice is **present** where its guard holds (or it has none), **absent** where
its guard is false or refused (its scope is not in play: a stored value there is
dormant), and of **unknown** presence while its guard waits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from ._world import World, public_value
from .errors import EvaluationError, RequestError
from .ir import LinkedModel, NodeId
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    ReadinessAssessment,
    Unresolved,
    ViewAssessment,
)
from .semantics import recognize, unrecognized
from .semantics import snapshot as snapshot_value

if TYPE_CHECKING:
    from ._configuration import Space
    from ._world import Evaluator
    from .compiler import Model

Assessment: TypeAlias = ViewAssessment[object] | ConstraintAssessment
Presence: TypeAlias = Literal["present", "absent", "unknown"]


@dataclass(frozen=True, slots=True, eq=False)
class State:
    """A configuration: its model, and the world of its facts and stored choices.
    ``answers``: what inspection answered of it, kept by question and scope."""

    model: Model[Space]
    world: World
    answers: dict[tuple[str, int], object] = field(default_factory=dict, repr=False)

    @property
    def linked(self) -> LinkedModel:
        return self.model.linked

    @property
    def evaluator(self) -> Evaluator:
        return self.world.evaluator

    @property
    def facts(self) -> Mapping[NodeId, object]:
        """The facts bound, by root formal (an optional one omitted is absent)."""
        return MappingProxyType(self.world.facts)

    @property
    def stored(self) -> Mapping[NodeId, object]:
        """Every stored choice, dormant ones included, by Decision."""
        return MappingProxyType(self.world.stored)


def prepare(
    linked: LinkedModel, pending: Mapping[NodeId, tuple[object, str]]
) -> dict[NodeId, object]:
    """Values as a configuration holds them: the entire request recognized by each
    node's semantics (each value with the role it is supplied in) before any snapshot
    adapter is invoked, then each frozen."""
    for index, (value, role) in pending.items():
        node = linked.nodes[index]
        assert node.semantics is not None
        if not recognize(node.semantics, value, owner=node.owner, role=f"{role} recognition"):
            raise RequestError(f"{node.key}: {unrecognized(node.semantics)}")
    prepared: dict[NodeId, object] = {}
    for index, (value, role) in pending.items():
        node = linked.nodes[index]
        assert node.semantics is not None
        prepared[index] = snapshot_value(
            node.semantics, value, owner=node.owner, role=f"{role} snapshot"
        )
    return prepared


def bind(model: Model[Space], facts: Mapping[NodeId, object]) -> State:
    """The design space of ``model`` over ``facts`` (recognized and frozen): every choice
    open."""
    return State(model, World.start(model.evaluator, facts, {}))


def presence(world: World, node: NodeId) -> tuple[Presence, QueryResult[object]]:
    """Whether the node ``node`` is present in ``world``, and its guard's answer: present
    where its guard holds (or it has none); absent where the guard is false, does not
    apply, or is refused (a guard refused through a refused input puts its scope out of
    play); unknown while it waits."""
    guard = world.evaluator.nodes[node].guard
    if guard is None:
        return "present", Available(True)
    answer = world.result(guard)
    if isinstance(answer, Available):
        if type(answer.value) is not bool:
            owner = world.evaluator.nodes[node].owner
            raise EvaluationError(owner, "applicability", "guard must return bool")
        return ("present" if answer.value else "absent"), answer
    if isinstance(answer, Unresolved):
        return "unknown", answer
    return "absent", answer


def decision_state(state: State, node: NodeId) -> QueryResult[DecisionState[object]]:
    """A Decision's state: the commitment, not the answer. Every stored choice is
    ``committed`` with its stored value, whatever a read of it answers (its value in
    play, inapplicable while dormant, unresolved while it waits). An open Decision is
    unassigned where it is present, and answers as its guard otherwise."""
    spec = state.linked.nodes[node]
    if spec.kind != "decision":
        raise EvaluationError(spec.owner, "decision state", "reference is not an owning decision")
    stored = state.world.stored
    if node in stored:
        assert spec.semantics is not None
        value = snapshot_value(
            spec.semantics, stored[node], owner=spec.owner, role="decision state"
        )
        return Available(DecisionState(spec.owner, "committed", value))
    found, answer = presence(state.world, node)
    if found == "present":
        return Available(DecisionState(spec.owner))
    if isinstance(answer, Available):
        return Inapplicable()
    return cast(QueryResult[DecisionState[object]], answer)


def copy_result(
    linked: LinkedModel, node: NodeId, answer: QueryResult[object]
) -> QueryResult[object]:
    """A public answer detached from the entry it was read from, by its semantics."""
    if not isinstance(answer, Available):
        return answer
    spec = linked.nodes[node]
    if spec.kind == "validity" and type(answer.value) is bool:
        return answer  # an immutable bool, the pieces' AND: shared as it is
    return Available(public_value(spec, answer))


def _copy_readiness(linked: LinkedModel, assessment: ReadinessAssessment) -> ReadinessAssessment:
    return ReadinessAssessment(
        {
            key: copy_result(linked, linked.keys[key], answer)
            for key, answer in assessment.results.items()
        },
        assessment.result,
    )


def copy_assessment(linked: LinkedModel, node: NodeId, assessment: Assessment) -> Assessment:
    """An assessment detached from its entry: every value in it copied by its semantics
    (a constraint's answers are exact bools and findings, shared as they are)."""
    if isinstance(assessment, ConstraintAssessment):
        return assessment
    output_node = linked.nodes[node].output
    output = copy_result(
        linked, output_node if output_node is not None else node, assessment.output_result
    )
    accepted = (
        output
        if assessment.accepted_result is assessment.output_result
        else copy_result(linked, node, assessment.accepted_result)
    )
    return ViewAssessment(
        output, _copy_readiness(linked, assessment.readiness), assessment.validity, accepted
    )


__all__ = [
    "Presence",
    "State",
    "bind",
    "copy_assessment",
    "copy_result",
    "decision_state",
    "prepare",
    "presence",
]
