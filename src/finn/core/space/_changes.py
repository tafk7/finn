# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Atomic choice replacement through private admission trials.

Every retained and supplied choice is revalidated before publication. Requests
are checked as a batch; only a fully admitted trial becomes a configuration.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypeVar, cast

from . import _execution, _runtime
from ._configuration import ChoiceView, Space
from .declarations import Decision, DecisionRef, SubspaceChoice
from .edits import Change, ChangeOutcome, ChangeRequest, ConfigurationResult
from .errors import ConfigurationError, EvaluationError, RequestError
from .ir import Node
from .occurrence import (
    _attach,
    _case_scope,
    _choice_record,
    _decision,
    _prepare_values,
    root,
    state,
)
from .results import Available, Inapplicable, QueryResult

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def change(point: Space, reference: Decision[T] | DecisionRef[T], value: T) -> Change[T]:
    _execution.driver_only("change construction")
    current = state(point)
    index = _decision(point, reference)
    return Change(current, current.linked.nodes[index].scope, index, value)


def clear(point: Space, reference: Decision[T] | DecisionRef[T]) -> Change[T]:
    _execution.driver_only("clear construction")
    current = state(point)
    index = _decision(point, reference)
    return Change(current, current.linked.nodes[index].scope, index, remove=True)


def _normalize_changes(
    point: Space, changes: tuple[ChangeRequest, ...]
) -> dict[int, Change[object]]:
    current = state(point)
    pending: dict[int, Change[object]] = {}
    for item in changes:
        if not isinstance(item, Change):
            raise RequestError("changes must come from a bound decision field")
        if item.base is not current:
            raise RequestError("all changes must target this exact base snapshot")
        if type(item.scope) is not int or not 0 <= item.scope < len(current.linked.scopes):
            raise RequestError("change scope does not belong to this model")
        if type(item.node) is not int or not 0 <= item.node < len(current.linked.nodes):
            raise RequestError("change node does not belong to this model")
        node = current.linked.nodes[item.node]
        if node.kind != "decision" or node.scope != item.scope:
            raise RequestError("change does not identify an owned decision in its scope")
        if item.node in pending:
            raise RequestError(f"duplicate change for {node.key}")
        pending[item.node] = item
    return pending


def _keyword_change(point: Space, name: str, value: object) -> Change[object]:
    current = state(point)
    scope = current.linked.scopes[point._scope]
    index = scope.named_members.get(name)
    if index is None:
        for declaration, choice_index in scope.choices.items():
            if isinstance(declaration, SubspaceChoice) and declaration.name == name:
                choice = current.linked.choices[choice_index]
                if choice.selector is None:
                    if type(value) is not str or value not in tuple(
                        case for case, _ in choice.cases
                    ):
                        raise RequestError(f"{choice.key}: unknown choice case {value!r}")
                    raise RequestError(f"{choice.key} is a singleton structural choice")
                index = choice.selector
                break
    if index is None:
        raise RequestError(f"unknown direct choice {name!r}")
    node = current.linked.nodes[index]
    if node.kind != "decision" or node.scope != point._scope:
        raise RequestError(f"{node.key} is not a direct owned choice in this scope")
    return Change(current, node.scope, index, value)


def _values_equal(node: Node, left: object, right: object) -> bool:
    assert node.semantics is not None
    try:
        return node.semantics.values_equal(
            node.semantics.freeze(left), node.semantics.freeze(right)
        )
    except Exception as cause:
        raise EvaluationError(node.owner, "configuration equality", str(cause)) from cause


def _admission(trial: _runtime.Snapshot, index: int) -> QueryResult[bool]:
    """The decision frame publishes a candidate only after its membership succeeds."""
    result = _runtime.evaluate(trial, index).result
    return Available(True) if isinstance(result, Available) else result


def try_with_choices(
    point: S, /, *changes: ChangeRequest, **choices: object
) -> ConfigurationResult[S]:
    """Build and validate a replacement choice set over the same frozen facts."""

    _execution.driver_only("configuration replacement")
    current = state(point)
    keyword_changes = tuple(_keyword_change(point, name, value) for name, value in choices.items())
    all_changes = (*changes, *keyword_changes)
    normalized = _normalize_changes(point, all_changes)
    with current.lock:
        prepared = _prepare_values(
            current.linked,
            {index: item.value for index, item in normalized.items() if not item.remove},
            "choice",
        )
        merged = dict(current.assignments)
        for index, item in normalized.items():
            if item.remove:
                merged.pop(index, None)
            else:
                merged[index] = prepared[index]

        semantically_same = len(merged) == len(current.assignments) and all(
            index in current.assignments
            and _values_equal(current.linked.nodes[index], value, current.assignments[index])
            for index, value in merged.items()
        )
        if semantically_same:
            no_op_outcomes = tuple(
                ChangeOutcome(
                    current.linked.nodes[item.node].owner,
                    Available(True),
                    "unchanged",
                )
                for item in all_changes
            )
            return ConfigurationResult(point, True, no_op_outcomes)

        trial = _runtime._TrialSnapshot(current, merged)
        validation = {
            index: _admission(trial, index)
            for index in sorted(merged, key=current.linked.ranks.__getitem__)
        }

        refused = {
            index
            for index, result in validation.items()
            if not (isinstance(result, Available) and result.value is True)
        }
        outcomes: list[ChangeOutcome] = []
        requested_nodes = set(normalized)
        for request in all_changes:
            item = normalized[request.node]
            node = current.linked.nodes[item.node]
            result = validation.get(item.node, Available(True))
            status: Literal["refused", "admissible"] = (
                "refused" if item.node in refused else "admissible"
            )
            outcomes.append(ChangeOutcome(node.owner, result, status))
        for index in sorted(refused - requested_nodes, key=current.linked.ranks.__getitem__):
            node = current.linked.nodes[index]
            outcomes.append(
                ChangeOutcome(node.owner, validation[index], "refused", requested=False)
            )
        if refused:
            return ConfigurationResult(point, False, tuple(outcomes))

        snapshot = trial.publish()
        successor = cast(S, _attach(snapshot, point._scope))
        published: list[ChangeOutcome] = []
        for request, outcome in zip(all_changes, outcomes):
            item = normalized[request.node]
            if item.remove and item.node in current.assignments:
                final_status: Literal["removed", "unchanged", "changed"] = "removed"
            elif item.remove:
                final_status = "unchanged"
            elif item.node in current.assignments and _values_equal(
                current.linked.nodes[item.node],
                prepared[item.node],
                current.assignments[item.node],
            ):
                final_status = "unchanged"
            else:
                final_status = "changed"
            published.append(replace(outcome, status=final_status))
        return ConfigurationResult(successor, True, tuple(published))


def with_choices(point: S, /, *changes: ChangeRequest, **choices: object) -> S:
    report = try_with_choices(point, *changes, **choices)
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance


def select(view: ChoiceView, case: str) -> ChoiceView:
    _execution.driver_only("configuration selection")
    current, declaration = _choice_record(view)
    _case_scope(declaration, case)
    if declaration.selector is None:
        # A singleton case is structurally selected, but an enclosing guard
        # still determines whether selection applies at this snapshot.
        if declaration.guard is not None:
            guard = _runtime.evaluate(current, declaration.guard).result
            if not isinstance(guard, Available) or guard.value is not True:
                refusal: QueryResult[bool] = (
                    Inapplicable()
                    if isinstance(guard, Available)
                    else cast(QueryResult[bool], guard)
                )
                report = ConfigurationResult(
                    root(view.instance),
                    False,
                    (ChangeOutcome(declaration.key, refusal, "refused"),),
                )
                raise ConfigurationError(report)
        return view
    selector = current.linked.nodes[declaration.selector]
    report = try_with_choices(
        root(view.instance),
        Change(current, selector.scope, selector.index, case),
    )
    if not report.accepted:
        raise ConfigurationError(report)
    successor = state(report.instance)
    if successor is current:
        return view
    owner = _attach(successor, view.instance._scope)
    return ChoiceView(owner, view.declaration)
