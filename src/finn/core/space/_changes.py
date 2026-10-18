# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Atomic choice replacement through private admission trials.

Every supplied choice, and every retained choice the change can reach, is checked
again before publication; a retained choice whose admission read nothing the change
reaches keeps the admission its base recorded (``_runtime._TrialSnapshot``).
Requests are checked as a batch, and a trial reads the forced values of the configuration
it would publish. A choice whose own check is false refuses the batch. One whose own
check cannot conclude (it reads a choice still open, or an input not supplied) is kept
**unchecked**: stored, read unresolved, and reported. Which choices are unchecked also
decides what the configuration forces, so the batch is checked again with the choices
found unchecked declared, until the same ones are found. A configuration with other
values for its formals (``try_with_facts``) is the same replacement over other facts: it
keeps what read none of the formals that changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, TypeVar, cast

from . import _execution, _runtime
from ._configuration import Space
from .edits import Change, ChangeOutcome, ChangeRequest, ConfigurationResult
from .errors import ConfigurationError, RequestError
from .ir import Node
from .occurrence import attach, decision_index, prepare_values, state
from .results import Available, QueryResult, Unresolved
from .semantics import equal, snapshot

T = TypeVar("T")
S = TypeVar("S", bound=Space)


def change(point: Space, reference: object, value: T) -> Change[T]:
    _execution.driver_only("change construction")
    current = state(point)
    index = decision_index(point, reference)
    return Change(current, current.linked.nodes[index].scope, index, value)


def clear(point: Space, reference: object) -> Change[object]:
    _execution.driver_only("clear construction")
    current = state(point)
    index = decision_index(point, reference)
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
        raise RequestError(f"unknown direct choice {name!r}")
    node = current.linked.nodes[index]
    if node.kind != "decision" or node.scope != point._scope:
        raise RequestError(f"{node.key} is not a direct owned choice in this scope")
    return Change(current, node.scope, index, value)


def _requests(
    point: Space, items: tuple[ChangeRequest | Mapping[Any, object], ...]
) -> list[Change[object]]:
    """Expand ``{reference: value}`` mappings in order; Change objects pass through."""
    current = state(point)
    result: list[Change[object]] = []
    for item in items:
        if isinstance(item, Mapping):
            for reference, value in item.items():
                index = decision_index(point, reference)
                result.append(Change(current, current.linked.nodes[index].scope, index, value))
        else:
            result.append(item)
    return result


def _values_equal(node: Node, left: object, right: object) -> bool:
    semantics = node.semantics
    assert semantics is not None
    role = "configuration equality"
    left = snapshot(semantics, left, owner=node.owner, role=role)
    right = snapshot(semantics, right, owner=node.owner, role=role)
    return equal(semantics, left, right, owner=node.owner, role=role)


def _admission(trial: _runtime._TrialSnapshot, index: int) -> QueryResult[bool]:
    """A candidate's own check: true once it is admitted, else its refusal, or unresolved
    where it cannot conclude."""
    _runtime.evaluate(trial, index)
    return cast(QueryResult[bool], trial.own(index))


def _checked(
    base: _runtime.Snapshot, merged: Mapping[int, object], parameters: Mapping[int, object] | None
) -> tuple[_runtime._TrialSnapshot, dict[int, QueryResult[bool]]]:
    """The trial of ``merged`` over ``base`` and each candidate's own check in it, with the
    choices found unchecked declared to it: a trial reads a declared one unresolved, as the
    configuration it would publish does, and forces by that. Starting from none, a pass
    declares the choices whose own check it found unresolved and runs again, until a pass
    finds the choices it was given. Less is read each time, so a check that was unresolved
    stays so and the choices only grow, at most one pass for each (a pass that declares
    nothing, the usual one, is the only one)."""
    ranked = sorted(merged, key=base.linked.ranks.__getitem__)
    unchecked = frozenset[int]()
    for _ in range(len(ranked) + 1):
        trial = _runtime._TrialSnapshot(base, merged, parameters, unchecked=unchecked)
        validation = {index: _admission(trial, index) for index in ranked}
        found = frozenset(
            index for index, result in validation.items() if isinstance(result, Unresolved)
        )
        if found == unchecked:
            return trial, validation
        unchecked = found
    raise RuntimeError("the choices found unchecked did not settle")


def try_with_choices(
    point: S, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
) -> ConfigurationResult[S]:
    """Build and validate a replacement choice set over the same frozen facts."""

    _execution.driver_only("configuration replacement")
    keyword_changes = tuple(_keyword_change(point, name, value) for name, value in choices.items())
    all_changes = (*_requests(point, changes), *keyword_changes)
    return _replace(point, {}, all_changes)


def try_with_facts(
    point: S, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
) -> ConfigurationResult[S]:
    """Build and validate a replacement over other values of the root's formals, and
    the choices ``changes`` make. What read none of the formals that changed is kept,
    and the retained choices are admitted as a trial admits them; a trial reads the
    forced values of the configuration it would publish."""

    _execution.driver_only("configuration replacement")
    return _replace(point, _fact_nodes(point, facts), _requests(point, changes))


def _fact_nodes(point: Space, facts: Mapping[Any, object]) -> dict[int, object]:
    """``{formal: value}`` by node: each formal a root formal the model exposes."""
    current = state(point)
    nodes: dict[int, object] = {}
    for reference, value in facts.items():
        index = current.model.resolve(point._scope, reference)
        if index not in current.linked.parameters:
            raise RequestError(f"{current.linked.nodes[index].key} is not an exposed parameter")
        if index in nodes:
            raise RequestError(f"duplicate fact for {current.linked.nodes[index].key}")
        nodes[index] = value
    return nodes


def _replace(
    point: S,
    facts: Mapping[int, object],
    all_changes: Sequence[Change[object]],
) -> ConfigurationResult[S]:
    current = state(point)
    normalized = _normalize_changes(point, tuple(all_changes))
    with current.lock:
        given = prepare_values(current.linked, facts, "parameter")
        parameters = {**current.parameters, **given}
        changed = _runtime.changed_facts(current, parameters) if given else frozenset[int]()
        fact_status: list[ChangeOutcome] = [
            ChangeOutcome(
                current.linked.nodes[index].owner,
                Available(True),
                "unchanged",
            )
            for index in given
        ]
        prepared = prepare_values(
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

        semantically_same = (
            not changed
            and len(merged) == len(current.assignments)
            and all(
                index in current.assignments
                and (
                    value is current.assignments[index]
                    or _values_equal(current.linked.nodes[index], value, current.assignments[index])
                )
                for index, value in merged.items()
            )
        )
        # A change that names an unchecked choice is checked again, and reports it as unchecked
        # where it still is, as any other does.
        if semantically_same and current.unchecked.isdisjoint(normalized):
            no_op_outcomes = tuple(
                ChangeOutcome(
                    current.linked.nodes[item.node].owner,
                    Available(True),
                    "unchanged",
                )
                for item in all_changes
            )
            return ConfigurationResult(point, True, (*fact_status, *no_op_outcomes))

        trial, validation = _checked(current, merged, parameters if changed else None)
        waiting = {index for index, result in validation.items() if isinstance(result, Unresolved)}
        refused = {
            index
            for index, result in validation.items()
            if index not in waiting and not (isinstance(result, Available) and result.value is True)
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
            outcomes.append(
                ChangeOutcome(node.owner, result, status, unchecked=item.node in waiting)
            )
        for index in sorted(refused - requested_nodes, key=current.linked.ranks.__getitem__):
            node = current.linked.nodes[index]
            outcomes.append(
                ChangeOutcome(node.owner, validation[index], "refused", requested=False)
            )
        if refused:
            admissible = [replace(outcome, status="admissible") for outcome in fact_status]
            return ConfigurationResult(point, False, (*admissible, *outcomes))

        snapshot = trial.publish()
        successor = cast(S, attach(snapshot, point._scope))
        published: list[ChangeOutcome] = [
            replace(outcome, status="changed" if index in changed else "unchanged")
            for index, outcome in zip(given, fact_status)
        ]
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
        # A retained choice this change leaves unchecked, which its base held checked.
        for index in sorted(
            waiting - requested_nodes - current.unchecked, key=current.linked.ranks.__getitem__
        ):
            published.append(
                ChangeOutcome(
                    current.linked.nodes[index].owner,
                    validation[index],
                    "unchanged",
                    requested=False,
                    unchecked=True,
                )
            )
        return ConfigurationResult(successor, True, tuple(published))


def with_choices(
    point: S, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
) -> S:
    report = try_with_choices(point, *changes, **choices)
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance


def with_facts(
    point: S, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
) -> S:
    report = try_with_facts(point, facts, *changes)
    if not report.accepted:
        raise ConfigurationError(report)
    return report.instance
