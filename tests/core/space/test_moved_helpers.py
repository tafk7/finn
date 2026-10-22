# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The helpers the running type takes from today's engine, copied into ``_running`` by role,
each answering as its original while both exist:

- ``answers``: how an answer is formed and detached (from ``_world`` and ``_state``),
  asked of every node of FINN's toys, each at its design space and after its change;
- ``depth``: the depth guard's limit and exception;
- ``purity``: the purity check's switch, read at each ask;
- ``facts``: the facts' preparation and a design space opened on them (``FactsLayer.bare``);
- ``requests``: a change's requests normalized (from ``_assign``), read against a design
  space at a scope (``surface.At``), where today's read a Space value's state.

One thing differs from today's, as decided: a ``Change`` holds the design space it was made
on, and applies to any design space equal to it (today's: only to the state it was made on,
by identity).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
from support.space_oracle import toys
from support.space_oracle.today import TodayAnswers

from finn.core.space import (
    Available,
    RequestError,
    Space,
    _assign,
    _state,
    _world,
    design_space,
    inspection,
    reject,
)
from finn.core.space._running import answers, depth, facts, purity, questions, requests
from finn.core.space._running.surface import At
from finn.core.space._running.value import FactsLayer
from finn.core.space._structure import Structure
from finn.core.space.edits import Change
from finn.core.space.ir import Node
from finn.core.space.occurrence import prepare_values, state

TOYS = toys.TOYS


def _points(name: str) -> list[Space]:
    """The toy's design space, and the one its change publishes."""
    toy = TOYS[name]
    point = toy.make()
    changed = TodayAnswers(point).change(toy.change, ()).report
    assert changed.accepted
    return [point, changed.instance]


def _nodes(point: Space) -> Iterator[tuple[Structure, Node]]:
    linked = cast(Structure, state(point).model.linked)
    for index in tuple(linked.all_nodes()):
        yield linked, linked.raw_nodes[index]


def _outcome(call: Callable[[], object]) -> object:
    """What ``call`` returns, or the type and message of what it raised."""
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - what was raised is the answer compared
        return (type(error), str(error))


def _called(found: object) -> object:
    """A ``Called`` (today's or the copy) by its fields; anything else as it is."""
    if isinstance(found, (answers.Called, _world.Called)):
        return (
            found.result,
            found.assessment,
            found.own_refusal,
            found.via,
            found.existence,
        )
    return found


# -- answers ----------------------------------------------------------------------------------


def test_the_answers_constants_are_todays() -> None:
    assert answers.INAPPLICABLE == _world.INAPPLICABLE
    assert [field.name for field in answers.Called.__dataclass_fields__.values()] == [
        field.name for field in _world.Called.__dataclass_fields__.values()
    ]


@pytest.mark.parametrize("name", sorted(TOYS))
def test_each_answer_is_detached_as_todays(name: str) -> None:
    for point in _points(name):
        world = state(point).world
        for linked, node in _nodes(point):
            entry = world.entry(node.index)
            found = answers.copy_result(linked, node.index, entry.result)
            assert found == _state.copy_result(linked, node.index, entry.result), node.key
            if isinstance(entry.result, Available):
                assert answers.public_value(node, entry.result) == _world.public_value(
                    node, entry.result
                ), node.key
            if entry.assessment is not None:
                assert answers.copy_assessment(
                    linked, node.index, entry.assessment
                ) == _state.copy_assessment(linked, node.index, entry.assessment), node.key


@pytest.mark.parametrize("name", sorted(TOYS))
def test_each_answer_is_formed_as_todays(name: str) -> None:
    base, changed = _points(name)
    worlds = [state(base).world, state(changed).world]
    for _, node in _nodes(base):
        results = [world.result(node.index) for world in worlds]
        if all(isinstance(result, Available) for result in results):
            left, right = (cast(Available[object], result).value for result in results)
            assert answers._same_value(node, left, right) == _world._same_value(
                node, left, right
            ), node.key
        if node.guard is not None:
            guard = worlds[0].result(node.guard)
            assert _outcome(lambda: answers._guard_result(node, guard)) == _outcome(
                lambda: _world._guard_result(node, guard)
            ), node.key
        given = [worlds[0].result(target) for target in node.dependencies]
        assert answers._blocked(given) == _world._blocked(given), node.key
        assert answers._present(node, given) == _world._present(node, given), node.key
        for result in results:
            if not isinstance(result, Available):
                assert answers._guard_assessment(node, result) == _world._guard_assessment(
                    node, result
                ), node.key
            returned = result.value if isinstance(result, Available) else result
            for value in (returned, result, True, reject("bad", "refused", owner=node.owner)):
                assert _called(_outcome(lambda: answers._finished(node, value))) == _called(
                    _outcome(lambda: _world._finished(node, value))
                ), node.key
        assert answers._unassigned(node) == _world._unassigned(node)
        assert answers._input_missing(node) == _world._input_missing(node)
        assert answers._empty_domain(node) == _world._empty_domain(node)
        assert answers._inputs_absent(node, [node.owner]) == _world._inputs_absent(
            node, [node.owner]
        )
        refusal = reject("bad", "refused", owner=node.owner)
        assert answers._guard_piece(node.owner, refusal) == _world._guard_piece(node.owner, refusal)
        assert answers.membership_arguments(node) == _world.membership_arguments(node)


def test_a_domains_members_are_listed_as_todays() -> None:
    listed = 0
    for point in (point for name in sorted(TOYS) for point in _points(name)):
        world = state(point).world
        for _, node in _nodes(point):
            domain, semantics = node.domain, node.semantics
            if node.kind != "decision" or domain is None or semantics is None:
                continue
            inputs = {
                argument.name: world.result(argument.node)
                for argument in _world.membership_arguments(node)
            }
            if not all(isinstance(answer, Available) for answer in inputs.values()):
                continue
            values = {
                name: cast(Available[object], answer).value for name, answer in inputs.items()
            }
            enumerated = domain.enumerate(values, semantics=semantics, owner=node.owner)
            if not isinstance(enumerated, Available):
                continue
            candidates = tuple(enumerated.value)
            keywords: dict[str, Any] = {"semantics": semantics, "owner": node.owner}
            assert answers._listed_members(
                domain, candidates, values, **keywords
            ) == _world._listed_members(domain, candidates, values, **keywords), node.key
            assert answers._first_member(
                domain, candidates, values, **keywords
            ) == _world._first_member(domain, candidates, values, **keywords), node.key
            listed += 1
    assert listed


# -- the depth guard and the purity check --------------------------------------------------------


def test_the_depth_guard_is_todays() -> None:
    assert depth.DEPTH == _world.DEPTH
    found = depth.Deeper(7)
    assert isinstance(found, BaseException) and not isinstance(found, Exception)
    assert found.node == 7 and found.args == (7,)


def test_the_purity_switch_is_read_at_each_ask_and_answers_as_todays() -> None:
    # The tests' switch flips both (``support.purity``): on for every test here.
    assert purity._PURITY is _world._PURITY is True
    first = purity.check_purity(True)
    try:
        today = _world.check_purity(True)
        for on in (False, True, False):
            assert purity.check_purity(on) == _world.check_purity(on)
            assert purity._PURITY is on is _world._PURITY
    finally:
        purity.check_purity(first)
        _world.check_purity(today)


# -- the facts and a design space opened on them -------------------------------------------------


@pytest.mark.parametrize("name", sorted(TOYS))
def test_a_design_space_opened_on_its_facts_is_the_layers_with_nothing_stored(name: str) -> None:
    point = TOYS[name].make()
    current = state(point)
    model = current.model
    ds = facts.open(model, dict(current.facts))
    assert ds == FactsLayer.of(model, current.facts).bare and not ds.stored
    assert ds.facts == current.facts
    assert ds is facts.open(model, dict(current.facts))  # interned with its layer
    pending = {index: value for index, value in current.facts.items()}
    assert facts.prepare_values(model.linked, pending, "parameter") == prepare_values(
        model.linked, pending, "parameter"
    )
    pairs = {index: (value, "choice") for index, value in current.stored.items()}
    assert facts.prepare(model.linked, pairs) == _state.prepare(model.linked, pairs)


def test_opening_a_design_space_checks_its_facts_as_todays() -> None:
    from finn.core.space.occurrence import _world_open  # noqa: PLC0415 - today's opener

    model = state(design_space(toys.Boundary(width=3))).model
    width, inner = model.linked.keys["width"], model.linked.keys["outside.width"]
    for given in ({width: 4}, {}, {width: 3, inner: 3}, {width: "wide"}):
        mine = _outcome(lambda: dict(facts.open(model, given).facts))
        todays = _outcome(lambda: dict(state(_world_open(model, given)).facts))
        assert mine == todays, given


# -- requests --------------------------------------------------------------------------------


def _at(point: Space, scope: int = 0) -> At:
    """``point``'s design space as a value, read at ``scope``: its facts opened, its stored
    choices bound."""
    current = state(point)
    model = current.model
    ds = facts.open(model, dict(current.facts))
    if current.stored:
        report = questions.bind(ds, list(current.stored.items()), model.per_class.session())
        assert report.published is not None
        ds = report.published
    return At(ds, scope)


def _fields(found: object) -> object:
    """A change by its fields (its base apart); a mapping of them by node; else as it is."""
    if isinstance(found, Change):
        return (found.scope, found.node, found.value, found.remove)
    if isinstance(found, dict):
        return {key: _fields(value) for key, value in found.items()}
    if isinstance(found, list):
        return [_fields(item) for item in found]
    return found


@pytest.mark.parametrize("name", sorted(TOYS))
def test_each_request_names_the_choices_todays_names(name: str) -> None:
    for point in _points(name):
        at = _at(point)
        decisions = inspection.decisions(point)
        for info in decisions:
            reference, value = info.reference, object()
            made = requests.change(at, reference, value)
            assert made.base == at.ds
            assert _fields(made) == _fields(_assign.change(point, reference, value))
            assert _fields(requests.clear(at, reference)) == _fields(
                _assign.clear(point, reference)
            )
        mapping = {info.reference: 1 for info in decisions}
        mine = requests._requests(at, (mapping,))
        todays = _assign._requests(point, (mapping,))
        assert _fields(mine) == _fields(todays)
        assert requests._requests(at, (mine[0],)) == [mine[0]]  # a Change passes through
        assert _fields(requests.normalize(at, mine)) == _fields(
            _assign.normalize(state(point), todays)
        )


@pytest.mark.parametrize("name", sorted(TOYS))
def test_a_keyword_change_names_a_scopes_own_choice_as_todays(name: str) -> None:
    point = TOYS[name].make()
    today = TodayAnswers(point)
    for scope_name in today.scopes():
        try:
            scoped = today.configuration(scope_name)
        except KeyError:
            continue
        at = _at(point, scoped._scope)
        linked = state(point).model.linked
        names = [*linked.scopes[scoped._scope].named_members, "nothing"]
        for member in names:
            assert _fields(_outcome(lambda: requests._keyword_change(at, member, 1))) == _fields(
                _outcome(lambda: _assign._keyword_change(scoped, member, 1))
            ), (scope_name, member)


@pytest.mark.parametrize("name", sorted(TOYS))
def test_the_facts_a_change_names_are_todays(name: str) -> None:
    point = TOYS[name].make()
    at = _at(point)
    for info in inspection.members(point):
        given = {info.reference: 1}
        assert _outcome(lambda: requests._fact_nodes(at, given)) == _outcome(
            lambda: _assign._fact_nodes(point, given)
        ), info.key


def test_normalize_refuses_what_todays_refuses() -> None:
    point = TOYS["house"].make()
    current, at = state(point), _at(point)
    decisions = inspection.decisions(point)
    first = decisions[0].reference
    other = design_space(type(point)(budget=400))  # type: ignore[call-arg]
    foreign = _at(other)
    cases: list[tuple[list[Any], list[Any]]] = [
        ([object()], [object()]),
        ([requests.change(at, first, 1)] * 2, [_assign.change(point, first, 1)] * 2),
        ([requests.change(foreign, first, 1)], [_assign.change(other, first, 1)]),
        (
            [Change(at.ds, 99, 0, 1)],
            [Change(current, 99, 0, 1)],
        ),
        (
            [Change(at.ds, 0, 10_000, 1)],
            [Change(current, 0, 10_000, 1)],
        ),
        (
            [Change(at.ds, 0, current.model.linked.keys["budget"], 1)],
            [Change(current, 0, current.model.linked.keys["budget"], 1)],
        ),
    ]
    for mine, todays in cases:
        found = _outcome(lambda: requests.normalize(at, mine))
        assert isinstance(found, tuple) and found[0] is RequestError
        assert found == _outcome(lambda: _assign.normalize(current, todays))


def test_a_change_applies_to_any_design_space_equal_to_its_own() -> None:
    """Today's applies a change to the state it was made on alone; a change made on
    a design space applies to any design space equal to it (here, the same toy changed
    again)."""
    point, copy = _points("house")[1], _points("house")[1]
    first = inspection.decisions(point)[0].reference
    made = requests.change(_at(point), first, True)
    again = _at(copy)
    assert again.ds == made.base and again.ds is not made.base
    assert list(requests.normalize(again, [made])) == [made.node]
    with pytest.raises(RequestError, match="exact base snapshot"):
        _assign.normalize(state(copy), [_assign.change(point, first, True)])


def test_a_change_made_at_a_childs_scope_applies_at_the_root() -> None:
    point = TOYS["house"].make()
    kitchen = point.kitchen  # type: ignore[attr-defined]
    reference = type(kitchen).finish
    child = _at(point, kitchen._scope)
    made = requests.change(child, reference, 2)
    todays = _assign.change(kitchen, reference, 2)
    assert _fields(made) == _fields(todays)
    root = _at(point)
    assert _fields(requests.normalize(root, [made])) == _fields(
        _assign.normalize(state(point), [todays])
    )
