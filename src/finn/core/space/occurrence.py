# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The read boundary: typed operations on a design space read at one of its scopes, or on
a method's receiver.

A Space object that reads values is one of two things:

- **a design space read at one of its scopes** (``design_space``, a change): it holds the
  design space value and the scope (``_running.surface.At``). Each read is a driver's
  demand, one question of that value asked of the running type (``_running.surface``,
  ``_running.questions``) in the model's one session, under the model's lock;
- **a receiver**, the ``self`` a ``self`` method runs with: it reads through the method's
  reader (``_world.MethodReader``). Each read is evaluated where the method runs, recorded,
  and checked against what the method's source may read.

A read of a design space during an authored call is that call's failure (a method reads
through its receiver), and so is a driver-only operation.

A change (``try_with_choices``, ``try_with_facts``, ``bind``) normalizes its requests
(``_running.requests``: every form ``with_choices`` takes, as node ids) and asks the
running type's change or bind. Where that publishes the receiver's own design space, the
receiver is returned; a refused change returns the receiver too.

**The world engine, until it is deleted.** ``state`` is the world engine's state of the
design space a Space object holds (``state_of``: the one it published as that object, else
one rebuilt from its facts and stored choices, kept on the object), and
``attach`` places a state of the world engine as a Space object holding its design space
value. The world engine's bodies of the read questions are kept privately, as
``_world_<question>``, taking the same Space objects, so that the two engines can be asked
the same question of the same design space.
"""

# Lazy imports break the declaration/compiled-model/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from . import _state
from ._configuration import BoundDecision, BoundValue, Space
from ._failure import check_transformation, current, driver_only, fault
from ._state import State
from ._world import MethodReader, World
from .compiler import Model
from .declarations import (
    Constraint,
    Declaration,
    Param,
    ValueRef,
    View,
    declared_path,
)
from .edits import Change, ChangeRequest, ConfigurationResult
from .errors import ConfigurationError, EvaluationError, RequestError
from .ir import Choice, Linked, NodeId
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    Unresolved,
    ViewAssessment,
)
from .semantics import snapshot

if TYPE_CHECKING:
    from ._running.evaluate import Session
    from ._running.questions import Report
    from ._running.surface import At
    from ._running.value import DesignSpace

T = TypeVar("T")
S = TypeVar("S", bound=Space)
A = TypeVar("A")


# -- what a Space object holds ------------------------------------------------------------


def _reader(point: Space) -> MethodReader | None:
    """The method reader ``point`` reads through, where it is a method's receiver."""
    found = vars(point).get("_state")
    return found if isinstance(found, MethodReader) else None


def held(point: Space) -> At:
    """The design space value ``point`` holds, read at its scope."""
    from ._running.surface import At

    found = vars(point).get("_state")
    if not isinstance(found, At):
        if declared_path(point) is not None:
            raise RequestError(
                f"{point!r} is a node declaration, not a configuration; open its root's "
                "design space with design_space()"
            )
        raise RequestError("a configuration is created by design_space()")
    return found


def _driver(point: Space) -> At:
    """The design space ``point`` holds, for a read: refused in a value transformation,
    and inside an authored call (it reads through its receiver), sticky there."""
    found = held(point)
    check_transformation()
    active = current()
    if active is not None:
        error = EvaluationError(active.owner, "field read", "cross-configuration access")
        fault(active, error)
        raise error
    return found


def model_of(point: Space) -> Model[Space]:
    """The compiled model a design space or a receiver reads."""
    reader = _reader(point)
    if reader is not None:
        return reader.world.evaluator.model
    found = vars(point).get("_state")
    if isinstance(found, State):  # the world engine's, of a table built by hand
        return found.model
    return held(point).ds.model


def place(at: At) -> Space:
    """The Space object of ``at``: its scope's Space class, holding the design space read
    at that scope."""
    space_type = at.ds.model.linked.scopes[at.scope].space_type
    instance = object.__new__(space_type)
    object.__setattr__(instance, "_state", at)
    object.__setattr__(instance, "_scope", at.scope)
    return instance


def _placed(found: QueryResult[object]) -> QueryResult[object]:
    """An answer whose value is a design space read at a scope, as its Space object."""
    from ._running.surface import At

    if isinstance(found, Available) and isinstance(found.value, At):
        return Available(place(found.value))
    return found


def ask(model: Model[Space], question: Callable[[Session], A]) -> A:
    """``question`` asked in the model's one session (``PerClass.session``), under the
    model's lock: one run of a model at a time, different models at once."""
    with model.lock:
        return question(model.per_class.session())


def opened(model: Model[S], parameters: Mapping[NodeId, object]) -> S:
    """The design space of ``model`` over ``parameters`` (by root formal), every choice
    open, read at its root: all root inputs checked, recognized and frozen before any
    evaluation (``_running.facts.open``)."""
    from ._running.facts import open as open_facts
    from ._running.surface import At

    compiled = cast(Model[Space], model)
    with compiled.lock:
        return cast(S, place(At(open_facts(compiled, parameters), 0)))


def prepare_values(
    linked: Linked,
    pending: Mapping[NodeId, object],
    role: str,
) -> dict[NodeId, object]:
    """Recognize every value before any is snapshot, then snapshot each."""
    return _state.prepare(linked, {index: (value, role) for index, value in pending.items()})


def _compiled(point: Space) -> Model[Space]:
    """The compiled model of a design space (or of a table the world engine holds)."""
    found = vars(point).get("_state")
    if isinstance(found, State):
        return found.model
    return held(point).ds.model


# -- reading a design space -----------------------------------------------------------------


def query(point: Space, reference: object) -> QueryResult[object]:
    """A member's answer, or a node's Space object where the node is present."""
    from ._running import surface

    driver_only("query inspection")
    at = held(point)
    return _placed(ask(at.ds.model, lambda session: surface.query(at, reference, session)))


def present(point: Space, node: object) -> bool:
    """Whether a node, or a value input, is present, read like a value (undecided presence
    raises)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.present(point, node)
    at = _driver(point)
    return ask(at.ds.model, lambda session: surface.present(at, node, session))


def read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    """A member read as a value: the design space's, or, on a receiver, the method's read
    (recorded and checked)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return cast(T, reader.read_value(point, reference))
    at = _driver(point)
    return cast(T, ask(at.ds.model, lambda session: surface.value(at, reference, session)))


@overload
def inspect(point: Space, reference: View[T]) -> ViewAssessment[T]: ...


@overload
def inspect(point: Space, reference: Constraint) -> ConstraintAssessment: ...


def inspect(
    point: Space,
    reference: View[T] | Constraint,
) -> ViewAssessment[T] | ConstraintAssessment:
    from ._running import surface

    driver_only("assessment inspection")
    at = held(point)
    found = ask(at.ds.model, lambda session: surface.inspect(at, reference, session))
    return cast(ViewAssessment[T] | ConstraintAssessment, found)


def decision_state(point: Space, reference: object) -> QueryResult[DecisionState[object]]:
    from ._running import surface

    driver_only("decision state")
    at = held(point)
    return ask(at.ds.model, lambda session: surface.decision_state(at, reference, session))


def decision_index(point: Space, reference: object) -> NodeId:
    return _compiled(point).decision(point._scope, reference)


def candidates(point: Space, reference: object) -> QueryResult[tuple[object, ...]] | None:
    """A choice's listed values (its domain's enumeration, each value its membership
    accepts), in their listed order (None: known by membership only), under its guard."""
    from ._running import surface

    driver_only("domain enumeration")
    at = held(point)
    return ask(at.ds.model, lambda session: surface.candidates(at, reference, session))


def bind_field(point: Space, reference: ValueRef[T] | View[T]) -> BoundValue[T] | BoundDecision[T]:
    """A decision accessor for a Decision, a value accessor for anything else read as a value.

    A view binds as a value accessor: ``get()`` is its accepted value and
    ``query()`` its accepted result, exactly the attribute read and
    ``point.query``; its assessment is ``point.inspect``.
    """
    reader = _reader(point)
    if reader is not None:
        reader._enter()
        model = reader.world.evaluator.model
    else:
        model = _driver(point).ds.model
    model.resolve(point._scope, reference)
    try:
        model.decision(point._scope, reference)
    except RequestError:
        return BoundValue(point, reference)
    return BoundDecision(point, reference)


def root(point: Space) -> Space:
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.root(point)
    at = _driver(point)
    return point if point._scope == 0 else place(surface.root(at))


def child(point: Space, record: Declaration) -> Space:
    """A child node's Space object (on a receiver, the child's receiver)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.child(point, record)
    at = _driver(point)
    return place(surface.child(at, getattr(record, "instance", record)))


def _scope_choice(linked: Linked, scope: int, decision: object) -> Choice:
    """The Choice of a Decision over nodes in compiled scope ``scope``."""
    from ._nodes import unwrap

    record = unwrap(decision)
    choices = linked.scopes[scope].choices
    try:
        return linked.choices[choices[record]]
    except (KeyError, TypeError) as error:
        raise RequestError("the selector is not part of this compiled scope") from error


def selected_candidate(point: Space, decision: Declaration) -> Space | None:
    """The Space object of the candidate a Decision over nodes selects, or None."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.selected_candidate(point, decision)
    at = _driver(point)
    found = ask(at.ds.model, lambda session: surface.selected_candidate(at, decision, session))
    return None if found is None else place(found)


def candidate(point: Space, decision: object, case: str) -> Space | None:
    """A candidate's Space object whether or not it is selected (None for a None case)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.candidate(point, decision, case)
    at = _driver(point)
    found = surface.candidate(at, decision, case)
    return None if found is None else place(found)


# -- changes ----------------------------------------------------------------------------------


def change(point: Space, reference: object, value: T) -> Change[T]:
    """A bound accessor's change: ``reference``'s Decision set to ``value``, on the design
    space ``point`` holds (``Change.base``)."""
    from ._running import requests

    driver_only("change construction")
    return requests.change(held(point), reference, value)


def clear(point: Space, reference: object) -> Change[object]:
    """A bound accessor's clear of ``reference``'s Decision, on the design space ``point``
    holds."""
    from ._running import requests

    driver_only("clear construction")
    return requests.clear(held(point), reference)


def _normalized(at: At, items: Sequence[Change[object]]) -> list[tuple[NodeId, object]]:
    """The requests as the running type's change takes them: each choice set (to its
    value) or cleared (``REMOVE``), in order, by node."""
    from ._running import questions, requests

    return [
        (index, questions.REMOVE if item.remove else item.value)
        for index, item in requests.normalize(at, items).items()
    ]


def _report(point: S, at: At, report: Report) -> ConfigurationResult[S]:
    """A change's report: the Space object of the design space it publishes (the receiver
    where it publishes the receiver's own design space, or is refused), each item's
    outcome, the rules it is refused by or, where accepted, those newly false in the
    published design space, and what it moved."""
    from ._running.surface import At

    if report.published is None:
        return ConfigurationResult(point, False, report.outcomes, report.broken)
    published = (
        point if report.published is at.ds else cast(S, place(At(report.published, at.scope)))
    )
    return ConfigurationResult(
        published,
        True,
        report.outcomes,
        report.broken,
        report.dormant,
        report.in_play,
        report.opened,
    )


def try_with_choices(
    point: S, /, *changes: ChangeRequest | Mapping[Any, object], **choices: object
) -> ConfigurationResult[S]:
    """Set, replace and clear choices as one batch, over the same facts."""
    from ._running import questions, requests

    driver_only("configuration replacement")
    at = held(point)
    keyword_changes = tuple(
        requests._keyword_change(at, name, value) for name, value in choices.items()
    )
    pending = _normalized(at, (*requests._requests(at, changes), *keyword_changes))
    report = ask(at.ds.model, lambda session: questions.change(at.ds, {}, pending, session))
    return _report(point, at, report)


def try_with_facts(
    point: S, facts: Mapping[Any, object], /, *changes: ChangeRequest | Mapping[Any, object]
) -> ConfigurationResult[S]:
    """A new design space from other values for root formals, keeping the choices that
    still fall within their domains, then the choices ``changes`` make on it, as one
    batch. A change of facts is never refused: every stored choice the batch does not
    name, dormant ones included, is checked again against its domain under the new facts,
    and one outside it, or whose domain's input refuses or does not apply, is dropped
    (``"dropped"``, not requested); one waiting on a missing fact keeps waiting. A
    constraint the new facts make false reads refused. The report's ``broken`` names each
    rule newly false in the published design space, by the new facts or revealed by a
    clear beside them, with the stored choices standing in its way. ``changes`` are a
    change of choices on the new design space, refused by the rules of
    ``try_with_choices`` judged against it; a refused batch publishes nothing."""
    from ._running import questions, requests

    driver_only("configuration replacement")
    at = held(point)
    given = requests._fact_nodes(at, facts)
    pending = _normalized(at, requests._requests(at, changes))
    report = ask(at.ds.model, lambda session: questions.change(at.ds, given, pending, session))
    return _report(point, at, report)


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


def bind(point: S, /, *choices: ChangeRequest | Mapping[Any, object]) -> ConfigurationResult[S]:
    """``point``'s facts with exactly the saved ``choices`` (``{reference: value}``
    mappings or ``Change`` objects of ``point``; a clear is a ``RequestError``). Binding
    facts to a Space's params instantiates a design space, and binding saved choices onto
    those facts is the same instantiation, keeping the choices that fall within their
    domains: each value is checked under ``point``'s facts as if its choice existed, and
    one outside its domain, or whose domain's input refuses or does not apply, is dropped
    (``"dropped"``); one waiting on a missing fact is kept, waiting. Each saved choice kept
    is ``"changed"``, as set on the facts with nothing stored, whatever ``point`` holds.
    Never refused for constraints: the report's ``broken`` names every rule the saved
    choices break under these facts (false through an origin the facts with nothing stored
    are not), each with the stored choices standing in its way, and the design space reads
    refused. Its ``dormant``, ``in_play`` and ``opened`` read against the same baseline, so
    the whole report (each outcome, the rules broken, what moved) is the same from any
    design space with these facts. A change of facts (``try_with_facts``) is this
    instantiation under the new facts. The point's own stored choices are replaced and not
    reported, so bind from a design space with nothing stored, as FINN will."""
    from ._running import questions, requests

    driver_only("configuration binding")
    at = held(point)
    normalized = requests.normalize(at, requests._requests(at, choices))
    if any(item.remove for item in normalized.values()):
        raise RequestError("a bind sets saved choices; it clears none")
    saved = [(index, item.value) for index, item in normalized.items()]
    report = ask(at.ds.model, lambda session: questions.bind(at.ds, saved, session))
    return _report(point, at, report)


# -- the world engine's state of a design space (until the world engine is deleted) ----------


_WORLD = ("the world engine's state", None)
"""Where a design space object keeps the world engine's state of it, beside what its
evaluations answered (``DesignSpace.evaluations``): it lives and dies with the object, and
the world engine tells its states apart by the object, as it does."""


def state_of(ds: DesignSpace) -> State:
    """The world engine's state of the design space object ``ds``: the one it published as
    that object (``attach``), else one rebuilt from its facts and stored choices
    (``World.start``), kept on the object."""
    model = ds.model
    with model.lock:
        found = ds.evaluations.get(_WORLD)
        if found is None:
            found = ds.evaluations[_WORLD] = State(
                model, World.start(model.evaluator, ds.facts, ds.stored)
            )
        return cast(State, found)


def state(point: Space) -> State:
    """The world engine's state of the design space ``point`` holds (``state_of``); of a
    table built by hand, the state it holds."""
    found = vars(point).get("_state")
    if isinstance(found, State):
        return found
    return state_of(held(point).ds)


def attach(current: State, scope: int) -> Space:
    """The Space object of ``scope`` in a state of the world engine, holding its design
    space value: its facts layer's design space with nothing stored, with the state's
    stored choices (``with_stored``), ``current`` kept on that new object as its state
    (on the layer's own design space where nothing is stored, unless it holds one already).
    A table built by hand has no design space value: its Space object holds the state."""
    if current.model.structure is None:
        space_type = current.linked.scopes[scope].space_type
        instance = object.__new__(space_type)
        object.__setattr__(instance, "_state", current)
        object.__setattr__(instance, "_scope", scope)
        return instance
    from ._running.surface import At
    from ._running.value import FactsLayer

    model = current.model
    with model.lock:
        ds = FactsLayer.of(model, current.world.facts).bare
        if current.world.stored:
            ds = ds.with_stored(current.world.stored)
        ds.evaluations.setdefault(_WORLD, current)
    return place(At(ds, scope))


def _world_open(model: Model[S], parameters: Mapping[NodeId, object]) -> S:
    """The world engine's opening of a design space (a table built by hand included)."""
    driver_only("configuration binding")
    for index in parameters:
        if index not in model.linked.parameters:
            raise RequestError(f"{model.linked.nodes[index].key} is not an exposed parameter")
    missing = [
        model.linked.nodes[i].key
        for i in model.linked.parameters
        if model.linked.nodes[i].required and i not in parameters
    ]
    if missing:
        raise RequestError(f"missing required parameters: {', '.join(missing)}")
    frozen = prepare_values(model.linked, parameters, "parameter")
    return cast(S, attach(_state.bind(cast(Model[Space], model), frozen), 0))


def _world_driver(point: Space) -> State:
    found = state(point)
    check_transformation()
    active = current()
    if active is not None:
        error = EvaluationError(active.owner, "field read", "cross-configuration access")
        fault(active, error)
        raise error
    return found


def _node_scope(linked: Linked, point: Space, reference: object) -> int | None:
    """For a node reference, the node's scope; None for anything else."""
    from ._nodes import NodeDecl
    from .references import descend

    if not isinstance(reference, Space):
        return None
    path = declared_path(reference)
    if path is None:
        return None
    *outer, last = path
    if not isinstance(last, NodeDecl):
        return None
    scopes = linked.scopes
    holder = descend(scopes, point._scope, outer)
    scope = scopes[holder]
    if last is scope.record:
        return holder
    target = scope.children.get(last)
    if target is None:
        raise RequestError("node is not part of this compiled scope")
    return target


def _world_query(point: Space, reference: object) -> QueryResult[object]:
    driver_only("query inspection")
    current = state(point)
    linked = current.linked
    target = _node_scope(linked, point, reference)
    if target is not None:
        presence = linked.scopes[target].guard
        if presence is not None:
            answer = current.world.result(presence)
            if not isinstance(answer, Available) or answer.value is not True:
                return answer if not isinstance(answer, Available) else Inapplicable()
        return Available(attach(current, target))
    index = current.model.resolve(point._scope, reference)
    return _state.answer(current, index)


def _world_present(point: Space, node: object) -> bool:
    reader = _reader(point)
    if reader is not None:
        return reader.present(point, node)
    current = _world_driver(point)
    target = _node_scope(current.linked, point, node)
    if target is None:
        if isinstance(node, Param):
            return _world_supplied(current, point, node)
        raise RequestError("present() takes a node (a child or a selector's case) or a value input")
    presence = current.linked.scopes[target].guard
    if presence is None:
        return True
    answer = current.world.result(presence)
    if isinstance(answer, Available):
        return answer.value is True
    if isinstance(answer, Inapplicable) or (isinstance(answer, Unresolved) and _unsupplied(answer)):
        return False
    return bool(_state.value(current, presence))  # undecided: raises like any value read


def _world_supplied(current: State, point: Space, formal: Param[object]) -> bool:
    applies = _state.source_applies(current, current.model.resolve(point._scope, formal))
    if isinstance(applies, bool):
        return applies
    _state.value(current, applies)  # undecided: raises like any value read
    raise AssertionError("an undecided presence read halts")


def _unsupplied(answer: Unresolved) -> bool:
    findings = answer.findings
    return bool(findings) and all(finding.code == "input-unsupplied" for finding in findings)


def _world_read_value(point: Space, reference: ValueRef[T] | View[T]) -> T:
    reader = _reader(point)
    if reader is not None:
        return cast(T, reader.read_value(point, reference))
    current = _world_driver(point)
    return cast(T, _state.value(current, current.model.resolve(point._scope, reference)))


def _world_inspect(
    point: Space,
    reference: View[T] | Constraint,
) -> ViewAssessment[T] | ConstraintAssessment:
    driver_only("assessment inspection")
    current = state(point)
    index = current.model.resolve(point._scope, reference)
    node = current.linked.nodes[index]
    if node.kind not in {"view", "constraint"}:
        raise RequestError(
            f"{node.key} is not a view or a constraint: inspect() assesses those; "
            "read or query() any other member"
        )
    assessment = current.world.entry(index).assessment
    assert assessment is not None
    return cast(
        ViewAssessment[T] | ConstraintAssessment,
        _state.copy_assessment(current.linked, index, assessment),
    )


def _world_decision_state(point: Space, reference: object) -> QueryResult[DecisionState[object]]:
    driver_only("decision state")
    return _state.decision_state(state(point), decision_index(point, reference))


def _world_candidates(point: Space, reference: object) -> QueryResult[tuple[object, ...]] | None:
    from ._world import candidates as listed

    driver_only("domain enumeration")
    current = state(point)
    index = decision_index(point, reference)
    result = listed(current.world, index)
    if isinstance(result, Available) and result.value is None:
        return None
    if isinstance(result, Available):
        node = current.linked.nodes[index]
        assert node.semantics is not None
        values = cast(tuple[object, ...], result.value)
        return Available(
            tuple(
                snapshot(node.semantics, value, owner=node.owner, role="public value snapshot")
                for value in values
            )
        )
    return cast(QueryResult[tuple[object, ...]], result)


def _world_root(point: Space) -> Space:
    reader = _reader(point)
    if reader is not None:
        return reader.root(point)
    current = _world_driver(point)
    return point if point._scope == 0 else attach(current, 0)


def _world_child(point: Space, record: Declaration) -> Space:
    reader = _reader(point)
    if reader is not None:
        return reader.child(point, record)
    current = _world_driver(point)
    scope = current.linked.scopes[point._scope]
    child_scope = scope.children.get(record)
    if child_scope is None:
        raise RequestError("child node is not part of this compiled scope")
    return attach(current, child_scope)


def _world_selected_candidate(point: Space, decision: Declaration) -> Space | None:
    reader = _reader(point)
    if reader is not None:
        return reader.selected_candidate(point, decision)
    current = _world_driver(point)
    choice = _scope_choice(current.linked, point._scope, decision)
    case = _state.value(current, choice.selector)
    for key, candidate in choice.cases:
        if key == case:
            return None if candidate is None else attach(current, candidate)
    raise EvaluationError(choice.key, "selection", "selector is not a declared case")


def _world_candidate(point: Space, decision: object, case: str) -> Space | None:
    reader = _reader(point)
    if reader is not None:
        return reader.candidate(point, decision, case)
    current = _world_driver(point)
    if type(case) is not str:
        raise RequestError("a case name must be a string")
    choice = _scope_choice(current.linked, point._scope, decision)
    for key, index in choice.cases:
        if key == case:
            return None if index is None else attach(current, index)
    raise RequestError(f"{choice.key}: unknown case {case!r}")
