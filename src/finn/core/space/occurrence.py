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
  reader (``_running.receiver.MethodReader``). Each read is evaluated where the method
  runs, recorded, and checked against what the method's source may read.

A read of a design space during an authored call is that call's failure (a method reads
through its receiver), and so is a driver-only operation.

A change (``try_with_choices``, ``try_with_facts``, ``bind``) normalizes its requests
(``_running.requests``: every form ``with_choices`` takes, as node ids) and asks the
running type's change or bind. Where that publishes the receiver's own design space, the
receiver is returned; a refused change returns the receiver too.
"""

# Lazy imports break the declaration/compiled-model/evaluation cycle.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ._configuration import BoundDecision, BoundValue, Space, _Namespace
from ._failure import check_transformation, current, driver_only, fault
from ._running.receiver import MethodReader
from .compiler import Model
from .declarations import (
    Constraint,
    Declaration,
    ValueRef,
    View,
    declared_path,
)
from .edits import Change, ChangeRequest, ConfigurationResult
from .errors import ConfigurationError, EvaluationError, RequestError
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    QueryResult,
    ViewAssessment,
)

if TYPE_CHECKING:
    from ._running.evaluate import Session
    from ._running.questions import Report
    from ._running.surface import At
    from ._scopes import Address

T = TypeVar("T")
S = TypeVar("S", bound=Space)
A = TypeVar("A")


# -- what a Space object holds ------------------------------------------------------------


def _reader(point: _Namespace) -> MethodReader | None:
    """The method reader ``point`` reads through, where it is a method's receiver."""
    found = vars(point).get("_held")
    return found if isinstance(found, MethodReader) else None


def held(point: _Namespace) -> At:
    """The design space value ``point`` holds, read at its scope."""
    from ._running.surface import At

    found = vars(point).get("_held")
    if not isinstance(found, At):
        if declared_path(point) is not None:
            raise RequestError(
                f"{point!r} is a node declaration, not a configuration; open its root's "
                "design space with design_space()"
            )
        raise RequestError("a configuration is created by design_space()")
    return found


def _driver(point: _Namespace) -> At:
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


def model_of(point: _Namespace) -> Model[Space]:
    """The compiled model a design space or a receiver reads."""
    reader = _reader(point)
    if reader is not None:
        return reader.model
    return held(point).ds.model


def place(at: At) -> _Namespace:
    """The Space object of ``at``: its scope's Space class, holding the design space read
    at that scope."""
    space_type = at.scope.space_type
    instance = object.__new__(space_type)
    object.__setattr__(instance, "_held", at)
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


def opened(model: Model[S], parameters: Mapping[Address, object]) -> S:
    """The design space of ``model`` over ``parameters`` (by root formal), every choice
    open, read at its root: all root inputs checked, recognized and frozen before any
    evaluation (``_running.facts.open``)."""
    from ._running.facts import open as open_facts
    from ._running.surface import At

    compiled = cast(Model[Space], model)
    with compiled.lock:
        return cast(S, place(At(open_facts(compiled, parameters), compiled.scopes.root)))


# -- reading a design space -----------------------------------------------------------------


def query(point: _Namespace, reference: object) -> QueryResult[object]:
    """A member's answer, or a node's Space object where the node is present."""
    from ._running import surface

    driver_only("query inspection")
    at = held(point)
    return _placed(ask(at.ds.model, lambda session: surface.query(at, reference, session)))


def present(point: _Namespace, node: object) -> bool:
    """Whether a node is present, read like a value (undecided presence raises); a value
    input is no node (it is known: test it with ``is not None``)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.present(point, node)
    at = _driver(point)
    return ask(at.ds.model, lambda session: surface.present(at, node, session))


def read_value(point: _Namespace, reference: ValueRef[T] | View[T]) -> T:
    """A member read as a value: the design space's, or, on a receiver, the method's read
    (recorded and checked)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return cast(T, reader.read_value(point, reference))
    at = _driver(point)
    return cast(T, ask(at.ds.model, lambda session: surface.value(at, reference, session)))


@overload
def inspect(point: _Namespace, reference: View[T]) -> ViewAssessment[T]: ...


@overload
def inspect(point: _Namespace, reference: Constraint) -> ConstraintAssessment: ...


def inspect(
    point: _Namespace,
    reference: View[T] | Constraint,
) -> ViewAssessment[T] | ConstraintAssessment:
    from ._running import surface

    driver_only("assessment inspection")
    at = held(point)
    found = ask(at.ds.model, lambda session: surface.inspect(at, reference, session))
    return cast(ViewAssessment[T] | ConstraintAssessment, found)


def decision_state(point: _Namespace, reference: object) -> QueryResult[DecisionState[object]]:
    from ._running import surface

    driver_only("decision state")
    at = held(point)
    return ask(at.ds.model, lambda session: surface.decision_state(at, reference, session))


def decision_index(point: _Namespace, reference: object) -> Address:
    return held(point).ds.model.decision(point._scope, reference)


def candidates(point: _Namespace, reference: object) -> QueryResult[tuple[object, ...]] | None:
    """A choice's listed values (its domain's enumeration, each value its membership
    accepts), in their listed order (None: known by membership only), under its guard."""
    from ._running import surface

    driver_only("domain enumeration")
    at = held(point)
    return ask(at.ds.model, lambda session: surface.candidates(at, reference, session))


def bind_field(
    point: _Namespace, reference: ValueRef[T] | View[T]
) -> BoundValue[T] | BoundDecision[T]:
    """A decision accessor for a Decision, a value accessor for anything else read as a value.

    A view binds as a value accessor: ``get()`` is its accepted value and
    ``query()`` its accepted result, exactly the attribute read and
    ``point.query``; its assessment is ``point.inspect``.
    """
    reader = _reader(point)
    if reader is not None:
        reader._enter()
        model = reader.model
    else:
        model = _driver(point).ds.model
    model.resolve(point._scope, reference)
    try:
        model.decision(point._scope, reference)
    except RequestError:
        return BoundValue(point, reference)
    return BoundDecision(point, reference)


def root(point: _Namespace) -> Space:
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return cast(Space, reader.root(point))
    at = _driver(point)
    return cast(Space, point if point._scope.parent is None else place(surface.root(at)))


def child(point: _Namespace, record: Declaration) -> _Namespace:
    """A child node's Space object (on a receiver, the child's receiver)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return reader.child(point, record)
    at = _driver(point)
    return place(surface.child(at, getattr(record, "instance", record)))


def selected_candidate(point: _Namespace, decision: Declaration) -> Space | None:
    """The Space object of the candidate a Decision over nodes selects, or None."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return cast(Space | None, reader.selected_candidate(point, decision))
    at = _driver(point)
    found = ask(at.ds.model, lambda session: surface.selected_candidate(at, decision, session))
    return None if found is None else cast(Space, place(found))


def candidate(point: _Namespace, decision: object, case: str) -> Space | None:
    """A candidate's Space object whether or not it is selected (None for a None case)."""
    from ._running import surface

    reader = _reader(point)
    if reader is not None:  # a method reads through the evaluation it runs in
        return cast(Space | None, reader.candidate(point, decision, case))
    at = _driver(point)
    found = surface.candidate(at, decision, case)
    return None if found is None else cast(Space, place(found))


# -- changes ----------------------------------------------------------------------------------


def change(point: _Namespace, reference: object, value: T) -> Change[T]:
    """A bound accessor's change: ``reference``'s Decision set to ``value``, on the design
    space ``point`` holds (``Change.base``)."""
    from ._running import requests

    driver_only("change construction")
    return requests.change(held(point), reference, value)


def clear(point: _Namespace, reference: object) -> Change[object]:
    """A bound accessor's clear of ``reference``'s Decision, on the design space ``point``
    holds."""
    from ._running import requests

    driver_only("clear construction")
    return requests.clear(held(point), reference)


def _normalized(at: At, items: Sequence[Change[object]]) -> list[tuple[Address, object]]:
    """The requests as the running type's change takes them: each choice set (to its
    value) or cleared (``REMOVE``), in order, by address."""
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
    (``"dropped"``, not requested). A constraint the new facts make false reads refused.
    The report's ``broken`` names each rule newly false in the published design space, by
    the new facts or revealed by a clear beside them, with the stored choices standing in
    its way. ``changes`` are a change of choices on the new design space, refused by the
    rules of ``try_with_choices`` judged against it; a refused batch publishes nothing."""
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
    (``"dropped"``). Each saved choice kept is ``"changed"``, as set on the facts with
    nothing stored, whatever ``point`` holds.
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
