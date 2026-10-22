# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The switch's shadow run: the public API answering from the running type, with today's
engine asked every question again and compared, a pytest plugin.

    python -m pytest -p no:xdist -p core.space._switch_shadow -q core/space/test_house.py

(from ``tests``). It is the other way round from ``core.space._running_shadow``. While it
is active, each public entry point below answers from the public API, which is the running
type (``finn.core.space._running``), and that answer is the one returned to the test, a
raise included. Today's engine (the world engine, kept privately until it is deleted) is
then asked the same question of the same design space value, and the two answers are
compared. A test sees the public API's behaviour, plus a failure where the two differ.

**What it compares** (each question's kind, as the count names it), as
``core.space._running_shadow`` does, and today's answer through the world engine's private
paths:

- ``design_space``: the design space opened, against today's opening of the same model and
  root values (``occurrence._world_open``);
- a change (``change``: ``try_with_choices``, which ``with_choices`` asks through; ``change
  of facts``: ``try_with_facts``, which ``with_facts`` asks through) and ``bind``: each
  report (every outcome, the rules broken, what moved) and the design space it publishes,
  by value, against today's assignment (``_assign.assign``, ``_assign.bound``) of the same
  requests;
- ``inspection``'s answers about a design space: ``validity``, ``states``, ``viable``,
  ``explain`` and ``candidate`` (``inspection._world_<question>``,
  ``occurrence._world_candidate``);
- a Space object's own questions: ``query``, ``present``, ``inspect``, a member read as a
  value (``value``), ``field`` (whether the accessor is a Decision's), a Decision's
  ``state`` and ``candidates``, ``root``, ``child`` and the candidate a Decision over
  nodes selects (``selected``) (``occurrence._world_<question>``).

A Space object in an answer (a child's, a candidate's, a change's) compares as its design
space value and scope: the public API's holds a design space (``surface.At``); today's
holds a state of the world engine, read as its model, facts and stored choices.

**Today's world.** Today's engine answers each design space from a state of its own, made
the way ``occurrence.state_of`` makes one (``World.start`` of the design space's facts and
stored choices) but held by the shadow, by the design space object, for the test (a weak
reference: it goes with the design space). It is never ``occurrence.state(p)``'s, which
the shadow would plant on the test's own design space object and share with the tests
that read today's engine. Where a change the two agree on publishes a design space, the
state today's change published answers it from then on, so today's engine reads its own
history. While today's side runs (and only there, and only in the shadow's thread), the
world engine's ``occurrence.state``, ``state_of`` and ``attach`` (and ``_assign``'s and
``inspection``'s names for them) read the shadow's states, and ``attach`` places a state
in a Space object of its own, which holds the state and touches no design space value.

**What the test sees is unchanged, and so are its counts.** The public API's answer is
returned as it was given, and today's side runs after it, on its own states, never in
the model's session (``PerClass.session``): a public question asked inside today's
evaluation (an authored function opening a design space of its own) answers in a scratch
session made for that run of today's side, never the model's. Each run of today's side
is checked: the model's session (whether it exists, and its counters, which the
count-of-work tests read) and the keys of the asked design space's evaluations must be as
they were before it. A change is an **isolation break**, failing the test as a
disagreement does.

**The caller's input** to a change or a bind is read before the public API is asked (the
requests by node, ``_running.requests``), and each value snapshotted as a design space
holds it (``_running.facts.prepare``): the public API's run may change the caller's
objects, and today's engine is asked what the public API was. A root's values, for
``design_space``, likewise. Where recognizing or snapshotting raises, the values go as
given.

**Comparing**, by ``core.space._running_shadow``'s forms and rules: the oracle's canonical
forms where they apply, else field by field, in order or sorted, and a value with no form
by ``==``. Its one named difference holds: ``explain``'s fact reads compare as a set (in
key order in the running type, in the process's first-read order in today's), counted
apart. A raise on one side and an answer on the other is a disagreement; both raising
compare by the error's type and message.

**A disagreement** fails the test at the end of the phase it was met in (setup, call or
teardown), naming the question, the member or members asked, the design space (its Space
class, its scope and its stored choices) and both answers, short (``today's | running``).
One met outside any test fails the session.

**The purity check** is on for every test, both switches (``support.purity``), unless the
test turns it off (``purity_unchecked``).

**The count**, at the session's end: the questions left to the public API alone, by
reason, and the fact-read lists compared as sets; then the questions compared, by kind,
the disagreements and the isolation breaks (both 0 to pass)::

    switch shadow: 1234 questions compared (change 12, ...); disagreements: 0; isolation breaks: 0

**What it leaves out**, as ``core.space._running_shadow`` does: the structural questions;
a question a method asks of its receiver, or one asked during an authored call or a value
transformation; a question asked while another is answered, or inside today's side; one
asked from another thread; a Space object holding a state of today's engine (a table built
by hand); a request refused before any question is asked (a malformed or foreign change, a
clear in a bind, an unknown fact) and a root the front end refuses to open; and every
question of the tests that make the running type answer wrongly on purpose
(``_running_shadow.UNCOMPARED``).
"""

from __future__ import annotations

import dataclasses
import enum
import sys
import threading
import weakref
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from functools import wraps
from types import ModuleType
from typing import Any, cast

import pytest
from support.purity import check_purity, restore_purity
from support.space_oracle import forms

from core.space._running_shadow import (
    _RENDERED,
    ASKED,
    CHANGES,
    UNCOMPARED,
    Disagreement,
    Outcome,
    Requests,
    Same,
    _asked,
    _differences,
    _facts_as_a_set,
    _short,
    _sorted,
    _t0,
)
from finn.core.reporting.findings import Finding
from finn.core.space import _assign, _failure, compiler, inspection, occurrence
from finn.core.space import _world as W
from finn.core.space._configuration import BoundDecision, Space
from finn.core.space._nodes import unsupplied_formals
from finn.core.space._running import facts as F
from finn.core.space._running import questions as Q
from finn.core.space._running import requests as R
from finn.core.space._running import surface as S
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Session
from finn.core.space._running.per_class import PerClass
from finn.core.space._running.value import DesignSpace, _raw, _same
from finn.core.space._state import State
from finn.core.space._structure import Structure
from finn.core.space.edits import Change, ConfigurationResult
from finn.core.space.errors import RequestError
from finn.core.space.ir import NodeId
from finn.core.space.references import ValueHandle
from finn.core.space.results import Available, Inapplicable, Rejected, Unresolved

OUTSIDE = "(outside any test)"
"""Where a question asked between tests is counted."""


class SwitchDisagreement(AssertionError):
    """Questions today's engine answered otherwise than the public API, or a run of today's
    side that changed what the test reads, in one phase of a test."""


# -- canonical forms ---------------------------------------------------------------------


Value = tuple[Any, Mapping[NodeId, object], Mapping[NodeId, object]]
"""A design space value's parts: its model, facts and stored choices."""


class SameValue(Same):
    """A design space value: its model, facts and stored choices (``(model, facts,
    stored)``), compared as ``DesignSpace`` compares (the same model, each fact and stored
    choice by its member's semantics); where an authored equality raises, by the facts' and
    stored choices' forms."""

    __slots__ = ()

    @property
    def parts(self) -> Value:
        return cast(Value, self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SameValue):
            return NotImplemented
        mine, theirs = self.parts, other.parts
        if mine[0] is not theirs[0]:
            return False
        try:
            nodes = _raw(mine[0])
            return bool(_same(nodes, mine[1], theirs[1]) and _same(nodes, mine[2], theirs[2]))
        except Exception:  # noqa: BLE001 - an authored equality that raises
            return all(
                form(dict(left)) == form(dict(right))
                for left, right in ((mine[1], theirs[1]), (mine[2], theirs[2]))
            )

    __hash__ = None

    def __repr__(self) -> str:
        model, facts, stored = self.parts
        key = model.linked.header

        def shown(values: Mapping[NodeId, object]) -> str:
            return ", ".join(f"{key(index).key}={value!r}" for index, value in values.items())

        return (
            f"{model.space_type.__qualname__}(facts {{{shown(facts)}}}, stored {{{shown(stored)}}})"
        )


@dataclass(frozen=True)
class Held:
    """A design space value at one of its scopes, in a form: equal by value."""

    space: SameValue
    scope: int


def _held(point: Space) -> Held | None:
    """What a Space object holds, as a value: the public API's design space, or a state of
    today's engine; ``None`` for a receiver or a declaration."""
    found = vars(point).get("_state")
    if isinstance(found, S.At):
        return Held(SameValue((found.ds.model, found.ds.facts, found.ds.stored)), found.scope)
    if isinstance(found, State):
        return Held(SameValue((found.model, found.facts, found.stored)), point._scope)
    return None


def form(found: object) -> object:
    """``found`` in a form compared by ``==`` (``core.space._running_shadow.form``'s rules):
    a Space object as the design space value it holds and its scope."""
    if isinstance(found, Space):
        held = _held(found)
        return Same(found) if held is None else held
    if isinstance(found, S.At):
        return Held(SameValue((found.ds.model, found.ds.facts, found.ds.stored)), found.scope)
    if isinstance(found, Available):
        return {"kind": "value", "value": form(found.value)}
    if isinstance(found, (Unresolved, Rejected, Inapplicable)):
        rendered = _t0(forms.answer, found)
        if rendered is not None:
            return rendered
        return {"kind": type(found).__name__, "findings": _sorted(map(form, found.findings))}
    for kinds, render in _RENDERED:
        if isinstance(found, kinds):
            rendered = _t0(render, found)
            if rendered is not None:
                return rendered
    if isinstance(found, ValueHandle):
        header = found._linked.header(found._node)
        return {"handle": header.key, "kind": header.kind}
    if isinstance(found, Finding):
        rendered = _t0(forms.finding, found)
        return Same(found) if rendered is None else rendered
    return _plain(found)


def _plain(found: object) -> object:
    """A value's form: the oracle's rules (``forms.value``), each part in ``form``."""
    if isinstance(found, enum.Enum):
        return {"enum": found.name}
    if found is None or isinstance(found, (bool, str, int)):
        return found
    if isinstance(found, float):
        return {"float": repr(found)}
    if dataclasses.is_dataclass(found) and not isinstance(found, type):
        return {
            "type": type(found).__qualname__,
            "fields": {
                item.name: form(getattr(found, item.name))
                for item in dataclasses.fields(found)
                if item.compare
            },
        }
    if isinstance(found, Mapping):
        return {"mapping": _sorted([form(key), form(item)] for key, item in found.items())}
    if isinstance(found, AbstractSet):
        return {"set": _sorted(map(form, found))}
    if isinstance(found, Sequence) and not isinstance(found, (bytes, bytearray)):
        return [form(item) for item in found]
    if not isinstance(found, type) and all(
        hasattr(found, name) for name in ("dtype", "shape", "tolist")
    ):
        array = cast(Any, found)
        return {
            "array": {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "contents": form(array.tolist()),
            }
        }
    rendered = _t0(forms.value, found)
    return Same(found) if rendered is None else rendered


def _describe(model: Any, stored: Mapping[NodeId, object], scope: int) -> str:
    """A design space, short: its Space class, its scope and its stored choices."""
    linked = model.linked
    held = linked.scopes[scope]
    shown = ", ".join(f"{linked.header(index).key}={value!r}" for index, value in stored.items())
    where = f" at {held.name}" if scope else ""
    return _short(f"{held.space_type.__qualname__}{where} {{{shown}}}")


# -- today's answer to each question -----------------------------------------------------


Today = Callable[..., object]
"""Today's answer to a question: of the Space object asked and the question's arguments
after it, as the public entry point takes them."""


def _today_field(point: Space, reference: object) -> bool:
    """Today's ``bind_field``, as the world engine's driver read it: whether the accessor
    is a Decision's."""
    model = occurrence._world_driver(point).model
    model.resolve(point._scope, reference)
    try:
        model.decision(point._scope, reference)
    except RequestError:
        return False
    return True


def _late(owner: ModuleType, name: str) -> Today:
    """``owner``'s ``name``, read when asked (so a test may replace it)."""

    def ask(*arguments: Any) -> object:
        return getattr(owner, name)(*arguments)

    ask.__qualname__ = f"{owner.__name__}.{name}"
    return ask


TODAY: dict[str, Today] = {
    "query": _late(occurrence, "_world_query"),
    "present": _late(occurrence, "_world_present"),
    "value": _late(occurrence, "_world_read_value"),
    "inspect": _late(occurrence, "_world_inspect"),
    "state": _late(occurrence, "_world_decision_state"),
    "candidates": _late(occurrence, "_world_candidates"),
    "field": _today_field,
    "root": _late(occurrence, "_world_root"),
    "child": _late(occurrence, "_world_child"),
    "selected": _late(occurrence, "_world_selected_candidate"),
    "candidate": _late(occurrence, "_world_candidate"),
    "validity": _late(inspection, "_world_validity"),
    "states": _late(inspection, "_world_states"),
    "viable": _late(inspection, "_world_viable"),
    "explain": _late(inspection, "_world_explain"),
}
"""Today's answer to each question, by kind: the world engine's body, kept privately."""

PUBLIC: dict[str, Callable[[object], object]] = {
    "field": lambda accessor: isinstance(accessor, BoundDecision),
}
"""The public API's answer as today's is compared, where they differ in type: a bound
accessor by whether it is a Decision's."""


@dataclass(frozen=True)
class Opening:
    """What ``design_space`` opens, read before the public API is asked: the root's model
    and its values by root formal, snapshotted."""

    model: Any
    parameters: dict[NodeId, object]


def _opening(node: object) -> Opening:
    """The model and root values ``design_space(node)`` opens, read by its front end, which
    raises where it refuses the root."""
    record = compiler.root_record(node)
    if any(True for _ in unsupplied_formals(record)):
        raise RequestError("a formal is unsupplied")
    model = compiler.compile_model(record.space_type)
    parameters = compiler.root_parameters(model, record)
    try:
        parameters = F.prepare_values(model.linked, parameters, "parameter")
    except Exception:  # noqa: BLE001 - today's engine meets it on the values as given
        pass
    return Opening(model, parameters)


def _requests(
    point: Space,
    changes: tuple[Any, ...],
    choices: Mapping[str, object] | None,
    facts: Mapping[Any, object] | None,
    *,
    bind: bool = False,
) -> Requests:
    """The items of a change of ``point`` or of a ``bind``, by node, as the public API's
    front end reads them (``_running.requests``), which raises where the request is
    refused before any question is asked (a clear in a bind too); each value snapshotted
    (``_snapshot``)."""
    at = occurrence.held(point)
    given = {} if facts is None else R._fact_nodes(at, facts)
    items = [
        *R._requests(at, changes),
        *(R._keyword_change(at, name, value) for name, value in (choices or {}).items()),
    ]
    normalized = R.normalize(at, items)
    if bind and any(item.remove for item in normalized.values()):
        raise RequestError("a bind sets saved choices; it clears none")
    requests = Requests(
        given,
        [(index, Q.REMOVE if item.remove else item.value) for index, item in normalized.items()],
    )
    return _snapshot(at.ds.model.linked, requests)


def _snapshot(linked: Any, requests: Requests) -> Requests:
    """``requests`` with each value as a design space holds it (``_running.facts.prepare``),
    taken before the public API is asked; where recognizing or snapshotting raises,
    ``requests`` as given, for today's engine to meet the same raise."""
    pending = {
        **{index: (value, "parameter") for index, value in requests.facts.items()},
        **{index: (value, "choice") for index, value in requests.choices if value is not Q.REMOVE},
    }
    try:
        held = F.prepare(linked, pending)
    except Exception:  # noqa: BLE001 - today's engine meets it on the values as given
        return requests
    return Requests(
        {index: held[index] for index in requests.facts},
        [(index, value if value is Q.REMOVE else held[index]) for index, value in requests.choices],
    )


def _first(arguments: tuple[Any, ...]) -> Iterable[object]:
    return arguments[:1]


def _same_answer(answer: object) -> object:
    return answer


def _placed(current: State, scope: int) -> Space:
    """A Space object of ``scope`` holding the state ``current`` of today's engine (no
    design space value: nothing of the running type's is touched)."""
    space_type = current.linked.scopes[scope].space_type
    found = object.__new__(space_type)
    object.__setattr__(found, "_state", current)
    object.__setattr__(found, "_scope", scope)
    return found


# -- the shadow ----------------------------------------------------------------------------


def _keys(keys: Iterable[object]) -> str:
    """Evaluation keys, short: a session's by the guards it takes as holding."""
    shown = [
        f"(a session, {key[1]!r})"
        if isinstance(key, tuple) and key and isinstance(key[0], Session)
        else repr(key)
        for key in keys
    ]
    return _short("{" + ", ".join(sorted(shown)) + "}", 200) if shown else "nothing"


@dataclass(frozen=True)
class Marks:
    """What today's side must leave as it found it: the model's session (whether it is
    made, and its counters) and the keys of the asked design space's evaluations."""

    session: object
    counters: object
    evaluations: frozenset[object]

    def changes(self, after: Marks) -> list[str]:
        """What differs in ``after``, each line ``what: before | after``."""
        found = []
        if self.session != after.session:
            now = "none" if after.session is None else "another" if self.session else "made"
            found.append(f"session: {'none' if self.session is None else 'made'} | {now}")
        mine, theirs = (
            cast(dict[str, object], self.counters),
            cast(dict[str, object], after.counters),
        )
        if mine is not None and theirs is not None:
            found += [
                f"counters: {name} {mine[name]!r} | {theirs[name]!r}"
                for name in mine
                if mine[name] != theirs[name]
            ]
        if self.evaluations != after.evaluations:
            found.append(
                f"evaluations: {_keys(self.evaluations - after.evaluations)} | "
                f"{_keys(after.evaluations - self.evaluations)}"
            )
        return found

    @staticmethod
    def of(ds: DesignSpace | None) -> Marks | None:
        if ds is None:
            return None
        session = ds.model.per_class._session
        counters = None if session is None else dataclasses.asdict(session.counters)
        return Marks(None if session is None else id(session), counters, frozenset(ds.evaluations))


@dataclass
class Shadow:
    """The public entry points wrapped (``apply``) so that each answers from the public API
    and today's engine is asked again and compared, until ``restore``. ``today`` replaces
    today's answer to a question by kind (``TODAY``'s by default)."""

    today: dict[str, Today] = field(default_factory=dict)
    compared: Counter[str] = field(default_factory=Counter)
    left: Counter[str] = field(default_factory=Counter)
    facts_as_sets: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)
    breaks: list[Disagreement] = field(default_factory=list)
    pending: list[Disagreement] = field(default_factory=list)
    test: str = OUTSIDE
    depth: int = 0
    active: bool = False
    saved: list[tuple[object, str, object]] = field(default_factory=list)
    worlds: dict[int, tuple[weakref.ref[DesignSpace], State]] = field(default_factory=dict)
    scratch: dict[int, Session] = field(default_factory=dict)
    thread: int = field(default_factory=threading.get_ident)

    def __post_init__(self) -> None:
        self.today = {**TODAY, **self.today}

    # -- a test ------------------------------------------------------------------------

    def begin(self, test: str) -> None:
        """A test's start: today's states of its own."""
        self.end()
        self.test = test

    def end(self) -> None:
        """A test's end: today's states it made dropped."""
        self.worlds.clear()
        self.scratch.clear()
        self.test = OUTSIDE

    def take_pending(self) -> list[Disagreement]:
        found, self.pending = self.pending, []
        return found

    # -- today's states ----------------------------------------------------------------

    def here(self) -> bool:
        """Whether today's side runs, in this thread."""
        return self.active and threading.get_ident() == self.thread

    def world(self, ds: DesignSpace) -> State:
        """Today's state of the design space object ``ds``: the one the shadow keeps for it,
        else one started from its facts and stored choices (``World.start``), kept."""
        found = self.worlds.get(id(ds))
        if found is not None and found[0]() is ds:
            return found[1]
        model = ds.model
        with model.lock:
            made = State(model, W.World.start(model.evaluator, ds.facts, ds.stored))
        self.keep(ds, made)
        return made

    def keep(self, ds: DesignSpace, current: State) -> None:
        """``current`` as today's state of ``ds`` while ``ds`` lives (or until the test's
        end)."""
        key = id(ds)
        worlds = self.worlds

        def gone(reference: weakref.ref[DesignSpace]) -> None:
            entry = worlds.get(key)
            if entry is not None and entry[0] is reference:
                del worlds[key]

        worlds[key] = (weakref.ref(ds, gone), current)

    # -- wrapping ------------------------------------------------------------------------

    def apply(self) -> None:
        """Wrap every entry point compared where it is defined, and wherever a module
        outside the engine holds it by name; route today's engine's state, state_of and
        attach to the shadow's states, and the model's session to a scratch one, while
        today's side runs."""
        wrappers: dict[int, Callable[..., Any]] = {}
        for owner, name, kind in ENTRY_POINTS:
            original = getattr(owner, name)
            wrapped = wrappers.get(id(original))
            if wrapped is None:
                wrapped = wrappers[id(original)] = self._wrapped(kind, original)
            self._set(owner, name, wrapped)
        for module in list(sys.modules.values()):
            if not isinstance(module, ModuleType) or module.__name__.startswith(_ENGINE):
                continue
            for name, value in list(vars(module).items()):
                wrapped = wrappers.get(id(value))
                if wrapped is not None and getattr(wrapped, "__wrapped__", None) is value:
                    self._set(module, name, wrapped)
        for owner, name, route in (
            (occurrence, "state", self._routed_state),
            (occurrence, "state_of", self._routed_state_of),
            (occurrence, "attach", self._routed_attach),
            (_assign, "state", self._routed_state),
            (_assign, "state_of", self._routed_state_of),
            (_assign, "attach", self._routed_attach),
            (inspection, "state", self._routed_state),
        ):
            self._set(owner, name, route(getattr(owner, name)))
        self._set(PerClass, "session", self._routed_session(PerClass.session))

    def _set(self, owner: object, name: str, value: object) -> None:
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self) -> None:
        """Every entry point and route as it was."""
        while self.saved:
            owner, name, value = self.saved.pop()
            setattr(owner, name, value)

    def _routed_state(self, original: Callable[[Space], State]) -> Callable[[Space], State]:
        shadow = self

        @wraps(original)
        def state(point: Space) -> State:
            if not shadow.here():
                return original(point)
            found = vars(point).get("_state")
            if isinstance(found, State):
                return found
            return shadow.world(occurrence.held(point).ds)

        return state

    def _routed_state_of(
        self, original: Callable[[DesignSpace], State]
    ) -> Callable[[DesignSpace], State]:
        shadow = self

        @wraps(original)
        def state_of(ds: DesignSpace) -> State:
            return shadow.world(ds) if shadow.here() else original(ds)

        return state_of

    def _routed_attach(
        self, original: Callable[[State, int], Space]
    ) -> Callable[[State, int], Space]:
        shadow = self

        @wraps(original)
        def attach(current: State, scope: int) -> Space:
            return _placed(current, scope) if shadow.here() else original(current, scope)

        return attach

    def _routed_session(
        self, original: Callable[[PerClass], Session]
    ) -> Callable[[PerClass], Session]:
        shadow = self

        @wraps(original)
        def session(record: PerClass) -> Session:
            if not shadow.here():
                return original(record)
            found = shadow.scratch.get(id(record))
            if found is None:
                found = shadow.scratch[id(record)] = Session(cache=NodeCache())
            return found

        return session

    def _wrapped(self, kind: str, original: Callable[..., Any]) -> Callable[..., Any]:
        shadow = self

        @wraps(original)
        def shadowed(*arguments: Any, **keywords: Any) -> Any:
            reason = shadow._leaves(kind, arguments)
            if reason is not None:
                shadow.left[reason] += 1
                return original(*arguments, **keywords)
            if kind == "viable" and len(arguments) > 1:
                # the keys read twice: an iterator becomes a tuple
                arguments = (arguments[0], tuple(arguments[1]), *arguments[2:])
            shadow.depth += 1
            try:
                before = shadow._before(kind, arguments, keywords)
                try:
                    public = Outcome(original(*arguments, **keywords))
                except Exception as error:  # noqa: BLE001 - compared, then raised again
                    public = Outcome(error=error)
                except BaseException:
                    # a cancellation is no answer: carried through as raised, its object
                    # (which today's engine would annotate again) never asked of today's
                    shadow.left["a cancellation, carried through"] += 1
                    raise
                # Outside the handler: today's side runs with no exception being handled.
                shadow._guarded(kind, arguments, before, public)
                if public.error is not None:
                    raise public.error
                return public.answer
            finally:
                shadow.depth -= 1

        return shadowed

    def _leaves(self, kind: str, arguments: tuple[Any, ...]) -> str | None:
        """Why a call is left to the public API alone, or ``None`` to compare it."""
        if threading.get_ident() != self.thread:
            return "asked from another thread"
        if kind != "design_space":
            point = arguments[0] if arguments else None
            found = vars(point).get("_state") if isinstance(point, Space) else None
            if isinstance(found, State):
                return "a Space object holding today's state"
            if not isinstance(found, S.At):
                return "a receiver's or a declaration's"
        if self.active:
            return "asked inside today's engine"
        if _failure.current() is not None or _failure._TRANSFORM.get() is not None:
            return "inside an authored call"
        if self.depth:
            return "inside another question"
        if self.test in UNCOMPARED:
            return "a test making the running type wrong on purpose"
        return None

    def _before(self, kind: str, arguments: tuple[Any, ...], keywords: dict[str, Any]) -> object:
        """What today's engine is asked, read before the public API's call may change it: a
        root's model and values (``design_space``, which freezes the root); a change's or a
        bind's items, snapshotted (``_requests``). ``None`` where the front end refuses
        it, and it is not compared."""
        try:
            if kind in CHANGES:
                point = cast(Space, arguments[0])
                return _requests(
                    point, *CHANGES[kind](arguments[1:], keywords), bind=kind == "bind"
                )
            if kind == "design_space" and arguments:
                return _opening(arguments[0])
            return None
        except Exception:  # noqa: BLE001 - the front end refuses it, and it is not compared
            return None

    # -- today's side --------------------------------------------------------------------

    def _today(self, ds: DesignSpace | None, ask: Callable[[], object], kind: str) -> Outcome:
        """``ask`` as today's side: the shadow's states and a scratch session routed in for
        its run; what it raised, as its outcome. The model's session and ``ds``'s
        evaluations checked unchanged after it (an isolation break, else)."""
        before = Marks.of(ds)
        self.active = True
        try:
            try:
                found = Outcome(ask())
            except BaseException as error:  # noqa: BLE001 - a raise is an answer to compare
                found = Outcome(error=cast(Exception, error))
        finally:
            self.active = False
            self.scratch.clear()
        after = Marks.of(ds)
        if before != after:
            assert ds is not None and before is not None and after is not None
            broken = Disagreement(
                self.test,
                kind,
                "(today's side changed what the test reads: before | after)",
                _describe(ds.model, ds.stored, 0),
                before.changes(after),
            )
            self.breaks.append(broken)
            self.pending.append(broken)
        return found

    # -- comparing -----------------------------------------------------------------------

    def _guarded(
        self, kind: str, arguments: tuple[Any, ...], before: object, public: Outcome
    ) -> None:
        """``_compare``, never raising into the caller: a failure of the comparison itself
        is a disagreement, so the public API's answer always reaches the caller."""
        try:
            self._compare(kind, arguments, before, public)
        except Exception as error:  # noqa: BLE001 - reported, never raised into the test
            found = Disagreement(
                self.test, kind, "(not compared)", "?", [f"the comparison raised {error!r}"]
            )
            self.disagreements.append(found)
            self.pending.append(found)

    def _compare(
        self, kind: str, arguments: tuple[Any, ...], before: object, public: Outcome
    ) -> None:
        if kind == "design_space":
            self._compare_opened(before, public)
            return
        point = cast(Space, arguments[0])
        at = occurrence.held(point)
        rest = arguments[1:]
        if kind in CHANGES:
            if not isinstance(before, Requests):
                self.left["refused before any question"] += 1
                return
            self._compare_change(kind, point, at, before, public)
            return
        # the public API's answer in its form first, before today's side runs
        held = public.form(lambda answer: form(PUBLIC.get(kind, _same_answer)(answer)))
        ask = self.today[kind]
        today = self._today(at.ds, lambda: ask(point, *rest), kind)
        self.compared[kind] += 1
        answered = today.form(form)
        if held == answered:
            return
        if kind == "explain" and _facts_as_a_set(held) == _facts_as_a_set(answered):
            self.facts_as_sets += 1
            return
        asked = _asked(at, ASKED.get(kind, _first)(rest))
        self._disagree(kind, asked, _describe(at.ds.model, at.ds.stored, at.scope), answered, held)

    def _compare_change(
        self, kind: str, point: Space, at: S.At, requests: Requests, public: Outcome
    ) -> None:
        def ask() -> tuple[_assign.Assigned, ConfigurationResult[Space]]:
            current = self.world(at.ds)
            nodes = current.linked.nodes
            items = [
                Change(current, nodes[index].scope, index, remove=True)
                if value is Q.REMOVE
                else Change(current, nodes[index].scope, index, value)
                for index, value in requests.choices
            ]
            if kind == "bind":
                assigned = _assign.bound(current, items)
            else:
                assigned = _assign.assign(current, requests.facts, items)
            return assigned, _assign._report(point, assigned)

        def shaped(found: Any) -> object:
            if isinstance(found, tuple):
                found = found[1]
            result = cast(ConfigurationResult[Space], found)
            return {"report": forms.report(result), "space": form(result.instance)}

        held = public.form(shaped)
        today = self._today(at.ds, ask, kind)
        self.compared[kind] += 1
        answered = today.form(shaped)
        if held != answered:
            linked = at.ds.model.linked
            asked = ", ".join(
                linked.header(index).key for index in (*requests.facts, *dict(requests.choices))
            )
            where = _describe(at.ds.model, at.ds.stored, at.scope)
            self._disagree(kind, asked or "(nothing)", where, answered, held)
            return
        if public.error is None and today.error is None:
            assigned = cast(tuple[_assign.Assigned, Any], today.answer)[0]
            published = vars(cast(ConfigurationResult[Space], public.answer).instance).get("_state")
            if assigned.state is not None and isinstance(published, S.At):
                self.keep(published.ds, assigned.state)

    def _compare_opened(self, before: object, public: Outcome) -> None:
        if not isinstance(before, Opening):
            self.left["refused by the front end"] += 1
            return
        if not isinstance(before.model.linked, Structure):
            self.left["a model built as a table"] += 1
            return
        opened = None
        if public.error is None:
            found = vars(cast(Space, public.answer)).get("_state")
            opened = found.ds if isinstance(found, S.At) else None
        held = public.form(form)
        today = self._today(
            opened, lambda: occurrence._world_open(before.model, before.parameters), "design_space"
        )
        self.compared["design_space"] += 1
        answered = today.form(form)
        if held != answered:
            where = _describe(before.model, {}, 0)
            self._disagree("design_space", "(the design space)", where, answered, held)

    def _disagree(self, kind: str, asked: str, where: str, today: object, public: object) -> None:
        found = Disagreement(self.test, kind, asked, where, _differences(today, public))
        self.disagreements.append(found)
        self.pending.append(found)

    # -- the count -----------------------------------------------------------------------

    def count(self) -> str:
        """The summary's line: the questions compared, by kind, the disagreements and the
        isolation breaks."""
        total = sum(self.compared.values())
        kinds = ", ".join(f"{kind} {number}" for kind, number in sorted(self.compared.items()))
        return (
            f"switch shadow: {total} questions compared ({kinds or 'none'}); "
            f"disagreements: {len(self.disagreements)}; isolation breaks: {len(self.breaks)}"
        )

    def left_alone(self) -> str:
        """The summary's other line: the questions left to the public API alone, by reason,
        and the fact-read lists compared as sets."""
        reasons = ", ".join(f"{reason} {number}" for reason, number in sorted(self.left.items()))
        return (
            f"switch shadow: left to the public API alone: {reasons or 'none'}; "
            f"explain's fact reads compared as a set: {self.facts_as_sets}"
        )


_ENGINE = "finn.core.space."
"""The engine's own modules, whose names for the entry points are left as they are: a
question the engine asks itself is part of the one asked."""

_SPACE = sys.modules["finn.core.space"]

ENTRY_POINTS: tuple[tuple[object, str, str], ...] = (
    (compiler, "design_space", "design_space"),
    (_SPACE, "design_space", "design_space"),
    (occurrence, "try_with_choices", "change"),
    (occurrence, "try_with_facts", "change of facts"),
    (occurrence, "bind", "bind"),
    (_SPACE, "bind", "bind"),
    (inspection, "validity", "validity"),
    (inspection, "states", "states"),
    (inspection, "viable", "viable"),
    (inspection, "explain", "explain"),
    (inspection, "candidate", "candidate"),
    (occurrence, "query", "query"),
    (occurrence, "present", "present"),
    (occurrence, "read_value", "value"),
    (occurrence, "inspect", "inspect"),
    (occurrence, "decision_state", "state"),
    (occurrence, "candidates", "candidates"),
    (occurrence, "bind_field", "field"),
    (occurrence, "root", "root"),
    (occurrence, "child", "child"),
    (occurrence, "selected_candidate", "selected"),
)
"""Each entry point wrapped: where it is defined (or re-exported), its name, its kind."""


# -- the plugin ----------------------------------------------------------------------------

_KEY = pytest.StashKey[Shadow]()


def pytest_configure(config: pytest.Config) -> None:
    shadow = Shadow()
    shadow.apply()
    config.stash[_KEY] = shadow


def pytest_unconfigure(config: pytest.Config) -> None:
    shadow = config.stash.get(_KEY, None)
    if shadow is not None:
        shadow.restore()


def pytest_report_header(config: pytest.Config) -> str:
    return (
        "switch shadow: the public API answering, today's engine asked again "
        "(core.space._switch_shadow)"
    )


@pytest.fixture(autouse=True)
def _switch_shadow_purity() -> Iterator[None]:
    """The purity check on for the test, both switches (``purity_unchecked`` turns it off
    after this)."""
    previous = check_purity(True)
    yield
    restore_purity(previous)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    shadow = item.config.stash[_KEY]
    shadow.begin(item.nodeid)
    try:
        return (yield)
    finally:
        shadow.end()


def _phase(item: pytest.Item) -> Generator[None, object, object]:
    """A phase of a test, failed by the disagreements and isolation breaks met in it."""
    shadow = item.config.stash[_KEY]
    try:
        return (yield)
    finally:
        found = shadow.take_pending()
        if found:
            raise SwitchDisagreement("\n".join(map(str, found)))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _phase(item))


def pytest_sessionfinish(session: pytest.Session) -> None:
    """The session fails where any question was answered otherwise, or today's side
    changed what a test reads."""
    shadow = session.config.stash[_KEY]
    if (shadow.disagreements or shadow.breaks) and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    shadow = config.stash[_KEY]
    for found in [*shadow.disagreements, *shadow.breaks][:20]:
        terminalreporter.write_line(f"switch shadow: {found.test}: {found.question} {found.asked}")
    terminalreporter.write_line(shadow.left_alone())
    terminalreporter.write_line(shadow.count())


__all__ = [
    "ENTRY_POINTS",
    "TODAY",
    "Shadow",
    "SwitchDisagreement",
    "form",
]
