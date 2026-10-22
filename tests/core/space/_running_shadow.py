# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The shadow run: today's public API answering every question both ways, a pytest plugin.

    python -m pytest -p no:xdist -p core.space._running_shadow -q kernels/test_channel_source.py

(from ``tests``). While it is active, each public entry point below answers as today's
engine does, and the running type (``finn.core.space._running``) then answers the same
question on the same design space value: the configuration's facts and stored choices,
read once (``DesignSpace``), at the configuration's scope (``surface.At``). The two answers
are compared, and a test fails on any disagreement. Today's answer is always the one
returned, so a test sees today's behaviour, plus a failure where the two differ.

**What it compares** (each question's kind, as the count names it):

- ``design_space``: the design space opened, against the value the running type makes from
  the class and its facts, prepared as a change's values are;
- a change (``change``: ``try_with_choices``, which ``with_choices`` asks through; ``change
  of facts``: ``try_with_facts``, which ``with_facts`` asks through) and ``bind``: each
  report (every outcome, the rules broken, what moved) and the design space it publishes,
  by value;
- ``inspection``'s answers about a configuration: ``validity``, ``states``, ``viable``,
  ``explain`` and ``candidate``;
- a configuration's own questions: ``query``, ``present``, ``inspect``, a member read as a
  value (``value``: ``point.member``, and a bound accessor's ``get``), ``field`` (whether the
  accessor is a Decision's), a Decision's ``state`` and ``candidates`` (its bound
  accessor's), ``root``, ``child`` (``point.child``) and the candidate a Decision over nodes
  selects (``selected``).

A configuration in an answer (a child's, a candidate's, a change's) compares as its design
space value and scope. A design space a change published is then answered from the running
type's own value, so each later question of it reads the running type's history.

**Comparing.** By the class-based oracle's canonical forms (``support.space_oracle.forms``)
where they apply: an answer, its findings sorted; a validity; a choice's state; the possible
values; a change's report. Anything else compares field by field (a dataclass), in order (a
sequence), sorted (a set or a mapping), and a value with no form by ``==``. One named
difference is not a disagreement: ``explain`` lists the facts a node read (a domain's fact
reads) in node order in the running type, and in the order the process first read them in
today's engine, so each node's fact reads compare as a set, the rest of its reads in order
(counted apart). A raise on one side and an answer on the other is a disagreement; both
raising compare by the error's type and message.

**A disagreement** fails the test at the end of the phase it was met in (setup, call or
teardown), naming the question, the member or members asked, the configuration (its Space
class, its scope and its stored choices) and both answers, short. One met outside any test
fails the session.

**The purity check** is on for every test (``_world.check_purity``, the switch today's
engine and the running type share: each memo hit and each cache hit is evaluated again and
compared), unless the test turns it off (``purity_unchecked``). The running type answers
through one session per test, with a node cache, dropped at the test's end.

**The count**, at the session's end, in the terminal summary: the questions compared, by
kind, and the disagreements (0 to pass); then the questions left to today's engine alone,
by reason, and how many fact-read lists compared as sets::

    running shadow: 1234 questions compared (change 12, explain 3, ...); disagreements: 0

**What it leaves out**, and why:

- the structural questions, each a function of the compiled Space class alone, which both
  engines read from the same compiled model: ``inspection.members``, ``decisions``,
  ``choices``, ``statistics``, ``model``, the handle factories (``decision_handle``,
  ``value_handle``) and a change's construction (a bound Decision's ``change`` and
  ``clear``);
- a question a method asks of its receiver (``self``), or one asked during an authored call
  or a value transformation: it is part of today's evaluation, which the running type makes
  in its own way (and today's refuses a configuration read there);
- a question asked while another is answered (``with_choices`` asking ``try_with_choices``,
  an authored function opening a design space of its own, the running type's own
  evaluation): the outermost is compared, the rest are part of it;
- a question asked from another thread than the one the shadow started in: the running
  type is single-threaded (and a test of today's locks makes its threads wait on each
  other);
- a configuration of a model built as a table (``LinkedModel``, as a test builds one by
  hand or collapsed): the running type reads a compiled structure only;
- a request refused before any question is asked (a malformed or foreign change, a clear in
  a bind, an unknown fact) and a declaration today's front end refuses to open: those are
  the front end's, which both engines share.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
import threading
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from functools import wraps
from types import ModuleType
from typing import Any, cast
from weakref import WeakKeyDictionary

import pytest
from support.space_oracle import forms

from finn.core.reporting.findings import Finding
from finn.core.space import _assign, _failure, compiler, inspection, occurrence
from finn.core.space import _world as W
from finn.core.space._configuration import BoundDecision, Space
from finn.core.space._running import questions as Q
from finn.core.space._running import surface as S
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.value import DesignSpace, FactsLayer
from finn.core.space._state import State
from finn.core.space._structure import Structure
from finn.core.space.edits import ConfigurationResult
from finn.core.space.errors import RequestError
from finn.core.space.ir import NodeId
from finn.core.space.references import ValueHandle
from finn.core.space.results import Available, Inapplicable, Rejected, Unresolved

OUTSIDE = "(outside any test)"
"""Where a question asked between tests is counted."""


class ShadowDisagreement(AssertionError):
    """Questions the running type answered otherwise than today's engine, in one phase of a
    test."""


class Same:
    """A value with no canonical form, compared by ``==``."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Same):
            return NotImplemented
        if self.value is other.value:
            return True
        try:
            return bool(self.value == other.value)
        except Exception:  # noqa: BLE001 - an equality that raises is no equality
            return False

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(self.value)


class SameSpace(Same):
    """A design space value, compared by value (``DesignSpace.__eq__``: each fact and stored
    choice by its member's semantics); where an authored equality raises, by its model, and
    its facts' and stored choices' forms."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SameSpace):
            return NotImplemented
        mine, theirs = cast(DesignSpace, self.value), cast(DesignSpace, other.value)
        try:
            return mine == theirs
        except Exception:  # noqa: BLE001 - an authored equality that raises
            return mine.model is theirs.model and all(
                form(dict(left)) == form(dict(right))
                for left, right in ((mine.facts, theirs.facts), (mine.stored, theirs.stored))
            )

    __hash__ = None


@dataclass(frozen=True)
class Configuration:
    """A design space value at one of its scopes, in a form: equal by value."""

    space: SameSpace
    scope: int


# -- the configuration as a design space value -------------------------------------------


_SPACES: WeakKeyDictionary[State, DesignSpace] = WeakKeyDictionary()
"""Each configuration's state in today's engine, with its design space value: the running
type's own where it published that design space."""


def _state(point: object) -> State | None:
    """A configuration's state; ``None`` for a receiver or a declaration."""
    found = vars(point).get("_state") if isinstance(point, Space) else None
    return found if isinstance(found, State) else None


def _space(current: State) -> DesignSpace:
    found = _SPACES.get(current)
    if found is None:
        world = current.world
        layer = FactsLayer.of(current.model, world.facts)
        found = _SPACES[current] = DesignSpace(layer, world.stored)
    return found


def _at(point: Space) -> S.At:
    """A configuration of today's as the running type reads it: its design space value at
    its scope."""
    current = _state(point)
    assert current is not None, "a configuration"
    return S.At(_space(current), point._scope)


# -- canonical forms ---------------------------------------------------------------------


def _t0(render: Callable[[Any], forms.Form], found: object) -> forms.Form | None:
    """``found`` in the oracle's form, or ``None`` where a value in it has none."""
    try:
        return render(found)
    except forms.AddressInRepr:
        return None


_STATES = (inspection.Set, inspection.Waiting, inspection.Dormant, inspection.Open)
_RENDERED: tuple[tuple[type | tuple[type, ...], Callable[[Any], forms.Form]], ...] = (
    (inspection.Validity, forms.validity),
    ((*_STATES, inspection.Absent), forms.state),
    (inspection.Viable, forms.viable),
)


def _sorted(items: Iterable[object]) -> list[object]:
    return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=repr))


def form(found: object) -> object:
    """``found`` in a form compared by ``==``: the oracle's where it applies; a
    configuration as its design space value and scope; a handle as its member's key and
    kind; else field by field, in order or sorted, and a value with no form by ``==``."""
    if isinstance(found, Space):
        if _state(found) is None:
            return Same(found)
        found = _at(found)
    if isinstance(found, S.At):
        return Configuration(SameSpace(found.ds), found.scope)
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


def _facts_as_a_set(evidence: object) -> object:
    """An evidence's form with each node's fact reads as a set, the rest of its reads in
    order (``explain``'s one named difference)."""
    if not isinstance(evidence, dict) or "fields" not in evidence:
        return evidence
    nodes = []
    for node in evidence["fields"]["nodes"]:
        reads = node["fields"]["dependencies"]
        split = {
            "in order": [read for read in reads if read["kind"] != "param"],
            "facts": _sorted(read for read in reads if read["kind"] == "param"),
        }
        nodes.append({**node, "fields": {**node["fields"], "dependencies": split}})
    return {**evidence, "fields": {**evidence["fields"], "nodes": nodes}}


# -- what each question asks of the running type -----------------------------------------


Answer = Callable[[Session, S.At, Space, tuple[Any, ...]], object]
"""The running type's answer to a question: of its session, the configuration's design
space value at its scope, the configuration (to address a key) and the question's
arguments after the configuration."""


def _viable(session: Session, at: S.At, point: Space, keys: Iterable[object]) -> object:
    indices = [occurrence.decision_index(point, key) for key in keys]
    return tuple(Q.viable(at.ds, index, session) for index in indices)


RUNNING: dict[str, Answer] = {
    "query": lambda s, at, p, a: S.query(at, a[0], s),
    "present": lambda s, at, p, a: S.present(at, a[0], s),
    "value": lambda s, at, p, a: S.value(at, a[0], s),
    "inspect": lambda s, at, p, a: S.inspect(at, a[0], s),
    "state": lambda s, at, p, a: S.decision_state(at, a[0], s),
    "candidates": lambda s, at, p, a: S.candidates(at, a[0], s),
    "field": lambda s, at, p, a: isinstance(S.field(at, a[0], s), S.DecisionField),
    "root": lambda s, at, p, a: S.root(at),
    # the node as its class declares it: its record's declaration
    "child": lambda s, at, p, a: S.child(at, getattr(a[0], "instance", a[0])),
    "selected": lambda s, at, p, a: S.selected_candidate(at, a[0], s),
    "validity": lambda s, at, p, a: Q.validity(evaluation(at.ds, s), at.scope),
    "states": lambda s, at, p, a: Q.states(evaluation(at.ds, s), at.scope),
    "viable": lambda s, at, p, a: _viable(s, at, p, a[0]),
    "explain": lambda s, at, p, a: S.explain(at, a[0], s),
    "candidate": lambda s, at, p, a: S.candidate(at, a[0], a[1]),
}
"""The running type's answer to each question, by kind."""

TODAY: dict[str, Callable[[object], object]] = {
    "field": lambda accessor: isinstance(accessor, BoundDecision),
}
"""Today's answer as the running type's is compared, where they differ in type: a bound
accessor by whether it is a Decision's."""

ASKED: dict[str, Callable[[tuple[Any, ...]], Iterable[object]]] = {
    "viable": lambda a: cast(Iterable[object], a[0]),
    "root": lambda a: (),
    "validity": lambda a: (),
    "states": lambda a: (),
}
"""The members a question asks, after the configuration (the first argument by default)."""


@dataclass(frozen=True)
class Requests:
    """A change's or a bind's items by node, as today's front end reads them: the facts,
    then each choice set (to its value) or cleared (``Q.REMOVE``), in order."""

    facts: dict[NodeId, object]
    choices: list[tuple[NodeId, object]]


def _requests(
    point: Space,
    changes: tuple[Any, ...],
    choices: Mapping[str, object] | None,
    facts: Mapping[Any, object] | None,
) -> Requests:
    """The items of a change of ``point`` (``try_with_choices``' keywords ``choices``,
    ``try_with_facts``' ``facts``), read by today's front end, which raises where the
    request is refused before any question is asked."""
    given = {} if facts is None else _assign._fact_nodes(point, facts)
    items = [
        *_assign._requests(point, changes),
        *(_assign._keyword_change(point, name, value) for name, value in (choices or {}).items()),
    ]
    normalized = _assign.normalize(occurrence.state(point), items)
    return Requests(
        given,
        [(index, Q.REMOVE if item.remove else item.value) for index, item in normalized.items()],
    )


CHANGES: dict[str, Callable[[tuple[Any, ...], dict[str, Any]], tuple[Any, Any, Any]]] = {
    "change": lambda a, k: (a, k, None),
    "change of facts": lambda a, k: (a[1:], None, a[0]),
    "bind": lambda a, k: (a, None, None),
}
"""Each change's items as ``_requests`` reads them, from its arguments after the
configuration: the changes, the keywords and the facts."""


def _made(model: Any, record: Any) -> DesignSpace:
    """The design space a root declaration opens, as the running type makes it: its facts
    (the root's values and value defaults) prepared as a change's values are, nothing
    stored."""
    linked = model.linked
    given = compiler.root_parameters(model, record)
    missing = [
        linked.header(index).key
        for index in linked.parameters
        if linked.raw_nodes[index].required and index not in given
    ]
    if missing:
        raise RequestError(f"missing required parameters: {', '.join(missing)}")
    facts = occurrence.prepare_values(linked, given, "parameter")
    return DesignSpace(FactsLayer.of(model, facts), {})


# -- the shadow ----------------------------------------------------------------------------


def _short(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class Disagreement:
    """One question answered otherwise: in which test, the question, the members asked, the
    configuration, and where the two answers differ (by path: today's | the running
    type's)."""

    test: str
    question: str
    asked: str
    configuration: str
    differences: list[str]

    def __str__(self) -> str:
        shown = "".join(f"\n    {line}" for line in self.differences[:6])
        more = len(self.differences) - 6
        if more > 0:
            shown += f"\n    ... {more} more"
        return (
            f"{self.question} {self.asked} of {self.configuration}: today's engine and the "
            f"running type answer otherwise (where: today's | running){shown}"
        )


def _describe(at: S.At) -> str:
    """A configuration, short: its Space class, its scope and its stored choices."""
    scope = at.ds.model.linked.scopes[at.scope]
    stored = ", ".join(f"{at.ds.key(index)}={value!r}" for index, value in at.ds.stored.items())
    where = f" at {scope.name}" if at.scope else ""
    return _short(f"{scope.space_type.__qualname__}{where} {{{stored}}}")


def _asked(at: S.At, references: Iterable[object]) -> str:
    """The members asked, by key where they resolve."""
    found = []
    for reference in references:
        try:
            index = at.ds.model.resolve(at.scope, reference)
            found.append(at.ds.model.linked.header(index).key)
        except Exception:  # noqa: BLE001 - only to name it
            found.append(_short(repr(reference), 60))
    return ", ".join(found) or "(the configuration)"


def _differences(today: object, running: object) -> list[str]:
    def text(item: object) -> str:
        return _short(json.dumps(item, sort_keys=True, default=repr), 240)

    return [
        f"{item.path or '(whole)'}: {text(item.base)} | {text(item.other)}"
        for item in forms.differences(today, running)
    ]


@dataclass(frozen=True)
class Outcome:
    """What asking gave: an answer, or what it raised."""

    answer: object = None
    error: Exception | None = None

    def form(self, render: Callable[[Any], object]) -> object:
        if self.error is not None:
            return {"raised": type(self.error).__name__, "message": str(self.error)}
        return render(self.answer)


@dataclass
class Shadow:
    """Today's public entry points wrapped (``apply``) so that each answers both ways and
    compares, until ``restore``. ``running`` replaces the running type's answer to a
    question by kind (``RUNNING``'s by default)."""

    running: dict[str, Answer] = field(default_factory=dict)
    compared: Counter[str] = field(default_factory=Counter)
    left: Counter[str] = field(default_factory=Counter)
    facts_as_sets: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)
    pending: list[Disagreement] = field(default_factory=list)
    test: str = OUTSIDE
    session: Session = field(default_factory=lambda: Session(cache=NodeCache()))
    depth: int = 0
    saved: list[tuple[object, str, object]] = field(default_factory=list)
    touched: list[DesignSpace] = field(default_factory=list)
    thread: int = field(default_factory=threading.get_ident)

    def __post_init__(self) -> None:
        self.running = {**RUNNING, **self.running}

    # -- a test's session ----------------------------------------------------------------

    def begin(self, test: str) -> None:
        """A test's start: a session of its own."""
        self.end()
        self.test = test

    def end(self) -> None:
        """A test's end: its session's evaluations dropped from every design space it
        read, and a new session."""
        for ds in self.touched:
            for held in (ds, ds.layer._bare):
                if held is not None:
                    keys = cast("dict[tuple[object, object], object]", held.evaluations)
                    for key in [key for key in keys if key[0] is self.session]:
                        del held.evaluations[key]
        self.touched.clear()
        self.test = OUTSIDE
        self.session = Session(cache=NodeCache())

    def take_pending(self) -> list[Disagreement]:
        found, self.pending = self.pending, []
        return found

    # -- wrapping ------------------------------------------------------------------------

    def apply(self) -> None:
        """Wrap every entry point compared where it is defined, and wherever a module
        outside the engine holds it by name."""
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

    def _set(self, owner: object, name: str, value: object) -> None:
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self) -> None:
        """Every entry point as it was."""
        while self.saved:
            owner, name, value = self.saved.pop()
            setattr(owner, name, value)

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
            before = shadow._before(kind, arguments)
            shadow.depth += 1
            try:
                try:
                    answer = original(*arguments, **keywords)
                except Exception as error:
                    shadow._guarded(kind, arguments, keywords, before, Outcome(error=error))
                    raise
                shadow._guarded(kind, arguments, keywords, before, Outcome(answer))
                return answer
            finally:
                shadow.depth -= 1

        return shadowed

    def _leaves(self, kind: str, arguments: tuple[Any, ...]) -> str | None:
        """Why a call is left to today's engine alone, or ``None`` to compare it."""
        if threading.get_ident() != self.thread:
            return "asked from another thread"
        if kind != "design_space":
            current = _state(arguments[0]) if arguments else None
            if current is None:
                return "a receiver's or a declaration's"
            if not isinstance(current.linked, Structure):
                return "a model built as a table"
        if _failure.current() is not None or _failure._TRANSFORM.get() is not None:
            return "inside an authored call"
        if self.depth:
            return "inside another question"
        return None

    def _before(self, kind: str, arguments: tuple[Any, ...]) -> tuple[Any, Any] | None:
        """What the running type reads before today's call changes it: a root declaration's
        model and record (``design_space``, which freezes the record)."""
        if kind != "design_space" or not arguments:
            return None
        try:
            record = compiler.root_record(arguments[0])
            return compiler.compile_model(record.space_type), record
        except Exception:  # noqa: BLE001 - today's refuses it, and it is not compared
            return None

    # -- comparing -----------------------------------------------------------------------

    def _guarded(
        self,
        kind: str,
        arguments: tuple[Any, ...],
        keywords: dict[str, Any],
        before: tuple[Any, Any] | None,
        today: Outcome,
    ) -> None:
        """``_compare``, never raising into the caller: a failure of the comparison itself
        is a disagreement, so today's answer always reaches the caller."""
        try:
            self._compare(kind, arguments, keywords, before, today)
        except Exception as error:  # noqa: BLE001 - reported, never raised into the test
            found = Disagreement(
                self.test, kind, "(not compared)", "?", [f"the comparison raised {error!r}"]
            )
            self.disagreements.append(found)
            self.pending.append(found)

    def _compare(
        self,
        kind: str,
        arguments: tuple[Any, ...],
        keywords: dict[str, Any],
        before: tuple[Any, Any] | None,
        today: Outcome,
    ) -> None:
        if kind == "design_space":
            self._compare_opened(before, today)
            return
        point = cast(Space, arguments[0])
        at = _at(point)
        self.touched.append(at.ds)
        rest = arguments[1:]
        if kind in CHANGES:
            try:
                requests = _requests(point, *CHANGES[kind](rest, keywords))
            except Exception:  # noqa: BLE001 - refused before any question, by the front end
                self.left["refused before any question"] += 1
                return
            self._compare_change(kind, at, requests, today)
            return
        try:
            running = Outcome(self.running[kind](self.session, at, point, rest))
        except Exception as error:  # noqa: BLE001 - a raise is an answer to compare
            running = Outcome(error=error)
        self.compared[kind] += 1
        held = today.form(lambda answer: form(TODAY.get(kind, _same)(answer)))
        answered = running.form(form)
        if held == answered:
            return
        if kind == "explain" and _facts_as_a_set(held) == _facts_as_a_set(answered):
            self.facts_as_sets += 1
            return
        asked = ASKED.get(kind, _first)(rest)
        self._disagree(kind, _asked(at, asked), at, held, answered)

    def _compare_change(self, kind: str, at: S.At, requests: Requests, today: Outcome) -> None:
        try:
            if kind == "bind":
                report = Q.bind(at.ds, requests.choices, self.session)
            else:
                report = Q.change(at.ds, requests.facts, requests.choices, self.session)
            running = Outcome(report)
        except Exception as error:  # noqa: BLE001 - a raise is an answer to compare
            running = Outcome(error=error)
        self.compared[kind] += 1

        def shaped(found: Any) -> object:
            if isinstance(found, ConfigurationResult):
                published = found.instance
            else:
                published = S.At(at.ds if found.published is None else found.published, at.scope)
            return {"report": forms.report(found), "configuration": form(published)}

        held, answered = today.form(shaped), running.form(shaped)
        if held != answered:
            asked = ", ".join(
                at.ds.key(index) for index in (*requests.facts, *dict(requests.choices))
            )
            self._disagree(kind, asked or "(nothing)", at, held, answered)
            return
        if today.error is None:
            current = _state(cast(ConfigurationResult[Space], today.answer).instance)
            published = cast(Q.Report, running.answer).published
            if current is not None and published is not None:
                _SPACES[current] = published
                self.touched.append(published)

    def _compare_opened(self, before: tuple[Any, Any] | None, today: Outcome) -> None:
        if today.error is not None or before is None:
            self.left["refused by the front end"] += 1
            return
        if not isinstance(before[0].linked, Structure):
            self.left["a model built as a table"] += 1
            return
        at = _at(cast(Space, today.answer))
        self.touched.append(at.ds)
        try:
            running = Outcome(S.At(_made(*before), 0))
        except Exception as error:  # noqa: BLE001 - a raise is an answer to compare
            running = Outcome(error=error)
        self.compared["design_space"] += 1
        held, answered = today.form(form), running.form(form)
        if held != answered:
            self._disagree("design_space", "(the design space)", at, held, answered)

    def _disagree(self, kind: str, asked: str, at: S.At, held: object, answered: object) -> None:
        found = Disagreement(self.test, kind, asked, _describe(at), _differences(held, answered))
        self.disagreements.append(found)
        self.pending.append(found)

    # -- the count -----------------------------------------------------------------------

    def count(self) -> str:
        """The summary's line: the questions compared, by kind, and the disagreements."""
        total = sum(self.compared.values())
        kinds = ", ".join(f"{kind} {number}" for kind, number in sorted(self.compared.items()))
        return (
            f"running shadow: {total} questions compared ({kinds or 'none'}); "
            f"disagreements: {len(self.disagreements)}"
        )

    def left_alone(self) -> str:
        """The summary's second line: the questions left to today's engine, by reason, and
        the fact-read lists compared as sets."""
        reasons = ", ".join(f"{reason} {number}" for reason, number in sorted(self.left.items()))
        return (
            f"running shadow: left to today's engine: {reasons or 'none'}; "
            f"explain's fact reads compared as a set: {self.facts_as_sets}"
        )


def _same(answer: object) -> object:
    return answer


def _first(arguments: tuple[Any, ...]) -> Iterable[object]:
    return arguments[:1]


_ENGINE = "finn.core.space."
"""The engine's own modules, whose names for the entry points are left as they are: a
question the engine asks itself is part of the one asked."""

ENTRY_POINTS: tuple[tuple[object, str, str], ...] = (
    (compiler, "design_space", "design_space"),
    (sys.modules["finn.core.space"], "design_space", "design_space"),
    (_assign, "try_with_choices", "change"),
    (_assign, "try_with_facts", "change of facts"),
    (_assign, "bind", "bind"),
    (sys.modules["finn.core.space"], "bind", "bind"),
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
    return "running shadow: today's public API answered both ways (core.space._running_shadow)"


@pytest.fixture(autouse=True)
def _running_shadow_purity() -> Iterator[None]:
    """The purity check on for the test, both engines' (``purity_unchecked`` turns it off
    after this)."""
    previous = W.check_purity(True)
    yield
    W.check_purity(previous)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    shadow = item.config.stash[_KEY]
    shadow.begin(item.nodeid)
    try:
        return (yield)
    finally:
        shadow.end()


def _phase(item: pytest.Item) -> Generator[None, object, object]:
    """A phase of a test, failed by the disagreements met in it."""
    shadow = item.config.stash[_KEY]
    try:
        return (yield)
    finally:
        found = shadow.take_pending()
        if found:
            raise ShadowDisagreement("\n".join(map(str, found)))


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
    """The session fails where any question was answered otherwise."""
    if session.config.stash[_KEY].disagreements and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    shadow = config.stash[_KEY]
    for found in shadow.disagreements[:20]:
        terminalreporter.write_line(f"running shadow: {found.test}: {found.question} {found.asked}")
    terminalreporter.write_line(shadow.left_alone())
    terminalreporter.write_line(shadow.count())


__all__ = [
    "ENTRY_POINTS",
    "RUNNING",
    "Disagreement",
    "Shadow",
    "ShadowDisagreement",
    "form",
]
