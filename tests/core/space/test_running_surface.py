# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type's answers to the public questions the oracle's interface does not ask
(``finn.core.space._running.surface``), asked in a session of its own with no cache, against
the public API's (its model's session, the node cache on); and its failure completion.

- **The surface**: evidence read through a formal at its source, its values detached; a
  design space at a scope (a child, a candidate, the root); the selected candidate of a
  chosen and of a picked selector; the evidence of a choice whose presence is unknown,
  the facts its list read in key order, the same in a permuted numbering.
- **Failure completion** (``test_native_cleanup``'s cases at the evaluator's own points):
  cleanup runs and may read; the first failure is primary and sticky; later ones are its
  cleanup failures, in order; a cancellation keeps its identity; failures are kept for one
  run only, and an answer that failed is kept nowhere; an interruption at each of the
  evaluator's own points (before a call, a read inside a method, publishing an answer,
  recording a failure, delivering one) completes the run.
"""

from __future__ import annotations

import inspect as _inspect
import sys
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

import pytest

from core.space._evaluation_support import source_line
from core.space._running_shapes import ByArgument, PickedByAChoice
from core.space._running_type import Running
from core.space._toys_support import Boiler, House
from finn.core.space import (
    Available,
    Decision,
    EvaluationError,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    ValueUnavailableError,
    _failure,
    bind,
    constraint,
    derived,
    design_space,
    divisors_of,
    domain,
    inspection,
    occurrence,
    reject,
    view,
)
from finn.core.space._failure import NativeEvaluationError, cancellation_details
from finn.core.space._running import evaluate
from finn.core.space._running import surface as S
from finn.core.space._running.evaluate import Evaluation, Session
from finn.core.space._running.facts import prepare_values
from finn.core.space._running.value import DesignSpace
from finn.core.space._scopes import PERMUTE, Address, ScopeRecord
from finn.core.space.occurrence import held
from finn.core.space.references import ValueHandle
from finn.core.space.results import ConstraintAssessment

if TYPE_CHECKING:
    from _typeshed import TraceFunction


# -- every question, everywhere --------------------------------------------------------


def _form(found: object) -> object:
    """An answer in a form both ways share: a configuration as its scope."""
    if isinstance(found, Space):
        return ("configuration", found._scope)
    if isinstance(found, S.At):
        return ("configuration", found.scope)
    if isinstance(found, Available) and isinstance(found.value, (Space, S.At)):
        return Available(_form(found.value))
    return found


def _outcome(ask: Callable[[], object]) -> object:
    """What asking gives: its answer (in ``_form``), or what it raises (type and text)."""
    try:
        return ("answers", _form(ask()))
    except (EvaluationError, ValueUnavailableError) as error:
        return ("raises", type(error), str(error))


# The evidence tests' shapes (``test_evidence``), and the selectors of ``_running_shapes``:
# chosen and picked, given as a binding's argument, and picked by a choice.


class Good(Space):
    size: int = Param()

    @view
    def physical(*, size: int) -> int:
        return size


class Bad(Space):
    @view
    def physical() -> int:
        raise AssertionError("inactive evaluator ran")


class Alternatives(Space):
    size: int = Param()
    implementation = Select(good=Good(size=size), bad=Bad())
    physical = view(implementation.physical)


class OptionalChild(Space):
    size: int | None = Param(default=None)

    @derived
    def doubled(self) -> int:
        return 0 if self.size is None else self.size * 2


class Optional(Space):
    size: int | None = Param(default=None)
    nil: None = Param(default=None)
    child = OptionalChild(size=size)


class Inner(Space):
    value: int = Param()


class Forwarded(Space):
    value: int = Param()

    @derived
    def enabled(self) -> bool:
        return False

    child = Inner(value=value, when=enabled)


class Literal(Space):
    @derived
    def enabled(self) -> bool:
        return False

    child = Inner(value=12345, when=enabled)


class Refusing(Space):
    chosen: int = Decision(values=(1, 2))

    @derived
    def value(self) -> int:
        return 3

    @constraint
    def supported() -> bool:
        return False

    @derived
    def selected(self) -> int:
        return self.value + self.chosen

    physical = view(selected)


# A selector whose cases' choices list values read from the facts (``Kernel``): with the
# selector open, each case's choices are of unknown presence, and their pieces are read from
# their lists as if they existed. ``plain.simd`` lists the divisors of a derived value
# reading a fact.


class Packed(Space):
    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def fits(self) -> bool | Rejected:
        return True if self.pe <= 8 else reject("pe-wide", "pe past 8")


class Plain(Space):
    depth: int | None = Param(default=None)

    @derived
    def lanes(self) -> int | Rejected:
        if self.depth is None:
            return reject("depth-unknown", "no depth is given")
        return self.depth * 4

    simd: int = Decision(domain=divisors_of(lanes))


class Kernel(Space):
    width: int = Param()
    depth: int | None = Param(default=None)
    style = Select(packed=Packed(width=width), plain=Plain(depth=depth))

    @view
    def bits(self) -> int:
        return self.width * 2


SHAPES: dict[str, Callable[[], Space]] = {
    "alternatives open": lambda: design_space(Alternatives(size=12)),
    "alternatives, good": lambda: design_space(Alternatives(size=12)).with_choices(
        {Alternatives.implementation: "good"}
    ),
    "alternatives, bad": lambda: design_space(Alternatives(size=12)).with_choices(
        {Alternatives.implementation: "bad"}
    ),
    "an optional input omitted": lambda: design_space(Optional(nil=None)),
    "an optional input given": lambda: design_space(Optional(size=3)),
    "an inactive formal, forwarded": lambda: design_space(Forwarded(value=12345)),
    "an inactive formal, literal": lambda: design_space(Literal()),
    "a refusal": lambda: design_space(Refusing()),
    "an open selector's cases listing from the facts": lambda: design_space(
        Kernel(width=12, depth=3)
    ),
    "an open selector's cases, a fact missing": lambda: design_space(Kernel(width=12)),
    "a selector's case listing from the facts, chosen": lambda: design_space(
        Kernel(width=12, depth=3)
    ).with_choices({Kernel.style: "plain"}),
    "selectors given as arguments, open": lambda: design_space(ByArgument()),
    "selectors given as arguments, a None case": lambda: (
        bind(design_space(ByArgument()), {ByArgument.x: 0, ByArgument.chosen: "direct"}).instance
    ),
    "selectors given as arguments, a case each": lambda: (
        bind(design_space(ByArgument()), {ByArgument.x: 1, ByArgument.chosen: "stage"}).instance
    ),
    "a selector picked by a choice, open": lambda: design_space(PickedByAChoice(f=2)),
    "a selector picked by a choice, its case placed": lambda: (
        bind(design_space(PickedByAChoice(f=2)), {PickedByAChoice.x: 0}).instance
    ),
    "a selector picked by a choice, its None case": lambda: (
        bind(design_space(PickedByAChoice(f=2)), {PickedByAChoice.x: 1}).instance
    ),
}


def test_evidence_reads_through_a_formal_at_its_source_and_copies_a_container() -> None:
    """The house's thermostat's cost reads its ``kw`` at its source (a formal bound by
    reference, read within its scope): named in ``via``, with no cache as through the public
    API. A mutable container a derived value answers (a list) is handed out as a copy, in
    the evidence and read as a value: a reader changing it changes nothing held."""
    point = design_space(House(budget=100)).with_choices({House.heating: "boiler"})
    running = Running()
    at = S.At(cast(Any, running.open(point)).ds)
    evidence = S.explain(at, House.thermostat.cost, running.session)
    assert evidence == inspection.explain(point, House.thermostat.cost)
    assert evidence.result == Available(5)
    via = {item.declaration.key: [each.key for each in item.via] for item in evidence.nodes}
    assert via["thermostat.cost"] == ["thermostat.kw"]

    class Listed(Space):
        n: int = Param()

        @derived
        def source(self) -> list[int]:
            return [1, self.n]

        physical = view(source)

    listed = design_space(Listed(n=2))
    opened = Running()
    here = S.At(cast(Any, opened.open(listed)).ds)
    first = S.explain(here, Listed.physical, opened.session)
    assert isinstance(first.result, Available)
    cast(list[int], first.result.value).append(3)
    assert S.explain(here, Listed.physical, opened.session).result == Available([1, 2])
    read = listed.source
    read.append(4)
    assert listed.source == [1, 2] and listed.physical == [1, 2]


def test_a_configuration_is_its_design_space_at_a_scope() -> None:
    """``query`` of a node, ``child``, ``candidate`` and ``root`` give the design space at
    a scope, equal by value; ``candidate`` whatever is selected."""
    point = design_space(House(budget=100))
    running = Running()
    at = S.At(cast(Any, running.open(point)).ds)
    kitchen = S.child(at, House.kitchen)
    assert S.query(at, House.kitchen, running.session) == Available(kitchen)
    assert kitchen.scope == point.kitchen._scope and S.root(kitchen) == at
    boiler = S.candidate(at, House.heating, "boiler")
    public = inspection.candidate(point, House.heating, "boiler")
    assert boiler is not None and public is not None and boiler.scope == public._scope
    assert S.query(boiler, Boiler.kw, running.session) == public.query(Boiler.kw)
    assert isinstance(public.query(Boiler.kw), Unresolved)  # its case is open
    with pytest.raises(ValueUnavailableError):
        S.selected_candidate(at, House.heating, running.session)
    assert S.query(at, House.garage, running.session) == point.query(House.garage)


@pytest.mark.parametrize(
    ("x", "chosen"),
    [(None, None), (1, None), (0, "direct"), (1, "lane"), (2, "stage")],
    ids=["both open", "picked set, chosen open", "None cases", "lane", "stage"],
)
def test_the_selected_candidate_of_a_chosen_and_a_picked_selector(
    x: int | None, chosen: str | None
) -> None:
    """The candidate a selector selects, a chosen one (``chosen``) and one picked by a
    derived value (``picked``, ``Select(by=)``), with no cache as through the public API:
    open, it raises as a value
    unavailable; its ``None`` case selected, ``None``; else its case's configuration, the
    one ``candidate`` names."""
    point = design_space(ByArgument())
    given: dict[object, object] = {}
    if x is not None:
        given[ByArgument.x] = x
    if chosen is not None:
        given[ByArgument.chosen] = chosen
    if given:
        point = bind(point, given).instance
    running = Running()
    at = S.At(cast(Any, running.open(point)).ds)
    picked = None if x is None else ("direct", "lane", "stage")[x]
    for decision, case in ((ByArgument.chosen, chosen), (ByArgument.picked, picked)):
        public = _outcome(lambda: occurrence.selected_candidate(point, decision))
        assert _outcome(lambda: S.selected_candidate(at, decision, running.session)) == public
        if case is None:
            assert isinstance(public, tuple) and public[:2] == ("raises", ValueUnavailableError)
        elif case == "direct":
            assert public == ("answers", None)
        else:
            named = inspection.candidate(point, decision, case)
            assert named is not None and public == ("answers", ("configuration", named._scope))


# -- evidence for a choice whose presence is unknown ---------------------------------------
#
# With ``style`` open, each case's choice is a piece read from its list as if it existed:
# what that list read (the facts, through anything) is in validity's evidence. Asked of the
# root's validity, each case's validity, a piece and a choice inside a case. ``plain``'s
# list reads two facts, ``depth`` first: they are given in key order (scope path, member),
# the same in every numbering and process (never the order a model first read them, a
# history). A permuted model's evidence is the default one's.


def _kernel() -> type[Space]:
    """``Kernel``, its plain case's list reading ``depth`` then ``width``: a class of its
    own each time."""

    class Packed(Space):
        width: int = Param()
        pe: int = Decision(domain=divisors_of(width))

        @constraint
        def fits(self) -> bool | Rejected:
            return True if self.pe <= 8 else reject("pe-wide", "pe past 8")

    class Plain(Space):
        width: int = Param()
        depth: int | None = Param()

        @derived
        def lanes(self) -> int | Rejected:
            if self.depth is None:
                return reject("depth-unknown", "no depth is given")
            return self.depth * self.width

        simd: int = Decision(domain=divisors_of(lanes))

    class Kernel(Space):
        width: int = Param()
        depth: int | None = Param(default=None)
        style = Select(packed=Packed(width=width), plain=Plain(width=width, depth=depth))

    return Kernel


def _by_key(point: Space, evidence: object) -> object:
    """``evidence`` (a ``QueryEvidence``) with every node named by its key, so two models
    compare: each node's key, answer, the keys it read in order, its state, selector and
    guard flags, and the keys it read through."""
    found = cast(Any, evidence)
    return (
        found.query.key,
        found.result,
        tuple(
            (
                node.declaration.key,
                node.result,
                tuple(each._node.key for each in node.dependencies),
                node.decision_state,
                node.selector,
                node.is_guard,
                tuple(item.key for item in node.via),
            )
            for node in found.nodes
        ),
    )


def _reads_unordered(point: Space, evidence: object) -> object:
    """``_by_key``'s form with each node's reads as a set: where only their order differs."""
    nodes = cast(Any, _by_key(point, evidence))
    return (
        *nodes[:2],
        tuple((*node[:2], frozenset(node[2]), *node[3:]) for node in nodes[2]),
    )


def _handle(point: Space, key: str | ScopeRecord) -> ValueHandle[object]:
    """The member at ``key``, or the validity of a scope."""
    model = held(point).ds.model
    index = model.scopes.find(key) if isinstance(key, str) else key.validity
    return ValueHandle[object](model, index)


FACTS = {"given": {"width": 12, "depth": 3}, "missing": {"width": 12}}
"""``Kernel``'s facts: ``plain``'s list reads both, or refuses on a ``depth`` left out (None)."""


def _explained(facts: dict[str, int]) -> tuple[Space, dict[str, object], dict[str, object]]:
    """A fresh ``Kernel`` with ``facts``, its scopes made in the order the environment asks
    for: its values, and each question's evidence, with no cache and through the
    public API."""
    kernel = _kernel()
    point = design_space(kernel(**cast(Any, facts)))
    running = Running()
    at = S.At(cast(Any, running.open(point)).ds)
    cases = {
        case: cast(Space, inspection.candidate(point, getattr(kernel, "style"), case))._scope
        for case in ("packed", "plain")
    }
    asked = {
        "the root's validity": _handle(point, held(point).ds.model.root),
        "packed's validity": _handle(point, cases["packed"]),
        "plain's validity": _handle(point, cases["plain"]),
        "a piece inside a case": _handle(point, "style.packed.fits"),
        "a choice inside a case": _handle(point, "style.plain.simd"),
    }
    mine = {what: S.explain(at, reference, running.session) for what, reference in asked.items()}
    public = {what: inspection.explain(point, reference) for what, reference in asked.items()}
    return point, cast(dict[str, object], mine), cast(dict[str, object], public)


def _reads(point: Space, evidence: object) -> dict[str, list[str]]:
    """Each evidence node's reads, in order, by key."""
    return {
        node.declaration.key: [each._node.key for each in node.dependencies]
        for node in cast(Any, evidence).nodes
    }


@pytest.mark.parametrize(
    ("facts", "plain"),
    [("given", ["depth", "width"]), ("missing", ["depth"])],
    ids=["given", "missing"],
)
def test_evidence_of_a_choice_of_unknown_presence_holds_what_its_list_read(
    facts: str, plain: list[str]
) -> None:
    point, mine, public = _explained(FACTS[facts])
    for what, evidence in mine.items():
        assert evidence == public[what], what
    read = _reads(point, mine["the root's validity"])
    # each case's piece: its guard, then the facts its list read in key order (``plain``'s
    # refuses on a ``depth`` left out), then the rest of its validity
    assert read["style.packed.$validity"][:2] == ["style.packed.$selected", "width"], read
    assert read["style.plain.$validity"] == ["style.plain.$selected", *plain], read


@pytest.mark.parametrize("facts", sorted(FACTS))
def test_evidence_of_a_choice_of_unknown_presence_is_the_same_in_a_permuted_order(
    facts: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same questions on a model whose scopes are made in the permuted order
    (``FINN_SPACE_PERMUTE``: reversed by path): every answer's evidence, by key, the default
    order's. Addresses are ordered by path, so the two formals with one source are named in
    one order whichever scope was made first."""
    point, mine, _ = _explained(FACTS[facts])
    with monkeypatch.context() as permuted:
        permuted.setenv(PERMUTE, "1")  # read when the model is made
        other, theirs, _ = _explained(FACTS[facts])
    made = [[r.path for r in held(each).ds.model.scopes.records()] for each in (point, other)]
    assert made[0] != made[1]  # the scopes made in another order
    for what, evidence in mine.items():
        assert _by_key(point, evidence) == _by_key(other, theirs[what]), what


# -- failure completion ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_call_is_left_active() -> Iterator[None]:
    assert _failure.current() is None
    yield None
    assert _failure.current() is None


WAYS = ("running",)
"""The ways a failure case is asked: the running type's evaluation, in a session of its own."""


@dataclass
class Opened:
    """One design space opened: the running type's design space, its evaluation and
    session."""

    way: str
    ds: DesignSpace | None = None
    session: Session | None = None

    def node(self, key: str) -> Address:
        assert self.ds is not None
        return self.ds.node(key)

    @property
    def evaluation(self) -> Evaluation:
        assert self.ds is not None and self.session is not None
        return evaluate.evaluation(self.ds, self.session)

    def read(self, key: str) -> None:
        """Ask ``key``'s answer (a failure raises)."""
        self.evaluation.get(self.node(key))

    def answer(self, key: str) -> object:
        assert self.ds is not None and self.session is not None
        return S.query(S.At(self.ds), self._reference(key), self.session)

    def value(self, key: str) -> object:
        """``key`` read as a value (through ``surface.value``)."""
        assert self.ds is not None and self.session is not None
        return S.value(S.At(self.ds), self._reference(key), self.session)

    def _reference(self, key: str) -> object:
        assert self.ds is not None
        return ValueHandle[object](self.ds.model, self.ds.node(key))

    def kept(self, key: str) -> bool:
        """Whether ``key``'s answer is kept (in the evaluation's memo)."""
        return self.node(key) in self.evaluation.memo

    def calls(self, kind: str) -> int:
        assert self.session is not None
        return self.session.counters.calls.get(kind, 0)

    def changed(self, choices: Mapping[str, object]) -> Opened:
        """The same facts with ``choices`` stored (others kept)."""
        assert self.ds is not None
        values = dict(self.ds.stored)
        values.update(prepare_values({self.ds.node(k): v for k, v in choices.items()}, "choice"))
        return Opened(self.way, ds=self.ds.with_stored(values), session=self.session)


def opened(way: str, declaration: Space, choices: Mapping[str, object] | None = None) -> Opened:
    running = Running()
    ds = cast(Any, running.open(design_space(declaration))).ds
    made = Opened(way, ds=ds, session=running.session)
    return made.changed(choices) if choices else made


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("warm", (False, True))
def test_nested_finally_and_manager_finish_uncached_reads(warm: bool, way: str) -> None:
    events: list[tuple[str, int]] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def cleanup_one(self) -> int:
            return self.fact + 1

        @derived
        def cleanup_two(self) -> int:
            return self.cleanup_one + 1

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def middle(self) -> int:
            try:
                with Cleanup(self):
                    return self.broken
            finally:
                events.append(("middle", self.cleanup_two))

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(("outer", self.fact))

    class Cleanup:
        def __init__(self, point: Example) -> None:
            self.point = point

        def __enter__(self) -> Cleanup:
            return self

        def __exit__(self, *error: object) -> None:
            events.append(("manager", self.point.cleanup_two))

    space = opened(way, Example(fact=7))
    if warm:
        assert space.value("cleanup_two") == 9
    with pytest.raises(NativeEvaluationError) as caught:
        space.value("outer")
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()
    assert events == [("manager", 9), ("middle", 9), ("outer", 7)]
    assert space.value("fact") == 7
    assert not space.kept("outer") and not space.kept("broken")


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("kind", ("unavailable", "inapplicable", "rejected"))
@pytest.mark.parametrize("warm", (False, True))
def test_nonvalue_cleanup_preserves_primary_and_outer_cleanup(
    kind: str, warm: bool, way: str
) -> None:
    events: list[object] = []

    class Example(Space):
        fact: int = Param()
        enabled: bool = Param()
        unavailable: int = Decision(values=(1, 2))
        inapplicable: int = Decision(values=(1,), when=enabled)

        @derived
        def rejected(self) -> int | Rejected:
            return reject("refused", "a derived value that refuses")

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def middle(self) -> int:
            try:
                return self.broken
            finally:
                events.append("middle started")
                if kind == "rejected":
                    _ = self.rejected
                else:
                    _ = self.unavailable if kind == "unavailable" else self.inapplicable
                events.append("middle finished")

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact)

    space = opened(way, Example(fact=7, enabled=False))
    if warm:
        space.read(kind)
    with pytest.raises(NativeEvaluationError) as caught:
        space.read("outer")
    assert isinstance(caught.value.__cause__, LookupError)
    issues = caught.value.cleanup_failures
    assert len(issues) == 1 and issues[0].owner == kind and issues[0].error is None
    assert isinstance(
        issues[0].result,
        {"unavailable": Unresolved, "inapplicable": Inapplicable, "rejected": Rejected}[kind],
    )
    assert events == ["middle started", 7]


@pytest.mark.parametrize("way", WAYS)
def test_secondary_exceptions_keep_order_and_primary(way: str) -> None:
    events: list[int] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def middle(self) -> int:
            try:
                return self.broken
            finally:
                events.append(self.fact)
                raise ValueError("middle cleanup")

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact + 1)
                raise ZeroDivisionError("outer cleanup")

    with pytest.raises(NativeEvaluationError) as caught:
        opened(way, Example(fact=7)).value("outer")
    assert caught.value.primary.owner == "broken"
    assert isinstance(caught.value.__cause__, LookupError)
    assert [
        (issue.owner, type(issue.error.__cause__))
        for issue in caught.value.cleanup_failures
        if issue.error is not None
    ] == [("middle", ValueError), ("outer", ZeroDivisionError)]
    assert events == [7, 8]


@pytest.mark.parametrize("way", WAYS)
def test_local_body_and_cleanup_exception_chain(way: str) -> None:
    class Example(Space):
        @derived
        def value(self) -> int:
            try:
                raise LookupError("body")
            finally:
                raise ValueError("cleanup")

    with pytest.raises(NativeEvaluationError) as caught:
        opened(way, Example()).value("value")
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    secondary = caught.value.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)


@pytest.mark.parametrize("way", WAYS)
def test_failed_prerequisite_rereads_are_query_local(way: str) -> None:
    starts: list[int] = []
    events: list[object] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def broken(self) -> int:
            starts.append(1)
            raise LookupError("primary")

        @derived
        def output(self) -> int:
            try:
                return self.broken
            finally:
                for _ in range(3):
                    try:
                        _ = self.broken
                    except EvaluationError:
                        events.append("caught")
                events.append(self.fact)

    space = opened(way, Example(fact=7))
    with pytest.raises(NativeEvaluationError) as caught:
        space.value("output")
    assert starts == [1] and events == ["caught", "caught", "caught", 7]
    assert caught.value.cleanup_failures == ()
    with pytest.raises(NativeEvaluationError):
        space.value("broken")
    assert starts == [1, 1]


@pytest.mark.parametrize("way", WAYS)
def test_cleanup_cycle_keeps_primary_and_finishes_parent(way: str) -> None:
    events: list[int] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def cycle(self) -> int:
            return self.cycle

        @derived
        def middle(self) -> int:
            try:
                return self.broken
            finally:
                _ = self.cycle

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact)

    with pytest.raises(NativeEvaluationError) as caught:
        opened(way, Example(fact=7)).value("outer")
    assert isinstance(caught.value.__cause__, LookupError)
    assert len(caught.value.cleanup_failures) == 1
    cycle = caught.value.cleanup_failures[0].error
    assert cycle is not None and cycle.owner == "cycle" and cycle.role == "dependency cycle"
    assert events == [7]


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("cancellation", (KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("warm", (False, True))
def test_cancellation_drains_and_retains_identity(
    cancellation: type[BaseException], warm: bool, way: str
) -> None:
    events: list[object] = []
    original = cancellation("cancel")

    class Example(Space):
        fact: int = Param()

        @derived
        def cancelled(self) -> int:
            raise original

        @derived
        def middle(self) -> int:
            try:
                return self.cancelled
            except cancellation:
                events.append("caught cancellation")
                return 99
            finally:
                events.append(self.fact)

        @derived
        def outer(self) -> int:
            try:
                return self.middle
            finally:
                events.append(self.fact + 1)
                raise ValueError("secondary cleanup")

    space = opened(way, Example(fact=7))
    if warm:
        assert space.value("fact") == 7
    with pytest.raises(cancellation) as caught:
        space.value("outer")
    assert caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and details.primary.owner == "cancelled"
    assert details.primary.__cause__ is original and len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and isinstance(secondary.__cause__, ValueError)
    assert events == ["caught cancellation", 7, 8] and space.value("fact") == 7


@pytest.mark.parametrize("way", WAYS)
def test_cancellation_cleanup_keeps_earlier_programmer_failure(way: str) -> None:
    original = KeyboardInterrupt("cleanup cancellation")

    class Example(Space):
        @derived
        def broken(self) -> int:
            raise LookupError("first")

        @derived
        def cancelled(self) -> int:
            raise original

        @derived
        def output(self) -> int:
            try:
                return self.broken
            finally:
                _ = self.cancelled

    with pytest.raises(KeyboardInterrupt) as caught:
        opened(way, Example()).read("output")
    assert caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and isinstance(details.primary.__cause__, LookupError)
    assert len(details.cleanup_failures) == 1
    secondary = details.cleanup_failures[0].error
    assert secondary is not None and secondary.__cause__ is original


# Interruptions at the evaluator's own points (``_running.evaluate``).

POINTS: dict[tuple[str, str], tuple[Callable[..., object], str]] = {
    ("running", "before-call"): (evaluate.invoke, "outcome = context.copy().run("),
    ("running", "suspended-read"): (evaluate.Reader._take, "found = self.ev.get(index)"),
    ("running", "publish"): (Evaluation._compute, "self.memo[index] = found\nexcept Deeper:"),
    ("running", "record"): (Evaluation._compute, "session.failures[key] = failing.failure"),
    ("running", "delivery"): (
        evaluate.Reader._take,
        "remember_failure(self.call, failing.failure)",
    ),
}
"""Each interruption point, by way: the function and the line (by a marker in it)."""

NODE = {"running": "index"}
"""The name the evaluating function (``Evaluation._compute``) gives the node it
evaluates."""


def _interrupt_at(
    way: str, point: str, original: BaseException, when: Callable[[FrameType], bool] | None = None
) -> Any:
    """A trace function raising ``original`` once, at ``point`` of ``way``'s evaluator."""
    function, marker = POINTS[(way, point)]
    code = cast(Any, function).__code__
    line = _line(function, marker)
    fired: list[bool] = []

    def trace(frame: FrameType, event: str, argument: object) -> TraceFunction:
        if (
            not fired
            and event == "line"
            and frame.f_code is code
            and frame.f_lineno == line
            and (when is None or when(frame))
        ):
            fired.append(True)
            raise original
        return trace

    trace.fired = fired  # type: ignore[attr-defined]
    return trace


def _line(function: Callable[..., object], marker: str) -> int:
    """The line of ``function``'s source where ``marker`` (its lines in a row) starts,
    once (``_evaluation_support.source_line``, for a marker of more than one line)."""
    first, *rest = marker.split("\n")
    if not rest:
        return source_line(function, first)
    source, start = _inspect.getsourcelines(function)
    found = [
        start + offset
        for offset in range(len(source) - len(rest))
        if first in source[offset]
        and all(line in source[offset + 1 + at] for at, line in enumerate(rest))
    ]
    assert len(found) == 1, f"expected one interruption boundary for {marker!r}: {found}"
    return found[0]


def _traced(trace: object, run: Callable[[], object]) -> None:
    prior = sys.gettrace()
    try:
        sys.settrace(cast(Any, trace))
        run()
    finally:
        sys.settrace(prior)


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("boundary", ("before-call", "suspended-read"))
def test_driver_cancellation_respects_callback_start(boundary: str, way: str) -> None:
    """An interruption before the call starts never enters it; one inside a read of the
    running method runs the method's cleanup."""
    events: list[object] = []
    original = KeyboardInterrupt("driver boundary")

    class Example(Space):
        fact: int = Param()
        cleanup: int = Param()

        @derived
        def output(self) -> int:
            events.append("entered")
            try:
                return self.fact
            finally:
                events.append(("cleanup", self.cleanup))

    space = opened(way, Example(fact=7, cleanup=8))
    trace = _interrupt_at(way, boundary, original)
    with pytest.raises(KeyboardInterrupt) as caught:
        _traced(trace, lambda: space.read("output"))
    assert trace.fired and caught.value is original
    details = cancellation_details(caught.value)
    assert details is not None and details.cleanup_failures == ()
    assert details.primary.role == "derived" and details.primary.owner == "output"
    calls = space.calls("derived")
    if boundary == "suspended-read":
        assert events == ["entered", ("cleanup", 8)] and calls == 1
    else:
        assert events == [] and calls == 1  # counted, never entered
    assert space.value("fact") == 7


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize(("boundary", "primary_failure"), (("publish", False), ("record", True)))
def test_interrupted_completion_drains_parent_and_preserves_failure(
    boundary: str, primary_failure: bool, way: str
) -> None:
    """An evaluation's answer published, or, where it failed, its failure recorded for the
    run: interrupted there, the readers' cleanups run and the failure completes."""
    events: list[object] = []
    original = KeyboardInterrupt("completion boundary")
    first = LookupError("original callback failure")

    class Example(Space):
        cleanup: int = Param()

        @derived
        def fact(self) -> int:
            events.append("fact")
            if primary_failure:
                raise first
            return 7

        @derived
        def middle(self) -> int:
            events.append("entered")
            try:
                return self.fact
            finally:
                events.append(("middle cleanup", self.cleanup))

        @derived
        def output(self) -> int:
            try:
                return self.middle
            finally:
                events.append(("outer cleanup", self.cleanup + 1))
                raise ValueError("outer cleanup failure")

    space = opened(way, Example(cleanup=8))
    fact = space.node("fact")
    trace = _interrupt_at(
        way, boundary, original, lambda frame: frame.f_locals.get(NODE[way]) == fact
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        _traced(trace, lambda: space.read("output"))
    assert trace.fired and caught.value is original
    assert events == ["entered", "fact", ("middle cleanup", 8), ("outer cleanup", 9)]
    details = cancellation_details(caught.value)
    assert details is not None
    if primary_failure:
        assert details.primary.owner == "fact" and details.primary.__cause__ is first
        assert len(details.cleanup_failures) == 2
        interrupted = details.cleanup_failures[0].error
        assert interrupted is not None and interrupted.__cause__ is original
    else:
        assert details.primary.owner == "fact" and details.primary.__cause__ is original
        assert len(details.cleanup_failures) == 1
    last = details.cleanup_failures[-1].error
    assert last is not None and isinstance(last.__cause__, ValueError)
    for key in ("fact", "middle", "output"):
        assert not space.kept(key)
    assert space.value("cleanup") == 8


@pytest.mark.parametrize("way", WAYS)
def test_interruption_before_failure_delivery_keeps_pending_primary(way: str) -> None:
    """Interrupted as a read delivers a failure to its method, the failure stays primary."""
    events: list[int] = []
    first = LookupError("provider failure")
    original = KeyboardInterrupt("before failure delivery")

    class Example(Space):
        cleanup: int = Param()

        @derived
        def failed(self) -> int:
            raise first

        @derived
        def output(self) -> int:
            try:
                return self.failed
            finally:
                events.append(self.cleanup)

    space = opened(way, Example(cleanup=8))
    trace = _interrupt_at(way, "delivery", original)
    with pytest.raises(KeyboardInterrupt) as caught:
        _traced(trace, lambda: space.read("output"))
    assert trace.fired and caught.value is original and events == [8]
    details = cancellation_details(original)
    assert details is not None and details.primary.__cause__ is first
    assert len(details.cleanup_failures) == 1
    interrupted = details.cleanup_failures[0].error
    assert interrupted is not None and interrupted.__cause__ is original


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("field", ("caught", "finally_return", "blocked"))
def test_caught_failure_and_nonvalue_cannot_publish_fallback(field: str, way: str) -> None:
    class Example(Space):
        choice: int = Decision(values=(1, 2))

        @derived
        def broken(self) -> int:
            raise LookupError("primary")

        @derived
        def caught(self) -> int:
            try:
                return self.broken
            except EvaluationError:
                return 99

        @derived
        def finally_return(self) -> int:
            try:
                return self.broken
            finally:
                return 99  # noqa: B012 - the case under test

        @derived
        def blocked(self) -> int:
            try:
                return self.choice
            except ValueUnavailableError:
                return 99

    space = opened(way, Example())
    if field == "blocked":
        assert isinstance(space.answer("blocked"), Unresolved)
        assert space.changed({"choice": 1}).value("blocked") == 1
    else:
        with pytest.raises(NativeEvaluationError):
            space.read(field)
        assert not space.kept(field)


@pytest.mark.parametrize("way", WAYS)
def test_driver_only_and_foreign_reads_remain_sticky_when_caught(way: str) -> None:
    receivers: list[Space] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def kept(self) -> int:
            receivers.append(self)
            return self.fact

        @derived
        def driver(self) -> int:
            try:
                _failure.driver_only("query inspection")
            except EvaluationError:
                return 99
            return 0

        @derived
        def foreign(self) -> int:
            try:
                return int(getattr(receivers[0], "fact"))
            except EvaluationError:
                return 99

    other = opened(way, Example(fact=8))
    assert other.value("kept") == 8
    space = opened(way, Example(fact=7))
    with pytest.raises(NativeEvaluationError, match="driver-only"):
        space.read("driver")
    with pytest.raises(NativeEvaluationError, match="cross-configuration"):
        space.read("foreign")
    assert space.value("fact") == 7
    with pytest.raises(EvaluationError, match="after it returned"):
        _ = getattr(receivers[0], "fact")


@pytest.mark.parametrize("way", WAYS)
def test_context_variables_survive_suspension_without_leaking(way: str) -> None:
    """Each call runs in a copy of its driver's context, never in its reader's."""
    marker: ContextVar[str] = ContextVar("test_running_marker", default="driver")
    observed: list[str] = []

    class Example(Space):
        fact: int = Param()

        @derived
        def inner(self) -> int:
            observed.append(marker.get())
            marker.set("inner")
            return self.fact

        @derived
        def outer(self) -> int:
            token = marker.set("outer")
            try:
                value = self.inner
                observed.append(marker.get())
                return value
            finally:
                marker.reset(token)

    assert opened(way, Example(fact=7)).value("outer") == 7 and observed == ["driver", "outer"]
    assert marker.get() == "driver"


@pytest.mark.parametrize("way", WAYS)
def test_contextual_domain_error_is_one_primary_failure(way: str) -> None:
    """A stored value's type check whose membership raises."""

    def accepts(*, candidate: int) -> bool:
        raise LookupError("membership failure")

    class Example(Space):
        choice: int = Decision(domain=domain(accepts=accepts))

    with pytest.raises(NativeEvaluationError) as caught:
        opened(way, Example(), {"choice": 1}).read("choice")
    assert caught.value.owner == "choice" and caught.value.role == "domain membership"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == ()


@pytest.mark.parametrize("way", WAYS)
def test_value_transformations_cannot_read_a_design_space_even_when_caught(way: str) -> None:
    class Facts(Space):
        value: int = Param()

    facts = opened(way, Facts(value=7))
    with pytest.raises(EvaluationError, match="pure value transformation"):
        with _failure.transformation("value recognition"):
            try:
                facts.read("value")
            except EvaluationError:
                pass
    assert facts.value("value") == 7


def test_a_blocked_self_constraint_keeps_its_inspectable_assessment() -> None:
    class Example(Space):
        choice: int = Decision(values=(1, 2))

        @constraint
        def positive(self) -> bool:
            return self.choice > 0

    point = design_space(Example())
    running = Running()
    at = S.At(cast(Any, running.open(point)).ds)
    assessment = S.inspect(at, Example.positive, running.session)
    assert assessment == point.inspect(Example.positive)
    assert isinstance(assessment, ConstraintAssessment)
    assert isinstance(assessment.result, Unresolved)
    assert isinstance(assessment.results["positive"], Unresolved)


@pytest.mark.parametrize("way", WAYS)
def test_self_cycles_are_reached_through_scopes_and_guards(way: str) -> None:
    class Child(Space):
        source: int = Param()

        @derived
        def value(self) -> int:
            return self.source

    class Parent(Space):
        @derived
        def output(self) -> int:
            return self.child.value

        child = Child(source=output)

    class Guard(Space):
        @derived
        def enabled(self) -> bool:
            return self.choice > 0

        choice: int = Decision(values=(1,), when=enabled)

    with pytest.raises(EvaluationError, match="dependency cycle") as scoped:
        opened(way, Parent()).read("output")
    assert "child.value" in str(scoped.value)
    with pytest.raises(EvaluationError, match="dependency cycle"):
        opened(way, Guard(), {"choice": 1}).read("choice")


@pytest.mark.parametrize("way", WAYS)
def test_a_failure_in_another_design_space_runs_its_readers_cleanup(way: str) -> None:
    events: list[int] = []

    class Example(Space):
        count: int = Decision(values=(1, 2, 3))

        @derived
        def inner(self) -> int:
            if self.count == 3:
                raise LookupError("three")
            return self.count

        @derived
        def outer(self) -> int:
            try:
                return self.inner
            finally:
                events.append(self.count)

    space = opened(way, Example(), {"count": 1})
    assert space.value("outer") == 1
    with pytest.raises(NativeEvaluationError) as caught:
        space.changed({"count": 3}).read("outer")
    assert caught.value.primary.owner == "inner" and events == [1, 3]


def test_a_failure_is_kept_for_its_run_only_and_raised_by_every_question() -> None:
    """A failure is raised by every question that reaches it, as the public API raises it,
    and kept for its run only: each question runs the function again."""
    starts: list[int] = []

    class Example(Space):
        choice: int = Decision(values=(1, 2))

        @derived
        def broken(self) -> int:
            starts.append(1)
            raise LookupError("broken")

        @constraint
        def holds(self) -> bool:
            return self.broken > self.choice

    point = design_space(Example())
    running = Running()
    at = S.At(cast(Any, running.open(point)).ds)
    asks: list[Callable[[], object]] = [
        lambda: S.query(at, Example.broken, running.session),
        lambda: S.inspect(at, Example.holds, running.session),
        lambda: S.explain(at, Example.holds, running.session),
        lambda: S.value(at, Example.broken, running.session),
    ]
    for ask in asks:
        with pytest.raises(NativeEvaluationError) as caught:
            ask()
        assert caught.value.primary.owner == "broken" and caught.value.cleanup_failures == ()
    assert len(starts) == len(asks)
    with pytest.raises(NativeEvaluationError, match="broken"):
        point.inspect(Example.holds)
