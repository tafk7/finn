# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The method reader taken into ``_running`` (``receiver.MethodReader``) reads what today's
(``_world.MethodReader``) reads.

The running type's reader subclasses today's, through a stand-in for the world it reads
(``evaluate.Reader``). Rebased here on the moved reader, which reads the compiled model and
its per-class record alone, every member of each design space below answers alike: the same
answer, the same reads in order (its trace), the same reads made only to know whether
something exists, and the same public answers (``support.space_oracle.forms.space``):

- FINN's toys, each at its design space and after its change;
- random composite Space classes (``_compose_support``): children, selectors, formals bound
  to members, presence tests;
- a Space whose methods read a value input's presence, a child's presence, a selector's
  selected candidate's member, a candidate selected or not, the root and a child.

The rebasing is made for the test alone, as the move will wire it: the running type's reader
on the moved base, the receiver it makes, the read boundary recognizing it (``occurrence``),
and the depth guard it passes through (the running type's, today's subclass, until then).
The moved ``scope_choice`` is both of today's (``occurrence._scope_choice`` and the reader's
``_choice``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import pytest
from support.space_oracle import Implementation, forms, load, toys
from support.space_oracle.today import TodayAnswers

from core.space._compose_support import composite_space
from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    _world,
    bind,
    derived,
    design_space,
    inspection,
    occurrence,
)
from finn.core.space._failure import Call, CleanupFailure, Failing, exception, remember_failure
from finn.core.space._running import answers, evaluate, receiver
from finn.core.space._running.evaluate import Evaluation, Session, evaluation
from finn.core.space._running.value import DesignSpace
from finn.core.space._structure import Structure
from finn.core.space.errors import RequestError, ValueUnavailableError
from finn.core.space.ir import Node, NodeId
from finn.core.space.results import Available, Inapplicable, QueryResult

RUNNING: Implementation = load("core.space._running_type:RUNNING")


class Rebased(receiver.MethodReader):
    """The running type's reader (``evaluate.Reader``), its overrides as they are, on the
    moved base."""

    __slots__ = ("ev", "reader")

    def __init__(self, ev: Evaluation, node: Node, call: Call, reader: Node) -> None:
        super().__init__(ev.model, node, call)
        self.ev = ev
        self.reader = reader

    @property
    def world(self) -> object:
        """What today's read boundary reads of a reader (``occurrence.model_of``,
        ``bind_field``) until the move repoints it to ``model``."""
        return SimpleNamespace(evaluator=SimpleNamespace(model=self.model))

    def _source(self, index: NodeId) -> NodeId:
        return self.ev.source(index, self.reader)

    def _take(self, index: NodeId) -> Any:
        try:
            found = self.ev.get(index)
        except evaluate.Deeper as deeper:
            self.call.abandoned = deeper
            raise
        except Failing as failing:
            remember_failure(self.call, failing.failure)
            raise exception(failing.failure) from None
        self.trace.append((index, found.result))
        return found

    def _read_value(self, point: Any, reference: object) -> object:
        self._enter()
        index = self.ev.model.resolve(point._scope, reference)
        self._check(index)
        index = self._source(index)
        self.valued.add(index)
        return self._value(index)

    def _answer(self, index: NodeId) -> QueryResult[object]:
        self._check(index)
        index = self._source(index)
        self.existence.add(index)
        return cast(QueryResult[object], self._take(index).result)

    def _value(self, index: NodeId) -> object:
        index = self._source(index)
        answer = self._take(index).result
        node = self.ev.raw[index]
        if isinstance(answer, Available):
            return answers.public_value(node, answer)
        self.call.blocked[index] = answer
        if self.call.failure is not None:
            self.call.failure.add(CleanupFailure(node.owner, "cleanup read", result=answer))
        raise ValueUnavailableError(answer, context=node.owner)

    def _source_applies(self, index: NodeId) -> bool | NodeId:
        index = self._source(index)
        node = self.ev.raw[index]
        if node.guard is not None:
            holds = self._holds(node.guard)
            if holds is not True:
                return holds
        if node.kind == "param":
            return isinstance(self._answer(index), Available)
        if node.kind in ("alias", "guard", "view"):
            return self._source_applies(cast(NodeId, node.output))
        if node.kind == "select":
            selector = cast(NodeId, node.selector)
            case = self._answer(selector)
            if not isinstance(case, Available):
                return False if isinstance(case, Inapplicable) else selector
            target = cast(Mapping[str, NodeId], node.selection_index).get(cast(str, case.value))
            return False if target is None else self._source_applies(target)
        if node.kind == "present":
            found = [self._source_applies(target) for _, target in node.alternatives]
            if any(answer is True for answer in found):
                return True
            return next((answer for answer in found if answer is not False), False)
        return True


@contextmanager
def _rebased(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Rebased]]:
    """The running type reading through ``Rebased``, for the block: each reader made."""
    world = SimpleNamespace(**{**vars(_world), "receiver": receiver.receiver})
    made: list[Rebased] = []

    def reader(ev: Evaluation, node: Node, call: Call, reader: Node) -> Rebased:
        made.append(Rebased(ev, node, call, reader))
        return made[-1]

    with monkeypatch.context() as patched:
        patched.setattr(evaluate, "Reader", reader)
        patched.setattr(evaluate, "W", world)
        patched.setattr(occurrence, "MethodReader", receiver.MethodReader)
        patched.setattr(receiver, "Deeper", _world.Deeper)
        yield made


# -- the design spaces asked ----------------------------------------------------------------


class Part(Space):
    v: int = Param(required=False)
    pe: int = Decision(values=(1, 2, 4))

    @derived
    def given(self) -> bool:
        return self.present(Part.v)


class Spare(Space):
    v: int = Param(required=False)


class Reads(Space):
    w: int = Param()
    pick: int = Decision(values=(0, 1))

    @derived
    def on(self) -> bool:
        return self.pick == 1

    part = Part(v=w, when=on)
    unit: Part | Spare = Select(part=Part(v=w), spare=Spare())

    @derived
    def placed(self) -> bool:
        return self.present(Reads.part)

    @derived
    def chosen(self) -> int:
        return self.unit.v

    @derived
    def wider(self) -> int:
        return int(self.part.pe * self.root.w)  # type: ignore[attr-defined]

    @derived
    def spare(self) -> bool:
        return inspection.candidate(self, Reads.unit, "spare") is not None


# Random composites that open (most draws are refused by the domain rule, or a cycle).
COMPOSITES = (33, 34, 36, 51, 58, 60, 71, 82, 100, 112, 129, 206)

READS: dict[str, dict[object, object]] = {
    "nothing stored": {},
    "part placed": {Reads.pick: 1, Reads.part.pe: 2},
    "part placed, unit its part": {Reads.pick: 1, Reads.part.pe: 2, Reads.unit: "part"},
    "unit spare": {Reads.unit: "spare"},
    "part placed, unit spare": {Reads.pick: 1, Reads.unit: "spare"},
}


def _toy(name: str, changed: bool) -> Space:
    toy = toys.TOYS[name]
    point = toy.make()
    if not changed:
        return point
    report = TodayAnswers(point).change(toy.change, ()).report
    assert report.accepted
    return cast(Space, report.instance)


CASES: dict[str, Callable[[], Space]] = {
    **{
        f"{name}{', changed' if changed else ''}": partial(_toy, name, changed)
        for name in sorted(toys.TOYS)
        for changed in (False, True)
    },
    **{
        f"composite {seed}": partial(
            lambda seed: design_space(composite_space(seed)(a=3)),  # type: ignore[call-arg]
            seed,
        )
        for seed in COMPOSITES
    },
    **{
        f"reads, {label}": partial(
            lambda stored: bind(design_space(Reads(w=2)), stored).instance, stored
        )
        for label, stored in READS.items()
    },
}


def _form(found: object) -> object:
    try:
        return forms.answer(cast(QueryResult[Any], found))
    except Exception as error:  # noqa: BLE001 - a form that cannot be made, by its error
        return repr(type(error))


def _public(point: Space) -> object:
    """Every public answer of ``point``'s design space, by the running type; or what asking
    raised (a random composite's method may read what its Space does not hold)."""
    try:
        return forms.space(RUNNING.open(point), forms.ALL)
    except Exception as error:  # noqa: BLE001 - a raise is the answer compared
        return forms.raised(error)


def _members(ds: DesignSpace) -> dict[str, object]:
    """Every member's answer, trace and reads made only to know whether something exists,
    in a session of its own (no cache): what each method read, in order."""
    ev = evaluation(ds, Session())
    linked = cast(Structure, ds.model.linked)
    found: dict[str, object] = {}
    for index in tuple(linked.all_nodes()):
        key = linked.header(index).key
        try:
            got = ev.get(index)
        except Exception as error:  # noqa: BLE001 - a raise is the answer compared
            found[key] = forms.raised(error)
            continue
        found[key] = (
            _form(got.result),
            [(node, _form(result)) for node, result in got.trace],
            sorted(got.existence),
        )
    return found


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_moved_reader_reads_what_todays_reads(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = CASES[name]()
    ds = cast(Any, RUNNING.open(point)).ds
    base = _members(ds)
    public = _public(point)
    with _rebased(monkeypatch):
        moved = _members(ds)
        assert _public(point) == public
    assert moved == base
    assert any(isinstance(entry, tuple) and entry[1] for entry in base.values())


def test_the_methods_read_through_every_part_of_the_moved_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = CASES["reads, part placed, unit its part"]()
    ds = cast(Any, RUNNING.open(point)).ds
    with _rebased(monkeypatch) as made:
        found = _members(ds)
    assert made and not any(isinstance(item, _world.MethodReader) for item in made)
    asked = ("placed", "chosen", "wider", "spare", "part.given")
    assert {key: cast(tuple[object, ...], found[key])[0] for key in asked} == {
        "placed": {"kind": "value", "value": True},
        "chosen": {"kind": "value", "value": 2},
        "wider": {"kind": "value", "value": 4},
        "spare": {"kind": "value", "value": True},
        "part.given": {"kind": "value", "value": True},
    }


def test_one_scope_choice_is_both_of_todays() -> None:
    point = toys.TOYS["house"].make()
    linked = occurrence.state(point).model.linked
    model = occurrence.state(point).model
    today = _world.MethodReader(
        cast(Any, SimpleNamespace(evaluator=model.evaluator)), cast(Any, None), cast(Any, None)
    )

    def outcome(call: Any) -> object:
        try:
            return call()
        except RequestError as error:
            return (RequestError, str(error))

    heating = type(point).heating  # type: ignore[attr-defined]
    for scope in range(len(occurrence.state(point).model.linked.scopes)):
        for decision in (heating, object(), None, [1]):
            moved = outcome(lambda: receiver.scope_choice(linked, scope, decision))
            assert moved == outcome(lambda: occurrence._scope_choice(linked, scope, decision))
            assert moved == outcome(lambda: today._choice(scope, decision))
