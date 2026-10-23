# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The method reader (``_running.receiver.MethodReader``), on which the running type's
reader (``evaluate.Reader``) reads the compiled model and its per-class record alone.

Every member of each design space below answers alike through the public API (its model's
session, the node cache on) and through the public API with no cache: the same public
answers (``support.space_oracle.forms.space``), and in a session of its own each method
reads something, in order (its trace):

- FINN's toys, each at its design space and after its change;
- random composite Space classes (``_compose_support``): children, selectors, formals bound
  to members, presence tests;
- a Space whose methods read a value input's presence, a child's presence, a selector's
  selected candidate's member, a candidate selected or not, the root and a child.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import pytest
from support.space_oracle import Implementation, forms, load, toys
from support.space_oracle.public import PUBLIC, PublicAnswers

from core.space._compose_support import composite_space
from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    bind,
    derived,
    design_space,
    inspection,
)
from finn.core.space._running import evaluate, receiver
from finn.core.space._running.evaluate import Session, evaluation
from finn.core.space._running.value import DesignSpace
from finn.core.space.errors import RequestError
from finn.core.space.occurrence import held
from finn.core.space.results import QueryResult

RUNNING: Implementation = load("core.space._running_type:RUNNING")


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
    report = PublicAnswers(point).change(toy.change, ()).report
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


def _public(implementation: Implementation, point: Space) -> object:
    """Every public answer of ``point``'s design space, by ``implementation``; or what
    asking raised (a random composite's method may read what its Space does not hold)."""
    try:
        return forms.space(implementation.open(point), forms.ALL)
    except Exception as error:  # noqa: BLE001 - a raise is the answer compared
        return forms.raised(error)


def _members(ds: DesignSpace) -> dict[str, object]:
    """Every member's answer, trace and reads made only to know whether something exists,
    in a session of its own (no cache): what each method read, in order."""
    ev = evaluation(ds, Session())
    found: dict[str, object] = {}
    for index in tuple(ds.model.scopes.all_addresses()):
        key = index.key
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
def test_every_method_reads_alike_with_and_without_the_cache(name: str) -> None:
    point = CASES[name]()
    assert _public(PUBLIC, point) == _public(RUNNING, point)
    found = _members(held(point).ds)
    assert any(isinstance(entry, tuple) and entry[1] for entry in found.values())


def test_the_methods_read_through_every_part_of_the_reader() -> None:
    point = CASES["reads, part placed, unit its part"]()
    made: list[evaluate.Reader] = []
    reader = evaluate.Reader

    def recorded(*arguments: Any) -> evaluate.Reader:
        made.append(reader(*arguments))
        return made[-1]

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(evaluate, "Reader", recorded)
        found = _members(held(point).ds)
    assert made and all(isinstance(item, receiver.MethodReader) for item in made)
    asked = ("placed", "chosen", "wider", "spare", "part.given")
    assert {key: cast(tuple[object, ...], found[key])[0] for key in asked} == {
        "placed": {"kind": "value", "value": True},
        "chosen": {"kind": "value", "value": 2},
        "wider": {"kind": "value", "value": 4},
        "spare": {"kind": "value", "value": True},
        "part.given": {"kind": "value", "value": True},
    }


def test_a_scope_choice_names_a_decision_over_nodes_of_its_scope() -> None:
    point = toys.TOYS["house"].make()
    root = held(point).ds.model.root
    heating = type(point).heating  # type: ignore[attr-defined]
    found = receiver.scope_choice(root, heating)
    assert found.key == "heating"
    for decision in (object(), None, [1]):
        with pytest.raises(RequestError, match="not part of this compiled scope"):
            receiver.scope_choice(root, decision)
