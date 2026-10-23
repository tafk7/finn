# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Laziness at evaluation: a scope that is absent evaluates no member of its own.

A Space's validity reads a placed Space's piece from the guard its placing scope places it
under first, and where that guard does not hold, the placed Space is absent: its validity is
not evaluated, and the piece answers inapplicable, as the placed Space's validity would
(``_running.evaluate``'s validity frame). A choice's presence reads its guards outermost
first, by address, so a choice in an absent scope is found absent from the guard its placing
scope places it under, its own member and guard not evaluated
(``_running.questions.presence``). Answers and refusals stay as they were; stage 7's chain
is the kernel tree's (``kernel_ops/test_space_templates``)."""

from __future__ import annotations

from typing import Any

import pytest

from finn.core.space import (
    Available,
    ConstraintAssessment,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.core.space._running import facts, questions
from finn.core.space._running.answers import INAPPLICABLE
from finn.core.space._running.evaluate import Ans, Session, evaluation
from finn.core.space._running.value import DesignSpace
from finn.core.space._scopes import EAGER, Address
from finn.core.space.occurrence import ask, held


class Fast(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2, 4))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.lanes > self.width:
            return reject("too-wide", "more lanes than width")
        return True


class Slow(Space):
    depth: int = Decision(values=(1, 2))


def _blank() -> Space:
    """A design space of a root class of its own: a model (and its scope records and
    session) is kept per class, so what one test evaluates would show in the next."""

    class Kernel(Space):
        width: int = Param()
        compute: Fast | Slow = Select(fast=Fast(width=width), slow=Slow())

    return design_space(Kernel(width=2))


class Gated(Space):
    """A candidate whose choice has its own guard (``when``), inside the candidate's."""

    lanes: int = Decision(values=(1, 2))

    @derived
    def wide(self) -> bool:
        return self.lanes > 1

    deep: int = Decision(values=(1, 2), when=wide)


class GatedKernel(Space):
    compute: Gated | Slow = Select(gated=Gated(), slow=Slow())


@pytest.fixture(autouse=True)
def _lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode reads everything at once


def _ds(point: Space | DesignSpace) -> DesignSpace:
    return point if isinstance(point, DesignSpace) else held(point).ds


def _evaluated(point: Space | DesignSpace) -> set[Address]:
    """Every address an evaluation of the design space ``point`` holds has an answer for."""
    return {index for ev in _ds(point).evaluations.values() for index in ev.memo}


def _in_play(point: Space | DesignSpace) -> set[str]:
    """The scopes with a member evaluated in the design space ``point`` holds."""
    return {index.scope.name for index in _evaluated(point)}


def _handles(point: Space) -> dict[str, Any]:
    return {item.key: item.reference for item in inspection.decisions(point)}


def _answer(point: Space, index: Address) -> Ans:
    """``index``'s answer in the design space ``point`` holds, in its model's session."""
    ds = held(point).ds
    return ask(ds.model, lambda session: evaluation(ds, session).get(index))


def test_an_absent_placed_spaces_validity_is_read_from_its_guard() -> None:
    blank = _blank()
    scopes = inspection.model(blank).scopes
    handles = _handles(blank)
    slow = blank.with_choices({handles["compute"]: "slow", handles["compute.slow.depth"]: 1})
    assert inspection.validity(slow).status == "valid"
    # The case not selected is placed (its record made, to list its choices) and evaluates
    # nothing: the scopes with a member evaluated are the present ones.
    assert "compute.fast" in {record.name for record in scopes.records()}
    assert _in_play(slow) == {"", "compute.slow"}
    # The root's validity read the case's guard, false, and not the case's validity.
    validity = scopes.root.validity
    guard = scopes.find("compute.fast.$selected")
    fast = scopes.find("compute.fast.$validity")
    found = _answer(slow, validity)
    read = [index for index, _ in found.trace]
    assert guard in read and fast not in read
    assessment = found.assessment
    assert isinstance(assessment, ConstraintAssessment)
    assert isinstance(assessment.results["compute.fast.$validity"], Inapplicable)
    assert fast not in _evaluated(slow)


def test_a_placed_space_selected_later_is_read_and_refuses_as_before() -> None:
    blank = _blank()
    scopes = inspection.model(blank).scopes
    handles = _handles(blank)
    slow = blank.with_choices({handles["compute"]: "slow", handles["compute.slow.depth"]: 1})
    assert "compute.fast" not in _in_play(slow)
    # The root's validity found from its guard alone is evaluated again where the guard
    # holds (early cutoff reads the guard it recorded): the case's constraint refuses.
    report = slow.try_with_choices({handles["compute"]: "fast", handles["compute.fast.lanes"]: 4})
    assert not report.accepted
    assert [rule.rule for rule in report.broken] == ["compute.fast.fits"]
    fast = slow.with_choices({handles["compute"]: "fast", handles["compute.fast.lanes"]: 2})
    assert inspection.validity(fast).status == "valid"
    assert "compute.fast" in _in_play(fast)
    # And back: the case's validity, read where its guard held, is not read where it fails.
    back = fast.with_choices({handles["compute"]: "slow"})
    assert inspection.validity(back).status == "valid"
    read = [index for index, _ in _answer(back, scopes.root.validity).trace]
    assert scopes.find("compute.fast.$validity") not in read
    assert inspection.validity(fast.with_choices({handles["compute.fast.lanes"]: 1})).status == (
        "valid"
    )


def test_an_absent_scopes_choices_are_found_absent_from_its_guard() -> None:
    blank = _blank()
    handles = _handles(blank)
    slow = blank.with_choices({handles["compute"]: "slow"})
    states = dict(inspection.states(slow))
    assert isinstance(states["compute.fast.lanes"], inspection.Absent)
    assert "compute.fast" not in _in_play(slow)


def test_presence_reads_the_guards_outermost_first() -> None:
    model = inspection.model(GatedKernel)
    scopes = model.scopes
    deep = scopes.find("compute.gated.deep")
    own = deep.guard  # its ``when``, a member of the absent scope
    assert own is not None
    selector = scopes.find("compute")
    opened = facts.open(model, {})

    def changed(requests: dict[Address, object]) -> DesignSpace:
        report = questions.change(opened, {}, list(requests.items()), Session())
        assert report.published is not None
        return report.published

    def presence(ds: DesignSpace, index: Address) -> tuple[str, object]:
        ev = evaluation(ds, Session())
        return ev.run(lambda: questions.presence(ev, index))

    slow = changed({selector: "slow"})
    # Absent from the case's guard: neither it nor its own guard evaluated.
    assert presence(slow, deep) == ("absent", INAPPLICABLE)
    assert "compute.gated" not in _in_play(slow)
    assert deep not in _evaluated(slow) and own not in _evaluated(slow)
    # What its own guard answers, read now: the same answer, the same object.
    assert evaluation(slow, Session()).result(own) is INAPPLICABLE
    # Where the case's guard holds or waits, its own guard decides, as before.
    lanes = scopes.find("compute.gated.lanes")
    narrow = changed({selector: "gated", lanes: 1})
    assert presence(narrow, deep) == ("absent", Available(False))
    wide = changed({selector: "gated", lanes: 2})
    assert presence(wide, deep) == ("present", Available(True))
    found, answer = presence(opened, deep)
    assert found == "unknown" and not isinstance(answer, (Available, Inapplicable))
