# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Laziness at evaluation: a scope that is absent resolves no node of its own.

A Space's validity reads a placed Space's piece from the guard its placing scope places it
under first, and where that guard does not hold, the placed Space is absent: its validity is
neither read nor resolved, and the piece answers inapplicable, as the placed Space's
validity would (``_world._validity``). A choice's presence reads its guards outermost first,
from the nodes' headers, so a choice in an absent scope is found absent from the guard its
placing scope places it under, its own node and guard not resolved (``_state.presence``).
Answers, refusals and early cutoff stay as they were; stage 7's chain is the kernel tree's
(``kernel_ops/test_space_templates``)."""

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
from finn.core.space._state import bind, presence
from finn.core.space._structure import EAGER, Structure
from finn.core.space._world import INAPPLICABLE
from finn.core.space.occurrence import state


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
    """A design space of a root class of its own: a model (and its structure) is kept per
    class, so what one test resolves would show in the next."""

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
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode resolves everything at once


def _structure(point: Space) -> Structure:
    structure = inspection.model(point).structure
    assert structure is not None
    return structure


def _in_play(structure: Structure) -> set[str]:
    return {structure.records[index].name for index in structure._in_play}


def _handles(point: Space) -> dict[str, Any]:
    return {item.key: item.reference for item in inspection.decisions(point)}


def test_an_absent_placed_spaces_validity_is_read_from_its_guard() -> None:
    blank = _blank()
    structure = _structure(blank)
    handles = _handles(blank)
    slow = blank.with_choices({handles["compute"]: "slow", handles["compute.slow.depth"]: 1})
    assert inspection.validity(slow).status == "valid"
    # The case not selected is placed (its record made, to list its choices) and resolves
    # nothing: the scopes with a node resolved are the present ones.
    assert "compute.fast" in {record.name for record in structure.records}
    assert _in_play(structure) == {"", "compute.slow"}
    # The root's validity read the case's guard, false, and not the case's validity.
    validity = structure.scopes[0].validity
    assert validity is not None
    guard = structure.find("compute.fast.$selected")
    fast = structure.find("compute.fast.$validity")
    world = state(slow).world
    read = [index for index, _ in world.entry(validity).trace]
    assert guard in read and fast not in read
    assessment = world.entry(validity).assessment
    assert isinstance(assessment, ConstraintAssessment)
    assert isinstance(assessment.results["compute.fast.$validity"], Inapplicable)
    assert fast not in structure._raw


def test_a_placed_space_selected_later_is_read_and_refuses_as_before() -> None:
    blank = _blank()
    structure = _structure(blank)
    handles = _handles(blank)
    slow = blank.with_choices({handles["compute"]: "slow", handles["compute.slow.depth"]: 1})
    assert "compute.fast" not in _in_play(structure)
    # The root's validity found from its guard alone is evaluated again where the guard
    # holds (early cutoff reads the guard it recorded): the case's constraint refuses.
    report = slow.try_with_choices({handles["compute"]: "fast", handles["compute.fast.lanes"]: 4})
    assert not report.accepted
    assert [rule.rule for rule in report.broken] == ["compute.fast.fits"]
    fast = slow.with_choices({handles["compute"]: "fast", handles["compute.fast.lanes"]: 2})
    assert inspection.validity(fast).status == "valid"
    assert "compute.fast" in _in_play(structure)
    # And back: the case's validity, read where its guard held, is not read where it fails.
    back = fast.with_choices({handles["compute"]: "slow"})
    assert inspection.validity(back).status == "valid"
    validity = structure.scopes[0].validity
    assert validity is not None
    world = state(back).world
    read = [index for index, _ in world.entry(validity).trace]
    assert structure.find("compute.fast.$validity") not in read
    assert inspection.validity(fast.with_choices({handles["compute.fast.lanes"]: 1})).status == (
        "valid"
    )


def test_an_absent_scopes_choices_are_found_absent_from_its_guard() -> None:
    blank = _blank()
    structure = _structure(blank)
    handles = _handles(blank)
    slow = blank.with_choices({handles["compute"]: "slow"})
    states = dict(inspection.states(slow))
    assert isinstance(states["compute.fast.lanes"], inspection.Absent)
    assert "compute.fast" not in _in_play(structure)


def test_presence_reads_the_guards_outermost_first() -> None:
    model = inspection.model(GatedKernel)
    structure = model.structure
    assert structure is not None
    deep = structure.find("compute.gated.deep")
    own = structure.header(deep).guard  # its ``when``, a node of the absent scope
    assert own is not None
    selector = structure.find("compute")
    open_world = bind(model, {}).world
    slow = open_world.successor(stored={selector: "slow"})
    # Absent from the case's guard: neither its node nor its own guard resolved.
    assert presence(slow, deep) == ("absent", INAPPLICABLE)
    assert "compute.gated" not in _in_play(structure)
    assert deep not in structure._raw and own not in structure._raw
    # What its own guard answers, read now: the same answer, the same object.
    assert slow.result(own) is INAPPLICABLE
    # Where the case's guard holds or waits, its own guard decides, as before.
    lanes = structure.find("compute.gated.lanes")
    narrow = open_world.successor(stored={selector: "gated", lanes: 1})
    assert presence(narrow, deep) == ("absent", Available(False))
    wide = open_world.successor(stored={selector: "gated", lanes: 2})
    assert presence(wide, deep) == ("present", Available(True))
    found, answer = presence(open_world, deep)
    assert found == "unknown" and not isinstance(answer, (Available, Inapplicable))
