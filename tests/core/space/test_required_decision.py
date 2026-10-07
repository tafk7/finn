# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``Decision(required=True)``: a choice with no safe baseline, which inspection reports
and nothing in the Space's evaluation reads."""

from __future__ import annotations

import pytest

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Space,
    design_space,
    domain,
    inspection,
)


class Buffer(Space):
    depth: int = Param()
    style: str = Decision(values=("auto", "block"))


class Shared(Space):
    style: str = Decision(values=("auto", "block"), required=True)


class Port(Space):
    width: int = Param()


class Core(Space):
    lanes: int = Decision(values=(1, 2, 4), ordered=True, required=True)
    stages: int = Decision(values=(0, 1), ordered=True)
    reach: int = Decision(
        domain=domain(accepts=lambda *, candidate: candidate >= 2, ordered=True), required=True
    )
    # Inline, supplying a formal: keyed by the formal's path.
    buffer = Buffer(depth=Decision(values=(2, 4), required=True))
    end: Port = Decision({"narrow": Port(width=8), "wide": Port(width=64)}, required=True)
    other: Port = Decision({"narrow": Port(width=8), "wide": Port(width=64)})


def _required() -> dict[str, bool]:
    return {item.key: item.required for item in inspection.decisions(Core)}


def test_inspection_reports_which_decisions_are_required() -> None:
    assert _required() == {
        "buffer.depth": True,
        "buffer.style": False,
        "end": True,
        "lanes": True,
        "other": False,
        "reach": True,
        "stages": False,
    }


def test_a_required_decision_evaluates_as_any_other() -> None:
    point = design_space(Core())
    viable = {item.key: item.cases for item in inspection.viable(point)}
    assert viable["lanes"] == (1, 2, 4) and viable["buffer.depth"] == (2, 4)
    assert [item.key for item in inspection.open(point)] == ["reach"]
    committed = point.with_choices({Core.lanes: 2, Core.reach: 8})
    assert "lanes" not in {item.key for item in inspection.viable(committed)}
    assert inspection.open(committed) == ()


def test_an_overriding_decision_states_its_own_requirement() -> None:
    class Narrowed(Space):
        # The enclosing body replaces the required Decision with one of its own,
        # which states a baseline: it is not required.
        shared = Shared(style=Decision(values=("block",)))

    class Kept(Space):
        shared = Shared()

    def required(space: type[Space]) -> bool:
        return {item.key: item.required for item in inspection.decisions(space)}["shared.style"]

    assert required(Kept) and not required(Narrowed)


def test_required_is_a_bool() -> None:
    with pytest.raises(DefinitionError, match="required= is True or False"):
        Decision(values=(1, 2), required=1)  # type: ignore[call-overload]
