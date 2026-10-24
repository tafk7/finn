# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two cores sharing a ``width``, the helper that commits choices on a point by key, the
one that binds a design space's facts with exactly the stored choices given (``built``), and a chain
whose changes reach far (``chain``), which the possible values' tests, the successor's
tests and the bounded change's walks share."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    constraint,
    divisors_of,
    inspection,
    occurrence,
    reject,
)
from finn.core.space._running import questions
from finn.core.space._running.evaluate import Session
from finn.core.space._running.surface import At


class Packed(Space):
    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed core takes at most 64 lanes")
        return True


class Stub(Space):
    width: int = Param()
    rows: int = Decision(values=(1, 2, 4))

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub core takes an even width")
        return True


CORES: dict[str, type[Space] | Space] = {"packed": Packed, "stub": Stub}


def commit(point: Any, choices: dict[str, object]) -> Any:
    owned = {item.key: item.reference for item in inspection.decisions(inspection.model(point))}
    return point.with_choices({owned[key]: value for key, value in choices.items()})


def built(point: Any, choices: Mapping[str, object]) -> Any:
    """The design space of ``point``'s facts with exactly these stored choices (by key),
    read at ``point``'s scope: bound as a saved graph is (``questions.bind``, in a session
    of its own with no cache: a recomputation sharing nothing with the model's session),
    not reached by a change, so no rule is checked: one that breaks a rule reads refused (a
    change to it would be refused). Each value must be of its choice's type, dormant ones
    included."""
    at = occurrence.held(point)
    saved = [(at.ds.node(key), value) for key, value in choices.items()]
    with at.ds.model.lock:
        report = questions.bind(at.ds.layer.bare, saved, Session())
    for outcome in report.outcomes:
        if outcome.status == "dropped":
            raise ValueError(f"{outcome.owner} is not of its type: {outcome.result}")
    assert report.published is not None
    return occurrence.place(At(report.published, at.scope))


class Stage(Space):
    """A stage of a chain, one or two wide."""

    size: int = Decision(values=(1, 2))


class Hop(Space):
    """Between two stages: two wide stages side by side are refused, and its depth is the
    upstream stage's size. Its constraints read both stages' sizes, so a change to one
    stage reaches the hops next to it only."""

    up: int = Param()
    down: int = Param()
    depth: int = Decision(values=(1, 2))

    @constraint
    def narrow(self) -> bool | Rejected:
        if self.up == 2 and self.down == 2:
            return reject("hop-wide", "two wide stages side by side")
        return True

    @constraint
    def deep(self) -> bool | Rejected:
        if self.depth != self.up:
            return reject("hop-depth", "the depth is the upstream stage's size")
        return True


def chain(length: int, *, stages_first: bool = True) -> type[Space]:
    """A chain of ``length`` stages (``s0``, ``s1``, ...) and a hop between each two
    (``h0`` between ``s0`` and ``s1``, ...), the stages declared first or the hops."""
    stages = {f"s{i}": Stage() for i in range(length)}
    hops = {
        f"h{i}": Hop(up=stages[f"s{i}"].size, down=stages[f"s{i + 1}"].size)
        for i in range(length - 1)
    }
    members = {**stages, **hops} if stages_first else {**hops, **stages}
    return type("Chain", (Space,), {"__module__": __name__, **members})
