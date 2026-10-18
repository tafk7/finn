# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two cores sharing a ``width``, the helpers that commit choices on a point and read its
forced ones, Spaces whose forced Decisions read each other (``Exclusive``, ``Mutual``,
``Gated``), and a chain whose changes reach far and move few forced answers (``chain``),
which the forcing tests, the successor's cache tests and the bounded change's walks share."""

from __future__ import annotations

from typing import Any

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    divisors_of,
    inspection,
    reject,
)


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
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({owned[key]: value for key, value in choices.items()})


def forced(point: Any) -> dict[str, object]:
    return {item.key: item.value for item in inspection.forced(point)}


class Free(Space):
    peer: bool = Param()


class Never(Space):
    peer: bool = Param()

    @constraint
    def never(self) -> bool | Rejected:
        return reject("never-viable", "this case is never viable")


class Unless(Space):
    """Refuses while its peer (a Boolean about another Decision) holds."""

    peer: bool = Param()

    @constraint
    def alone(self) -> bool | Rejected:
        if self.peer:
            return reject("peer-holds", "this case is not viable beside its peer")
        return True


class Exclusive(Space):
    """``x`` has one viable case while ``y`` is open, and ``y`` one while ``x`` is: the two
    together are refused. The plain rounds force the one they visit first and refuse the
    other."""

    @derived
    def x_is_a(self) -> bool:
        return isinstance(self.x, Unless)

    @derived
    def y_is_b(self) -> bool:
        return isinstance(self.y, Unless)

    x: Unless | Never = Decision({"a": Unless, "b": Never}, peer=y_is_b)
    y: Never | Unless = Decision({"a": Never, "b": Unless}, peer=x_is_a)
    other: int = Decision(values=(1, 2))


class Mutual(Space):
    """Given ``y`` at b, ``x`` has one case (a); given ``x`` at a, ``y`` has one (b);
    open, neither does."""

    @derived
    def x_is_a(self) -> bool:
        return isinstance(self.x, Free)

    @derived
    def y_is_b(self) -> bool:
        return isinstance(self.y, Free)

    x: Free | Unless = Decision({"a": Free, "b": Unless}, peer=y_is_b)
    y: Unless | Free = Decision({"a": Unless, "b": Free}, peer=x_is_a)


class Gated(Space):
    """``x`` is forced while ``y`` is open, and its verdict reads ``y`` open. A change to
    ``gate`` forces ``y`` too, which ``x``'s verdict then did not see. ``x``'s case reads
    ``y`` and then ``gate``, so while ``gate`` is open no case of ``y`` refuses it, and
    ``y`` stays open."""

    @derived
    def y_is_a(self) -> bool:
        return isinstance(self.y, Unless) and self.gate == 2

    @derived
    def gate_shut(self) -> bool:
        return self.gate == 2

    x: Unless | Never = Decision({"a": Unless, "b": Never}, peer=y_is_a)
    gate: int = Decision(values=(1, 2))
    y: Unless | Free = Decision({"a": Unless, "b": Free}, peer=gate_shut)


class Stage(Space):
    """A stage of a chain, one or two wide."""

    size: int = Decision(values=(1, 2))


class Hop(Space):
    """Between two stages: two wide stages side by side are refused, and its depth is the
    upstream stage's size. Its constraints read both stages' sizes, so a stage's verdict
    reads its neighbours' and theirs read theirs: a change to one stage reaches every open
    Decision of the chain, and moves the forced answers next to it only."""

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
