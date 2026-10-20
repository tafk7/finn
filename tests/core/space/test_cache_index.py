# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A world made from another takes exactly the entries that read nothing it reads
otherwise, whatever was changed: choices or facts.

Over a Space of sums of a few choices and facts each, and random changes to them: every
entry of the parent that read none of the changed choices and facts is the child's own
(the same entry object); every other is found again in the child (an answer kept by
early cutoff is a new entry). The parent is left as it was, and each world gains entries
the other never asked for.
"""

from __future__ import annotations

from random import Random

import pytest

from core.space._world_support import open_world, stored
from finn.core.space import Decision, Param, Space, derived
from finn.core.space._world import World


class Sums(Space):
    """Six choices, two facts, and sums over them, each reading a few."""

    p: int = Param()
    q: int = Param()
    a: int = Decision(values=(1, 2))
    b: int = Decision(values=(1, 2))
    c: int = Decision(values=(1, 2))
    d: int = Decision(values=(1, 2))
    e: int = Decision(values=(1, 2))
    f: int = Decision(values=(1, 2))

    @derived
    def ab(self) -> int:
        return self.a + self.b

    @derived
    def cd(self) -> int:
        return self.c + self.d

    @derived
    def ef(self) -> int:
        return self.e + self.f

    @derived
    def abcd(self) -> int:
        return self.ab + self.cd

    @derived
    def pa(self) -> int:
        return self.p + self.a

    @derived
    def qf(self) -> int:
        return self.q * self.f

    @derived
    def total(self) -> int:
        return self.abcd + self.ef + self.pa + self.qf


CHOICES = ("a", "b", "c", "d", "e", "f")


def _warm(world: World) -> None:
    for node in world.evaluator.linked.nodes:
        world.entry(node.index)


@pytest.mark.parametrize("seed", range(20))
def test_a_child_takes_exactly_what_read_nothing_it_changed(seed: int) -> None:
    rng = Random(seed)
    chosen = {name: rng.choice((1, 2)) for name in CHOICES}
    base = open_world(Sums(p=1, q=2), chosen)
    _warm(base)
    held = dict(base.cache)
    changed = rng.sample(CHOICES, rng.randint(1, 3))
    values = dict(base.stored)
    values.update(stored(base.evaluator, {name: 3 - chosen[name] for name in changed}))
    facts = dict(base.facts)
    moved = rng.random() < 0.5
    if moved:
        facts[base.evaluator.linked.keys["p"]] = 5
    child = base.successor(facts=facts, stored=values)
    reached = child.delta
    nodes = base.evaluator.linked.nodes
    for node in nodes:
        entry = child.entry(node.index)
        if not held[node.index].reads & reached:
            assert entry is held[node.index], node.key
        else:
            assert entry is not held[node.index], node.key
    # The parent is as it was, and a world gains only what it is asked for.
    assert base.cache == held
    fresh = base.successor(facts=facts, stored=values)
    assert not fresh.cache
    total = base.evaluator.linked.keys["total"]
    assert fresh.result(total) == child.result(total)
