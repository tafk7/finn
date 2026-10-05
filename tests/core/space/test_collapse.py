# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forwarding chains collapse at compilation; scopes, keys, values and diagnostics do not change.

A formal bound to a reference, a formal forwarded through composites, and a
class-body alias are ``alias`` nodes. Collapse points each alias at the end of
its chain, and points a reader straight at that source when the alias applies
whenever the reader does. Each test opens the same declaration with and
without collapse and compares the answer of every node.
"""

from __future__ import annotations

from typing import TypeVar, cast

from core.space._collapse_support import answers, counts, open_space
from core.space.test_house import House
from core.space.test_references import Budget, Company, Department

from finn.core.space import (
    Decision,
    Inapplicable,
    Param,
    Space,
    composite,
    design_space,
    inspection,
    view,
)
from finn.core.space.occurrence import state


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))

    @view
    def bits(self) -> int:
        return self.width * self.lanes


class Middle(Space):
    width: int = Param()
    leaf = Leaf(width=width)  # forwards the enclosing formal


class Outer(Space):
    width: int = Param()
    enabled: bool = Decision(values=(False, True))
    middle = Middle(width=width)  # a chain: middle.leaf.width -> middle.width -> width
    guarded = Middle(width=width, when=enabled)
    # Read from outside the guarded node: its alias must still answer "inapplicable".
    outside = Leaf(width=guarded.leaf.width)


class Division(Space):
    budget: Budget = Param()
    team = Department(budget=budget)  # forwards its reference input


class Holding(Space):
    shared = Budget(limit=100, rate=10)
    division = Division(budget=shared)


S = TypeVar("S", bound=Space)


def _choices(point: S, **values: object) -> S:
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({handles[key]: value for key, value in values.items()})


def test_every_node_answers_the_same_with_and_without_collapse() -> None:
    cases: list[tuple[Space, dict[str, object]]] = [  # declarations and their choices
        (Outer(width=3), {}),
        (Outer(width=3), {"enabled": False}),
        (Outer(width=3), {"enabled": True, "middle.leaf.lanes": 2}),
        (House(budget=80), {}),
        (
            House(budget=80),
            {
                "want_garage": True,
                "heating": "boiler",
                "hall.finish": 1,
                "kitchen.finish": 1,
                "dining.finish": 3,
                "garage.finish": 1,
            },
        ),
        (Company(limit=40), {}),
        (Company(limit=40), {"sales.staff": 2, "research.staff": 3, "open_lab": True}),
        (Holding(), {"division.team.staff": 2}),
    ]
    for declaration, choices in cases:
        collapsed = open_space(declaration, collapsed=True)
        plain = open_space(declaration, collapsed=False)
        if choices:
            collapsed = _choices(collapsed, **choices)
            plain = _choices(plain, **choices)
        assert answers(collapsed) == answers(plain), type(declaration).__name__


def test_a_chain_collapses_to_its_source_but_keeps_every_scope_and_key() -> None:
    collapsed = open_space(Outer(width=3), collapsed=True)
    plain = open_space(Outer(width=3), collapsed=False)
    linked, before = state(collapsed).linked, state(plain).linked
    assert [node.key for node in linked.nodes] == [node.key for node in before.nodes]
    assert [scope.name for scope in linked.scopes] == [scope.name for scope in before.scopes]
    chain = linked.keys["middle.leaf.width"]
    assert before.nodes[chain].output == before.keys["middle.width"]
    assert linked.nodes[chain].output == linked.keys["width"]  # straight to the source
    assert linked.forward[chain] == linked.keys["width"]
    # The alias of a guarded node, read from outside it, keeps its hop.
    reader = linked.keys["outside.width"]
    assert linked.nodes[reader].output == linked.keys["guarded.leaf.width"]
    assert isinstance(collapsed.outside.query(Leaf.width), type(plain.outside.query(Leaf.width)))
    off = _choices(collapsed, enabled=False)
    assert isinstance(off.outside.query(Leaf.width), Inapplicable)


def test_evaluation_skips_forwarding_nodes_and_explain_still_names_them() -> None:
    def read(point: Space) -> object:
        return cast(Outer, point).middle.leaf.query(Leaf.bits)

    lanes = {"middle.leaf.lanes": 2}
    before = counts(_choices(open_space(Outer(width=3), collapsed=False), **lanes), read)
    after = counts(_choices(open_space(Outer(width=3), collapsed=True), **lanes), read)
    assert after.nodes == before.nodes and after.aliases == before.aliases
    assert after.aliases_evaluated < before.aliases_evaluated
    assert after.evaluated < before.evaluated
    point = _choices(design_space(Outer(width=3)), **{"middle.leaf.lanes": 2})
    evidence = inspection.explain(point.middle.leaf, Leaf.bits)
    keys = {node.declaration.key for node in evidence.nodes}
    assert "width" in keys and "middle.leaf.width" not in keys
    via = {alias.key for node in evidence.nodes for alias in node.via}
    assert via == {"middle.leaf.width"}


def test_a_pipeline_built_as_data_answers_the_same() -> None:
    class Stage(Space):
        width_in: int = Param()
        growth: int = Decision(values=(0, 1))

        @view
        def width_out(self) -> int:
            return self.width_in + self.growth

    stages = [Stage(width_in=4), *(Stage() for _ in range(9))]
    for previous, current in zip(stages, stages[1:]):
        current.width_in = previous.width_out
    space_type = composite("Pipeline", {f"s{index}": stage for index, stage in enumerate(stages)})
    choices = {f"s{index}.growth": 1 for index in range(10)}
    collapsed = _choices(open_space(space_type(), collapsed=True), **choices)
    plain = _choices(open_space(space_type(), collapsed=False), **choices)
    assert answers(collapsed) == answers(plain)
