# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forwarding chains collapse at compilation; scopes and keys do not change.

A formal bound to a reference, a formal forwarded through composites, and a
class-body alias are ``alias`` nodes. Collapse points each alias at the end of
its chain, and points a reader straight at that source when the alias applies
whenever the reader does. The compiled forms are compared with and without collapse;
a design space is evaluated on its nodes as resolved, never collapsed, and its evidence
names the formals a read went through.
"""

from __future__ import annotations

from typing import TypeVar

from core.space._collapse_support import compiled, open_space
from finn.core.space import (
    Decision,
    Inapplicable,
    Param,
    Space,
    Unresolved,
    design_space,
    inspection,
    view,
)


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


S = TypeVar("S", bound=Space)


def _choices(point: S, **values: object) -> S:
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({handles[key]: value for key, value in values.items()})


def test_a_chain_collapses_to_its_source_but_keeps_every_scope_and_key() -> None:
    linked = compiled(Outer(width=3), collapsed=True).linked
    before = compiled(Outer(width=3), collapsed=False).linked
    # The two number their nodes apart: compared by key.
    assert sorted(node.key for node in linked.nodes) == sorted(node.key for node in before.nodes)
    assert sorted(scope.name for scope in linked.scopes) == sorted(
        scope.name for scope in before.scopes
    )
    chain = linked.keys["middle.leaf.width"]
    assert before.nodes[before.keys["middle.leaf.width"]].output == before.keys["middle.width"]
    assert linked.nodes[chain].output == linked.keys["width"]  # straight to the source
    assert linked.forward[chain] == linked.keys["width"]
    # The alias of a guarded node, read from outside it, keeps its hop.
    reader = linked.keys["outside.width"]
    assert linked.nodes[reader].output == linked.keys["guarded.leaf.width"]
    point = open_space(Outer(width=3))
    assert isinstance(point.outside.query(Leaf.width), Unresolved)  # while ``enabled`` is open
    off = _choices(point, enabled=False)
    assert isinstance(off.outside.query(Leaf.width), Inapplicable)


def test_explain_names_a_forwarding_formal_a_read_went_through() -> None:
    point = _choices(design_space(Outer(width=3)), **{"middle.leaf.lanes": 2})
    evidence = inspection.explain(point.middle.leaf, Leaf.bits)
    keys = {node.declaration.key for node in evidence.nodes}
    assert "width" in keys and "middle.leaf.width" not in keys
    via = {alias.key for node in evidence.nodes for alias in node.via}
    assert via == {"middle.leaf.width"}
