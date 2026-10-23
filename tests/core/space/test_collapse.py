# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Forwarding chains: a formal bound to a reference, a formal forwarded through composites,
and a class-body alias are ``alias`` members, each read as itself, hop by hop, its output
followed. A design space is evaluated on its members as their scopes read them, never
collapsed; a guarded node's alias read from outside it answers as its guard does; and the
evidence names the formals a read went through.
"""

from __future__ import annotations

from typing import TypeVar

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


def test_a_guarded_nodes_alias_read_from_outside_it_answers_as_its_guard() -> None:
    point = design_space(Outer(width=3))
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
