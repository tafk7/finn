# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a read through an alias answers, pinned by its public answers.

Two reads that an alias stands in: a Decision reached through a class-body alias of a
child's Decision (an edit names the alias, and is stored under the Decision); and
``explain`` through a class-body alias. Each answer is the engine's as it stands, so a
change in how aliases are followed shows here first.
"""

from __future__ import annotations

from finn.core.space import (
    Available,
    BoundDecision,
    Decision,
    Param,
    Space,
    bind,
    derived,
    design_space,
    inspection,
)

# -- a Decision reached through a class-body alias ---------------------------------------------


class Chosen(Space):
    d: int = Decision(values=(1, 2))


class ChosenThroughAlias(Space):
    leaf = Chosen()
    e = leaf.d  # a class-body alias of a child's Decision


def test_a_decision_reached_through_a_class_body_alias_is_the_decision() -> None:
    """An edit naming the alias edits the Decision it stands for: stored under the
    Decision's key, one choice; its handle is the Decision's. A read naming the child's
    member by path finds the alias first (the scope's members by declaration)."""
    point = design_space(ChosenThroughAlias())
    by_alias = point.with_choices({ChosenThroughAlias.e: 2})
    by_path = point.with_choices({ChosenThroughAlias.leaf.d: 2})
    for chosen in (by_alias, by_path):
        assert chosen.leaf.d == 2
        assert [key for key, _ in inspection.states(chosen)] == ["leaf.d"]
        assert isinstance(inspection.states(chosen)[0][1], inspection.Set)
    bound = bind(point, {ChosenThroughAlias.e: 2})
    assert bound.accepted
    assert [key for key, _ in inspection.states(bound.instance)] == ["leaf.d"]
    assert bound.instance.leaf.d == 2

    assert isinstance(point.field(ChosenThroughAlias.e), BoundDecision)
    handle = inspection.decision_handle(point, ChosenThroughAlias.e)
    assert repr(handle) == "DecisionHandle('leaf.d')"
    assert handle == inspection.decision_handle(point, ChosenThroughAlias.leaf.d)

    read = inspection.value_handle(point, ChosenThroughAlias.leaf.d)
    assert repr(read) == "ValueHandle('e')"
    assert read == inspection.value_handle(point, ChosenThroughAlias.e)
    assert by_alias.query(ChosenThroughAlias.e) == Available(2)
    assert by_alias.query(ChosenThroughAlias.e) == by_alias.query(ChosenThroughAlias.leaf.d)


# -- explain through a class-body alias -------------------------------------------------------


class Bits(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))

    @derived
    def bits(*, width: int, lanes: int) -> int:
        return width * lanes


class ExplainedThroughAlias(Space):
    w: int = Param()
    leaf = Bits(width=w)
    b = leaf.bits  # a class-body alias of the child's derived value

    @derived
    def total(*, b: int) -> int:
        return b + 1


def test_explain_through_a_class_body_alias_reads_the_alias_as_a_node_of_its_own() -> None:
    """The alias is a node of the evidence, read by its reader and reading its source; only
    a formal bound by reference, read within its scope, is named as read through (``via``)."""
    point = design_space(ExplainedThroughAlias(w=3)).with_choices(
        {ExplainedThroughAlias.leaf.lanes: 2}
    )
    evidence = inspection.explain(point, ExplainedThroughAlias.total)
    assert evidence.result == Available(7)
    nodes = {item.declaration.key: item for item in evidence.nodes}
    # The formal read at its source is no node of its own: its source is.
    assert sorted(nodes) == ["b", "leaf.bits", "leaf.lanes", "total", "w"]
    dependencies = {
        key: [repr(handle) for handle in item.dependencies] for key, item in nodes.items()
    }
    assert dependencies["total"] == ["ValueHandle('b')"]
    assert dependencies["b"] == ["ValueHandle('leaf.bits')"]
    assert dependencies["leaf.bits"] == ["ValueHandle('w')", "ValueHandle('leaf.lanes')"]
    via = {key: [each.key for each in item.via] for key, item in nodes.items()}
    assert via == {"b": [], "leaf.bits": ["leaf.width"], "leaf.lanes": [], "total": [], "w": []}
    assert nodes["b"].declaration.kind == "alias"
    assert nodes["b"].result == Available(6)
