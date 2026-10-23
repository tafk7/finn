# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a read through an alias answers, pinned by its public answers.

Three reads that an alias stands in: a Decision reached through a class-body alias of a
child's Decision (an edit names the alias, and is stored under the Decision); a view's
output copied by value semantics where an alias on the way states other semantics than
its source (a formal bound by reference, and a class-body alias of such a formal); and
``explain`` through a class-body alias. Each answer is the engine's as it stands, so a
change in how aliases are followed shows here first.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from finn.core.space import (
    Available,
    BoundDecision,
    Decision,
    Param,
    Space,
    ValueSemantics,
    View,
    ViewAssessment,
    bind,
    derived,
    design_space,
    inspection,
)

# A list held as it is: the same type token as a plain list's, but no copy is taken.
KEPT: ValueSemantics[list[int]] = ValueSemantics(
    type_token=list,
    name="list",
    recognizes=lambda value: type(value) is list,
    equal=lambda left, right: left == right,
    snapshot=lambda value: value,
)
# A list copied on every read (a plain list's semantics, stated).
COPIED: ValueSemantics[list[int]] = ValueSemantics(
    type_token=list,
    name="list",
    recognizes=lambda value: type(value) is list,
    equal=lambda left, right: left == right,
    snapshot=deepcopy,
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


# -- a view's output copied by value semantics ------------------------------------------------


class Kept(Space):
    """A formal that keeps its value as it is, read by a view of its own."""

    width: list[int] = Param(semantics=KEPT)
    physical = View(width)


class KeptByReference(Space):
    supplier: list[int] = Param(semantics=COPIED)
    child = Kept(width=supplier)  # bound by reference: the formal forwards its supplier


class KeptThroughAlias(Space):
    supplier: list[int] = Param(semantics=COPIED)
    leaf = Kept(width=supplier)
    e = leaf.width  # a class-body alias: it takes the child's formal's semantics
    physical = View(e)


class KeptGuarded(Space):
    enabled: bool = Param()
    supplier: list[int] = Param(semantics=COPIED)
    leaf = Kept(width=supplier, when=enabled)
    physical = View(leaf.width)  # the formal's guard is not implied by the view's


def _value(result: object) -> object:
    assert isinstance(result, Available)
    return result.value


def _output(assessment: object) -> object:
    assert isinstance(assessment, ViewAssessment)
    return _value(assessment.output_result)


def _kept(point: Space, view: Any) -> tuple[bool, bool]:
    """Whether ``view``'s output, inspected and explained, is the same object at each read:
    copied by the kept semantics (the object as it is held), or by the copying ones (a copy
    each read)."""
    output = _output(point.inspect(view))
    again = _output(point.inspect(view))
    explained = _output(inspection.explain(point, view).assessment)
    return output is again, explained is again


def test_a_view_through_a_formal_bound_by_reference_copies_by_the_suppliers_semantics() -> None:
    """The child's view reads its formal, which states kept semantics, bound by reference
    to a supplier whose semantics copy: the output is copied (a new list each read)."""
    point = design_space(KeptByReference(supplier=[1, 2]))
    child: Any = point.child
    assert child.physical == [1, 2]
    assert _kept(child, Kept.physical) == (False, False)
    # Read as the formal itself, by its own (kept) semantics: the object as it is held.
    assert _value(child.query(Kept.width)) is _value(child.query(Kept.width))


def test_a_view_through_a_class_body_alias_copies_by_the_suppliers_semantics() -> None:
    """A class-body alias of the child's formal takes that formal's (kept) semantics, and a
    view of the alias copies its output by the supplier's (a new list each read)."""
    point = design_space(KeptThroughAlias(supplier=[1, 2]))
    assert point.physical == [1, 2]
    assert _kept(point, KeptThroughAlias.physical) == (False, False)
    # The alias read itself, by its own (kept) semantics: the object as it is held.
    held = point.query(KeptThroughAlias.e)
    assert _value(held) is _value(point.query(KeptThroughAlias.e))


def test_a_view_through_a_guarded_formal_copies_by_the_formals_semantics() -> None:
    """Where the formal's guard does not hold whenever the view's does, the view's output is
    copied by the formal's own (kept) semantics: the object as it is held."""
    point = design_space(KeptGuarded(enabled=True, supplier=[1, 2]))
    assert point.physical == [1, 2]
    assert _kept(point, KeptGuarded.physical) == (True, True)


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
