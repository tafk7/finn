# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A Space class's compiled form (``finn.core.space._template.Compiled``): computed once, with
its template, cached on the class; its members numbered within the class, every reference an
edge.

An edge names a member of the class (``Local``), a member of a class it places through its
slots (``Placed``), its own formal (``Formal``), the guard it is placed under (``Placing``),
or a selection of a choice's member in the choice's own scope (``Selected``). No record
stands for a member of a placed class (an import), for the guard the class is placed under,
or for a scope it places: what the class writes for a placed Space (a formal's binding by
reference, a selection it makes in that Space's choice) is a record of its own, addressed by
slot and formal or by the selection.
"""

from __future__ import annotations

from typing import cast

from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    derived,
    supplied,
)
from finn.core.space._table import Binding
from finn.core.space._template import Compiled, compiled, link_template
from finn.core.space.ir import PLACING, Formal, Local, Placed, Selected


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))

    @derived
    def bits(*, width: int, lanes: int) -> int:
        return width * lanes


class Middle(Space):
    depth: int = Param()
    leaf = Leaf(width=depth)


class Wide(Space):
    width: int = Param()
    extra: int = Param()


class Narrow(Space):
    width: int = Param()


class Top(Space):
    w: int = Param()
    enabled: bool = Decision(values=(False, True))
    middle = Middle(depth=w)  # bound by reference
    fixed = Leaf(width=4)  # bound to a literal
    guarded = Leaf(width=w, when=enabled)
    mode: Wide | Narrow = Select(wide=Wide(width=w, extra=1), narrow=Narrow(width=2))
    optional: Wide | None = Select(wide=Wide(width=w, extra=2), off=None)
    deep = middle.leaf.bits  # through two slots
    chosen = mode.width  # a selection of the own choice's member


class Outer(Space):
    top = Top(w=3)
    shared = top.mode.width  # a selection the candidates share, made by the placed class
    single = cast(Wide, top.optional).extra  # one candidate has it: this class's selection


def _slot(form: Compiled, name: str) -> int:
    return form.slot_names[name]


def _member(form: Compiled, key: str) -> int:
    return form.keys[key]


def _choice(form: Compiled, key: str) -> int:
    (found,) = (choice.index for choice in form.choices if choice.key == key)
    return found


def test_compiled_once_per_class_with_its_template_and_cached_on_the_class() -> None:
    template = link_template(Top)
    form = compiled(Top)
    assert compiled(Top) is form
    assert Top.__dict__["_space_compiled"] is form
    assert form.space_type is Top
    # Each placed class's compiled form is its own, shared by every slot placing it.
    leaves = [slot.compiled for slot in form.slots if slot.space_type is Leaf]
    assert len(leaves) == 2 and all(leaf is compiled(Leaf) for leaf in leaves)
    # Its records are its template's nodes but its imports and the guard it is placed under.
    kept = [node.index for node in template.nodes if node.kind != "import"]
    kept.remove(template.placement_guard)
    assert form.nodes == tuple(kept)
    assert "import" not in {record.kind for record in form.records}


def test_slots_in_placement_order_with_their_bindings_and_guards() -> None:
    form = compiled(Top)
    assert [slot.name for slot in form.slots] == [
        "middle",
        "fixed",
        "guarded",
        "mode.wide",
        "mode.narrow",
        "optional.wide",
    ]
    middle, fixed, guarded = (
        form.slots[_slot(form, name)] for name in ("middle", "fixed", "guarded")
    )
    # A reference: the record this class writes for it, reading this class's own formal.
    binding = middle.bindings["depth"]
    assert binding.kind == "reference"
    assert binding.node == form.bound[(middle.index, "depth")]
    written = form.records[binding.node]
    assert (written.key, written.kind, written.output) == ("middle.depth", "alias", Formal("w"))
    assert form.written[binding.node] == middle.index
    assert fixed.bindings["width"] == Binding("literal", 4)
    # Unguarded, a child is placed under the guard the class is placed under; guarded, under
    # its own guard, a member of the class.
    assert middle.guard == PLACING
    assert guarded.guard == Local(_member(form, "guarded.$guard"))
    mode = form.choices[_choice(form, "mode")]
    assert mode.cases == (
        ("wide", _slot(form, "mode.wide")),
        ("narrow", _slot(form, "mode.narrow")),
    )
    assert form.slots[_slot(form, "mode.wide")].candidate == (mode.index, "wide")


def test_references_into_placed_classes_are_slot_edges() -> None:
    form = compiled(Top)
    leaf = compiled(Leaf)
    middle = _slot(form, "middle")
    deep = form.records[_member(form, "deep")]
    assert deep.output == Placed((middle, compiled(Middle).slot_names["leaf"]), leaf.keys["bits"])
    # Each placed Space's validity is a piece of the class's own.
    validity = form.records[form.validity]
    placed = [piece for piece in validity.constraints if isinstance(piece, Placed)]
    assert placed == [Placed((slot.index,), slot.compiled.validity) for slot in form.slots]
    # A member of the class reads its own formal as a formal, and is placed under the guard
    # the class is placed under.
    enabled = form.records[_member(form, "enabled")]
    assert enabled.guard == PLACING
    assert Formal("w") in validity.constraints


def test_a_selection_is_addressed_in_its_choices_scope() -> None:
    top = compiled(Top)
    mode = _choice(top, "mode")
    assert top.records[_member(top, "chosen")].output == Selected((), mode, "width")
    shared = top.own_selection[(mode, "width")]
    assert top.ref(Selected((), mode, "width")) == shared

    outer = compiled(Outer)
    slot = _slot(outer, "top")
    # Shared by the candidates: the placed class's selection, read there.
    assert outer.records[_member(outer, "shared")].output == Selected((slot,), mode, "width")
    assert outer.ref(Selected((slot,), mode, "width")) == Placed((slot,), shared)
    # Made by this class only: the record it writes, addressed by the selection.
    optional = _choice(top, "optional")
    edge = Selected((slot,), optional, "extra")
    assert outer.records[_member(outer, "single")].output == edge
    record = outer.chosen[((slot,), optional, "extra")]
    assert outer.ref(edge) == record
    assert outer.records[record].key == "top.optional.$member.extra"
    assert outer.written[record] == slot


class Source(Space):
    contents: int = Param()


class Edge(Space):
    contents: int = Param(required=False)
    valued = supplied(contents)
    source: Source = Select(memory=Source, when=valued, contents=contents)


class Graph(Space):
    stored = Edge(contents=4)
    streamed = Edge()


def test_an_unsupplied_formal_and_the_choice_it_decides_are_carried() -> None:
    """``supplied()``'s rule as the template carries it: a choice guarded by the supply of a
    formal names it, and a slot leaving it unsupplied binds it so."""
    edge = compiled(Edge)
    (source,) = edge.choices
    assert source.never_formal == "contents"
    graph = compiled(Graph)
    assert graph.slots[_slot(graph, "streamed")].bindings["contents"] == Binding("unsupplied")
    assert graph.slots[_slot(graph, "stored")].bindings["contents"] == Binding("literal", 4)
    # What exists only where the choice applies, by record.
    candidate = edge.slots[source.cases[0][1] or 0]
    assert candidate.requires == source.index
