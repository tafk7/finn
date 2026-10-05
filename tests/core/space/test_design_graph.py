# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A graph of design spaces, stated only in terms of Space.

Calling a Space class declares a node; ``node.member`` is a reference to one of its
members. Edges are bindings at the call, or assignments to a formal after the
node is declared (``adder.back = register.q``: forward, around a cycle, or
built as data), and ``Present`` for whichever source is present. A
``Param(Located)`` formal receives a reference with its node and member names,
and ``Members(key)`` ranges over the present children that export ``key``. A
Decision over nodes is the structural choice. None of these cases is about
hardware.
"""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    Located,
    LocatedParam,
    Members,
    Param,
    Present,
    Rejected,
    Space,
    Unresolved,
    View,
    ViewKey,
    composite,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    selections,
    view,
)

COST = ViewKey("cost", int)
AGREED = ViewKey("agreed", int)
WIDTH = ViewKey("width", int)


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.code for finding in result.findings}


def owners(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.owner for finding in result.findings}


class Tiles(Space):
    extent: int = Param()
    factor: int = Decision(domain=divisors_of(extent))

    @view
    def cost(self) -> int:
        return self.extent // self.factor

    exports = {COST: cost}


# -- 1. a relation between siblings is an ordinary node ---------------------------------


class Same(Space):
    """A relation: two located values must be equal; its output is the agreed value."""

    left: LocatedParam[int] = LocatedParam()
    right: LocatedParam[int] = LocatedParam()

    @constraint
    def equal(self) -> bool | Rejected:
        left, right = self.left, self.right
        if left.value != right.value:
            return reject(
                "not-same",
                f"{left.node}.{left.member}={left.value}, "
                f"{right.node}.{right.member}={right.value}",
            )
        return True

    @view(requires=(equal,))
    def agreed(self) -> int:
        return self.left.value

    exports = {AGREED: agreed}


class Aligned(Space):
    extent: int = Param()
    first = Tiles(extent=extent)
    second = Tiles(extent=extent)
    # A plain reference supplies a located formal: node and member come from the graph.
    aligned = Same(left=first.factor, right=second.factor)
    relations = Members(AGREED)

    @view(requires=(relations,))
    def factor(self) -> int:
        return self.aligned.agreed


def choose(point: Aligned, first: int, second: int) -> Aligned:
    return point.with_choices({Aligned.first.factor: first, Aligned.second.factor: second})


def test_a_relation_node_reads_located_siblings_and_owns_its_refusal() -> None:
    point = design_space(Aligned(extent=12))
    assert isinstance(point.query(Aligned.factor), Unresolved)
    assert choose(point, 3, 3).factor == 3
    refused = choose(point, 3, 4).inspect(Aligned.factor)
    assert isinstance(refused.accepted_result, Rejected)
    # Identity comes from the graph, not from literals: node and member names.
    # A refusal reached by both the output and the obligation is reported once.
    assert [(f.owner, f.message) for f in refused.accepted_result.findings] == [
        ("aligned.equal", "first.factor=3, second.factor=4")
    ]
    assert set(refused.constraints.results) == {"aligned.agreed"}


# -- 2. quantification: a budget over whichever members are present -------------------


class Budgeted(Space):
    limit: int = Param()
    use_third: bool = Decision(values=(False, True))
    a = Tiles(extent=12)
    b = Tiles(extent=8)
    c = Tiles(extent=6, when=use_third)
    costs = Members(COST)

    @constraint
    def within(self) -> bool | Rejected:
        spent = {member.node: member.value for member in self.costs}
        if sum(spent.values()) > self.limit:
            return reject("over-budget", f"{spent} exceeds {self.limit}")
        return True

    @view(requires=(costs, within))
    def total(self) -> int:
        return sum(member.value for member in self.costs)


def test_members_range_over_present_nodes_with_per_member_obligations() -> None:
    point = design_space(Budgeted(limit=10)).with_choices(use_third=False)
    point = point.with_choices({Budgeted.a.factor: 4})
    assessment = point.inspect(Budgeted.total)
    # Only b is still open; the absent c is inapplicable and never refuses.
    results = assessment.constraints.results
    assert set(results) == {"a.cost", "b.cost", "c.cost", "within"}
    assert isinstance(results["b.cost"], Unresolved)
    assert isinstance(results["c.cost"], Inapplicable)
    point = point.with_choices({Budgeted.b.factor: 2})
    assert point.total == 3 + 4
    assert point.costs == (Located("a", "cost", 3), Located("b", "cost", 4))
    third = point.with_choices(use_third=True)
    third = third.with_choices({Budgeted.c.factor: 1})
    refused = third.query(Budgeted.total)
    assert codes(refused) == {"over-budget"} and owners(refused) == {"within"}


# -- 3. a graph built as data, with edges assigned after their nodes -------------------


class Stage(Space):
    width_in: int = Param()
    growth: int = Decision(values=(0, 1, 2))

    @view
    def width_out(self) -> int:
        return self.width_in + self.growth

    exports = {WIDTH: width_out}


def pipeline(count: int) -> type[Space]:
    """Nodes are plain Python values, joined by assignment; ``composite`` names them."""
    stages = [Stage(width_in=4), *(Stage() for _ in range(1, count))]
    for previous, current in zip(stages, stages[1:]):
        current.width_in = previous.width_out
    nodes = {f"s{index}": stage for index, stage in enumerate(stages)}
    return composite(f"Pipeline{count}", {**nodes, "widths": Members(WIDTH)})


def test_a_pipeline_is_built_as_data_and_its_edges_are_declarations() -> None:
    space_type = pipeline(5)
    point = design_space(space_type())
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices({handles[f"s{i}.growth"]: 1 for i in range(5)})
    members = cast("Members[int]", getattr(space_type, "widths"))
    widths = point.query(members)
    assert isinstance(widths, Available)
    assert [(item.node, item.value) for item in widths.value] == [
        ("s0", 5),
        ("s1", 6),
        ("s2", 7),
        ("s3", 8),
        ("s4", 9),
    ]
    evidence = inspection.explain(point, members)
    assert "s3.width_out" in {n.declaration.key for n in evidence.nodes}
    # s4.width_in forwards s3.width_out: evaluation reads the source directly,
    # and the evidence still names the formal it read through.
    assert "s4.width_in" in {alias.key for n in evidence.nodes for alias in n.via}


# -- 4. a formal supplied by whichever source is present --------------------------------


class Source(Space):
    width: int = Param()

    @view
    def out(self) -> int:
        return self.width


class Sink(Space):
    width: int = Param()
    limit: int = Param(required=False)


class Either(Space):
    mode: str = Decision(values=("a", "b"))

    @derived
    def is_a(self) -> bool:
        return self.mode == "a"

    @derived
    def is_b(self) -> bool:
        return self.mode == "b"

    a = Source(width=3, when=is_a)
    b = Source(width=5, when=is_b)
    sink = Sink(width=Present(a.out, b.out))


def test_present_supplies_a_formal_from_whichever_node_is_present() -> None:
    point = design_space(Either())
    assert isinstance(point.sink.query(Sink.width), Unresolved)  # not yet known which
    assert point.with_choices(mode="a").sink.width == 3
    assert point.with_choices(mode="b").sink.width == 5
    # An optional formal nobody supplies is unsupplied, not an error.
    assert codes(point.sink.query(Sink.limit)) == {"input-unsupplied"}


def test_two_present_sources_are_refused_where_the_value_is_read() -> None:
    class Both(Space):
        a = Source(width=3)
        b = Source(width=3)
        sink = Sink()
        sink.width = Present(a.out, b.out)  # alternative suppliers, assigned

    answer = design_space(Both()).sink.query(Sink.width)
    assert codes(answer) == {"multiple-suppliers"} and owners(answer) == {"sink.width"}


# -- 5. a cycle in the graph, anchored; and an unanchored value cycle ------------------


class Adder(Space):
    inp: int = Param()
    back: int = Param()

    @view
    def out(self) -> int:
        return max(self.inp, self.back) + 1


class Register(Space):
    """Holds a declared width: this anchors the value flow around a loop.

    ``q`` is a raw value. Were it a view obliging ``fits``, its acceptance would
    read ``d`` and so close the cycle: acceptance is part of the value flow.
    """

    width: int = Param()
    d: int = Param()

    @constraint
    def fits(self) -> bool:
        return self.d <= 2 * self.width

    @derived
    def q(self) -> int:
        return self.width


class Accumulator(Space):
    width: int = Param()
    total: int = Param()
    adder = Adder(inp=width)  # back is supplied by the loop edge below
    register = Register(width=total, d=adder.out)
    adder.back = register.q


class Echo(Space):
    d: int = Param()

    @view
    def q(self) -> int:
        return self.d


def test_a_cyclic_graph_evaluates_when_its_value_flow_is_anchored() -> None:
    point = design_space(Accumulator(width=4, total=12))
    assert point.adder.out == 13
    assert point.register.inspect(Register.fits).verdict is True
    wide = design_space(Accumulator(width=30, total=12))  # 31 does not fit twice 12
    assert isinstance(wide.register.inspect(Register.fits).result, Rejected)


def test_an_unanchored_value_cycle_fails_with_its_path() -> None:
    class Unanchored(Space):
        width: int = Param()
        adder = Adder(inp=width)
        echo = Echo(d=adder.out)
        adder.back = echo.q

    with pytest.raises(EvaluationError, match="cycle"):
        design_space(Unanchored(width=4)).adder.out


# -- 6. closure: a composite is a node like any other ----------------------------------


class Pair(Space):
    """Two stages in series; its input is its own formal, its output an export."""

    width_in: int = Param()
    first = Stage(width_in=width_in)
    second = Stage(width_in=first.width_out)
    width_out = View(second.width_out)
    exports = {WIDTH: width_out}


class Chain(Space):
    head = Stage(width_in=2)
    body = Pair()
    body.width_in = head.width_out  # an edge, assigned
    widths = Members(WIDTH)


def test_a_composite_node_has_the_surface_of_a_leaf_node() -> None:
    point = design_space(Chain())
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices({handle: 1 for handle in handles.values()})
    assert [(m.node, m.value) for m in point.widths] == [("head", 3), ("body", 5)]


# -- 7. reducibility: the old structural choice is simply a Decision over nodes ----------


class Fixed(Space):
    width: int = Param()
    physical = View(width)


class Tuned(Space):
    base: int = Param()
    extra: int = Decision(values=(1, 2))

    @view
    def physical(self) -> int:
        return self.base + self.extra


class WithChoice(Space):
    """The structural choice: a Decision over nodes, read through ``choice.physical``."""

    base: int = Param(required=False)
    tuned = Tuned(base=base)
    choice: Fixed | Tuned = Decision({"fixed": Fixed(width=8), "tuned": tuned})
    physical = View(choice.physical)


class WithPrimitives(Space):
    """The same choice spelled with the primitives it lowers onto."""

    base: int = Param(required=False)
    choice: str = Decision(values=("fixed", "tuned"))

    @derived
    def is_fixed(self) -> bool:
        return self.choice == "fixed"

    @derived
    def is_tuned(self) -> bool:
        return self.choice == "tuned"

    fixed = Fixed(width=8, when=is_fixed)
    tuned = Tuned(base=base, when=is_tuned)
    physical = View(Present(fixed.physical, tuned.physical))


@pytest.mark.parametrize("space_type", (WithChoice, WithPrimitives))
def test_a_structural_choice_reduces_to_primitives(
    space_type: type[WithChoice] | type[WithPrimitives],
) -> None:
    start = design_space(space_type(base=4))
    physical = space_type.physical
    extra = space_type.tuned.extra
    assert isinstance(start.query(physical), Unresolved)
    tuned = start.with_choices({space_type.choice: "tuned", extra: 2})
    assert tuned.query(physical) == Available(6)
    # A candidate-local choice of an inactive candidate is refused, not stored.
    assert not tuned.try_with_choices({space_type.choice: "fixed"}).accepted
    fixed = tuned.with_choices({space_type.choice: "fixed"}, tuned.tuned.field(Tuned.extra).clear())
    assert fixed.query(physical) == Available(8)
    assert isinstance(fixed.query(extra), Inapplicable)
    replayed = selections.restore(design_space(space_type(base=4)), selections.capture(tuned))
    assert replayed.accepted and replayed.instance.query(physical) == Available(6)


def test_candidates_are_ordinary_nodes_with_declaration_path_keys() -> None:
    keys = {item.key for item in inspection.decisions(WithChoice)}
    assert keys == {"choice", "choice.tuned.extra"}
    start = design_space(WithChoice(base=4))
    point = start.with_choices(choice="tuned").with_choices({WithChoice.tuned.extra: 1})
    assert point.physical == 5
    selected = point.choice
    assert isinstance(selected, Tuned) and selected.extra == 1
    fixed = inspection.candidate(point, WithChoice.choice, "fixed")
    assert isinstance(fixed, Fixed) and isinstance(fixed.query(Fixed.physical), Inapplicable)


def test_a_formal_of_the_enclosing_space_locates_at_the_space_itself() -> None:
    class Own(Space):
        width: int = Param()
        here = Same(left=width, right=width)

    assert design_space(Own(width=3)).here.left == Located(None, "width", 3)


def test_assigning_a_formal_twice_in_one_body_is_a_definition_error() -> None:
    with pytest.raises(DefinitionError, match="already assigned") as caught:

        class Twice(Space):
            a = Source(width=3)
            sink = Sink(width=1)
            sink.width = a.out

    # Both sites are named: the assignment, and the call that supplied the formal.
    message = str(caught.value)
    assert "assigned at test_design_graph.py:" in message
    assert "already assigned at test_design_graph.py:" in message


def test_a_required_formal_left_unsupplied_is_reported_when_prepared() -> None:
    sink = Sink()  # legal: an assignment may still supply the width
    assert inspection.declaration(sink).unsupplied == ("width",)

    class Dangling(Space):
        sink = Sink()

    with pytest.raises(DefinitionError, match=r"sink\.width is not supplied") as caught:
        design_space(Dangling())
    message = str(caught.value)
    # The formal's declaration line and the node's call line.
    assert "Sink.width (declared at test_design_graph.py:" in message
    assert "for the node sink (declared at test_design_graph.py:" in message
    with pytest.raises(DefinitionError, match=r"width is not supplied"):
        design_space(Sink())
