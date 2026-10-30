# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A graph of design spaces, stated only in terms of Space.

Calling a Space class declares a node; ``node.member`` is a reference to one of its
members. Edges are bindings at the call, or assignments to a formal after the
node is declared (``adder.back = register.q``: forward, around a cycle, or
built as data). A relation between siblings is their parent's constraint, and
``Members(Base.member)`` ranges over the present children deriving from ``Base``, each
read at its member. A
Decision over nodes is the structural choice. None of these cases is about
hardware.
"""

from __future__ import annotations

from typing import cast

import pytest
from support.refusals import refused
from support.space_cores import built

from core.space._choices_support import committed, replay
from finn.core.space import (
    Available,
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    Located,
    Members,
    Param,
    Rejected,
    Select,
    Space,
    Unresolved,
    composite,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    reject,
    view,
)
from finn.core.space._nodes import NodeDecl, node_record, unsupplied_formals


def owners(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved))
    return {finding.owner for finding in result.findings}


class Costly(Space):
    """What a budget counts: a part states its ``cost``, whose own refuses."""

    @derived
    def cost(self) -> int | Rejected:
        return reject("no-cost", f"{type(self).__qualname__} states no cost")


class Tiles(Costly):
    extent: int = Param()
    factor: int = Decision(domain=divisors_of(extent))

    @derived
    def cost(self) -> int:
        return self.extent // self.factor


# -- 1. a relation between siblings is their parent's -----------------------------------


class Aligned(Space):
    extent: int = Param()
    first = Tiles(extent=extent)
    second = Tiles(extent=extent)

    @constraint
    def aligned(self) -> bool | Rejected:
        """A relation of two children, the parent's: their factors agree."""
        first, second = self.first.factor, self.second.factor
        if first != second:
            return reject("not-same", f"first.factor={first}, second.factor={second}")
        return True

    @view
    def factor(self) -> int:
        return self.first.factor


def choose(point: Aligned, first: int, second: int) -> Aligned:
    return point.with_choices({Aligned.first.factor: first, Aligned.second.factor: second})


def test_a_relation_of_siblings_is_their_parents_and_owns_its_refusal() -> None:
    point = design_space(Aligned(extent=12))
    assert isinstance(point.query(Aligned.factor), Unresolved)
    assert choose(point, 3, 3).factor == 3
    # Setting the two factors apart breaks ``aligned`` on the values the change names.
    report = point.try_with_choices({Aligned.first.factor: 3, Aligned.second.factor: 4})
    assert not report.accepted
    assert [(rule.rule, rule.standing) for rule in report.broken] == [("aligned", ())]
    apart = built(point, {"first.factor": 3, "second.factor": 4})
    refused = apart.inspect(Aligned.factor)
    assert isinstance(refused.result, Rejected)
    assert refused.output_result is None
    assert [(f.owner, f.message) for f in apart.query(Aligned.aligned).findings] == [
        ("aligned", "first.factor=3, second.factor=4")
    ]
    validity = inspection.validity(apart)
    assert [piece.owner for piece in validity.pieces] == ["aligned"]


# -- 2. quantification: a budget over whichever members are present -------------------


class Budgeted(Space):
    limit: int = Param()
    use_third: bool = Decision(domain=(False, True))
    a = Tiles(extent=12)
    b = Tiles(extent=8)
    c = Tiles(extent=6, when=use_third)
    costs = Members(Costly.cost)

    @constraint
    def within(self) -> bool | Rejected:
        spent = {member.node: member.value for member in self.costs}
        if sum(spent.values()) > self.limit:
            return reject("over-budget", f"{spent} exceeds {self.limit}")
        return True

    @view
    def total(self) -> int:
        return sum(member.value for member in self.costs)


def test_members_range_over_present_nodes_and_the_budget_counts_them() -> None:
    point = design_space(Budgeted(limit=10)).with_choices(use_third=False)
    point = point.with_choices({Budgeted.a.factor: 4})
    # Only b is still open; the absent c is inapplicable and never refuses.
    assert owners(point.query(Budgeted.costs)) == {"b.factor"}
    assert isinstance(point.query(Budgeted.c.cost), Inapplicable)
    validity = inspection.validity(point)
    assert validity.status == "open"
    assert [piece.owner for piece in validity.pieces] == ["within"]
    assert owners(point.query(Budgeted.total)) == {"b.factor"}
    point = point.with_choices({Budgeted.b.factor: 2})
    assert point.total == 3 + 4
    assert point.costs == (Located("a", "cost", 3), Located("b", "cost", 4))
    third = point.with_choices(use_third=True)
    # The third member's cost puts the budget over: ``within`` is false on its own values,
    # the first two factors standing in its way (``use_third`` decides whether ``c`` is
    # there, so it does not).
    report = third.try_with_choices({Budgeted.c.factor: 1})
    assert not report.accepted
    assert [(rule.rule, rule.standing) for rule in report.broken] == [
        ("within", ("a.factor", "b.factor"))
    ]
    over = built(third, {"use_third": True, "a.factor": 4, "b.factor": 2, "c.factor": 1})
    refused = over.query(Budgeted.total)
    assert isinstance(refused, Rejected)
    assert inspection.validity(over).status == "refused"


# -- 3. a graph built as data, with edges assigned after their nodes -------------------


class Wide(Space):
    """What a pipeline counts the width of: a part states its ``width_out``, whose own
    refuses."""

    @derived
    def width_out(self) -> int | Rejected:
        return reject("no-width", f"{type(self).__qualname__} states no width")


class Stage(Wide):
    width_in: int = Param()
    growth: int = Decision(domain=(0, 1, 2))

    @derived
    def width_out(self) -> int:
        return self.width_in + self.growth


def pipeline(count: int) -> type[Space]:
    """Nodes are plain Python values, joined by assignment; ``composite`` names them."""
    stages = [Stage(width_in=4), *(Stage() for _ in range(1, count))]
    for previous, current in zip(stages, stages[1:]):
        current.width_in = previous.width_out
    nodes = {f"s{index}": stage for index, stage in enumerate(stages)}
    return composite(f"Pipeline{count}", {**nodes, "widths": Members(Wide.width_out)})


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


# -- 4. a formal nobody supplies --------------------------------------------------------


class Source(Space):
    width: int = Param()

    @derived
    def out(self) -> int:
        return self.width


class Sink(Space):
    width: int = Param()
    limit: int | None = Param(default=None)


def test_an_optional_formal_nobody_supplies_is_none() -> None:
    class Holder(Space):
        sink = Sink(width=3)

    point = design_space(Holder()).sink
    assert point.width == 3
    assert point.limit is None and point.query(Sink.limit) == Available(None)


# -- 5. a cycle in the graph, anchored; and an unanchored value cycle ------------------


class Adder(Space):
    inp: int = Param()
    back: int = Param()

    @derived
    def out(self) -> int:
        return max(self.inp, self.back) + 1


class Register(Space):
    """Holds a declared width: this anchors the value flow around a loop.

    ``q`` is a derived value of ``width`` alone: it reads nothing of ``d``, so the loop
    through ``adder`` and ``register`` is anchored.
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

    @derived
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


class Pair(Wide):
    """Two stages in series; its input is its own formal, its output its ``width_out``."""

    width_in: int = Param()
    first = Stage(width_in=width_in)
    second = Stage(width_in=first.width_out)

    @derived
    def width_out(self) -> int:
        return self.second.width_out


class Chain(Space):
    head = Stage(width_in=2)
    body = Pair()
    body.width_in = head.width_out  # an edge, assigned
    widths = Members(Wide.width_out)


def test_a_composite_node_has_the_surface_of_a_leaf_node() -> None:
    point = design_space(Chain())
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices({handle: 1 for handle in handles.values()})
    assert [(m.node, m.value) for m in point.widths] == [("head", 3), ("body", 5)]


# -- 7. reducibility: the structural choice is simply a Decision over nodes --------------


class Fixed(Space):
    width: int = Param()
    physical = view(width)


class Tuned(Space):
    base: int = Param()
    extra: int = Decision(domain=(1, 2))

    @view
    def physical(self) -> int:
        return self.base + self.extra


class WithChoice(Space):
    """The structural choice: a selector, read through ``choice.physical``."""

    base: int = Param()
    choice: Fixed | Tuned = Select(fixed=Fixed(width=8), tuned=Tuned(base=base))
    physical = view(choice.physical)


class WithPrimitives(Space):
    """The same choice spelled with the primitives it lowers onto."""

    base: int = Param()
    choice: str = Decision(domain=("fixed", "tuned"))

    @derived
    def is_fixed(self) -> bool:
        return self.choice == "fixed"

    @derived
    def is_tuned(self) -> bool:
        return self.choice == "tuned"

    fixed = Fixed(width=8, when=is_fixed)
    tuned = Tuned(base=base, when=is_tuned)

    @view
    def physical(self) -> int:
        return self.fixed.physical if self.choice == "fixed" else self.tuned.physical


@pytest.mark.parametrize("space_type", (WithChoice, WithPrimitives))
def test_a_structural_choice_reduces_to_primitives(
    space_type: type[WithChoice] | type[WithPrimitives],
) -> None:
    start = design_space(space_type(base=4))
    physical = space_type.physical
    # The tuned node's choice: a case's path, or the primitive node's.
    extra = (
        WithChoice.choice["tuned"].extra  # type: ignore[index]
        if space_type is WithChoice
        else WithPrimitives.tuned.extra
    )
    assert isinstance(start.query(physical), Unresolved)
    tuned = start.with_choices({space_type.choice: "tuned", extra: 2})
    assert tuned.query(physical) == Available(6)
    # A candidate-local choice of an inactive candidate is kept, dormant.
    dormant = tuned.with_choices({space_type.choice: "fixed"})
    assert dormant.query(physical) == Available(8)
    assert isinstance(dormant.query(extra), Inapplicable)
    stored = committed(dormant)
    assert [stored[key] for key in stored if key.endswith("tuned.extra")] == [2]
    tuned_node = cast(Tuned, tuned.choice if isinstance(tuned, WithChoice) else tuned.tuned)
    fixed = tuned.with_choices({space_type.choice: "fixed"}, tuned_node.field(Tuned.extra).clear())
    assert fixed.query(physical) == Available(8)
    assert isinstance(fixed.query(extra), Inapplicable)
    replayed = replay(design_space(space_type(base=4)), committed(tuned))
    assert replayed.query(physical) == Available(6)


def test_candidates_are_ordinary_nodes_with_declaration_path_keys() -> None:
    keys = {item.key for item in inspection.decisions(WithChoice)}
    assert keys == {"choice", "choice.tuned.extra"}
    start = design_space(WithChoice(base=4))
    extra = WithChoice.choice["tuned"].extra  # type: ignore[index]
    point = start.with_choices(choice="tuned").with_choices({extra: 1})
    assert point.physical == 5
    selected = point.choice
    assert isinstance(selected, Tuned) and selected.extra == 1
    fixed = inspection.candidate(point, WithChoice.choice, "fixed")
    assert isinstance(fixed, Fixed) and isinstance(fixed.query(Fixed.physical), Inapplicable)


def test_assigning_a_formal_twice_in_one_body_is_a_definition_error() -> None:
    with refused("formal-assigned-twice", error=DefinitionError) as caught:

        class Twice(Space):
            a = Source(width=3)
            sink = Sink(width=1)
            sink.width = a.out

    # Both sites are named: the assignment, and the call that supplied the formal.
    message = str(caught.error)
    assert "assigned at test_design_graph.py:" in message
    assert "already assigned at test_design_graph.py:" in message


def test_a_required_formal_left_unsupplied_is_reported_when_prepared() -> None:
    sink = Sink()  # legal: an assignment may still supply the width
    assert unsupplied_formals(cast(NodeDecl, node_record(sink))) == ["width"]

    class Dangling(Space):
        sink = Sink()

    with refused("formal-unsupplied", error=DefinitionError, message=r"sink\.width") as caught:
        design_space(Dangling())
    message = str(caught.error)
    # The formal's declaration line and the node's call line.
    assert "Sink.width (declared at test_design_graph.py:" in message
    assert "for the node sink (declared at test_design_graph.py:" in message
    with refused("formal-unsupplied", error=DefinitionError, message=r"^width\b"):
        design_space(Sink())
