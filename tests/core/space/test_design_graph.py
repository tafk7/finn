# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A graph of design spaces, stated only in terms of Space.

Nodes are placements. Edges are the existing bindings, plus ``Bind`` for an
edge declared after its nodes (forward, around a cycle, or built as data) and
``Present`` for whichever source is present. Structure becomes observable to
computations through ``Located`` values (``.at`` / ``located``) and
``Members(key)``, which ranges over the present children that export ``key``.
None of these cases is about hardware.
"""

from __future__ import annotations

from typing import cast

import pytest

from finn.core.space import (
    Available,
    Bind,
    Decision,
    DefinitionError,
    EvaluationError,
    Inapplicable,
    Located,
    Members,
    Param,
    Present,
    Rejected,
    ScopeBuilder,
    Space,
    Subspace,
    SubspaceChoice,
    Unresolved,
    View,
    ViewKey,
    constraint,
    derived,
    divisors_of,
    inspection,
    located,
    reject,
    require_value,
    selections,
    view,
)
from finn.core.space.graph import LOCATED

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
    extent = Param(int)
    factor = Decision(int, domain=divisors_of(extent))

    @view
    def cost(self) -> int:
        return self.extent // self.factor

    exports = {COST: cost}


# -- 1. a relation between siblings is an ordinary node ---------------------------------


class Same(Space):
    """A relation: two located values must be equal; its output is the agreed value."""

    left = Param(LOCATED)
    right = Param(LOCATED)

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

    @view(constraints=(equal,))
    def agreed(self) -> int:
        return cast(int, self.left.value)

    exports = {AGREED: agreed}


class Aligned(Space):
    extent = Param(int)
    first = Subspace(Tiles, extent=extent)
    second = Subspace(Tiles, extent=extent)
    aligned = Subspace(Same, left=first.at(Tiles.factor), right=second.at(Tiles.factor))
    relations = Members(AGREED)

    @view(constraints=(relations,))
    def factor(self) -> int:
        return self.aligned.agreed()


def choose(point: Aligned, first: int, second: int) -> Aligned:
    return point.with_choices(
        point.first.field(Tiles.factor).change(first),
        point.second.field(Tiles.factor).change(second),
    )


def test_a_relation_node_reads_located_siblings_and_owns_its_refusal() -> None:
    point = Aligned(extent=12)
    assert isinstance(point.factor.query(), Unresolved)
    assert choose(point, 3, 3).factor() == 3
    refused = choose(point, 3, 4).factor.inspect()
    assert isinstance(refused.accepted_result, Rejected)
    # Identity comes from the graph, not from literals: node and member names.
    # (The reducer reports a refusal reached by both output and obligation twice.)
    assert {(f.owner, f.message) for f in refused.accepted_result.findings} == {
        ("aligned.equal", "first.factor=3, second.factor=4")
    }
    assert set(refused.constraints.results) == {"aligned.agreed"}


# -- 2. quantification: a budget over whichever members are present -------------------


class Budgeted(Space):
    limit = Param(int)
    use_third = Decision(bool, values=(False, True))
    a = Subspace(Tiles, extent=12)
    b = Subspace(Tiles, extent=8)
    c = Subspace(Tiles, extent=6, when=use_third)
    costs = Members(COST)

    @constraint
    def within(self) -> bool | Rejected:
        spent = {member.node: member.value for member in self.costs}
        if sum(spent.values()) > self.limit:
            return reject("over-budget", f"{spent} exceeds {self.limit}")
        return True

    @view(constraints=(costs, within))
    def total(self) -> int:
        return sum(member.value for member in self.costs)


def test_members_range_over_present_nodes_with_per_member_obligations() -> None:
    point = Budgeted(limit=10).with_choices(use_third=False)
    point = point.with_choices(point.a.field(Tiles.factor).change(4))
    assessment = point.total.inspect()
    # Only b is still open; the absent c is inapplicable and never refuses.
    results = assessment.constraints.results
    assert set(results) == {"a.cost", "b.cost", "c.cost", "within"}
    assert isinstance(results["b.cost"], Unresolved)
    assert isinstance(results["c.cost"], Inapplicable)
    point = point.with_choices(point.b.field(Tiles.factor).change(2))
    assert point.total() == 3 + 4
    assert point.costs == (Located("a", "cost", 3), Located("b", "cost", 4))
    third = point.with_choices(use_third=True)
    third = third.with_choices(third.c.field(Tiles.factor).change(1))
    refused = third.total.query()
    assert codes(refused) == {"over-budget"} and owners(refused) == {"within"}


# -- 3. a graph built as data, with edges declared after their nodes -------------------


class Stage(Space):
    width_in = Param(int)
    growth = Decision(int, values=(0, 1, 2))

    @view
    def width_out(self) -> int:
        return self.width_in + self.growth

    exports = {WIDTH: width_out}


def pipeline(count: int) -> type[Space]:
    builder = ScopeBuilder(Space, name=f"Pipeline{count}")
    stages = [builder.add("s0", Subspace(Stage, width_in=4))]
    for index in range(1, count):
        stages.append(builder.add(f"s{index}", Subspace(Stage)))  # width_in left open
        builder.add(
            f"e{index}",
            Bind(stages[index].ref(Stage.width_in), stages[index - 1].accepted(Stage.width_out)),
        )
    builder.add("widths", Members(WIDTH))
    return builder.finish()


def test_a_pipeline_is_built_as_data_and_its_edges_are_declarations() -> None:
    point = pipeline(5)()
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices(*(point.field(handles[f"s{i}.growth"]).change(1) for i in range(5)))
    members = cast("Members[int]", getattr(type(point), "widths"))
    widths = cast(tuple[Located[int], ...], require_value(point.query(members)))
    assert [(item.node, item.value) for item in widths] == [
        ("s0", 5),
        ("s1", 6),
        ("s2", 7),
        ("s3", 8),
        ("s4", 9),
    ]
    evidence = inspection.explain(point, members)
    assert {"e4", "s4.width_in", "s3.width_out"} <= {n.declaration.key for n in evidence.nodes}


# -- 4. a formal supplied by whichever source is present --------------------------------


class Source(Space):
    width = Param(int)

    @view
    def out(self) -> int:
        return self.width


class Sink(Space):
    width = Param(int)
    limit = Param(int, required=False)


class Either(Space):
    mode = Decision(str, values=("a", "b"))

    @derived
    def is_a(self) -> bool:
        return self.mode == "a"

    @derived
    def is_b(self) -> bool:
        return self.mode == "b"

    a = Subspace(Source, width=3, when=is_a)
    b = Subspace(Source, width=5, when=is_b)
    sink = Subspace(Sink, width=Present(a.accepted(Source.out), b.accepted(Source.out)))


def test_present_supplies_a_formal_from_whichever_node_is_present() -> None:
    point = Either()
    assert isinstance(point.sink.query(Sink.width), Unresolved)  # not yet known which
    assert point.with_choices(mode="a").sink.width == 3
    assert point.with_choices(mode="b").sink.width == 5
    # An optional formal nobody supplies is unsupplied, not an error.
    assert codes(point.sink.query(Sink.limit)) == {"input-unsupplied"}


def test_two_present_sources_are_refused_where_the_value_is_read() -> None:
    class Both(Space):
        a = Subspace(Source, width=3)
        b = Subspace(Source, width=3)
        sink = Subspace(Sink)
        from_a = Bind(sink.ref(Sink.width), a.accepted(Source.out))
        from_b = Bind(sink.ref(Sink.width), b.accepted(Source.out))

    answer = Both().sink.query(Sink.width)
    assert codes(answer) == {"multiple-suppliers"} and owners(answer) == {"sink.width"}


# -- 5. a cycle in the graph, anchored; and an unanchored value cycle ------------------


class Adder(Space):
    inp = Param(int)
    back = Param(int)

    @view
    def out(self) -> int:
        return max(self.inp, self.back) + 1


class Register(Space):
    """Holds a declared width: this anchors the value flow around a loop.

    ``q`` is a raw value. Were it a view obliging ``fits``, its acceptance would
    read ``d`` and so close the cycle: acceptance is part of the value flow.
    """

    width = Param(int)
    d = Param(int)

    @constraint
    def fits(self) -> bool:
        return self.d <= 2 * self.width

    @derived
    def q(self) -> int:
        return self.width


class Accumulator(Space):
    width = Param(int)
    total = Param(int)
    adder = Subspace(Adder, inp=width)  # back is supplied by the loop edge below
    register = Subspace(Register, width=total, d=adder.accepted(Adder.out))
    loop = Bind(adder.ref(Adder.back), register.ref(Register.q))


class Echo(Space):
    d = Param(int)

    @view
    def q(self) -> int:
        return self.d


def test_a_cyclic_graph_evaluates_when_its_value_flow_is_anchored() -> None:
    point = Accumulator(width=4, total=12)
    assert point.adder.out() == 13
    assert point.register.inspect(Register.fits).verdict is True
    wide = Accumulator(width=30, total=12)  # 31 does not fit twice 12
    assert isinstance(wide.register.inspect(Register.fits).result, Rejected)


def test_an_unanchored_value_cycle_fails_with_its_path() -> None:
    class Unanchored(Space):
        width = Param(int)
        adder = Subspace(Adder, inp=width)
        echo = Subspace(Echo, d=adder.accepted(Adder.out))
        loop = Bind(adder.ref(Adder.back), echo.accepted(Echo.q))

    with pytest.raises(EvaluationError, match="cycle"):
        Unanchored(width=4).adder.out()


# -- 6. closure: a composite is a node like any other ----------------------------------


class Pair(Space):
    """Two stages in series; its input is its own formal, its output an export."""

    width_in = Param(int)
    first = Subspace(Stage, width_in=width_in)
    second = Subspace(Stage, width_in=first.accepted(Stage.width_out))
    width_out = View(second.accepted(Stage.width_out))
    exports = {WIDTH: width_out}


class Chain(Space):
    head = Subspace(Stage, width_in=2)
    body = Subspace(Pair)  # width_in left open, supplied by an edge
    edge = Bind(body.ref(Pair.width_in), head.accepted(Stage.width_out))
    widths = Members(WIDTH)


def test_a_composite_node_has_the_surface_of_a_leaf_node() -> None:
    point = Chain()
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices(*(point.field(ref).change(1) for ref in handles.values()))
    assert [(m.node, m.value) for m in point.widths] == [("head", 3), ("body", 5)]


# -- 7. reducibility: a SubspaceChoice is a selector, guarded nodes and Present ---------

OUTPUT = ViewKey("output", int)


class Fixed(Space):
    width = Param(int)
    physical = View(width)
    exports = {OUTPUT: physical}


class Tuned(Space):
    base = Param(int)
    extra = Decision(int, values=(1, 2))

    @view
    def physical(self) -> int:
        return self.base + self.extra

    exports = {OUTPUT: physical}


class WithChoice(Space):
    base = Param(int, required=False)
    choice = SubspaceChoice(
        {"fixed": Subspace(Fixed, width=8), "tuned": Subspace(Tuned, base=base)},
        exports=(OUTPUT,),
    )
    physical = View(choice.accepted(OUTPUT))


class WithPrimitives(Space):
    base = Param(int, required=False)
    choice = Decision(str, values=("fixed", "tuned"))

    @derived
    def is_fixed(self) -> bool:
        return self.choice == "fixed"

    @derived
    def is_tuned(self) -> bool:
        return self.choice == "tuned"

    fixed = Subspace(Fixed, width=8, when=is_fixed)
    tuned = Subspace(Tuned, base=base, when=is_tuned)
    physical = View(Present(fixed.accepted(OUTPUT), tuned.accepted(OUTPUT)))


@pytest.mark.parametrize("family", (WithChoice, WithPrimitives))
def test_a_structural_choice_reduces_to_primitives(family: type[Space]) -> None:
    start = family(base=4)
    selector = next(item.reference for item in inspection.decisions(start) if item.key == "choice")
    extra = next(
        item.reference for item in inspection.decisions(start) if item.key.endswith("extra")
    )
    physical = cast(View[int], getattr(family, "physical"))
    assert isinstance(start.query(physical), Unresolved)
    tuned = start.with_choices(start.field(selector).change("tuned"), start.field(extra).change(2))
    assert tuned.query(physical) == Available(6)
    # A case-local choice of an inactive case is refused, not stored.
    assert not tuned.try_with_choices(tuned.field(selector).change("fixed")).accepted
    fixed = tuned.with_choices(tuned.field(selector).change("fixed"), tuned.field(extra).clear())
    assert fixed.query(physical) == Available(8)
    assert isinstance(fixed.query(extra), Inapplicable)
    replayed = selections.restore(family(base=4), selections.capture(tuned))
    assert replayed.accepted and replayed.instance.query(physical) == Available(6)


def test_case_members_are_ordinary_references_once_cases_are_nodes() -> None:
    # Previously a case-local choice was reachable only through inspection handles.
    tuned_extra = WithChoice.choice.alternatives["tuned"].decision_ref(Tuned.extra)
    start = WithChoice(base=4)
    point = start.with_choices(
        start.field(inspection.choices(start)[0].selector).change("tuned"),  # type: ignore[arg-type]
        start.field(tuned_extra).change(1),
    )
    assert point.physical() == 5
    located_case = WithChoice.choice.at(OUTPUT)
    del located_case  # the located form names the choice node: see the MVAU case


def test_the_enclosing_space_locates_its_own_members() -> None:
    class Own(Space):
        width = Param(int)
        here = located(width)

    assert Own(width=3).here == Located(None, "width", 3)


def test_a_bind_must_target_an_open_formal() -> None:
    with pytest.raises(DefinitionError, match="already supplied"):

        class Twice(Space):
            a = Subspace(Source, width=3)
            sink = Subspace(Sink, width=1)
            again = Bind(sink.ref(Sink.width), a.accepted(Source.out))

        Twice()
