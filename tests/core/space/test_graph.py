# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Graph composition over a toy domain: nets, ports, promotion, folds, cycles."""

from __future__ import annotations

import pytest

from finn.core.space import (
    Carried,
    Decision,
    DefinitionError,
    Ends,
    EvaluationError,
    Fold,
    Interface,
    Interpretation,
    Link,
    Net,
    Param,
    Port,
    Rejected,
    ScopeBuilder,
    Space,
    Subspace,
    Topology,
    Unresolved,
    ViewKey,
    constraint,
    derived,
    inspection,
    reject,
    selections,
    view,
)

# -- a toy domain: words of some width, and the label each end offers ------------------

WORD = Interface("word", carries=int, offers=str)
PART = ViewKey("part", str)
WIRE = ViewKey("wire", str)


class Wire(Link):
    width = Carried(WORD)
    ends = Ends(WORD)

    @constraint
    def one_driver(self) -> bool | Rejected:
        drivers = [end for end in self.ends if end.direction == "out"]
        if len(drivers) != 1:
            return reject("wire-drivers", f"{len(drivers)} ends drive this wire")
        return True

    @view(constraints=(one_driver,))
    def description(self) -> str:
        names = sorted(f"{end.node or '<top>'}.{end.port}" for end in self.ends)
        return f"{self.width}:" + ",".join(names)

    exports = {WIRE: description}


class Source(Space):
    width = Param(int)

    @view
    def label(self) -> str:
        return f"source{self.width}"

    out = Port(WORD, "out", carry=width, offer=label)
    exports = {PART: label}


class Sink(Space):
    limit = Param(int)

    @view
    def label(self) -> str:
        return "sink"

    inp = Port(WORD, "in", offer=label)

    @constraint
    def fits(self) -> bool | Rejected:
        if self.inp > self.limit:
            return reject("sink-width", f"{self.inp} bits exceed {self.limit}")
        return True

    @view(constraints=(fits,))
    def part(self) -> str:
        return f"sink{self.inp}"

    exports = {PART: part}


def describe(*, topology: Topology[str, str], name: str) -> str:
    nodes = ",".join(f"{node}={value}" for node, value in topology.nodes)
    nets = ";".join(f"{net.name}={net.value}" for net in topology.nets)
    ports = ",".join(f"{port.direction}:{port.name}" for port in topology.ports)
    return f"{name}[{nodes}|{nets}|{ports}]"


LISTING = Interpretation("listing", node=PART, net=WIRE, result=str, reduce=describe)


class Pair(Space):
    """One source fanned out to two sinks, and a promoted third sink input."""

    width = Param(int)
    data = Net(Wire)
    spare = Net(Wire)
    source = Subspace(Source, width=width, out=data)
    left = Subspace(Sink, limit=8, inp=data)
    right = Subspace(Sink, limit=4, inp=data)
    extra = Subspace(Sink, limit=16, inp=spare)
    feed = Port(WORD, "in", net=spare)
    listing = Fold(LISTING, name="pair")


def test_publish_adopt_fan_out_and_a_fold_over_the_declared_topology() -> None:
    point = Pair(width=3)
    assert point.left.inp == 3 and point.right.inp == 3
    assert point.data.width == 3
    assert point.data.description() == "3:left.inp,right.inp,source.out"
    ends = point.data.ends
    assert [(end.node, end.port, end.direction, end.publishes) for end in ends] == [
        ("source", "out", "out", True),
        ("left", "inp", "in", False),
        ("right", "inp", "in", False),
    ]
    assert ends[0].offer == "source3" and ends[1].offer == "sink"


def test_an_unanchored_promoted_input_relays_and_is_unresolved_when_standalone() -> None:
    point = Pair(width=3)
    # Nothing inside anchors ``spare``: ``feed`` relays it from outside, which is absent.
    answer = point.extra.query(Sink.inp)
    assert isinstance(answer, Unresolved)
    assert {f.code for f in answer.findings} == {"port-unconnected"}
    assert isinstance(point.listing.query(), Unresolved)


def test_independent_refusals_are_all_visible_and_owned_per_node_and_net() -> None:
    point = Pair(width=12)
    assessment = point.listing.inspect()
    refused = {
        key
        for key, answer in assessment.constraints.results.items()
        if isinstance(answer, Rejected)
    }
    assert {"left.part", "right.part"} <= refused
    owners = {
        finding.owner
        for answer in assessment.constraints.results.values()
        if isinstance(answer, Rejected)
        for finding in answer.findings
    }
    assert {"left.fits", "right.fits"} <= owners
    # the unresolved relay keeps the fold unresolved while the refusals stay visible
    assert isinstance(assessment.accepted_result, Unresolved)


class Closed(Space):
    """Pair with its spare input supplied by an outer source: a nested composite."""

    width = Param(int)
    feed = Net(Wire)
    source = Subspace(Source, width=5, out=feed)
    pair = Subspace(Pair, width=width, feed=feed)
    listing = Fold(LISTING, name="closed")


def test_a_composite_port_has_the_same_surface_as_a_primitive_port() -> None:
    point = Closed(width=3)
    # The inner relay adopts the outer net's value.
    assert point.pair.extra.inp == 5
    assert point.pair.feed == 5
    assert point.pair.listing() == (
        "pair[source=source3,left=sink3,right=sink3,extra=sink5|"
        "data=3:left.inp,right.inp,source.out;spare=5:<top>.feed,extra.inp|in:feed]"
    )
    # The inner fold reaches the outer net through the relay, and each net by name.
    evidence = inspection.explain(point.pair, Pair.listing)
    keys = {node.declaration.key for node in evidence.nodes}
    assert {"feed.$carried", "pair.spare.$carried", "pair.data.description"} <= keys
    ends = point.feed.ends
    assert [(end.node, end.port, end.direction) for end in ends] == [
        ("source", "out", "out"),
        ("pair", "feed", "in"),
    ]


def test_an_unconnected_adopting_port_is_unresolved_not_a_definition_error() -> None:
    sink = Sink(limit=4)
    answer = sink.query(Sink.inp)
    assert isinstance(answer, Unresolved)
    assert [f.code for f in answer.findings] == ["port-unconnected"]
    source = Source(width=2)
    assert source.out == 2


class Disagree(Space):
    data = Net(Wire)
    first = Subspace(Source, width=3, out=data)
    second = Subspace(Source, width=4, out=data)
    sink = Subspace(Sink, limit=8, inp=data)


def test_two_publishers_are_checked_and_the_refusal_belongs_to_the_net() -> None:
    point = Disagree()
    answer = point.sink.query(Sink.inp)
    assert isinstance(answer, Rejected)
    (finding,) = answer.findings
    assert (finding.code, finding.owner) == ("net-disagree", "data")
    driven = point.data.inspect(Wire.one_driver)
    assert isinstance(driven.result, Rejected)


def test_a_net_nobody_anchors_is_refused_at_preparation() -> None:
    class Floating(Space):
        data = Net(Wire)
        left = Subspace(Sink, limit=8, inp=data)
        right = Subspace(Sink, limit=8, inp=data)

    with pytest.raises(DefinitionError, match="no end publishes"):
        Floating()


def test_conditional_ends_follow_case_selection() -> None:
    class Delivery(Space):
        width = Param(int)
        data = Net(Wire, carry=width)
        mode = Decision(str, values=("external", "internal"))

        @derived
        def external(self) -> bool:
            return self.mode == "external"

        @derived
        def internal(self) -> bool:
            return self.mode == "internal"

        weights_in = Port(WORD, "in", net=data, when=external)
        rom = Subspace(Source, width=width, out=data, when=internal)
        sink = Subspace(Sink, limit=8, inp=data)
        listing = Fold(LISTING, name="delivery")

    point = Delivery(width=3)
    assert isinstance(point.data.query(Wire.ends), Unresolved)
    external = point.with_choices(mode="external")
    assert [e.node for e in external.data.ends] == [None, "sink"]
    assert external.data.description() == "3:<top>.weights_in,sink.inp"
    internal = point.with_choices(mode="internal")
    assert [e.node for e in internal.data.ends] == ["rom", "sink"]
    assert internal.listing() == ("delivery[rom=source3,sink=sink3|data=3:rom.out,sink.inp|]")
    assert external.listing() == (
        "delivery[sink=sink3|data=3:<top>.weights_in,sink.inp|in:weights_in]"
    )


def test_a_net_owned_decision_anchors_the_value_and_persists() -> None:
    class Negotiated(Space):
        data = Net(Wire, carry=Decision(int, values=(4, 8)))
        first = Subspace(Sink, limit=8, inp=data)
        sink = Subspace(Sink, limit=8, inp=data)

    point = Negotiated()
    keys = [item.key for item in inspection.decisions(point)]
    assert keys == ["data.carry"]
    handle = inspection.decisions(point)[0].reference
    chosen = point.with_choices(point.field(handle).change(8))
    assert chosen.sink.inp == 8
    saved = selections.capture(chosen)
    assert saved.keys == ("data.carry",)
    assert selections.restore(Negotiated(), saved).instance.sink.inp == 8


# -- a non-stream, cyclic topology: an accumulator with a feedback wire --------------


class Adder(Space):
    """Adds its input to the value fed back; the width grows by one bit."""

    @view
    def label(self) -> str:
        return "adder"

    inp = Port(WORD, "in", offer=label)
    back = Port(WORD, "in", offer=label)

    @derived
    def sum_width(self) -> int:
        return max(self.inp, self.back) + 1

    out = Port(WORD, "out", carry=sum_width, offer=label)
    exports = {PART: label}


class Register(Space):
    @view
    def label(self) -> str:
        return "register"

    d = Port(WORD, "in", offer=label)
    q = Port(WORD, "out", offer=label)
    exports = {PART: label}


class Accumulator(Space):
    """in -> adder -> register -> (back to adder). The declared topology has a cycle;
    the value flow is anchored by the parent on the feedback wire."""

    width = Param(int)
    total = Param(int)
    inp = Net(Wire, carry=width)
    result = Net(Wire)
    feedback = Net(Wire, carry=total)
    x = Port(WORD, "in", net=inp)
    adder = Subspace(Adder, inp=inp, back=feedback, out=result)
    register = Subspace(Register, d=result, q=feedback)
    listing = Fold(LISTING, name="acc")


def test_a_cyclic_topology_with_an_anchored_value_flow_evaluates() -> None:
    point = Accumulator(width=4, total=12)
    assert point.adder.out == 13 and point.register.d == 13
    assert point.listing() == (
        "acc[adder=adder,register=register|inp=4:<top>.x,adder.inp;"
        "result=13:adder.out,register.d;feedback=12:adder.back,register.q|in:x]"
    )


def test_an_unanchored_value_cycle_fails_evaluation_with_scoped_context() -> None:
    class Unanchored(Space):
        """The register would publish what the adder computes from what it publishes."""

        width = Param(int)
        inp = Net(Wire, carry=width)
        result = Net(Wire)
        feedback = Net(Wire)
        adder = Subspace(Adder, inp=inp, back=feedback, out=result)
        echo = Subspace(Echo, d=result, q=feedback)

    point = Unanchored(width=4)
    with pytest.raises(EvaluationError, match="cycl"):
        point.adder.out


class Echo(Space):
    """Publishes on q exactly what it adopts on d."""

    d = Port(WORD, "in")
    q = Port(WORD, "out", carry=d)


# -- a clock net built as data: one driver, many loads --------------------------------

CLOCK: Interface[int, None] = Interface("clock", carries=int)


class ClockNet(Link):
    period = Carried(CLOCK)
    ends = Ends(CLOCK)

    @view
    def loads(self) -> int:
        return sum(1 for end in self.ends if end.direction == "in")


class Timed(Space):
    clk = Port(CLOCK, "in")

    @derived
    def frequency(self) -> int:
        return 1000 // self.clk


def test_topology_is_buildable_as_data_through_scope_builder() -> None:
    builder = ScopeBuilder(Space, name="ClockTree")
    clk = builder.add("clk", Net(ClockNet, carry=4))
    for index in range(3):
        builder.add(f"unit{index}", Subspace(Timed, clk=clk))
    template = builder.finish()
    point = template()
    clock: ClockNet = getattr(point, "clk")
    assert clock.loads() == 3
    assert [getattr(point, f"unit{index}").frequency for index in range(3)] == [250] * 3


def test_a_value_cycle_through_declared_bindings_fails_preparation() -> None:
    class Mirror(Space):
        result = Net(Wire)
        feedback = Net(Wire)
        first = Subspace(Echo, d=result, q=feedback)
        second = Subspace(Echo, d=feedback, q=result)

    with pytest.raises(DefinitionError, match="cyclic"):
        Mirror()


def test_a_view_can_oblige_another_view_without_a_proxy_constraint() -> None:
    class Checked(Space):
        width = Param(int)
        data = Net(Wire)
        source = Subspace(Source, width=width, out=data)
        sink = Subspace(Sink, limit=4, inp=data)

        @view(constraints=(data.accepted(Wire.description),))
        def summary(self) -> int:
            return self.sink.inp

    assert Checked(width=3).summary() == 3


def test_ports_bind_only_to_nets_and_nets_join_one_interface() -> None:
    with pytest.raises(DefinitionError, match="binds to a Net"):

        class Wrong(Space):
            sink = Subspace(Sink, limit=4, inp=3)

        Wrong()
    with pytest.raises(DefinitionError, match="one interface"):

        class Mixed(Space):
            data = Net(Wire, carry=4)
            sink = Subspace(Sink, limit=4, inp=data)
            timed = Subspace(Timed, clk=data)

        Mixed()
