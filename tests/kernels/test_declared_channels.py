# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels as ordinary Spaces: each takes the ends of the ports on it as facts.

Every channel owns its refusals; a channel with one end is a boundary of the
network and presents its ``port`` name, by the boundary rule; a channel's tensor
is a fact, and a channel whose facts read each other is refused when its network
is declared.
"""

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, built, choose, module_of, with_direct_transports

from finn.core.space import (
    Available,
    Rejected,
    Space,
    Unresolved,
    inspection,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import NETLIST, PORT_END, LeafKernel, PortEnd
from finn.kernels.channels import Channel, boundary_contract
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Fact, Interface, Network, NetworkError, wire
from finn.kernels.transport import AxisBeat, StreamContract

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)
VECTOR = Tensor((4,), INT4)


def source(contents: tuple[int, ...], tensor: Tensor) -> MemStreamKernel:
    return MemStreamKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=contents,
        platform=FULL_DSP48E2,
    )


def unconfigured(first: Tensor = VECTOR, second: Tensor = VECTOR) -> Network:
    """Two constant vectors streamed to two outputs; each channel's tensor is given. Each
    channel has only its producer: it is a boundary, named by its port."""
    network = Network(
        {
            "first": wire(
                tensor=first,
                port="out0_V",
                producer=Interface("first_source", "output_end"),
                platform=FULL_DSP48E2,
            ),
            "second": wire(
                tensor=second,
                port="out1_V",
                producer=Interface("second_source", "output_end"),
                platform=FULL_DSP48E2,
            ),
            "first_source": source((1, 2, 3, 4), first),
            "second_source": source((5, 6, 7, -8), second),
        }
    )
    return with_direct_transports(network)


def constants(first: Tensor = VECTOR, second: Tensor = VECTOR) -> Network:
    return choose(
        unconfigured(first, second),
        {
            "first_source.ram_style": "auto",
            "first_source.pumped_memory": False,
            "second_source.ram_style": "distributed",
            "second_source.pumped_memory": False,
        },
    )


def test_matching_channels_compose_into_one_module():
    names = {port.name for port in built(constants(), "constants").abi.pins}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names


def test_each_channel_owns_its_refusal_and_independent_refusals_are_all_visible():
    # A source traversing four elements cannot carry an eight-element tensor.
    wide = Tensor((8,), INT4)
    point = constants(first=wide, second=wide)
    assert isinstance(point.query("first", NETLIST), Rejected)
    assert isinstance(point.query("second", NETLIST), Rejected)
    refusal = module_of(point, "constants")
    assert isinstance(refusal, Rejected)
    # Each finding keeps its owner, the channel's own declaration; the network names the
    # channel beside it.
    assert {(f.owner, dict(f.details)["tensor"]) for f in refusal.findings} == {
        ("well_formed", "first"),
        ("well_formed", "second"),
    }
    assert {f.code for f in refusal.findings} == {"channel-tensor"}
    # One channel refusing leaves the other channel's netlist accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.member("first").query(Channel.netlist), Rejected)
    assert isinstance(mixed.member("second").query(Channel.netlist), Available)


def test_explain_shows_per_channel_and_per_member_evidence():
    point = constants()
    channel = inspection.explain(point.member("first"), Channel.netlist)
    visited = {node.declaration.key for node in channel.nodes}
    assert {"netlist", "well_formed", "compatible", "ends"} <= visited
    member = inspection.explain(point.member("first_source"), LeafKernel.netlist)
    visited = {node.declaration.key for node in member.nodes}
    # A leaf's netlist is its own module, its ``codegen``: no members' netlists.
    assert {"output.presented", "module", "codegen"} <= visited


def test_a_channel_waits_for_its_own_endpoints_only():
    point = choose(
        unconfigured(),
        {"first_source.ram_style": "auto", "first_source.pumped_memory": False},
    )
    # The ROM choice feeds only the module, not either channel's contracts.
    assert isinstance(point.member("first").query(Channel.netlist), Available)
    assert isinstance(point.member("second").query(Channel.netlist), Available)
    assert isinstance(module_of(point, "constants"), Unresolved)
    # A channel takes its ends as facts, each labelled from the network.
    first = point.member("first")
    end = first.producer
    assert end.label == "first_source.output"
    assert end.contract.transport.endpoint is Endpoint.INITIATOR  # the source produces
    ends = first.endpoints
    assert (ends.source_owner, ends.sink_owner) == ("first_source.output", None)
    assert ends.sink.transport.name == "out0_V"


def test_boundary_ports_are_axis_and_byte_aligned():
    contract = boundary_contract(
        "in0_V", INT4, BeatSequence(vector_major((3,), 3)), Endpoint.TARGET
    )
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12


class Replaying(Space):
    """A consumer reading each two-beat group of its input three times, framed."""

    @view(semantics=PORT_END)
    def end(self) -> PortEnd:
        beat = AxisBeat("s_axis", DataType["INT4"], 2, endpoint=Endpoint.TARGET, last=True)
        transport = beat.native(clock="ap_clk", reset="ap_rst_n")
        form = vector_major((2, 4), 2).replayed(3, inner_beats=2)
        contract = StreamContract(transport, INT4, form, markers=(("s_axis_tlast", LevelEnd(2)),))
        return PortEnd("port", contract)


def test_a_boundary_presents_its_internal_end_without_the_replay_the_receiver_realizes():
    receiver = Network(
        {
            "edge": wire(
                tensor=Tensor((2, 4), INT4),
                port="in0_V",
                consumer=Interface("reader", "end"),
                platform=FULL_DSP48E2,
            ),
            "reader": Replaying(),
        }
    )
    ends = receiver.member("edge").endpoints
    assert ends.source_owner is None and ends.sink_owner == "reader.port"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert ends.source.form == vector_major((2, 4), 2)
    assert ends.source.transport.name == "in0_V" and not ends.source.rules
    assert ends.sink.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_two_producers_on_one_channel_are_refused_by_the_channel():
    clash = Network(
        {
            # b's end given as the consumer is a producer's all the same.
            "shared": wire(
                tensor=VECTOR,
                port="out0_V",
                producer=Interface("a", "output_end"),
                consumer=Interface("b", "output_end"),
                platform=FULL_DSP48E2,
            ),
            "a": source((1, 2, 3, 4), VECTOR),
            "b": source((1, 2, 3, 4), VECTOR),
        }
    )
    refused = clash.member("shared").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    assert {f.code for f in refused.findings} == {"channel-users"}
    assert "a.output, b.output" in refused.findings[0].message


def test_a_boundary_channel_needs_its_port_name():
    unnamed = Network(
        {
            "out": wire(
                tensor=VECTOR, producer=Interface("source", "output_end"), platform=FULL_DSP48E2
            ),
            "source": source((1, 2, 3, 4), VECTOR),
        }
    )
    refused = unnamed.member("out").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    assert {(f.code, f.owner) for f in refused.findings} == {("channel-boundary", "endpoints")}


# -- the anchoring rule: a channel's facts read the members before it --------------------


def test_channels_whose_facts_read_each_other_are_refused_where_the_network_is_declared():
    """A channel's facts are read from members ranked before it: kernels, which take
    plain facts, then channels after the ones they read. Two channels that read each
    other, or one that reads itself, can never be ranked."""
    with pytest.raises(NetworkError, match="read each other"):
        Network(
            {
                "a": wire(tensor=VECTOR, consumer=Interface("b", "set_end"), platform=FULL_DSP48E2),
                "b": wire(tensor=VECTOR, consumer=Interface("a", "set_end"), platform=FULL_DSP48E2),
            }
        )
    with pytest.raises(NetworkError, match="read each other"):
        Network({"a": wire(tensor=Fact("a", "set_end"), platform=FULL_DSP48E2)})
    with pytest.raises(NetworkError, match="no members"):
        Network({"a": wire(tensor=Fact("nowhere", "tensor"), platform=FULL_DSP48E2)})
