# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels as ordinary Spaces: each takes the contracts of the ports on it as facts.

Every channel owns its refusals; a channel with a port on one side only is a
boundary of the network and presents its ``port`` name, by the boundary rule; a
channel's tensor is a fact, and a channel whose facts read each other is refused when
its network is declared.
"""

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, built, choose, module_of, with_direct_transports

from finn.core.space import (
    Available,
    Rejected,
    Space,
    Unresolved,
    derived,
    inspection,
    view,
)
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import NETLIST, PORT_CONTRACT, KernelSpace, PortContract
from finn.kernels.channels import Channel, boundary_contract
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Fact, Interface, Network, NetworkError, wire
from finn.kernels.port import STREAM
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
                producer=Interface("first_source", "output_contract"),
                platform=FULL_DSP48E2,
            ),
            "second": wire(
                tensor=second,
                port="out1_V",
                producer=Interface("second_source", "output_contract"),
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
    assert {"netlist", "well_formed", "compatible", "streams"} <= visited
    member = inspection.explain(point.member("first_source"), KernelSpace.netlist)
    visited = {node.declaration.key for node in member.nodes}
    # A kernel's netlist with a module builds it from its ports' streams, its parameters,
    # its data and its bus: it places no member, whose netlists it reads none of.
    assert {"output_stream", "parameters", "init_file", "config_presented", "netlists"} <= visited


def test_a_channel_waits_for_its_own_sides_only():
    point = choose(
        unconfigured(),
        {"first_source.ram_style": "auto", "first_source.pumped_memory": False},
    )
    # The ROM choice feeds only the module, not either channel's contracts.
    assert isinstance(point.member("first").query(Channel.netlist), Available)
    assert isinstance(point.member("second").query(Channel.netlist), Available)
    assert isinstance(module_of(point, "constants"), Unresolved)
    # A channel takes its ports' streams as facts, each with its port's label from the
    # network, and its producing port's word width; the fact a stream is bound to is its side.
    first = point.member("first")
    produced = point.member("first_source")
    assert first.producer == produced.output_stream
    assert first.producer_label == "first_source.output"
    assert first.producer_width == produced.output_data_width == 8
    sides = first.sides
    assert (sides.source_owner, sides.sink_owner) == ("first_source.output", None)
    assert first.boundary.transport.name == "out0_V"
    assert sides.sink == Stream(first.boundary.element, first.boundary.sequence)


def test_boundary_ports_are_axis_and_byte_aligned():
    contract = boundary_contract(
        "in0_V", INT4, BeatSequence(vector_major((3,), 3)), Endpoint.TARGET
    )
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12


def replaying() -> StreamContract:
    """What a consumer reading each two-beat group of its input three times, framed,
    states at its port."""
    beat = AxisBeat("s_axis", DataType["INT4"], 2, endpoint=Endpoint.TARGET, last=True)
    transport = beat.native(clock="ap_clk", reset="ap_rst_n")
    form = vector_major((2, 4), 2).replayed(3, inner_beats=2)
    return StreamContract(transport, INT4, form, markers=(("s_axis_tlast", LevelEnd(2)),))


class Replaying(Space):
    """A consumer reading each two-beat group of its input three times, framed: its port's
    stream and its contract."""

    @derived(semantics=STREAM)
    def port_stream(self) -> Stream:
        contract = replaying()
        return Stream(contract.element, contract.sequence)

    @view(semantics=PORT_CONTRACT)
    def port_contract(self) -> PortContract:
        return PortContract("port", replaying())


def test_a_boundary_presents_its_free_side_without_the_replay_the_receiver_realizes():
    receiver = Network(
        {
            "edge": wire(
                tensor=Tensor((2, 4), INT4),
                port="in0_V",
                consumer=Interface("reader", "port_contract"),
                platform=FULL_DSP48E2,
            ),
            "reader": Replaying(),
        }
    )
    edge = receiver.member("edge")
    sides = edge.sides
    assert sides.source_owner is None and sides.sink_owner == "reader.port"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert sides.source == Stream(INT4, BeatSequence(vector_major((2, 4), 2)))
    assert edge.boundary.transport.name == "in0_V" and not edge.boundary.rules
    assert sides.sink.sequence.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_an_adapter_under_a_refused_guard_is_one_piece():
    # A selector whose ``when`` is refused counts once, at the highest refused guard of its
    # chain: one piece keyed by the selector, none for its cases. A correction: the engine
    # gave one piece more per case, against the reference model's reading 8. A channel with
    # a known value given a producer too has two producers (channel-users).
    clash = Network(
        {
            "shared": wire(
                tensor=VECTOR,
                port="out0_V",
                contents=(1, 2, 3, 4),
                producer=Interface("a", "output_contract"),
                platform=FULL_DSP48E2,
            ),
            "a": source((1, 2, 3, 4), VECTOR),
        }
    )
    validity = inspection.validity(clash.member("shared"))
    assert validity.status == "refused"
    for selector in ("adapter", "output_adapter"):
        (piece,) = [
            piece
            for piece in validity.pieces
            if piece.owner == selector or piece.owner.startswith(f"{selector}.")
        ]
        assert piece.owner == selector
        assert {(f.code, f.owner) for f in piece.result.findings} == {
            ("channel-users", "sides"),
            ("guard-refused", selector),
        }


def test_a_boundary_channel_needs_its_port_name():
    unnamed = Network(
        {
            "out": wire(
                tensor=VECTOR,
                producer=Interface("source", "output_contract"),
                platform=FULL_DSP48E2,
            ),
            "source": source((1, 2, 3, 4), VECTOR),
        }
    )
    refused = unnamed.member("out").query(Channel.netlist)
    assert isinstance(refused, Rejected)
    # The origin, and beside it each guard that reads it (guard-refused, by its scope).
    assert {f.code for f in refused.findings} == {"channel-boundary", "guard-refused"}
    origins = {(f.code, f.owner) for f in refused.findings if f.code != "guard-refused"}
    assert origins == {("channel-boundary", "sides")}


# -- the anchoring rule: a channel's facts read the members before it --------------------


def test_channels_whose_facts_read_each_other_are_refused_where_the_network_is_declared():
    """A channel's facts are read from members ranked before it: kernels, which take
    plain facts, then channels after the ones they read. Two channels that read each
    other, or one that reads itself, can never be ranked."""
    with pytest.raises(NetworkError, match="read each other"):
        Network(
            {
                "a": wire(
                    tensor=VECTOR, consumer=Interface("b", "set_contract"), platform=FULL_DSP48E2
                ),
                "b": wire(
                    tensor=VECTOR, consumer=Interface("a", "set_contract"), platform=FULL_DSP48E2
                ),
            }
        )
    with pytest.raises(NetworkError, match="read each other"):
        Network({"a": wire(tensor=Fact("a", "set_contract"), platform=FULL_DSP48E2)})
    with pytest.raises(NetworkError, match="no members"):
        Network({"a": wire(tensor=Fact("nowhere", "tensor"), platform=FULL_DSP48E2)})
