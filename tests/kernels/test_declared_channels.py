# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels as ordinary Spaces in a partition: each takes the sides of the ports on it.

Every channel owns its refusals; a channel with a port on one side only is a boundary of
the partition and presents the AXIS port its boundary side names, by the boundary rule; a
channel's tensor is a fact. A side on the wrong role, and channels whose facts read each
other, are the composer's to refuse (``tests/kernels/test_composer.py``).
"""

from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, built, by_hand, choose, module_of, with_direct_transports

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
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.partition import Partition
from finn.kernels.sides import Side
from finn.kernels.transport import AxisBeat, StreamContract
from finn.kernels.values.semantics import IntegerTensorValue

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)
VECTOR = Tensor((4,), INT4)


def source(contents: tuple[int, ...], tensor: Tensor) -> Facts:
    return Facts(
        MemStreamKernel,
        {
            "dtype": DataType["INT4"],
            "form": PRODUCED,
            "contents": IntegerTensorValue.of(contents),
            "platform": FULL_DSP48E2,
        },
    )


def unconfigured(first: Tensor = VECTOR, second: Tensor = VECTOR) -> Partition:
    """Two constant vectors streamed to two outputs; each channel's tensor is given. Each
    channel has only its producer: it is a boundary, named by its port."""
    network = by_hand(
        {
            "first": channel(
                Port("first_source", "output"),
                Side.boundary("out0_V"),
                tensor=first,
                platform=FULL_DSP48E2,
            ),
            "second": channel(
                Port("second_source", "output"),
                Side.boundary("out1_V"),
                tensor=second,
                platform=FULL_DSP48E2,
            ),
            "first_source": source((1, 2, 3, 4), first),
            "second_source": source((5, 6, 7, -8), second),
        }
    )
    return with_direct_transports(network)


def constants(first: Tensor = VECTOR, second: Tensor = VECTOR) -> Partition:
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
    # Each finding keeps its owner, the channel's own declaration; the partition names the
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
    visited = {node.declaration.key.removeprefix("first.") for node in channel.nodes}
    assert {"netlist", "well_formed", "compatible", "streams"} <= visited
    member = inspection.explain(point.member("first_source"), KernelSpace.netlist)
    visited = {node.declaration.key.removeprefix("first_source.") for node in member.nodes}
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
    # A channel takes each side as one value: a port's side, its stream, its label below
    # its member and, an output's, its word width; its other side the boundary's.
    first = point.member("first")
    produced = point.member("first_source")
    assert first.producer == Side.kernel(
        produced.output_stream, "first_source.output", produced.output_data_width
    )
    assert first.producer.width == 8
    assert first.consumer == Side.boundary("out0_V")
    sides = first.sides
    assert (sides.source_owner, sides.sink_owner) == ("first_source.output", "out0_V")
    assert sides.owners == ("first_source.output", None)
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

    @derived
    def port_stream(self) -> Stream:
        contract = replaying()
        return Stream(contract.element, contract.sequence)

    @derived
    def port_side(self) -> Side:
        return Side.kernel(self.port_stream, "port")

    @view(semantics=PORT_CONTRACT)
    def port_contract(self) -> PortContract:
        return PortContract("port", replaying())


def test_a_boundary_presents_its_free_side_without_the_replay_the_receiver_realizes():
    receiver = by_hand(
        {
            "edge": channel(
                Side.boundary("in0_V"),
                Port("reader", "port"),
                tensor=Tensor((2, 4), INT4),
                platform=FULL_DSP48E2,
            ),
            "reader": Facts(Replaying, {}),
        }
    )
    edge = receiver.member("edge")
    sides = edge.sides
    assert sides.producer == Side.boundary("in0_V") and sides.sink_owner == "reader.port"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert sides.source == Stream(INT4, BeatSequence(vector_major((2, 4), 2)))
    assert edge.boundary.transport.name == "in0_V" and not edge.boundary.rules
    assert sides.sink.sequence.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_an_adapter_under_a_refused_guard_is_one_piece():
    # A selector whose ``when`` is refused counts once, at the highest refused guard of its
    # chain: one piece keyed by the selector, none for its cases. A correction: the engine
    # gave one piece more per case, against the reference model's reading 8. A channel with
    # a known value given a producer too has two producers (channel-users).
    clash = by_hand(
        {
            "shared": channel(
                Port("a", "output"),
                Side.boundary("out0_V"),
                tensor=VECTOR,
                contents=IntegerTensorValue.of((1, 2, 3, 4)),
                platform=FULL_DSP48E2,
            ),
            "a": source((1, 2, 3, 4), VECTOR),
        }
    )
    validity = clash.validity("shared")
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
