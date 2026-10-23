# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's netlist: its stages below it, and each checked hop resolved into wires.

A channel wires the hops between its own ends; its network wires the hops to the ports
beside it (``Network.hops``) and places them around the channel's links, in the network's
netlist. A link's end belongs to the kernel whose port it is, beside the channel
(``^compute``, ``compute`` in the network); a stage sits at its label below the channel
(``adapter.input_gen.input_gen``, ``transport.fifo.buffer``); a link's end on the
boundary is the root's own pins (``None``).
"""

from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    choose,
    dotp_extents,
    with_channel_defaults,
    with_direct_transports,
)

from finn.core.space import Available, Rejected, Space, derived, view
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import Fragment, Leaf
from finn.kernels.base import PORT_CONTRACT, PortContract
from finn.kernels.channels import Channel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.port import STREAM
from finn.kernels.sides import SIDE, Side
from finn.kernels.transport import AxisBeat, StreamContract

INT3, INT8 = DataType["INT3"], DataType["INT8"]
X = Tensor((3, 4), ScalarEncoding(INT3))
W = Tensor((4, 4), ScalarEncoding(INT3))
Y = Tensor((3, 4), ScalarEncoding(INT8))


def placing(*, w_port: str | None = "in1_V") -> Network:
    """dotp between boundary channels: three rows of four, four outputs, PE = SIMD = 2
    (the weight channel's port ``w_port``)."""
    named = {} if w_port is None else {"port": w_port}
    return Network(
        {
            "x": wire(
                tensor=X,
                port="in0_V",
                consumer=Interface("compute", "x_contract"),
                platform=FULL_DSP48E2,
            ),
            "w": wire(
                platform=FULL_DSP48E2,
                tensor=W,
                consumer=Interface("compute", "w_contract"),
                **named,
            ),
            "y": wire(
                tensor=Y,
                port="out0_V",
                producer=Interface("compute", "y_contract"),
                platform=FULL_DSP48E2,
            ),
            "compute": PackedDotpKernel(
                result_range=ordinary_integer_bounds(INT8),
                x_tensor=X,
                w_tensor=W,
                **dotp_extents(W),
                pe=2,
                simd=2,
                platform=FULL_DSP48E2,
            ),
        }
    )


FOLDING = {
    "compute.compute_pumping": False,
    "compute.reducer": "tree",
}


def placed(**transport: object) -> Network:
    folded = choose(placing(), FOLDING)
    return with_channel_defaults(with_direct_transports(choose(folded, transport)))


def links(point: Network) -> list[tuple[str | None, str | None]]:
    """The ends of each link of ``point``'s netlist, in order."""
    found = point.netlist()
    assert isinstance(found, Available)
    fragment, _ = found.value
    return [(link.source.instance, link.sink.instance) for link in fragment.links]


def test_an_adapted_channel_places_its_stage_and_wires_each_hop() -> None:
    point = placed(**{"w.transport": "direct"})
    fragment = point.member("x").netlist
    ((label, leaf),) = fragment.instances
    assert label == "adapter.input_gen.input_gen"
    assert isinstance(leaf, Leaf) and leaf.name == "input_gen"
    # Its own hop, from its boundary to its stage; the hop to compute's port is the
    # network's, placed after it.
    (into,) = fragment.links
    assert (into.source.instance, into.sink.instance) == (None, label)
    assert (into.source.data, into.sink.data) == ("in0_V_tdata", "idat")
    hops = point.hops("x")
    assert not isinstance(hops, Rejected) and hops.producer is None and hops.consumer
    out = hops.consumer
    assert (out.source.instance, out.sink.instance) == (label, "^compute")
    assert out.sink.data == "s_axis_input_tdata" and out.lanes == (0, 1) and out.lane_bits == 3
    # The frame marker: the input_gen's olst bit closing each reduction drives TLAST.
    assert out.markers == (("olst", 1, "s_axis_input_tlast", None),)
    # A direct channel is one hop, the network's, and only a boundary channel presents
    # a bus. The network's netlist: each channel's hops in the order the stream runs.
    assert point.member("y").netlist.links == ()
    assert [bus.name for bus in point.member("y").boundary_bus] == ["out0_V"]
    staged = f"x.{label}"
    assert links(point) == [
        (None, staged),
        (staged, "compute"),
        (None, "compute"),
        ("compute", None),
    ]


def test_a_buffered_channel_places_its_fifo_below_its_transport() -> None:
    point = placed(
        **{
            "w.transport": "fifo",
            "w.transport.fifo.depth": 4,
            "w.transport.fifo.buffer.ram_style": "auto",
        }
    )
    fragment = point.member("w").netlist
    assert [label for label, _ in fragment.instances] == ["transport.fifo.buffer"]
    assert [(item.source.instance, item.sink.instance) for item in fragment.links] == [
        (None, "transport.fifo.buffer"),
    ]
    assert links(point)[2:4] == [
        (None, "w.transport.fifo.buffer"),
        ("w.transport.fifo.buffer", "compute"),
    ]


def test_a_boundary_no_port_names_is_refused() -> None:
    """A weight channel with one side and no port is a boundary nothing may cross: only an
    ONNX input or output of a partition is one."""

    # A side no port is wired to, and no port names, is no side the network can give: it
    # refuses the channel as it binds it, owned by the channel.
    validity = choose(placing(w_port=None), FOLDING).validity("w")
    assert validity.status == "refused" and isinstance(validity.result, Rejected)
    assert {(item.code, item.owner) for item in validity.result.findings} == {
        ("channel-boundary", "w")
    }
    # The same network with the port named is accepted.
    assert not isinstance(choose(placing(), FOLDING).member("w").sides, Rejected)


def reading() -> StreamContract:
    """What ``Reader`` states at its port."""
    beat = AxisBeat("s_axis", INT3, 4, endpoint=Endpoint.TARGET)
    transport = beat.native(clock="ap_clk", reset="ap_rst_n")
    return StreamContract(transport, ScalarEncoding(INT3), vector_major((3, 4), 4))


class Reader(Space):
    """A consumer that presents its stream and contract itself, not through a kernel's
    port."""

    @derived(semantics=STREAM)
    def port_stream(self) -> Stream:
        contract = reading()
        return Stream(contract.element, contract.sequence)

    @derived(semantics=SIDE)
    def port_side(self) -> Side:
        return Side.kernel(self.port_stream, "port")

    @view(semantics=PORT_CONTRACT)
    def port_contract(self) -> PortContract:
        return PortContract("", reading())


def test_a_contract_that_is_no_kernels_port_has_no_netlist() -> None:
    bare = Network(
        {
            "edge": wire(
                platform=FULL_DSP48E2,
                tensor=X,
                port="in0_V",
                consumer=Interface("reader", "port_contract"),
            ),
            "reader": Reader(),
        }
    )
    point = with_direct_transports(bare)
    edge = point.member("edge")
    assert edge.consumer.label == "reader.port"
    # The channel's own netlist has no hop to the reader; the network's refuses it.
    assert edge.query(Channel.netlist).value == Fragment()
    refused = point.netlist()
    assert isinstance(refused, Rejected)
    assert [
        (item.code, item.owner, item.message, dict(item.details)) for item in refused.findings
    ] == [
        (
            "channel-port",
            "edge",
            "reader presents a contract, but is no kernel's port",
            {"tensor": "edge"},
        )
    ]
