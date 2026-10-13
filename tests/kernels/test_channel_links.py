# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's netlist: its stages below it, and each checked hop resolved into wires.

An end belongs to the kernel whose port it is, beside the channel
(``^compute``); a stage sits at its label below the channel
(``adapter.input_gen.input_gen``, ``transport.fifo.buffer``); a boundary end is
the root's own pins (``None``).
"""

from qonnx.core.datatype import DataType

from finn.core.space import Rejected, Space, view
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import Leaf
from finn.kernels.base import PORT_END, PortEnd
from finn.kernels.channels import Channel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.transport import AxisBeat, StreamContract
from kernels.helpers import FULL_DSP48E2, choose, with_adapter_memories, with_direct_transports

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
                consumer=Interface("compute", "x_end"),
                platform=FULL_DSP48E2,
            ),
            "w": wire(
                platform=FULL_DSP48E2, tensor=W, consumer=Interface("compute", "w_end"), **named
            ),
            "y": wire(
                tensor=Y,
                port="out0_V",
                producer=Interface("compute", "y_end"),
                platform=FULL_DSP48E2,
            ),
            "compute": PackedDotpKernel(
                result_range=ordinary_integer_bounds(INT8),
                x_tensor=X,
                w_tensor=W,
                y_tensor=Y,
                platform=FULL_DSP48E2,
            ),
        }
    )


FOLDING = {
    "compute.pe": 2,
    "compute.simd": 2,
    "compute.compute_pumping": False,
    "compute.reducer": "tree",
}


def placed(**transport: object) -> Network:
    folded = choose(placing(), FOLDING)
    return with_adapter_memories(with_direct_transports(choose(folded, transport)))


def test_an_adapted_channel_places_its_stage_and_wires_each_hop() -> None:
    point = placed(**{"w.transport": "direct"})
    fragment = point.member("x").netlist
    ((label, leaf),) = fragment.instances
    assert label == "adapter.input_gen.input_gen"
    assert isinstance(leaf, Leaf) and leaf.name == "input_gen"
    into, out = fragment.links
    assert (into.source.instance, into.sink.instance) == (None, label)
    assert (into.source.data, into.sink.data) == ("in0_V_tdata", "idat")
    assert (out.source.instance, out.sink.instance) == (label, "^compute")
    assert out.sink.data == "s_axis_input_tdata" and out.lanes == (0, 1) and out.lane_bits == 3
    # The frame marker: the input_gen's olst bit closing each reduction drives TLAST.
    assert out.markers == (("olst", 1, "s_axis_input_tlast", None),)
    # A direct channel is one hop, and only a boundary channel presents a bus.
    (direct,) = point.member("y").netlist.links
    assert (direct.source.instance, direct.sink.instance) == ("^compute", None)
    assert [bus.name for bus in point.member("y").boundary_bus] == ["out0_V"]


def test_a_buffered_channel_places_its_fifo_below_its_transport() -> None:
    point = placed(
        **{
            "w.transport": "fifo",
            "w.transport.fifo.buffer.depth": 4,
            "w.transport.fifo.buffer.ram_style": "auto",
        }
    )
    fragment = point.member("w").netlist
    assert [label for label, _ in fragment.instances] == ["transport.fifo.buffer"]
    assert [(item.source.instance, item.sink.instance) for item in fragment.links] == [
        (None, "transport.fifo.buffer"),
        ("transport.fifo.buffer", "^compute"),
    ]


def test_a_boundary_no_port_names_is_refused() -> None:
    """A weight channel with one end and no port is a boundary nothing may cross: only an
    ONNX input or output of a partition is one."""

    refused = choose(placing(w_port=None), FOLDING).member("w").query(Channel.endpoints)
    assert isinstance(refused, Rejected)
    assert {(item.code, item.owner) for item in refused.findings} == {
        ("channel-boundary", "endpoints")
    }
    # The same network with the port named is accepted.
    assert not isinstance(choose(placing(), FOLDING).member("w").endpoints, Rejected)


class Reader(Space):
    """A consumer that presents its end itself, not through a kernel's port."""

    @view(semantics=PORT_END)
    def end(self) -> PortEnd:
        beat = AxisBeat("s_axis", INT3, 4, endpoint=Endpoint.TARGET)
        transport = beat.native(clock="ap_clk", reset="ap_rst_n")
        contract = StreamContract(transport, ScalarEncoding(INT3), vector_major((3, 4), 4))
        return PortEnd("", contract)


def test_an_end_that_is_no_kernels_port_has_no_netlist() -> None:
    bare = Network(
        {
            "edge": wire(
                platform=FULL_DSP48E2, tensor=X, port="in0_V", consumer=Interface("reader", "end")
            ),
            "reader": Reader(),
        }
    )
    edge = with_direct_transports(bare).member("edge")
    assert edge.consumer.label == "reader"
    refused = edge.query(Channel.netlist)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"channel-port"}
