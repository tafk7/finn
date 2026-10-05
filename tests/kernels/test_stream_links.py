# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stream's netlist: its stages below it, and each checked hop resolved into wires.

A user's end belongs to the kernel whose port it is, beside the stream
(``^compute``); a stage sits at its label below the stream
(``adapter.input_gen.input_gen``, ``transport.fifo.buffer``); a boundary end is
the root's own pins (``None``).
"""

from qonnx.core.datatype import DataType

from finn.core.space import Param, Rejected, Space, design_space, view
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.module import Leaf
from finn.kernels.base import PORT
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.transport import STREAM_CONTRACT, AxiStream, StreamContract
from kernels.helpers import FULL_DSP48E2, with_adapter_memories, with_direct_transports

INT3, INT8 = DataType["INT3"], DataType["INT8"]


class Placed(Space):
    """dotp between boundary streams: three rows of four, four outputs, PE = SIMD = 2."""

    x = Channel(tensor=Tensor((3, 4), ScalarEncoding(INT3)), port="in0_V", platform=FULL_DSP48E2)
    w = Channel(platform=FULL_DSP48E2, tensor=Tensor((4, 4), ScalarEncoding(INT3)), port="in1_V")
    y = Channel(tensor=Tensor((3, 4), ScalarEncoding(INT8)), port="out0_V", platform=FULL_DSP48E2)
    compute = PackedDotpKernel(
        result_dtype=INT8,
        x_channel=x,
        w_channel=w,
        y_channel=y,
        platform=FULL_DSP48E2,
    )


def placed(**transport: object) -> Placed:
    choices = {
        "compute.pe": 2,
        "compute.simd": 2,
        "compute.compute_pumping": False,
        "compute.reducer": "tree",
        **transport,
    }
    return with_adapter_memories(commit(with_direct_transports(design_space(Placed())), choices))


def test_an_adapted_stream_places_its_stage_and_wires_each_hop() -> None:
    point = placed(**{"w.transport": "direct"})
    fragment = point.x.netlist
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
    # A direct stream is one hop, and only a boundary stream presents a bus.
    (direct,) = point.y.netlist.links
    assert (direct.source.instance, direct.sink.instance) == ("^compute", None)
    assert [bus.name for bus in point.y.boundary_bus] == ["out0_V"]


def test_a_buffered_stream_places_its_fifo_below_its_transport() -> None:
    point = placed(
        **{
            "w.transport": "fifo",
            "w.transport.fifo.buffer.depth": 4,
            "w.transport.fifo.buffer.ram_style": "auto",
        }
    )
    fragment = point.w.netlist
    assert [label for label, _ in fragment.instances] == ["transport.fifo.buffer"]
    assert [(item.source.instance, item.sink.instance) for item in fragment.links] == [
        (None, "transport.fifo.buffer"),
        ("transport.fifo.buffer", "^compute"),
    ]


def test_a_boundary_no_port_names_is_refused() -> None:
    """A weight stream with one user and no port is a boundary nothing may cross: only an
    ONNX input or output of a partition is one (D4)."""

    class Unnamed(Space):
        x = Channel(
            tensor=Tensor((3, 4), ScalarEncoding(INT3)), port="in0_V", platform=FULL_DSP48E2
        )
        w = Channel(tensor=Tensor((4, 4), ScalarEncoding(INT3)), platform=FULL_DSP48E2)
        y = Channel(
            platform=FULL_DSP48E2, tensor=Tensor((3, 4), ScalarEncoding(INT8)), port="out0_V"
        )
        compute = PackedDotpKernel(
            result_dtype=INT8,
            x_channel=x,
            w_channel=w,
            y_channel=y,
            platform=FULL_DSP48E2,
        )

    folding = {
        "compute.pe": 2,
        "compute.simd": 2,
        "compute.compute_pumping": False,
        "compute.reducer": "tree",
    }
    refused = commit(design_space(Unnamed()), folding).w.query(Channel.endpoints)
    assert isinstance(refused, Rejected)
    assert {(item.code, item.owner) for item in refused.findings} == {
        ("channel-boundary", "w.endpoints")
    }
    # The same root with the port named is accepted.
    assert not isinstance(commit(design_space(Placed()), folding).w.endpoints, Rejected)


class Reader(Space):
    """A consumer that exports its end itself, not through a kernel's port."""

    input_channel: Channel = Param()

    @view(semantics=STREAM_CONTRACT)
    def port(self) -> StreamContract:
        stream = AxiStream("s_axis", INT3, 4, endpoint=Endpoint.TARGET)
        transport = stream.native(clock="ap_clk", reset="ap_rst_n")
        return StreamContract(transport, ScalarEncoding(INT3), vector_major((3, 4), 4))

    exports = {PORT: {input_channel: port}}


def test_a_user_that_is_no_kernels_port_has_no_netlist() -> None:
    class Bare(Space):
        edge = Channel(
            platform=FULL_DSP48E2, tensor=Tensor((3, 4), ScalarEncoding(INT3)), port="in0_V"
        )
        reader = Reader(input_channel=edge)

    point = with_direct_transports(design_space(Bare()))
    assert point.edge.users[0].node == "reader"
    refused = point.edge.query(Channel.netlist)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"channel-user"}
