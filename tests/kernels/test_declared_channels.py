# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels as ordinary Spaces: each takes the ends of the ports on it as facts.

Every channel owns its refusals; a channel with one end is a boundary of the
root and presents its ``port`` name, by the boundary rule; a channel's tensor
is anchored where it is declared, and deriving it from an end is refused as a
dependency cycle.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    EvaluationError,
    Param,
    Rejected,
    Space,
    Unresolved,
    design_space,
    inspection,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import PORT_END, Kernel, PortEnd
from finn.kernels.channels import Channel, boundary_contract, end_of
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.transport import AxisBeat, StreamContract
from kernels.helpers import FULL_DSP48E2, Root, view_of, with_direct_transports

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)
VECTOR = Tensor((4,), INT4)


class Constants(Root):
    """Two constant vectors streamed to two outputs; each channel's tensor is supplied."""

    first_tensor: Tensor = Param()
    second_tensor: Tensor = Param()
    first_end = end_of("first_source", "output_end")
    second_end = end_of("second_source", "output_end")
    # Each channel has only its producer: it is a boundary, named by its port.
    first = Channel(tensor=first_tensor, port="out0_V", producer=first_end, platform=FULL_DSP48E2)
    second = Channel(
        tensor=second_tensor, port="out1_V", producer=second_end, platform=FULL_DSP48E2
    )

    first_source = MemStreamKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=(1, 2, 3, 4),
        output_tensor=first_tensor,
        platform=FULL_DSP48E2,
    )
    second_source = MemStreamKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=(5, 6, 7, -8),
        output_tensor=second_tensor,
        platform=FULL_DSP48E2,
    )


def constants(first=VECTOR, second=VECTOR):
    point = with_direct_transports(
        design_space(Constants(first_tensor=first, second_tensor=second))
    )
    return point.with_choices(
        point.first_source.field(MemStreamKernel.ram_style).change("auto"),
        point.first_source.field(MemStreamKernel.pumped_memory).change(False),
        point.second_source.field(MemStreamKernel.ram_style).change("distributed"),
        point.second_source.field(MemStreamKernel.pumped_memory).change(False),
    )


def test_matching_channels_compose_into_one_module():
    built = constants().module
    names = {port.name for port in built.abi.pins}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names


def test_each_channel_owns_its_refusal_and_independent_refusals_are_all_visible():
    # A source traversing four elements cannot carry an eight-element tensor.
    wide = Tensor((8,), INT4)
    point = constants(first=wide, second=wide)
    assessment = point.inspect(Kernel.module)
    results = assessment.constraints.results
    assert isinstance(results["first.netlist"], Rejected)
    assert isinstance(results["second.netlist"], Rejected)
    refusal = assessment.accepted_result
    assert isinstance(refusal, Rejected)
    assert {f.owner for f in refusal.findings} == {"first.well_formed", "second.well_formed"}
    assert {f.code for f in refusal.findings} == {"channel-tensor"}
    # One channel refusing leaves the other channel's netlist accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.first.query(Channel.netlist), Rejected)
    assert isinstance(mixed.second.query(Channel.netlist), Available)


def test_explain_shows_per_channel_and_per_member_evidence():
    point = constants()
    evidence = inspection.explain(point, Kernel.module)
    visited = {node.declaration.key for node in evidence.nodes}
    assert {
        "first.netlist",
        "second.netlist",
        "first.well_formed",
        "first.compatible",
        "first.ends",
        "first_source.output.contract",
        "first_source.module",
        "netlists",
    } <= visited


def test_a_channel_waits_for_its_own_endpoints_only():
    point = with_direct_transports(
        design_space(Constants(first_tensor=VECTOR, second_tensor=VECTOR))
    )
    point = point.with_choices(
        point.first_source.field(MemStreamKernel.ram_style).change("auto"),
        point.first_source.field(MemStreamKernel.pumped_memory).change(False),
    )
    # The ROM choice feeds only the module, not either channel's contracts.
    assert isinstance(point.first.query(Channel.netlist), Available)
    assert isinstance(point.second.query(Channel.netlist), Available)
    assert isinstance(point.query(Kernel.module), Unresolved)
    # A channel takes its ends as facts, each labelled from the root.
    end = point.first.producer
    assert end.label == "first_source.output"
    assert end.contract.transport.endpoint is Endpoint.INITIATOR  # the source produces
    ends = point.first.endpoints
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
    class Receiver(Space):
        reader_end = end_of("reader", "end")
        edge = Channel(
            tensor=Tensor((2, 4), INT4), port="in0_V", consumer=reader_end, platform=FULL_DSP48E2
        )
        reader = Replaying()

    ends = design_space(Receiver()).edge.endpoints
    assert ends.source_owner is None and ends.sink_owner == "reader.port"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert ends.source.form == vector_major((2, 4), 2)
    assert ends.source.transport.name == "in0_V" and not ends.source.rules
    assert ends.sink.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_two_producers_on_one_channel_are_refused_by_the_channel():
    class Clash(Space):
        tensor: Tensor = Param()
        a_end = end_of("a", "output_end")
        b_end = end_of("b", "output_end")
        # b's end given as the consumer is a producer's all the same.
        shared = Channel(
            tensor=tensor, port="out0_V", producer=a_end, consumer=b_end, platform=FULL_DSP48E2
        )
        a = MemStreamKernel(
            dtype=DataType["INT4"],
            form=PRODUCED,
            contents=(1, 2, 3, 4),
            output_tensor=tensor,
            platform=FULL_DSP48E2,
        )
        b = MemStreamKernel(
            dtype=DataType["INT4"],
            form=PRODUCED,
            contents=(1, 2, 3, 4),
            output_tensor=tensor,
            platform=FULL_DSP48E2,
        )

    point = with_direct_transports(design_space(Clash(tensor=VECTOR)))
    refused = point.shared.query(Channel.netlist)
    assert isinstance(refused, Rejected)
    assert {f.code for f in refused.findings} == {"channel-users"}
    assert "a.output, b.output" in refused.findings[0].message


def test_a_boundary_channel_needs_its_port_name():
    class Unnamed(Space):
        tensor: Tensor = Param()
        source_end = end_of("source", "output_end")
        out = Channel(tensor=tensor, producer=source_end, platform=FULL_DSP48E2)
        source = MemStreamKernel(
            dtype=DataType["INT4"],
            form=PRODUCED,
            contents=(1, 2, 3, 4),
            output_tensor=tensor,
            platform=FULL_DSP48E2,
        )

    refused = with_direct_transports(design_space(Unnamed(tensor=VECTOR))).out.query(
        Channel.netlist
    )
    assert isinstance(refused, Rejected)
    assert {(f.code, f.owner) for f in refused.findings} == {("channel-boundary", "out.endpoints")}


# -- the anchoring rule: a channel's tensor must not depend on its ends -----------------


class ProducerTensorChannel(Space):
    """A channel that derives its tensor from its producer's end: not anchored."""

    producer: PortEnd = Param(semantics=PORT_END)

    @view
    def tensor(self) -> Tensor:
        contract = self.producer.contract
        return Tensor(contract.form.shape, contract.element)


class TensorReadingProducer(Space):
    """Builds its end from the tensor it is given, as every kernel does."""

    output_tensor: Tensor = Param()
    source = MemStreamKernel(
        platform=FULL_DSP48E2, dtype=DataType["INT4"], form=PRODUCED, contents=(1, 2, 3, 4)
    )

    @view(semantics=PORT_END)
    def end(self) -> PortEnd:
        contract = self.source.output.contract
        tensor = self.output_tensor  # the tensor shapes the port
        form = vector_major(tensor.shape, 2)
        return PortEnd("port", StreamContract(contract.transport, tensor.element, form))


def test_a_tensor_derived_from_its_ends_is_refused_with_the_cycle_path():
    class Unanchored(Space):
        edge_tensor = view_of("edge", "tensor")
        producer_end = end_of("producer", "end")
        edge = ProducerTensorChannel(producer=producer_end)
        producer = TensorReadingProducer(output_tensor=edge_tensor)

    point = design_space(Unanchored())
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        point.edge.tensor
    path = str(caught.value)
    # The cycle, in evaluation order: the tensor reads the end, the end reads the tensor.
    assert "edge.tensor" in path and "producer_end" in path and "producer.end" in path
