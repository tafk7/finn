# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces: kernels reference them, and each stream sees its users.

Every stream owns its refusals; a stream with one user is a boundary of the
root and presents its ``port`` name, by the boundary rule; a stream's tensor
is anchored where it is declared, and deriving it from a user is refused as a
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
    Users,
    derived,
    design_space,
    inspection,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd, vector_major
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import PORT, Kernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.streams import Stream, boundary_contract
from finn.kernels.transport import STREAM_CONTRACT, AxiStream, StreamContract
from kernels.helpers import FULL_DSP48E2, Root

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)
VECTOR = Tensor((4,), INT4)


class Constants(Root):
    """Two constant vectors streamed to two outputs; each stream's tensor is supplied."""

    first_tensor: Tensor = Param()
    second_tensor: Tensor = Param()
    # Each stream has only its producer: it is a boundary, named by its port.
    first = Stream(tensor=first_tensor, port="out0_V", platform=FULL_DSP48E2)
    second = Stream(tensor=second_tensor, port="out1_V", platform=FULL_DSP48E2)

    first_source = MemStreamKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=(1, 2, 3, 4),
        output_stream=first,
        platform=FULL_DSP48E2,
    )
    second_source = MemStreamKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=(5, 6, 7, -8),
        output_stream=second,
        platform=FULL_DSP48E2,
    )


def constants(first=VECTOR, second=VECTOR):
    point = design_space(Constants(first_tensor=first, second_tensor=second))
    return point.with_choices(
        point.first_source.field(MemStreamKernel.ram_style).change("auto"),
        point.first_source.field(MemStreamKernel.pumped_memory).change(False),
        point.second_source.field(MemStreamKernel.ram_style).change("distributed"),
        point.second_source.field(MemStreamKernel.pumped_memory).change(False),
    )


def test_matching_streams_compose_into_one_module():
    built = constants().module
    names = {port.name for port in built.pins.ports}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names


def test_each_stream_owns_its_refusal_and_independent_refusals_are_all_visible():
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
    assert {f.code for f in refusal.findings} == {"stream-tensor"}
    # One stream refusing leaves the other stream's netlist accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.first.query(Stream.netlist), Rejected)
    assert isinstance(mixed.second.query(Stream.netlist), Available)


def test_explain_shows_per_stream_and_per_member_evidence():
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


def test_a_stream_waits_for_its_own_endpoints_only():
    point = design_space(Constants(first_tensor=VECTOR, second_tensor=VECTOR))
    point = point.with_choices(
        point.first_source.field(MemStreamKernel.ram_style).change("auto"),
        point.first_source.field(MemStreamKernel.pumped_memory).change(False),
    )
    # The ROM choice feeds only the module, not either stream's contracts.
    assert isinstance(point.first.query(Stream.netlist), Available)
    assert isinstance(point.second.query(Stream.netlist), Available)
    assert isinstance(point.query(Kernel.module), Unresolved)
    # A stream sees its users by declaration name and by the input that references it.
    (end,) = point.first.users
    assert (end.node, end.member) == ("first_source.output", "stream")
    assert end.value.transport.endpoint is Endpoint.INITIATOR  # the source produces
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

    input_stream: Stream = Param()

    @view(semantics=STREAM_CONTRACT)
    def port(self) -> StreamContract:
        stream = AxiStream("s_axis", DataType["INT4"], 2, endpoint=Endpoint.TARGET, last=True)
        transport = stream.native(clock="ap_clk", reset="ap_rst_n")
        form = vector_major((2, 4), 2).replayed(3, inner_beats=2)
        return StreamContract(transport, INT4, form, markers={"s_axis_tlast": LevelEnd(2)})

    exports = {PORT: {input_stream: port}}


def test_a_boundary_presents_its_internal_end_without_the_replay_the_receiver_realizes():
    class Receiver(Space):
        edge = Stream(tensor=Tensor((2, 4), INT4), port="in0_V", platform=FULL_DSP48E2)
        reader = Replaying(input_stream=edge)

    ends = design_space(Receiver()).edge.endpoints
    assert ends.source_owner is None and ends.sink_owner == "reader"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert ends.source.form == vector_major((2, 4), 2)
    assert ends.source.transport.name == "in0_V" and not ends.source.rules
    assert ends.sink.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_two_producers_on_one_stream_are_refused_by_the_stream():
    class Clash(Space):
        tensor: Tensor = Param()
        shared = Stream(tensor=tensor, port="out0_V", platform=FULL_DSP48E2)
        a = MemStreamKernel(
            dtype=DataType["INT4"],
            form=PRODUCED,
            contents=(1, 2, 3, 4),
            output_stream=shared,
            platform=FULL_DSP48E2,
        )
        b = MemStreamKernel(
            dtype=DataType["INT4"],
            form=PRODUCED,
            contents=(1, 2, 3, 4),
            output_stream=shared,
            platform=FULL_DSP48E2,
        )

    point = design_space(Clash(tensor=VECTOR))
    refused = point.shared.query(Stream.netlist)
    assert isinstance(refused, Rejected)
    assert {f.code for f in refused.findings} == {"stream-users"}
    assert "a.output.stream, b.output.stream" in refused.findings[0].message


def test_a_boundary_stream_needs_its_port_name():
    class Unnamed(Space):
        tensor: Tensor = Param()
        out = Stream(tensor=tensor, platform=FULL_DSP48E2)
        source = MemStreamKernel(
            dtype=DataType["INT4"],
            form=PRODUCED,
            contents=(1, 2, 3, 4),
            output_stream=out,
            platform=FULL_DSP48E2,
        )

    refused = design_space(Unnamed(tensor=VECTOR)).out.query(Stream.netlist)
    assert isinstance(refused, Rejected)
    assert {(f.code, f.owner) for f in refused.findings} == {("stream-boundary", "out.endpoints")}


# -- the anchoring rule: a stream's tensor must not depend on its users ----------------


class ProducerTensorStream(Space):
    """A stream that derives its tensor from its producer's contract: not anchored."""

    ends = Users(PORT)

    @derived
    def tensor(self) -> Tensor:
        (end,) = self.ends
        return Tensor(end.value.form.shape, end.value.element)


class TensorReadingProducer(Space):
    """Builds its port contract from the stream's tensor, as every kernel does."""

    output_stream: ProducerTensorStream = Param()
    source = MemStreamKernel(
        platform=FULL_DSP48E2, dtype=DataType["INT4"], form=PRODUCED, contents=(1, 2, 3, 4)
    )

    @view(semantics=STREAM_CONTRACT)
    def port(self) -> StreamContract:
        contract = self.source.output.contract
        tensor = self.output_stream.tensor  # the stream's tensor shapes the port
        return StreamContract(contract.transport, tensor.element, vector_major(tensor.shape, 2))

    exports = {PORT: {output_stream: port}}


def test_a_tensor_derived_from_its_users_is_refused_with_the_cycle_path():
    class Unanchored(Space):
        edge = ProducerTensorStream()
        producer = TensorReadingProducer(output_stream=edge)

    point = design_space(Unanchored())
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        point.edge.tensor
    path = str(caught.value)
    # The cycle, in evaluation order: the tensor reads the users, a user's port reads the tensor.
    assert "edge.tensor" in path and "edge.ends" in path and "producer.port" in path
