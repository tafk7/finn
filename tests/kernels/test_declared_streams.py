# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces: kernels reference them, and each stream sees its users.

Every stream owns its refusals; a stream with one user is a boundary of the
composite and presents its ``port`` name, by the boundary rule; a stream's
tensor is anchored in the composite, and deriving it from a user is refused as
a dependency cycle.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    EvaluationError,
    Members,
    Param,
    Rejected,
    Space,
    Unresolved,
    Users,
    design_space,
    default_semantics,
    derived,
    inspection,
    view,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.build import ModuleBuildRequirements
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.dataflow.tensor import TENSOR, ScalarEncoding, Tensor
from finn.kernels.rom import RomKernel
from finn.dataflow.traversal import LevelEnd, BeatSequence, vector_major
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.streams import (
    CONNECTION,
    MODULE,
    PORT,
    Stream,
    boundary_contract,
    netlist,
)

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)
VECTOR = Tensor((4,), INT4)


class Constants(Space):
    """Two constant vectors streamed to two outputs; each stream's tensor is supplied."""

    first_tensor: Tensor = Param(semantics=TENSOR)
    second_tensor: Tensor = Param(semantics=TENSOR)
    # Each stream has only its producer: it is a boundary, named by its port.
    first = Stream(tensor=first_tensor, port="out0_V")
    second = Stream(tensor=second_tensor, port="out1_V")

    first_source = RomKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=(1, 2, 3, 4),
        output_stream=first,
    )
    second_source = RomKernel(
        dtype=DataType["INT4"],
        form=PRODUCED,
        contents=(5, 6, 7, -8),
        output_stream=second,
    )
    modules = Members(MODULE)
    streams = Members(CONNECTION)

    @view(
        semantics=default_semantics(ModuleBuildRequirements),
        requires=(modules, streams),
    )
    def build(self) -> ModuleBuildRequirements:
        composed = netlist(
            self.modules,
            self.streams,
            module="constants",
            producer=ProducerIdentity("test.constants", "1"),
        )
        assert not isinstance(composed, Rejected)
        return composed.requirements


def constants(first=VECTOR, second=VECTOR):
    point = design_space(Constants(first_tensor=first, second_tensor=second))
    return point.with_choices(
        point.first_source.field(RomKernel.rom_style).change("auto"),
        point.second_source.field(RomKernel.rom_style).change("distributed"),
    )


def test_matching_streams_compose_into_one_module():
    built = constants().build
    names = {port.name for port in built.abi.ports}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names


def test_each_stream_owns_its_refusal_and_independent_refusals_are_all_visible():
    # A source traversing four elements cannot carry an eight-element tensor.
    wide = Tensor((8,), INT4)
    point = constants(first=wide, second=wide)
    assessment = point.inspect(Constants.build)
    results = assessment.constraints.results
    assert isinstance(results["first.connection"], Rejected)
    assert isinstance(results["second.connection"], Rejected)
    refusal = assessment.accepted_result
    assert isinstance(refusal, Rejected)
    assert {f.owner for f in refusal.findings} == {"first.well_formed", "second.well_formed"}
    assert {f.code for f in refusal.findings} == {"stream-tensor"}
    # One stream refusing leaves the other stream's connection accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.first.query(Stream.connection), Rejected)
    assert isinstance(mixed.second.query(Stream.connection), Available)


def test_explain_shows_per_stream_and_per_member_evidence():
    point = constants()
    evidence = inspection.explain(point, Constants.build)
    visited = {node.declaration.key for node in evidence.nodes}
    assert {
        "first.connection",
        "second.connection",
        "first.well_formed",
        "first.compatible",
        "first.ends",
        "first_source.output.contract",
        "first_source.build_requirements",
        "modules",
        "streams",
    } <= visited


def test_a_stream_waits_for_its_own_endpoints_only():
    point = design_space(Constants(first_tensor=VECTOR, second_tensor=VECTOR))
    point = point.with_choices(point.first_source.field(RomKernel.rom_style).change("auto"))
    # The ROM choice feeds only the module, not either stream's contracts.
    assert isinstance(point.first.query(Stream.connection), Available)
    assert isinstance(point.second.query(Stream.connection), Available)
    assert isinstance(point.query(Constants.build), Unresolved)
    # A stream sees its users by declaration name and by the input that references it.
    (end,) = point.first.users
    assert (end.node, end.member) == ("first_source.output", "stream")
    assert end.value.transport.endpoint is Endpoint.INITIATOR  # the source produces
    connection = point.first.connection
    assert (connection.source_owner, connection.sink_owner) == ("first_source.output", None)
    assert connection.sink.transport.name == "out0_V"


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
        edge = Stream(tensor=Tensor((2, 4), INT4), port="in0_V")
        reader = Replaying(input_stream=edge)

    ends = design_space(Receiver()).edge.endpoints
    assert ends.source_owner is None and ends.sink_owner == "reader"
    # Each row once, no marker: the replay and the frame are the receiver's to realize.
    assert ends.source.form == vector_major((2, 4), 2)
    assert ends.source.transport.name == "in0_V" and not ends.source.rules
    assert ends.sink.form == vector_major((2, 4), 2).replayed(3, inner_beats=2)


def test_two_producers_on_one_stream_are_refused_by_the_stream():
    class Clash(Space):
        tensor: Tensor = Param(semantics=TENSOR)
        shared = Stream(tensor=tensor, port="out0_V")
        a = RomKernel(
            dtype=DataType["INT4"], form=PRODUCED, contents=(1, 2, 3, 4), output_stream=shared
        )
        b = RomKernel(
            dtype=DataType["INT4"], form=PRODUCED, contents=(1, 2, 3, 4), output_stream=shared
        )

    point = design_space(Clash(tensor=VECTOR))
    refused = point.shared.query(Stream.connection)
    assert isinstance(refused, Rejected)
    assert {f.code for f in refused.findings} == {"stream-users"}
    assert "a.output.stream, b.output.stream" in refused.findings[0].message


def test_a_boundary_stream_needs_its_port_name():
    class Unnamed(Space):
        tensor: Tensor = Param(semantics=TENSOR)
        out = Stream(tensor=tensor)
        source = RomKernel(
            dtype=DataType["INT4"], form=PRODUCED, contents=(1, 2, 3, 4), output_stream=out
        )

    waiting = design_space(Unnamed(tensor=VECTOR)).out.query(Stream.connection)
    assert isinstance(waiting, Unresolved)
    assert {f.owner for f in waiting.findings} == {"out.port"}


# -- the anchoring rule: a stream's tensor must not depend on its users ----------------


class ProducerTensorStream(Space):
    """A stream that derives its tensor from its producer's contract: not anchored."""

    ends = Users(PORT)

    @derived(semantics=TENSOR)
    def tensor(self) -> Tensor:
        (end,) = self.ends
        return Tensor(end.value.form.shape, end.value.element)


class TensorReadingProducer(Space):
    """Builds its port contract from the stream's tensor, as every kernel does."""

    output_stream: ProducerTensorStream = Param()
    source = RomKernel(dtype=DataType["INT4"], form=PRODUCED, contents=(1, 2, 3, 4))

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
