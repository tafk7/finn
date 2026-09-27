# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces: kernels reference them, and each stream sees its users.

Every stream owns its refusals; a stream with one user is a boundary of the
composite and presents its ``port`` name; a stream's spec is anchored in the
composite, and deriving it from a user is refused as a dependency cycle.
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
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.physical.forms import vector_major
from finn.kernels.streams import (
    CONNECTION,
    MODULE,
    PORTS,
    PORTS_SEMANTICS,
    STREAM_SPEC,
    Flow,
    Ports,
    Stream,
    StreamSpec,
    boundary_contract,
    netlist,
    produces,
)

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)


class Constants(Space):
    """Two constant vectors streamed to two outputs; each output's order is supplied."""

    first_spec: StreamSpec = Param(semantics=STREAM_SPEC)
    second_spec: StreamSpec = Param(semantics=STREAM_SPEC)
    # Each stream has only its producer: it is a boundary, named by its port.
    first = Stream(spec=first_spec, port="out0_V")
    second = Stream(spec=second_spec, port="out1_V")

    first_source = CyclicDelivery(
        dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4), output_stream=first
    )
    second_source = CyclicDelivery(
        dtype=DataType["INT4"], form=PRODUCED, values=(5, 6, 7, -8), output_stream=second
    )
    modules = Members(MODULE)
    streams = Members(CONNECTION)

    @view(semantics=default_semantics(ModuleBuildRequirements), requires=(modules, streams))
    def build(self) -> ModuleBuildRequirements:
        composed = netlist(
            self.modules,
            self.streams,
            module="constants",
            producer=ProducerIdentity("test.constants", "1"),
        )
        assert not isinstance(composed, Rejected)
        return composed.requirements


def constants(first=PRODUCED, second=PRODUCED):
    point = design_space(
        Constants(first_spec=StreamSpec(INT4, first), second_spec=StreamSpec(INT4, second))
    )
    return point.with_choices(
        point.first_source.field(CyclicDelivery.rom_style).change("auto"),
        point.second_source.field(CyclicDelivery.rom_style).change("distributed"),
    )


def test_matching_streams_compose_into_one_module():
    built = constants().build()
    names = {port.name for port in built.abi.ports}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names


def test_each_stream_owns_its_refusal_and_independent_refusals_are_all_visible():
    # Four lanes cannot be fed by a two-lane source: both streams refuse.
    wide = vector_major((4,), 4)
    point = constants(first=wide, second=wide)
    assessment = point.build.inspect()
    results = assessment.constraints.results
    assert isinstance(results["first.connection"], Rejected)
    assert isinstance(results["second.connection"], Rejected)
    refusal = assessment.accepted_result
    assert isinstance(refusal, Rejected)
    assert {f.owner for f in refusal.findings} == {"first.compatible", "second.compatible"}
    # One stream refusing leaves the other stream's connection accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.first.connection.query(), Rejected)
    assert isinstance(mixed.second.connection.query(), Available)


def test_explain_shows_per_stream_and_per_member_evidence():
    point = constants()
    evidence = inspection.explain(point, Constants.build)
    visited = {node.declaration.key for node in evidence.nodes}
    assert {
        "first.connection",
        "second.connection",
        "first.compatible",
        "first.ends",
        "first_source.ports",
        "first_source.build_requirements",
        "modules",
        "streams",
    } <= visited


def test_a_stream_waits_for_its_own_endpoints_only():
    point = design_space(
        Constants(first_spec=StreamSpec(INT4, PRODUCED), second_spec=StreamSpec(INT4, PRODUCED))
    )
    point = point.with_choices(point.first_source.field(CyclicDelivery.rom_style).change("auto"))
    # The ROM choice feeds only the module, not either stream's contracts.
    assert isinstance(point.first.connection.query(), Available)
    assert isinstance(point.second.connection.query(), Available)
    assert isinstance(point.build.query(), Unresolved)
    # A stream sees its users by declaration name and by the input that references it.
    (end,) = point.first.ends
    assert (end.node, end.member) == ("first_source", "output_stream")
    assert end.value["output_stream"].flow is Flow.OUT
    connection = point.first.connection()
    assert (connection.source_owner, connection.sink_owner) == ("first_source", None)
    assert connection.sink.transport.name == "out0_V"


def test_boundary_ports_are_axis_and_byte_aligned():
    contract = boundary_contract("in0_V", StreamSpec(INT4, vector_major((3,), 3)), Endpoint.TARGET)
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12


def test_two_producers_on_one_stream_are_refused_by_the_stream():
    class Clash(Space):
        spec: StreamSpec = Param(semantics=STREAM_SPEC)
        shared = Stream(spec=spec, port="out0_V")
        a = CyclicDelivery(
            dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4), output_stream=shared
        )
        b = CyclicDelivery(
            dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4), output_stream=shared
        )

    point = design_space(Clash(spec=StreamSpec(INT4, PRODUCED)))
    refused = point.shared.connection.query()
    assert isinstance(refused, Rejected)
    assert {f.code for f in refused.findings} == {"stream-users"}
    assert "a.output_stream, b.output_stream" in refused.findings[0].message


def test_a_boundary_stream_needs_its_port_name():
    class Unnamed(Space):
        spec: StreamSpec = Param(semantics=STREAM_SPEC)
        out = Stream(spec=spec)
        source = CyclicDelivery(
            dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4), output_stream=out
        )

    waiting = design_space(Unnamed(spec=StreamSpec(INT4, PRODUCED))).out.connection.query()
    assert isinstance(waiting, Unresolved)
    assert {f.owner for f in waiting.findings} == {"out.port"}


# -- the anchoring rule: a stream's spec must not depend on its users ------------------


class ProducerSpecStream(Space):
    """A stream that derives its spec from its producer's contract: not anchored."""

    ends = Users(PORTS)

    @derived(semantics=STREAM_SPEC)
    def spec(self) -> StreamSpec:
        (end,) = self.ends
        contract = end.value[end.member].contract
        return StreamSpec(contract.element, contract.form)


class SpecReadingProducer(Space):
    """Builds its port contract from the stream's spec, as dotp does."""

    output_stream: ProducerSpecStream = Param()
    source = CyclicDelivery(dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4))

    @view(semantics=PORTS_SEMANTICS)
    def ports(self) -> Ports:
        contract = self.source.output()
        spec = self.output_stream.spec  # the stream's spec shapes the port
        return Ports.of(
            output_stream=produces(type(contract)(contract.transport, spec.element, spec.form))
        )

    exports = {PORTS: ports}


def test_a_spec_derived_from_its_users_is_refused_with_the_cycle_path():
    class Unanchored(Space):
        edge = ProducerSpecStream()
        producer = SpecReadingProducer(output_stream=edge)

    point = design_space(Unanchored())
    with pytest.raises(EvaluationError, match="dependency cycle") as caught:
        point.edge.spec
    path = str(caught.value)
    # The cycle, in evaluation order: the spec reads the users, a user's ports read the spec.
    assert "edge.spec" in path and "edge.ends" in path and "producer.ports" in path
