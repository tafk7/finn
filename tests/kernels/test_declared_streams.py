# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declared streams: explicit endpoints, per-stream refusals, boundary ports, composition."""

from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    ConstraintGroup,
    Param,
    Rejected,
    Space,
    Subspace,
    Unresolved,
    default_semantics,
    inspection,
    view,
)
from finn.kernels.artifacts.build import ModuleBuildRequirements
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.physical.forms import vector_major
from finn.kernels.streams import (
    STREAM_SPEC,
    Stream,
    StreamLink,
    StreamSpec,
    TopInput,
    TopOutput,
    compose,
    connected,
)

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)


class Constants(Space):
    """Two constant vectors streamed to two top outputs; each output's order is supplied."""

    first_spec = Param(STREAM_SPEC)
    second_spec = Param(STREAM_SPEC)

    first_source = Subspace(
        CyclicDelivery, dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4)
    )
    second_source = Subspace(
        CyclicDelivery, dtype=DataType["INT4"], form=PRODUCED, values=(5, 6, 7, -8)
    )
    first_sink = Subspace(TopOutput, name="out0_V", input_stream=first_spec)
    second_sink = Subspace(TopOutput, name="out1_V", input_stream=second_spec)

    first = Stream(
        first_spec,
        source=("u_first", first_source.accepted(CyclicDelivery.output)),
        sink=("out0_V", first_sink.accepted(TopOutput.port)),
    )
    second = Stream(
        second_spec,
        source=("u_second", second_source.accepted(CyclicDelivery.output)),
        sink=("out1_V", second_sink.accepted(TopOutput.port)),
    )
    first_connected = connected(first)
    second_connected = connected(second)
    streams = ConstraintGroup(first_connected, second_connected)

    @view(semantics=default_semantics(ModuleBuildRequirements), constraints=(streams,))
    def build(self) -> ModuleBuildRequirements:
        return compose(
            module="constants",
            producer=ProducerIdentity("test.constants", "1"),
            instances={
                "u_first": self.first_source.build_requirements(),
                "u_second": self.second_source.build_requirements(),
            },
            connections=(self.first.connection(), self.second.connection()),
        ).requirements


def constants(first=PRODUCED, second=PRODUCED):
    point = Constants(first_spec=StreamSpec(INT4, first), second_spec=StreamSpec(INT4, second))
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
    assert isinstance(results["first_connected"], Rejected)
    assert isinstance(results["second_connected"], Rejected)
    owners = {
        cause.owner
        for result in results.values()
        for finding in result.findings
        for cause in (finding, *finding.causes)
    }
    assert {"first.compatible", "second.compatible"} <= owners
    assert not isinstance(assessment.accepted_result, Available)
    # One stream refusing leaves the other stream's connection accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.first.connection.query(), Rejected)
    assert isinstance(mixed.second.connection.query(), Available)


def test_explain_shows_per_stream_evidence_not_one_opaque_callback():
    point = constants()
    evidence = inspection.explain(point, Constants.build)
    visited = {node.declaration.key for node in evidence.nodes}
    assert {"first.connection", "second.connection", "first.compatible"} <= visited


def test_a_stream_waits_for_its_own_endpoints_only():
    point = Constants(first_spec=StreamSpec(INT4, PRODUCED), second_spec=StreamSpec(INT4, PRODUCED))
    point = point.with_choices(point.first_source.field(CyclicDelivery.rom_style).change("auto"))
    # The ROM choice feeds only the module, not either stream's contracts.
    assert isinstance(point.first.connection.query(), Available)
    assert isinstance(point.second.connection.query(), Available)
    assert isinstance(point.build.query(), Unresolved)
    assert point.first.query(StreamLink.name) == Available("first")


def test_boundary_ports_are_axis_and_byte_aligned():
    source = TopInput(name="in0_V", output_stream=StreamSpec(INT4, vector_major((3,), 3)))
    contract = source.port()
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12
