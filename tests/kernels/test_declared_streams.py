# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as relation nodes: located ends, per-stream refusals, members wired."""

from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Located,
    Members,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    configure,
    default_semantics,
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
    STREAM_SPEC,
    StreamLink,
    StreamSpec,
    boundary_contract,
    netlist,
)

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)


class Constants(Space):
    """Two constant vectors streamed to two outputs; each output's order is supplied."""

    first_spec: Param[StreamSpec] = Param(STREAM_SPEC)
    second_spec: Param[StreamSpec] = Param(STREAM_SPEC)
    out0_V = View(first_spec)
    out1_V = View(second_spec)

    first_source = CyclicDelivery(dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4))
    second_source = CyclicDelivery(dtype=DataType["INT4"], form=PRODUCED, values=(5, 6, 7, -8))
    # Plain references into the links' Param(Located) ends: a child's view, or one
    # of the composite's own members.
    first = StreamLink(spec=first_spec, source=first_source.output, sink=out0_V)
    second = StreamLink(spec=second_spec, source=second_source.output, sink=out1_V)
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
    point = configure(
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
        "first.source",
        "first_source.build_requirements",
        "modules",
        "streams",
    } <= visited


def test_a_stream_waits_for_its_own_endpoints_only():
    point = configure(
        Constants(first_spec=StreamSpec(INT4, PRODUCED), second_spec=StreamSpec(INT4, PRODUCED))
    )
    point = point.with_choices(point.first_source.field(CyclicDelivery.rom_style).change("auto"))
    # The ROM choice feeds only the module, not either stream's contracts.
    assert isinstance(point.first.connection.query(), Available)
    assert isinstance(point.second.connection.query(), Available)
    assert isinstance(point.build.query(), Unresolved)
    # Ends know where they are, by declaration name.
    assert point.first.source.node == "first_source"
    assert point.first.sink == Located(None, "out0_V", StreamSpec(INT4, PRODUCED))


def test_boundary_ports_are_axis_and_byte_aligned():
    contract = boundary_contract("in0_V", StreamSpec(INT4, vector_major((3,), 3)), Endpoint.TARGET)
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12
