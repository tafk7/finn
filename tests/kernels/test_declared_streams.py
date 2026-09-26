# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as nets: promoted boundary ports, per-net refusals, the netlist fold."""

from qonnx.core.datatype import DataType

from finn.core.space import (
    Available,
    Fold,
    Net,
    Param,
    Port,
    Rejected,
    Space,
    Subspace,
    Unresolved,
    inspection,
)
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.physical.forms import vector_major
from finn.kernels.streams import (
    NETLIST,
    STREAM,
    STREAM_SPEC,
    StreamLink,
    StreamSpec,
    boundary_contract,
)

INT4 = ScalarEncoding(DataType["INT4"])
PRODUCED = vector_major((4,), 2)


class Constants(Space):
    """Two constant vectors streamed to two outputs; each output's order is supplied."""

    first_spec = Param(STREAM_SPEC)
    second_spec = Param(STREAM_SPEC)

    first = Net(StreamLink)
    second = Net(StreamLink)
    out0_V = Port(STREAM, "out", net=first, carry=first_spec)
    out1_V = Port(STREAM, "out", net=second, carry=second_spec)
    first_source = Subspace(
        CyclicDelivery,
        dtype=DataType["INT4"],
        form=PRODUCED,
        values=(1, 2, 3, 4),
        output_stream=first,
    )
    second_source = Subspace(
        CyclicDelivery,
        dtype=DataType["INT4"],
        form=PRODUCED,
        values=(5, 6, 7, -8),
        output_stream=second,
    )
    build = Fold(NETLIST, module="constants", producer=ProducerIdentity("test.constants", "1"))


def constants(first=PRODUCED, second=PRODUCED):
    point = Constants(first_spec=StreamSpec(INT4, first), second_spec=StreamSpec(INT4, second))
    return point.with_choices(
        point.first_source.field(CyclicDelivery.rom_style).change("auto"),
        point.second_source.field(CyclicDelivery.rom_style).change("distributed"),
    )


def test_matching_streams_compose_into_one_module():
    built = constants().build()
    names = {port.name for port in built.requirements.abi.ports}
    assert {"ap_clk", "ap_rst_n", "out0_V", "out1_V"} <= names
    # Instance names are the declaration names of the nodes.
    assert [i.instance_id for i in built.structure.instances] == [
        "u_first_source",
        "u_second_source",
    ]


def test_each_stream_owns_its_refusal_and_independent_refusals_are_all_visible():
    # Four lanes cannot be fed by a two-lane source: both streams refuse.
    wide = vector_major((4,), 4)
    point = constants(first=wide, second=wide)
    assessment = point.build.inspect()
    results = assessment.constraints.results
    assert isinstance(results["first.connection"], Rejected)
    assert isinstance(results["second.connection"], Rejected)
    owners = {
        cause.owner
        for result in results.values()
        if isinstance(result, Rejected)
        for finding in result.findings
        for cause in (finding, *finding.causes)
    }
    assert {"first.compatible", "second.compatible"} <= owners
    assert isinstance(assessment.accepted_result, Rejected)
    assert {f.owner for f in assessment.accepted_result.findings} >= {
        "first.compatible",
        "second.compatible",
    }
    # One stream refusing leaves the other stream's connection accepted.
    mixed = constants(first=wide)
    assert isinstance(mixed.first.connection.query(), Rejected)
    assert isinstance(mixed.second.connection.query(), Available)


def test_explain_shows_per_net_evidence_not_one_opaque_callback():
    point = constants()
    evidence = inspection.explain(point, Constants.build)
    visited = {node.declaration.key for node in evidence.nodes}
    assert {
        "first.connection",
        "second.connection",
        "first.compatible",
        "first.$carried",
        "first.$ends",
        "first_source.build_requirements",
        "$topology.netlist",
    } <= visited


def test_a_stream_waits_for_its_own_endpoints_only():
    point = Constants(first_spec=StreamSpec(INT4, PRODUCED), second_spec=StreamSpec(INT4, PRODUCED))
    point = point.with_choices(point.first_source.field(CyclicDelivery.rom_style).change("auto"))
    # The ROM choice feeds only the module, not either stream's contracts.
    assert isinstance(point.first.connection.query(), Available)
    assert isinstance(point.second.connection.query(), Available)
    assert isinstance(point.build.query(), Unresolved)
    ends = point.first.ends
    assert [(end.node, end.port, end.direction) for end in ends] == [
        (None, "out0_V", "in"),
        ("first_source", "output_stream", "out"),
    ]


def test_boundary_ports_are_axis_and_byte_aligned():
    contract = boundary_contract("in0_V", StreamSpec(INT4, vector_major((3,), 3)), Endpoint.TARGET)
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12


def test_a_stream_kernel_configures_standalone_with_its_ports_unconnected():
    point = CyclicDelivery(dtype=DataType["INT4"], form=PRODUCED, values=(1, 2, 3, 4))
    assert isinstance(point.query(CyclicDelivery.output_stream), Unresolved)
    assert point.output().payload_bits == 8
