# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed AXIS authoring agrees with logical streams and existing lowering values."""

import pytest
from dataclasses import replace
from qonnx.core.datatype import DataType

from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.build import FixedModuleName, ModuleABIRequirements
from finn.dataflow.kernels.matmul.regions import construct_dot_product_region
from finn.kernels.physical.axi_stream import AxiStream
from finn.parked.dataflow.model.physical.axi_stream_binding import axi_stream_from_port, bind_axi_stream
from finn.parked.dataflow.model.physical.interface import (
    low_fields_binding,
    validate_kernel_stream_bindings,
)
from finn.kernels.physical.layout import (
    PeriodicLast,
    UnusedBitPolicy,
)


@pytest.fixture
def region():
    return construct_dot_product_region(
        repetitions=1,
        matrix_width=8,
        matrix_height=2,
        activation_element_type=DataType["INT3"],
        weight_element_type=DataType["INT3"],
        output_element_type=DataType["INT3"],
        pe=1,
        simd=2,
    )


def test_logical_port_derives_pins_and_existing_composition_bindings(region):
    specifications = (
        ("activation", "s_axis_input", PeriodicLast("tlast", 4, 3)),
        ("weight", "s_axis_weights", None),
        ("output", "m_axis_output", None),
    )
    streams = tuple(
        axi_stream_from_port(name, region, port, last=framing is not None)
        for port, name, framing in specifications
    )
    abi = ModuleABIRequirements(FixedModuleName("example"), tuple(s.bus() for s in streams), ())
    bindings = tuple(
        bind_axi_stream(stream, region, port, framing=framing)
        for stream, (port, _, framing) in zip(streams, specifications)
    )
    validate_kernel_stream_bindings(region, abi, bindings)
    for binding, (port, name, framing) in zip(bindings, specifications):
        previous = low_fields_binding(
            region=region,
            abi=abi,
            region_port_id=port,
            abi_bus_id=name,
            framing=framing,
        )
        assert binding.region_port_id == previous.region_port_id
        assert binding.abi_bus_id == previous.abi_bus_id
        assert binding.payload.fields == previous.payload.fields
        assert binding.framing == previous.framing
    # A producer may leave high padding unspecified; a receiver must ignore it.
    output = bindings[-1]
    bad_output = replace(
        output,
        payload=replace(
            output.payload,
            unused=(replace(output.payload.unused[0], policy=UnusedBitPolicy.IGNORE_ON_RECEIVE),),
        ),
    )
    with pytest.raises(ValueError, match="padding policy"):
        validate_kernel_stream_bindings(region, abi, (*bindings[:-1], bad_output))


@pytest.mark.parametrize(
    "dtype,lanes,endpoint",
    [
        ("UINT3", 2, Endpoint.TARGET),  # Same physical width, different value interpretation.
        ("INT3", 1, Endpoint.TARGET),
        ("INT3", 2, Endpoint.INITIATOR),
    ],
)
def test_binding_rejects_logical_disagreement(region, dtype, lanes, endpoint):
    stream = AxiStream("data", DataType[dtype], lanes, endpoint=endpoint)
    with pytest.raises(ValueError, match="disagrees with logical port"):
        bind_axi_stream(stream, region, "activation")


@pytest.mark.parametrize(
    "last,framing",
    [(True, None), (False, PeriodicLast("tlast", 4, 3)), (True, PeriodicLast("tlast", 3, 2))],
)
def test_binding_requires_consistent_explicit_framing(region, last, framing):
    stream = axi_stream_from_port("data", region, "activation", last=last)
    with pytest.raises(ValueError, match="framing"):
        bind_axi_stream(stream, region, "activation", framing=framing)
