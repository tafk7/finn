# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declared streams: topology rules, boundary ports, and the optional FIFO slot."""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Param, Rejected, Space, Subspace, derived, inspection, view
from finn.kernels.artifacts.build import ModuleBuildRequirements
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.physical.forms import vector_major
from finn.kernels.streams import (
    STREAM_SPEC,
    Stream,
    StreamSpec,
    TopInput,
    TopOutput,
    assemble_streams,
)
from finn.core.space import default_semantics
from finn.core.space.errors import EvaluationError

INT4 = ScalarEncoding(DataType["INT4"])


class Passthrough(Space):
    """A constant vector streamed straight to the top output."""

    @derived(semantics=STREAM_SPEC)
    def spec(self) -> StreamSpec:
        return StreamSpec(INT4, vector_major((4,), 2))

    values = Stream(spec)
    source = Subspace(
        CyclicDelivery,
        dtype=DataType["INT4"],
        form=vector_major((4,), 2),
        values=(1, 2, 3, 4),
        output_stream=values.spec,
    )
    sink = Subspace(TopOutput, name="out0_V", input_stream=values.spec)

    @view(semantics=default_semantics(ModuleBuildRequirements))
    def build(self) -> ModuleBuildRequirements | Rejected:
        return assemble_streams(
            self, module="passthrough", producer=ProducerIdentity("test", "1")
        ).requirements


def test_an_unbuffered_stream_has_no_transport_choice_and_assembles_directly():
    point = Passthrough()
    assert [choice.key for choice in inspection.choices(point)] == []
    requirements = point.with_choices(point.source.field(CyclicDelivery.rom_style).change("auto"))
    built = requirements.build()
    names = {port.name for port in built.abi.ports}
    assert {"ap_clk", "ap_rst_n", "out0_V"} <= names


def test_a_stream_needs_exactly_one_producer_and_one_consumer():
    class FanOut(Passthrough):
        second = Subspace(TopOutput, name="out1_V", input_stream=Passthrough.values.spec)

    point = FanOut()
    point = point.with_choices(point.source.field(CyclicDelivery.rom_style).change("auto"))
    with pytest.raises(EvaluationError, match="fan-out"):
        point.build()


def test_ports_must_bind_declared_streams():
    class Loose(Space):
        spec = Param(STREAM_SPEC)
        sink = Subspace(TopOutput, name="out0_V", input_stream=spec)

        @view(semantics=default_semantics(ModuleBuildRequirements))
        def build(self) -> ModuleBuildRequirements:
            return assemble_streams(
                self, module="loose", producer=ProducerIdentity("test", "1")
            ).requirements

    point = Loose(spec=StreamSpec(INT4, vector_major((4,), 2)))
    with pytest.raises(EvaluationError, match="declared stream"):
        point.build()


def test_boundary_ports_are_axis_and_byte_aligned():
    source = TopInput(name="in0_V", output_stream=StreamSpec(INT4, vector_major((3,), 3)))
    contract = source.component().ports["output_stream"]
    assert contract.transport.data_width == 16
    assert contract.payload_bits == 12
