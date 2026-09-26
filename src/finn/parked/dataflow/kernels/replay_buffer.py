# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Region adapter for the physical replay component.

Native interface and source authorship belongs to ``kernels.streaming``.
The direct physical assembly path does not import this analysis adapter.

The buffer presents each activation row once per neuron fold.  That expansion is
what the monolithic MVAU Region performed implicitly by scheduling its activation
input across ``nf``; naming it as its own Region makes it a composable unit and
leaves the dot-product half with nothing but arithmetic.

It owns no decision at all.  ``LEN``, ``REP``, and ``W`` are the folding restated
in the buffer's own vocabulary, derived from facts its Kernel supplies -- a
buffer that picked its own depth would be picking a fold.  It is kept even at one
neuron fold, where it is an identity: eliding the physical buffer is a choice for
this Kernel's own realization to make, not a reason for the Region to disappear.

The core predates FINN's AXI naming and takes ``clk`` with a synchronous,
active-high ``rst``, so the ABI says so rather than smoothing it over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from finn.kernels.artifacts.abi import (
    ComponentABI,
)
from finn.kernels.artifacts.build import ModuleABIRequirements, ScalarTable
from finn.kernels.physical.layout import PeriodicLast
from finn.parked.dataflow.model.physical.interface import KernelStreamBinding
from finn.parked.dataflow.model.physical.interface import low_fields_binding
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.space.declarations import Input, derived
from finn.parked.dataflow.model.kernel import Kernel
from finn.parked.dataflow.model.logical.authoring import RegionDeclaration
from finn.parked.dataflow.model.physical.authoring import ModuleParameter
from finn.dataflow.model.logical.region import DataflowRegion, NumericElementType, element_width
from finn.dataflow.kernels.matmul.regions import construct_activation_replay_region
from finn.kernels.streaming import REPLAY_BUFFER_SOURCES, replay_buffer_requirements
from finn.kernels.artifacts.requirements import FixedModuleName

FINNLIB_ROOT = "finnlib"
FINNLIB_SOURCES = ("rtl/infra/replay_buffer.sv",)


class ReplayBufferKernel(Kernel):
    """Present each activation row once per neuron fold."""

    id = "replay_buffer"
    version = "2"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    pe = Input(int)
    simd = Input(int)

    region = RegionDeclaration(
        family="mvau.activation_replay",
        version="2",
        construct=construct_activation_replay_region,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_element_type=activation_type,
        pe=pe,
        simd=simd,
    )

    @derived(int, matrix_width=matrix_width, simd=simd)
    def sequence_length(*, matrix_width: int, simd: int) -> int:
        return matrix_width // simd

    @derived(int, matrix_height=matrix_height, pe=pe)
    def replay_count(*, matrix_height: int, pe: int) -> int:
        return matrix_height // pe

    @derived(int, activation_type=activation_type, simd=simd)
    def data_width(*, activation_type: NumericElementType, simd: int) -> int:
        return simd * element_width(activation_type)

    LEN = ModuleParameter(sequence_length)
    REP = ModuleParameter(replay_count)
    W = ModuleParameter(data_width)

    sources = REPLAY_BUFFER_SOURCES

    @classmethod
    def local_stream_bindings(
        cls,
        *,
        region: DataflowRegion,
        parameters: ScalarTable,
        abi: ModuleABIRequirements,
    ) -> tuple[KernelStreamBinding, ...]:
        period = cast(int, dict(parameters)["LEN"])
        return (
            low_fields_binding(
                region=region, abi=abi, region_port_id="activation_in", abi_bus_id="in0"
            ),
            low_fields_binding(
                region=region,
                abi=abi,
                region_port_id="activation_out",
                abi_bus_id="out0",
                framing=PeriodicLast("tlast", period, period - 1),
            ),
        )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        requirements = replay_buffer_requirements(
            word_bits=cast(int, parameters["W"]),
            sequence_length=cast(int, parameters["LEN"]),
            replay_count=cast(int, parameters["REP"]),
        )
        abi = requirements.abi
        assert isinstance(abi.entry_point, FixedModuleName)
        return ComponentABI(abi.entry_point.value, abi.ports, abi.parameters, abi.clock_alignments)


__all__ = [
    "FINNLIB_ROOT",
    "FINNLIB_SOURCES",
    "ReplayBufferKernel",
]
