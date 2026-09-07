# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative one-Region model of FinnLib's ``replay_buffer``.

The buffer presents each activation row once per neuron fold.  That expansion is
what the monolithic MVAU Region performed implicitly by scheduling its activation
input across ``nf``; naming it as its own Region makes it a composable unit and
leaves the dot-product half with nothing but arithmetic.

It owns no decision at all.  ``LEN``, ``REP``, and ``W`` are the folding restated
in the buffer's own vocabulary, derived from facts its Design supplies -- a
buffer that picked its own depth would be picking a fold.  It is kept even at one
neuron fold, where it is an identity: eliding the physical buffer is a choice for
this Kernel's own realization to make, not a reason for the Region to disappear.

The core predates FINN's AXI naming and takes ``clk`` with an active-high
``rst``, so the ABI says so rather than smoothing it over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.space.declarations import Input, derived
from finn.dataflow.kernels.kernel import Kernel, ModuleParameter, RegionDeclaration
from finn.dataflow.model.region import NumericElementType, element_width
from finn.dataflow.ops.mvau.regions import construct_activation_replay_region

FINNLIB_ROOT = "finnlib"
FINNLIB_SOURCES = ("rtl/infra/replay_buffer.sv",)


class ReplayBufferKernel(Kernel):
    """Present each activation row once per neuron fold."""

    id = "replay_buffer"
    version = "1"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    pe = Input(int)
    simd = Input(int)

    region = RegionDeclaration(
        family="mvau.activation_replay",
        version="1",
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

    sources = (
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[0],
            provides=("module:replay_buffer",),
        ),
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        width = cast(int, parameters["W"])
        return ComponentABI(
            "replay_buffer",
            (
                Signal("clk", Direction.IN, 1, Clock(Free())),
                Signal("rst", Direction.IN, 1, Reset(active_low=False)),
                Bus(
                    "in0",
                    StandardProtocol.AXIS,
                    (
                        Member("tdata", "idat", width),
                        Member("tvalid", "ivld"),
                        Member("tready", "irdy"),
                    ),
                    endpoint=Endpoint.TARGET,
                    associated_clock="clk",
                    associated_reset="rst",
                ),
                Bus(
                    "out0",
                    StandardProtocol.AXIS,
                    (
                        Member("tdata", "odat", width),
                        Member("tvalid", "ovld"),
                        Member("tready", "ordy"),
                        Member("tlast", "olast"),
                    ),
                    endpoint=Endpoint.INITIATOR,
                    associated_clock="clk",
                    associated_reset="rst",
                ),
                # `ofin` marks the end of the whole replayed run rather than of
                # one sequence, so it is not an AXI-Stream member of `out0`.
                Signal("ofin", Direction.OUT, 1),
            ),
            tuple((name, str(value)) for name, value in parameters.items()),
        )


__all__ = [
    "FINNLIB_ROOT",
    "FINNLIB_SOURCES",
    "ReplayBufferKernel",
]
