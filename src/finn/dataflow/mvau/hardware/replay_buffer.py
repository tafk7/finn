# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``ReplayBufferKernel``: FINN's ``replay_buffer`` covering the replay Region.

Three parameters, no choices, and no target coverage.  ``LEN``, ``REP`` and
``W`` are the folding restated in the buffer's own vocabulary, so they are
derived from what the Region was built from and never decided.  A buffer that
picked its own depth would be picking a fold.

It is retained even at one neuron fold, where it is an identity.  Eliding the
physical buffer is a decision for this Kernel to make in its own elaboration if
it ever becomes worthwhile; it is not a reason for the Region to disappear.

The core predates FINN's AXI naming and takes ``clk`` with an active-high
``rst``.  That is a fact about this hardware, so it is recorded here rather
than smoothed over by a convention imposed on every core.
"""

from __future__ import annotations

from typing import cast

from finn.dataflow.authoring.scope import Ref
from finn.dataflow.hardware import (
    HardwareDesign,
    HardwareKernel,
    KernelBinding,
    PhysicalComponent,
    scalar_parameters,
)
from finn.dataflow.mvau.computation import ACTIVATION_REPLAY_COMPUTATION
from finn.dataflow.mvau.hardware.inputs import ActivationReplayHardwareInputs
from finn.dataflow.region import element_width

#: FINN's half of the composition, relative to the FINN root, in compile order.
FINN_ROOT = "finn"
FINN_SOURCES = (
    "finn-rtllib/mvu/mvu_pkg.sv",
    "finn-rtllib/mvu/replay_buffer.sv",
)

#: The physical module this Kernel instantiates.
REPLAY_BUFFER_MODULE = "finn-rtllib.mvu.replay_buffer"


class ReplayBufferKernel(HardwareKernel):
    """Present each activation row once per neuron fold."""

    id = "replay_buffer"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[ActivationReplayHardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "replay",
            region=facts.region,
            computation=facts.computation,
            implements=ACTIVATION_REPLAY_COMPUTATION,
            description="the compact-to-expanded activation sequence",
        )
        design.source(FINN_ROOT, *FINN_SOURCES)

        length = design.derived(
            "buffer_length",
            int,
            dependencies={"matrix_width": facts.matrix_width, "simd": facts.simd},
            evaluate=lambda matrix_width, simd: matrix_width // simd,
        )
        repetitions = design.derived(
            "buffer_repetitions",
            int,
            dependencies={"matrix_height": facts.matrix_height, "pe": facts.pe},
            evaluate=lambda matrix_height, pe: matrix_height // pe,
        )
        width = design.derived(
            "buffer_width",
            int,
            dependencies={
                "activation_element_type": facts.activation_element_type,
                "simd": facts.simd,
            },
            evaluate=lambda activation_element_type, simd: (
                simd * element_width(activation_element_type)
            ),
        )
        design.parameter("LEN", cast("Ref[object]", length))
        design.parameter("REP", cast("Ref[object]", repetitions))
        design.parameter("W", cast("Ref[object]", width))

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        """One ``replay_buffer`` instance."""

        return (
            PhysicalComponent(
                "replay",
                REPLAY_BUFFER_MODULE,
                scalar_parameters(dict(binding.parameters)),
            ),
        )


__all__ = [
    "FINN_ROOT",
    "FINN_SOURCES",
    "REPLAY_BUFFER_MODULE",
    "ReplayBufferKernel",
]
