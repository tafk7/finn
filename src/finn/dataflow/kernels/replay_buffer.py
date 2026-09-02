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

from dataclasses import dataclass
from finn.dataflow.authoring import (
    Covers,
    Imported,
    KernelInput,
    Parameter,
    RegionClaim,
    Sources,
)
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.computation import ACTIVATION_REPLAY_COMPUTATION, ComputationContract
from finn.dataflow.design import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.kernels import (
    Kernel,
    PhysicalComponent,
    scalar_parameters,
)
from finn.dataflow.region import DataflowRegion

#: FINN's half of the composition, relative to the FINN root, in compile order.
FINN_ROOT = "finn"
FINN_SOURCES = (
    "finn-rtllib/mvu/mvu_pkg.sv",
    "finn-rtllib/mvu/replay_buffer.sv",
)

#: The physical module this Kernel instantiates.
REPLAY_BUFFER_MODULE = "finn-rtllib.mvu.replay_buffer"


@dataclass(frozen=True)
class ReplayBufferInputs:
    """Operation-neutral traffic and folding facts consumed by replay hardware."""

    role: str
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    length: Ref[int]
    repetitions: Ref[int]
    width: Ref[int]


class ReplayBufferKernel(Kernel):
    """Present each activation row once per neuron fold."""

    id = "replay_buffer"
    version = "1"
    uses_class_authoring = True

    covered_region = Imported(DATAFLOW_REGION_SEMANTICS, stable_name="region")
    computation = Imported(ComputationContract)
    length = Imported(int)
    repetitions = Imported(int)
    width = Imported(int)
    coverage = Covers(
        RegionClaim(
            KernelInput("role"),
            covered_region,
            computation,
            ACTIVATION_REPLAY_COMPUTATION,
            "the compact-to-expanded activation sequence",
        )
    )
    source_files = Sources(FINN_ROOT, *FINN_SOURCES)
    length_parameter = Parameter("LEN", length)
    repetitions_parameter = Parameter("REP", repetitions)
    width_parameter = Parameter("W", width)

    @classmethod
    def elaborate(cls, kernel: Kernel) -> tuple[PhysicalComponent, ...]:
        """One ``replay_buffer`` instance."""

        return (
            PhysicalComponent(
                "replay_buffer",
                REPLAY_BUFFER_MODULE,
                scalar_parameters(dict(kernel.parameters)),
            ),
        )


__all__ = [
    "FINN_ROOT",
    "FINN_SOURCES",
    "REPLAY_BUFFER_MODULE",
    "ReplayBufferInputs",
    "ReplayBufferKernel",
]
