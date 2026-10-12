# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph preparation: the phase that takes a Brevitas export to the graph the kernel
path converts.

``finn.transformation.prepare.phase`` holds the sub-phases (P0 import, P2
quantization lowering, P1 the network's inputs and outputs, P3 streamlining, P4
topology), each over the transforms that own its work, and their options
(``GraphPreparation``, which a build states by its preparation steps' settings). The
builder runs them as ``phase_prepare_model``
(``finn.builder.build_dataflow_steps.build_dataflow_phases``).

The phase reads no target: nothing here imports ``finn.platform`` or the KernelOps
(``tests/layering.py``).
"""

from finn.transformation.prepare.phase import (
    SUB_PHASES,
    GraphPreparation,
    census,
    prepared,
    reference,
)

__all__ = [
    "SUB_PHASES",
    "GraphPreparation",
    "census",
    "prepared",
    "reference",
]
