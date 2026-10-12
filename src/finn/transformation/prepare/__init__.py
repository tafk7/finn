# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph preparation: the phase that takes a Brevitas export to the graph the kernel
path converts.

``finn.transformation.prepare.phase`` holds the sub-phases (P0 import, P2
quantization lowering, P1 the network's inputs and outputs, P3 streamlining, P4
topology, P6 containers), each over the transforms that own its work, and their
options (``GraphPreparation``, which a build states by its preparation steps' options);
``finn.transformation.prepare.containers`` widens the integer regions whose bounds pass
float32's exact integers. The builder runs them as ``phase_prepare_model``
(``finn.builder.build_dataflow_phases``).

The phase reads no target: nothing here imports ``finn.platform`` or the KernelOps
(``tests/layering.py``).
"""

from finn.transformation.prepare.phase import (
    RECIPE_TRANSFORMS,
    SUB_PHASES,
    GraphPreparation,
    census,
    prepared,
    reference,
)

__all__ = [
    "RECIPE_TRANSFORMS",
    "SUB_PHASES",
    "GraphPreparation",
    "census",
    "prepared",
    "reference",
]
