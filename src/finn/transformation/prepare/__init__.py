# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph preparation: the phase that takes a Brevitas export to the graph the kernel
path converts, and its checkpoint.

``finn.transformation.prepare.phase`` holds the sub-phases (P0 import, P2
quantization lowering, P1 the network's inputs and outputs, P3 streamlining, P4
topology, P6 containers), each over the transforms that own its work, and their
options (``GraphPreparation``, a KernelBuildConfig's ``preparation``);
``finn.transformation.prepare.containers`` is P6's rule, which widens the integer
regions whose bounds pass float32's exact integers;
``finn.transformation.prepare.checkpoint`` checks what the phase leaves (P7): its
structure, its annotations' soundness and its containers' exactness, by bound, and
declares where the phase computes otherwise than the export (``DEVIATIONS``). The
equivalence with the export executes both graphs and is the harness's
(``finn.harness.preparation``). The builder runs them as ``phase_graph_preparation``
(``finn.builder.kernel_build_steps``).

The phase reads no target: nothing here imports ``finn.platform`` or the KernelOps
(``tests/layering.py``); the checkpoint is given the KernelOps' anchors.
"""

from finn.transformation.prepare.checkpoint import (
    BOUND_RULES,
    DEVIATIONS,
    VALUE_DEVIATIONS,
    PreparationRefused,
    checkpoint,
    summary,
)
from finn.transformation.prepare.phase import (
    RECIPE_TRANSFORMS,
    SUB_PHASES,
    GraphPreparation,
    census,
    prepared,
    reference,
)

__all__ = [
    "BOUND_RULES",
    "DEVIATIONS",
    "RECIPE_TRANSFORMS",
    "SUB_PHASES",
    "VALUE_DEVIATIONS",
    "GraphPreparation",
    "PreparationRefused",
    "census",
    "checkpoint",
    "prepared",
    "reference",
    "summary",
]
