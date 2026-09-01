# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU generated-source requirements and staging boundary."""

from finn.dataflow.mvau.hardware.composition import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    decomposed_top_module_name,
    staged_layout,
    write_decomposed_artifact,
)

__all__ = [
    "MVAUDecomposedArtifactRequirements",
    "build_decomposed_artifact_requirements",
    "decomposed_top_module_name",
    "staged_layout",
    "write_decomposed_artifact",
]
