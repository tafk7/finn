# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU elaboration-origin construction."""

from __future__ import annotations

from finn.dataflow.authoring.composition import PhysicalCompositionProvenance
from finn.dataflow.design import QualifiedPath
from finn.dataflow.ops.mvau.physical import MVAUElaborationOrigin


def mvau_elaboration_origin(
    context: PhysicalCompositionProvenance,
) -> MVAUElaborationOrigin:
    """Construct the immutable identity of one production design realization."""

    return MVAUElaborationOrigin(
        context.family_version,
        context.problem_fingerprint,
        tuple((QualifiedPath(path), value) for path, value in context.assignments),
        tuple(origin.kernel_id for _name, origin in context.kernel_origins),
    )


__all__ = ["mvau_elaboration_origin"]
