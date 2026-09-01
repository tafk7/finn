# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU elaboration-origin construction."""

from __future__ import annotations

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.ops.mvau.physical import MVAUElaborationOrigin
from finn.dataflow.ops.mvau.persistence import (
    MVAU_DECLARATION_FAMILY_VERSION,
    mvau_problem_fingerprint,
)
from finn.dataflow.ops.mvau.projection import MVAUResolvedDesign


def mvau_elaboration_origin(
    resolved: MVAUResolvedDesign,
    realization: DesignRealization,
) -> MVAUElaborationOrigin:
    """Construct the immutable identity of one production design realization."""

    return MVAUElaborationOrigin(
        MVAU_DECLARATION_FAMILY_VERSION,
        mvau_problem_fingerprint(resolved.point.problem),
        tuple(sorted(resolved.point.assignments.items(), key=lambda item: item[0])),
        tuple(realization.kernel(name).kernel_id for name in realization.kernels),
    )


__all__ = ["mvau_elaboration_origin"]
