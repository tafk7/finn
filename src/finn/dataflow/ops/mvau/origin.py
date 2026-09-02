# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Production MVAU elaboration-origin construction."""

from __future__ import annotations

from finn.dataflow.authoring.realization import DesignRealization
from finn.dataflow.ops.mvau.associations import MVAUResolvedDataflowOp
from finn.dataflow.ops.mvau.physical import MVAUElaborationOrigin
from finn.dataflow.authoring.compiler import CompiledDataflowOperation
from finn.dataflow.op import dataflow_problem_fingerprint
from finn.dataflow.ops.mvau.contracts import MVAU_DATAFLOW_OP_FAMILY_VERSION


def mvau_elaboration_origin(
    resolved: MVAUResolvedDataflowOp,
    realization: DesignRealization,
) -> MVAUElaborationOrigin:
    """Construct the immutable identity of one production design realization."""

    compiled = resolved.compiled
    family_version = (
        getattr(compiled.owner, "dataflow_family_version")()
        if isinstance(compiled, CompiledDataflowOperation)
        else MVAU_DATAFLOW_OP_FAMILY_VERSION
    )
    return MVAUElaborationOrigin(
        family_version,
        dataflow_problem_fingerprint(resolved.point.problem),
        tuple(sorted(resolved.point.assignments.items(), key=lambda item: item[0])),
        tuple(realization.kernel(name).kernel_id for name in realization.kernels),
    )


__all__ = ["mvau_elaboration_origin"]
