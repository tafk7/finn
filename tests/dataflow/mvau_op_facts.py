# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The operation-owned facts a compute Kernel reads, for pool-only tests.

Tests that exercise the compute pool alone still need the operation context the
pool reads: its problem fields, and now its ``effective_narrow_weights``
property.  Assembling that context in one place keeps the pool tests honest
about what a Kernel actually depends on.
"""

from __future__ import annotations

from finn.dataflow.design import DesignSpaceSpec, ProblemSchema
from finn.dataflow.ops.mvau.problem import (
    MVAU_PROBLEM_SPEC,
    MVAUProblemPaths,
)


def compute_pool_context() -> DesignSpaceSpec:
    """Every operation-owned declaration the compute pool depends on.

    Everything the operation declares except the source description, which
    describes tensors a pool-only test has no graph to name.
    """

    fields = tuple(
        field
        for field in MVAU_PROBLEM_SPEC.problem_schema.fields
        if field.path != MVAUProblemPaths.SOURCE_DESCRIPTION
    )
    return DesignSpaceSpec(ProblemSchema(fields), properties=MVAU_PROBLEM_SPEC.properties)


__all__ = ["compute_pool_context"]
