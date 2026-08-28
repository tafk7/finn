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
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM_SPEC,
    MVAUProblemPaths,
)

#: Facts outside ``problem.mvau.*`` and ``problem.target.*`` that a compute
#: Kernel nonetheless reaches, directly or through an operation property.
_ALSO_READ = (MVAUProblemPaths.RUNTIME_WRITABLE,)


def compute_pool_context() -> DesignSpaceSpec:
    """Every operation-owned declaration the compute pool depends on."""

    fields = tuple(
        field
        for field in MVAU_PROBLEM_SPEC.problem_schema.fields
        if (
            str(field.path).startswith(("problem.mvau.", "problem.target."))
            or field.path in _ALSO_READ
        )
        and field.path != MVAUProblemPaths.SOURCE_DESCRIPTION
    )
    properties = tuple(
        item
        for item in MVAU_PROBLEM_SPEC.properties
        if item.path == MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS
    )
    return DesignSpaceSpec(ProblemSchema(fields), properties=properties)


__all__ = ["compute_pool_context"]
