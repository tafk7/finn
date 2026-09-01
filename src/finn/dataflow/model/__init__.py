# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative frontend for constructing ordinary dataflow design-space specs."""

from finn.dataflow.model.compiler import compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    Domain,
    Input,
    Problem,
    Readiness,
    Space,
    Use,
    constraint,
    derived,
    divisors_of,
    domain,
    finite,
    reject,
    unresolved,
)

__all__ = [
    "AuthoringError",
    "Constraint",
    "ConstraintGroup",
    "Decision",
    "Derived",
    "Domain",
    "Input",
    "Problem",
    "Readiness",
    "Space",
    "Use",
    "constraint",
    "compile_space",
    "derived",
    "divisors_of",
    "domain",
    "finite",
    "reject",
    "unresolved",
]
