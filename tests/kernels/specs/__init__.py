# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One spec per kernel (``KernelSpec``): its conformance cases, the numeric sweeps' cases
that place it, its unit tests, where its computational reference lives, and its lean
covering points and refused side, drawn from its design space (decision KT10;
``tests/kernels/test_specs.py`` checks them offline, ``test_conformance.py``
simulates the cases, ``tests/kernel_ops/test_parity.py`` the KernelOps' parity).
"""

from __future__ import annotations

from kernels.specs import (
    dotp,
    eltwise,
    matmul,
    memstream,
    pool,
    stages,
    thresholding,
    transpose,
)
from kernels.specs.base import Case, KernelSpec

SPECS: tuple[KernelSpec, ...] = (
    dotp.PACKED,
    dotp.INT8,
    thresholding.SPEC,
    eltwise.SPEC,
    transpose.SPEC,
    memstream.SPEC,
    stages.FIFO,
    stages.VPC,
    stages.INPUT_GEN,
    matmul.SPEC,
    pool.SPEC,
)


def conformance_cases() -> dict[str, Case]:
    """Every spec's conformance cases, by name; a name two specs state is refused."""
    found: dict[str, Case] = {}
    for spec in SPECS:
        twice = sorted(set(found) & set(spec.cases))
        if twice:
            raise ValueError(f"{spec.name} states conformance cases another spec states: {twice}")
        found |= spec.cases
    return found


__all__ = ["SPECS", "conformance_cases"]
