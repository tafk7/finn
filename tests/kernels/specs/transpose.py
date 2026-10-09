# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""transpose: rows in, columns out. A reorder, so its test-side reference is the identity
(no KernelOp reaches it)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from finn.kernels.transpose import TransposeKernel
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed
from kernels.specs.thresholding import tensor

MATRICES = (2, 6, 4)


def transpose() -> dict[str, Any]:
    """Rows in, columns out: the same tensor in another order, so the reference is identity.

    SIMD 2 divides both sides; 3 and 6 divide I alone, so input beats span rows,
    with FinnLib's write rotation off (gcd(J, SIMD) = 1) and on (2). The stalled
    input (and a ``vpc`` feeding the adapter sample) lets the output catch up with
    the input, which FinnLib's ``inner_shuffle`` survives only with its page-guard fix.
    """
    return dict(
        space_type=TransposeKernel,
        inputs={"input_channel": tensor(MATRICES, "INT4")},
        outputs={"output_channel": MATRICES},
        reference=lambda input_channel: {"output_channel": input_channel},
        factors=tuple({"simd": simd} for simd in (2, 3, 6)),
        choices={"ram_style": "auto"},
        facts={"platform": FULL_DSP48E2},
    )


SPEC = KernelSpec(
    kernel=TransposeKernel,
    reference=TEST_SIDE,
    cases={"transpose": transpose},
    space="transpose",
    probes=(
        Probe(
            "UltraRAM on a platform without it",
            lambda: placed(transpose(), platform=replace(FULL_DSP48E2, uram=False)),
            frozenset({"uram-absent"}),
            {"kernel.ram_style": "ultra"},
        ),
    ),
    sweeps=(SweepCases("kernels.sweeps.adapter_numeric", ("TRANSPOSES",), ("sweep-adapters",)),),
    unit=("tests/kernels/test_transpose.py",),
)

__all__ = ["MATRICES", "SPEC", "transpose"]
