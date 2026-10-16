# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""pwpolyf: GELU, SiLU, Sigmoid or Tanh, elementwise in FLOAT32, on DSP58. No KernelOp
reaches it yet, so its reference is test-side: ``activation``, the function itself,
which its conformance case and its numeric sweep (``kernels.sweeps.pwpolyf_numeric``)
both state.

The hardware approximates the function by polynomials, and gives the function's limit
for a magnitude of 8 and above, so its FLOAT32 results are compared within
``TOLERANCE[function]`` (absolute), not bit for bit. Those tolerances are provisional:
each is the largest error of the RTL's own polynomials and clamps against the function,
as measured over binary32 inputs, rounded up. Whether an approximating kernel states an
error bound or is checked to a tolerance is not decided.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.pwpolyf import PwPolyKernel
from support.helpers import FULL_DSP48E2, FULL_DSP58
from support.kernel_specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed

#: Where the hardware gives the function's limit in place of its polynomial: |x| >= 8.
CLAMP = 8.0

#: The absolute tolerance each function's results are compared within (provisional): the
#: RTL's largest error against the function over binary32 inputs, rounded up to one
#: significant digit. SiLU's and Sigmoid's are their clamps' (at |x| = 8 the function is
#: 8 / (1 + e**8) and 1 / (1 + e**8) from its limit: 2.7e-3 and 3.4e-4); GELU's and
#: Tanh's their polynomials' near zero, a quadratic over |x| < 0.25 (5.8e-5 and 2.0e-3).
TOLERANCE = {
    "gelu": 6e-5,
    "silu": 3e-3,
    "sigmoid": 4e-4,
    "tanh": 3e-3,
}

_erf = np.vectorize(math.erf, otypes=[np.float64])


def activation(function: str, x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """``function`` of each element of ``x``, in float64, of ``x`` as the hardware reads
    it (binary32); at an infinity, the function's limit. NaN stays NaN."""
    values = np.asarray(x).astype(np.float32).astype(np.float64)
    found: npt.NDArray[np.float64]
    with np.errstate(over="ignore", invalid="ignore"):
        if function == "gelu":
            found = values * 0.5 * (1.0 + _erf(values / math.sqrt(2.0)))
        elif function == "silu":
            found = values / (1.0 + np.exp(-values))
        elif function == "sigmoid":
            found = 1.0 / (1.0 + np.exp(-values))
        elif function == "tanh":
            found = np.tanh(values)
        else:
            raise ValueError(f"pwpolyf computes no {function!r}")
    if function in ("gelu", "silu"):  # x times a factor tending to 0: inf * 0 above
        found = np.where(np.isneginf(values), 0.0, found)
    return found


def clamped(function: str, x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """What the hardware states it gives for a magnitude of ``CLAMP`` and above, exactly:
    GELU and SiLU pass a positive input through and give 0 for a negative one; Sigmoid
    gives 1 and 0; Tanh 1 and -1. NaN where ``x`` is not clamped."""
    values = np.asarray(x).astype(np.float32).astype(np.float64)
    above = values if function in ("gelu", "silu") else np.ones_like(values)
    below = -1.0 if function == "tanh" else 0.0
    found: npt.NDArray[np.float64] = np.where(
        values >= CLAMP, above, np.where(values <= -CLAMP, below, np.nan)
    )
    return found


def case(function: str, shape: tuple[int, ...], dtype: str = "FLOAT32") -> dict[str, Any]:
    """``function`` of a ``dtype`` tensor of ``shape``, on DSP58."""
    return dict(
        space_type=PwPolyKernel,
        inputs={"input_channel": Tensor(shape, ScalarEncoding(DataType[dtype]))},
        outputs={"output_channel": shape},
        reference=lambda input_channel: {"output_channel": activation(function, input_channel)},
        facts={"platform": FULL_DSP58, "function": function},
        tolerance=TOLERANCE.get(function),
    )


def pwpolyf_gelu() -> dict[str, Any]:
    """GELU of rows of eight over two leading axes: PE 1, a divisor and 8 (a float input
    has no adapter sample)."""
    return case("gelu", (2, 3, 8))


SPEC = KernelSpec(
    kernel=PwPolyKernel,
    reference=TEST_SIDE,
    cases={"pwpolyf_gelu": pwpolyf_gelu},
    space="pwpolyf_gelu",
    probes=(
        Probe(
            "a function the RTL does not compute",
            lambda: placed(case("relu", (2, 8))),
            frozenset({"pwpolyf-function"}),
        ),
        Probe(
            "a platform without DSP58: no DSPFP32",
            lambda: placed(case("gelu", (2, 8)), platform=FULL_DSP48E2),
            frozenset({"pwpolyf-target"}),
        ),
        Probe(
            "a float other than FLOAT32: the lanes are binary32",
            lambda: placed(case("gelu", (2, 8), "FLOAT16")),
            frozenset({"dtype-family"}),
        ),
        Probe(
            "an integer input",
            lambda: placed(case("gelu", (2, 8), "INT8")),
            frozenset({"dtype-family"}),
        ),
    ),
    sweeps=(SweepCases("kernels.sweeps.pwpolyf_numeric", ("CASES",), ("sweep-pwpolyf",)),),
    unit=("tests/kernels/test_pwpolyf.py",),
)

__all__ = [
    "CLAMP",
    "SPEC",
    "TOLERANCE",
    "activation",
    "case",
    "clamped",
    "pwpolyf_gelu",
]
