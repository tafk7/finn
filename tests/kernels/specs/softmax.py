# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Softmax: over the last axis, on FinnLib's HLS ``SoftMax``. No KernelOp reaches it yet,
so its reference is test-side: ``softmax``, which its conformance cases and its numeric
sweep (``kernels.sweeps.softmax_numeric``) both state.

The hardware approximates the function, so its FLOAT32 results are compared within
``TOLERANCE`` (absolute), not bit for bit. That tolerance is provisional: it is
upstream FINN's for the same HLS core, and whether an approximating kernel states an
error bound or is checked to a tolerance is not decided.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.softmax import SoftmaxKernel
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed

#: The absolute tolerance its results are compared within (provisional: upstream FINN's
#: for its HLS softmax).
TOLERANCE = 1e-5


def softmax(x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Softmax over the last axis, in float64, of ``x`` as the hardware reads it: each
    value converted to binary32 first (an integer of more than 24 bits rounds). A row whose
    maximum is ``+inf`` shares one evenly among its infinite elements, the others zero:
    the function's limit, and what the hardware computes. A row of ``-inf`` alone, or with
    a NaN, has no value (NaN)."""
    values = np.asarray(x).astype(np.float32).astype(np.float64)
    peak = values.max(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore"):
        powers = np.where(np.isposinf(peak), np.isposinf(values), np.exp(values - peak))
        found: npt.NDArray[np.float64] = powers / powers.sum(axis=-1, keepdims=True)
    return found


def case(shape: tuple[int, ...], dtype: str) -> dict[str, Any]:
    """Softmax of a ``dtype`` tensor of ``shape``, over its last axis."""
    return dict(
        space_type=SoftmaxKernel,
        inputs={"input_channel": Tensor(shape, ScalarEncoding(DataType[dtype]))},
        outputs={"output_channel": shape},
        reference=lambda input_channel: {"output_channel": softmax(input_channel)},
        facts={"platform": FULL_DSP48E2},
        tolerance=TOLERANCE,
    )


def softmax_int8() -> dict[str, Any]:
    """Rows of eight INT8 values over two leading axes: SIMD 1, 4 and 8, and the adapter."""
    return case((2, 3, 8), "INT8")


def softmax_float() -> dict[str, Any]:
    """Rows of six FLOAT32 values: SIMD 1, 3 and 6 (a float input has no adapter sample)."""
    return case((4, 6), "FLOAT32")


SPEC = KernelSpec(
    kernel=SoftmaxKernel,
    reference=TEST_SIDE,
    cases={"softmax_int8": softmax_int8, "softmax_float": softmax_float},
    space="softmax_int8",
    probes=(
        Probe(
            "a float other than FLOAT32: the top's float lanes are binary32",
            lambda: placed(case((2, 8), "FLOAT16")),
            frozenset({"dtype-family"}),
        ),
        Probe(
            "an integer wider than 32 bits",
            lambda: placed(case((2, 8), "INT33")),
            frozenset({"dtype-maximum-bits"}),
        ),
        Probe(
            "a bipolar element: no ordinary integer",
            lambda: placed(case((2, 8), "BIPOLAR")),
            frozenset({"dtype-family"}),
        ),
    ),
    sweeps=(SweepCases("kernels.sweeps.softmax_numeric", ("CASES",), ("sweep-softmax",)),),
    unit=("tests/kernels/test_softmax.py",),
)

__all__ = ["SPEC", "TOLERANCE", "case", "softmax", "softmax_float", "softmax_int8"]
