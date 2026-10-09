# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""eltwise: element by element, a broadcast operand replayed. No KernelOp reaches it yet, so
its reference is test-side (decision KT12 A1): ``computed``, which its conformance case and
its numeric sweep (``kernels.sweeps.eltwise_numeric``) both state."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.kernels.eltwise import EltwiseKernel
from kernels.helpers import FULL_DSP48E2, FULL_DSP58
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed
from kernels.specs.thresholding import CHANNELS, PE_FACTORS, PIXELS, tensor


def computed(
    operation: str, lhs: npt.NDArray[Any], rhs: npt.NDArray[Any], scale: float = 1.0
) -> npt.NDArray[Any]:
    """What ``eltwise`` computes of ``lhs`` and ``rhs`` (``rhs`` broadcast over ``lhs``'s
    leading axes): ADD ``lhs + rhs``, SUB ``lhs - rhs``, SBR ``rhs - lhs``, MUL
    ``lhs * rhs``, ``rhs`` scaled by ``scale`` first.

    Integers (int64 arrays) exactly. With a float32 operand, in binary32: an integer
    operand converted (exactly: its magnitude within 2**24, beyond which the RTL truncates
    toward zero, so it is refused here), ``rhs * scale`` rounded, then the operation
    rounded, as FinnLib's ``binopf`` multiplies and adds in two steps."""
    floating = np.float32 in (lhs.dtype, rhs.dtype)
    if floating:
        for operand in (lhs, rhs):
            if operand.dtype != np.float32 and np.abs(operand).max() > 1 << 24:
                raise ValueError("an integer converted to binary32 beyond 2**24 is not exact")
        lhs, rhs = lhs.astype(np.float32), rhs.astype(np.float32)
        if scale != 1.0:
            rhs = rhs * np.float32(scale)
    elif scale != 1.0:
        raise ValueError("eltwise scales float arithmetic only")
    found: npt.NDArray[Any]
    if operation == "ADD":
        found = lhs + rhs
    elif operation == "SUB":
        found = lhs - rhs
    elif operation == "SBR":
        found = rhs - lhs
    elif operation == "MUL":
        found = lhs * rhs
    else:
        raise ValueError(f"eltwise computes no {operation}")
    return found


FACTS = dict(
    operation="ADD",
    lhs_dtype=DataType["INT4"],
    rhs_dtype=DataType["INT4"],
    b_scale=1.0,
    platform=FULL_DSP58,
)


def eltwise() -> dict[str, Any]:
    """lhs + rhs, rhs a channel vector broadcast over the rows."""
    return dict(
        space_type=EltwiseKernel,
        inputs={
            "lhs_channel": tensor((PIXELS, CHANNELS), "INT4"),
            "rhs_channel": tensor((CHANNELS,), "INT4"),
        },
        outputs={"result_channel": (PIXELS, CHANNELS)},
        reference=lambda lhs_channel, rhs_channel: {
            "result_channel": computed("ADD", lhs_channel, rhs_channel)
        },
        factors=PE_FACTORS,
        facts=FACTS,
    )


SPEC = KernelSpec(
    kernel=EltwiseKernel,
    reference=TEST_SIDE,
    cases={"eltwise": eltwise},
    space="eltwise",
    probes=(
        Probe(
            "an operation the RTL does not implement",
            lambda: placed(eltwise(), {"pe": 1}, operation="DIV"),
            frozenset({"eltwise-operation"}),
        ),
        Probe(
            "a scale on integer operands",
            lambda: placed(eltwise(), {"pe": 1}, b_scale=2.0),
            frozenset({"eltwise-scale"}),
        ),
        Probe(
            "floating-point arithmetic without DSP58",
            lambda: placed(
                eltwise(),
                {"pe": 1},
                lhs_dtype=DataType["FLOAT32"],
                rhs_dtype=DataType["FLOAT32"],
                platform=FULL_DSP48E2,
            ),
            frozenset({"eltwise-target"}),
        ),
    ),
    sweeps=(SweepCases("kernels.sweeps.eltwise_numeric", ("CASES",), ("sweep-eltwise",)),),
    unit=("tests/kernels/test_eltwise.py",),
)

__all__ = ["SPEC", "computed", "eltwise"]
