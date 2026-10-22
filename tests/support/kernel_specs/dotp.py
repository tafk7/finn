# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""dotp's two cores: the packed core and the INT8 DSP58 core.

A core inside MatMul (``compute``): the KernelOp MatMul reaches it there, so its
computation is checked at the op's boundary; placed alone, its conformance cases
state Y = X @ W (or per channel, depthwise) as their reference.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.dataflow.gemm import Form
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.dotp import Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.matmul import datatype_range
from finn.platform.capabilities import DspBlock
from support.helpers import dotp_extents, full_platform
from support.kernel_specs.base import ORACLE, KernelSpec, Probe, SweepCases, placed, refuses

ROWS, REDUCTION, OUTPUTS = 2, 6, 4
Array = npt.NDArray[Any]


def folding(outputs: int, reduction: int) -> tuple[dict[str, int], ...]:
    """PE over the divisors of N and SIMD over those of K, as MatMul's domains list them:
    the smallest of each, the one in the middle of each list, and the largest."""
    pes = [d for d in range(1, outputs + 1) if outputs % d == 0]
    simds = [d for d in range(1, reduction + 1) if reduction % d == 0]
    return tuple(
        {"pe": pes[pick(len(pes))], "simd": simds[pick(len(simds))]}
        for pick in (lambda _: 0, lambda size: size // 2, lambda size: size - 1)
    )


def dotp(
    space_type: type[Any],
    dsp: DspBlock,
    bits: int,
    form: Form = Form.DENSE,
    reducer: str | None = None,
    pumped: bool = False,
) -> dict[str, Any]:
    """Y = X @ W, or per channel (depthwise); the accumulator type is the parent's fact.
    ``reducer`` is the packed core's, which only it declares. PE and SIMD are given to a
    core (its role's choices, MatMul's): the smallest, an interior and the largest folding
    of the extents (``folding``). ``pumped`` compute runs at the doubled clock, which needs
    SIMD >= 2: one interior configuration."""
    a = w = DataType[f"INT{bits}"]
    depthwise = form is Form.DEPTHWISE
    reduction = 3 if depthwise else REDUCTION
    x_shape = (ROWS, reduction, OUTPUTS) if depthwise else (ROWS, reduction)

    def reference(x_channel: Array, w_channel: Array) -> dict[str, Array]:
        y = (x_channel * w_channel).sum(axis=1) if depthwise else x_channel @ w_channel
        return {"y_channel": y}

    inputs = {
        "x_channel": Tensor(x_shape, ScalarEncoding(a)),
        "w_channel": Tensor((reduction, OUTPUTS), ScalarEncoding(w)),
    }
    return dict(
        space_type=space_type,
        inputs=inputs,
        outputs={"y_channel": (ROWS, OUTPUTS)},
        reference=reference,
        factors=({"pe": 2, "simd": 3},) if pumped else folding(OUTPUTS, reduction),
        choices={"compute_pumping": pumped, **({"reducer": reducer} if reducer else {})},
        facts={
            "platform": full_platform(dsp),
            "form": form,
            "result_range": datatype_range(reduction, ScalarEncoding(a), ScalarEncoding(w)),
            # What a parent states from the operation's facts (``helpers.dotp_extents``).
            **dotp_extents(inputs["w_channel"]),
        },
    )


SWEEPS = SweepCases(
    "kernels.sweeps.pure_dot_product_numeric",
    ("CASES", "STRESS_CASES"),
    ("sweep-dotp", "sweep-dotp-stress"),
)

PACKED = KernelSpec(
    kernel=PackedDotpKernel,
    reference=ORACLE.format(op="MatMul"),
    cases={
        "dotp-packed": lambda: dotp(PackedDotpKernel, DspBlock.DSP48E2, 4, reducer="tree"),
        "dotp-packed-compressor": lambda: dotp(
            PackedDotpKernel, DspBlock.DSP48E2, 4, reducer="compressor"
        ),
    },
    space="dotp-packed",
    # Placed at its first folding, SIMD 1: pumping has no lanes to split.
    refuses=refuses(("kernel.compute_pumping", True, "dotp-pumping")),
    probes=(
        Probe(
            "the packed core broadcasts activations: no depthwise form",
            lambda: placed(dotp(PackedDotpKernel, DspBlock.DSP48E2, 4, Form.DEPTHWISE)),
            frozenset({"dotp-form"}),
        ),
        Probe(
            "activations wider than the DSP's signed B input",
            lambda: placed(dotp(PackedDotpKernel, DspBlock.DSP48E2, 19)),
            frozenset({"dotp-activation-width"}),
        ),
    ),
    sweeps=(SWEEPS,),
    unit=("tests/kernels/test_dotp.py",),
)

INT8 = KernelSpec(
    kernel=Int8Dsp58DotpKernel,
    reference=ORACLE.format(op="MatMul"),
    cases={
        "dotp-int8": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8),
        "dotp-int8-depthwise": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8, Form.DEPTHWISE),
        "dotp-int8-pumped": lambda: dotp(Int8Dsp58DotpKernel, DspBlock.DSP58, 8, pumped=True),
    },
    space="dotp-int8",
    refuses=refuses(("kernel.compute_pumping", True, "dotp-pumping")),
    probes=(
        Probe(
            "the INT8 core is a DSP58 mode",
            lambda: placed(dotp(Int8Dsp58DotpKernel, DspBlock.DSP48E2, 8)),
            frozenset({"dotp-target"}),
        ),
    ),
    sweeps=(SWEEPS,),
    unit=("tests/kernels/test_dotp.py",),
)

__all__ = ["INT8", "OUTPUTS", "PACKED", "REDUCTION", "ROWS", "dotp", "folding"]
