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
from finn.kernels.target import DspBlock
from kernels.helpers import full_platform
from kernels.specs.base import ORACLE, KernelSpec, Probe, SweepCases, placed

ROWS, REDUCTION, OUTPUTS = 2, 6, 4
Array = npt.NDArray[Any]


def dotp(
    space_type: type[Any],
    dsp: DspBlock,
    bits: int,
    form: Form = Form.DENSE,
    reducer: str | None = None,
    pumped: bool = False,
) -> dict[str, Any]:
    """Y = X @ W, or per channel (depthwise); the accumulator type is the parent's fact.
    ``reducer`` is the packed core's, which only it declares. ``pumped`` compute runs at
    the doubled clock, which needs SIMD >= 2: one interior configuration."""
    a = w = DataType[f"INT{bits}"]
    depthwise = form is Form.DEPTHWISE
    reduction = 3 if depthwise else REDUCTION
    x_shape = (ROWS, reduction, OUTPUTS) if depthwise else (ROWS, reduction)

    def reference(x_channel: Array, w_channel: Array) -> dict[str, Array]:
        y = (x_channel * w_channel).sum(axis=1) if depthwise else x_channel @ w_channel
        return {"y_channel": y}

    return dict(
        space_type=space_type,
        inputs={
            "x_channel": Tensor(x_shape, ScalarEncoding(a)),
            "w_channel": Tensor((reduction, OUTPUTS), ScalarEncoding(w)),
        },
        outputs={"y_channel": (ROWS, OUTPUTS)},
        reference=reference,
        **({"factors": ({"pe": 2, "simd": 3},)} if pumped else {}),
        choices={"compute_pumping": pumped, **({"reducer": reducer} if reducer else {})},
        facts={
            "platform": full_platform(dsp),
            "form": form,
            "result_range": datatype_range(reduction, a, w),
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
    probes=(
        Probe(
            "the packed core broadcasts activations: no depthwise form",
            lambda: placed(dotp(PackedDotpKernel, DspBlock.DSP48E2, 4, Form.DEPTHWISE)),
            frozenset({"dotp-form"}),
        ),
        Probe(
            "activations wider than the DSP's signed B input",
            lambda: placed(dotp(PackedDotpKernel, DspBlock.DSP48E2, 19), {"pe": 1, "simd": 1}),
            frozenset({"dotp-activation-width"}),
        ),
        Probe(
            "pumped compute with one SIMD lane",
            lambda: placed(dotp(PackedDotpKernel, DspBlock.DSP48E2, 4)),
            frozenset({"dotp-pumping"}),
            {"kernel.simd": 1, "kernel.compute_pumping": True},
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

__all__ = ["INT8", "OUTPUTS", "PACKED", "REDUCTION", "ROWS", "dotp"]
