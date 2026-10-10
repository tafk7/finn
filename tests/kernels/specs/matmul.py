# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMul: a kernel with children (the dotp cores) on the channels its parent declares,
bound by the KernelOp MatMul. Its conformance case states X @ W as its reference."""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import design_space
from finn.dataflow.gemm import Form, Window
from finn.kernels.matmul import MatMulKernel
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import ORACLE, KernelSpec, Probe, SweepCases, placed, refuses
from kernels.specs.dotp import OUTPUTS, REDUCTION, ROWS
from kernels.specs.thresholding import tensor

FACTS: dict[str, Any] = dict(
    m=ROWS,
    n=OUTPUTS,
    k=REDUCTION,
    activation_dtype=DataType["INT4"],
    weights_dtype=DataType["INT4"],
    platform=FULL_DSP48E2,
)


def matmul() -> dict[str, Any]:
    """Y = X @ W on the packed core, its weights streamed: its activations replayed and framed
    by the channel's adapter, as its core's port states; its result element its own view."""
    result = design_space(MatMulKernel(**FACTS)).result_tensor
    return dict(
        space_type=MatMulKernel,
        inputs={
            "x_channel": tensor((ROWS, REDUCTION), "INT4"),
            "w_channel": tensor((REDUCTION, OUTPUTS), "INT4"),
        },
        outputs={"y_channel": result},
        reference=lambda x_channel, w_channel: {"y_channel": x_channel @ w_channel},
        choices={
            "compute": "packed",
            "compute.packed.compute_pumping": False,
            "compute.packed.reducer": "tree",
        },
        facts=FACTS,
    )


def depthwise() -> dict[str, Any]:
    """Depthwise, its weights streamed: no value on the weight channel."""
    case = matmul()
    case["inputs"] = {
        "x_channel": tensor((ROWS, REDUCTION, OUTPUTS), "INT4"),
        "w_channel": tensor((REDUCTION, OUTPUTS), "INT4"),
    }
    case["outputs"] = {
        "y_channel": design_space(MatMulKernel(**FACTS, form=Form.DEPTHWISE)).result_tensor
    }
    case["facts"] = {**FACTS, "form": Form.DEPTHWISE}
    return case


MATMUL_SWEEPS = SweepCases(
    "kernels.sweeps.matmul_numeric",
    ("CASES", "DEPTHWISE_CASES"),
    ("sweep-dense", "sweep-depthwise", "sweep-memstream", "sweep-sets"),
)

SPEC = KernelSpec(
    kernel=MatMulKernel,
    reference=ORACLE.format(op="MatMul"),
    cases={"matmul": matmul},
    space="matmul",
    # On DSP48E2: the INT8 core is a DSP58 mode.
    refuses=refuses(("kernel.compute", "int8_dsp58", "dotp-target")),
    probes=(
        Probe(
            "a dense realization of depthwise weights it has no value of",
            lambda: placed(depthwise(), {"realization": "dense"}),
            frozenset({"matmul-realization", "matmul-tensor"}),
        ),
        Probe(
            "a native depthwise realization no core of the platform reads",
            lambda: placed(depthwise()),
            frozenset({"matmul-native"}),
            {"kernel.realization": "native"},
        ),
        # Its activations are then the window's image, which the channel does not carry.
        Probe(
            "a window whose pixels are not its rows",
            lambda: placed(matmul(), window=Window((2, 2), (1, 1))),
            frozenset({"matmul-window", "matmul-tensor"}),
        ),
        Probe(
            "a window whose taps do not divide its reduction",
            lambda: placed(matmul(), window=Window((1, 5), (1, 4))),
            frozenset({"matmul-window", "matmul-tensor"}),
        ),
    ),
    sweeps=(MATMUL_SWEEPS,),
    unit=(
        "tests/kernels/test_matmul_assembly.py",
        "tests/kernels/test_matmul_collapse.py",
        "tests/kernels/test_matmul_depthwise.py",
        "tests/kernels/test_matmul_replay.py",
    ),
)

__all__ = ["FACTS", "SPEC", "matmul"]
