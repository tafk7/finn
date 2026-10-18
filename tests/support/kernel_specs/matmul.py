# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMul: a kernel with children (the dotp cores) on the channels its parent declares,
bound by the KernelOp MatMul. Its conformance case states X @ W as its reference."""

from __future__ import annotations

from typing import Any

from finn.core.space import design_space
from finn.dataflow.gemm import Form, Window
from finn.dataflow.tensor import Tensor
from finn.kernels.matmul import MatMulKernel
from support.helpers import FULL_DSP48E2
from support.kernel_specs.base import ORACLE, KernelSpec, Probe, SweepCases, placed, refuses
from support.kernel_specs.dotp import OUTPUTS, REDUCTION, ROWS
from support.kernel_specs.thresholding import tensor

FACTS: dict[str, Any] = dict(platform=FULL_DSP48E2)
"""MatMul's facts beside its tensors: its extents and datatypes are its tensors'."""


def _results(x: Tensor, w: Tensor, **facts: Any) -> Tensor:
    """MatMul's statement of its results on the activations ``x`` and streamed weights
    ``w``: a value, whatever MatMul's validity."""
    found: Tensor = design_space(
        MatMulKernel(**FACTS, **facts, x_tensor=x, w_tensor=w)
    ).result_tensor
    return found


def matmul() -> dict[str, Any]:
    """Y = X @ W on the packed core, its weights streamed: its activations replayed and framed
    by the channel's adapter, as its core's port states; its result element its own view."""
    x, w = tensor((ROWS, REDUCTION), "INT4"), tensor((REDUCTION, OUTPUTS), "INT4")
    return dict(
        space_type=MatMulKernel,
        inputs={"x_channel": x, "w_channel": w},
        outputs={"y_channel": _results(x, w)},
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
    x, w = tensor((ROWS, REDUCTION, OUTPUTS), "INT4"), tensor((REDUCTION, OUTPUTS), "INT4")
    case["inputs"] = {"x_channel": x, "w_channel": w}
    case["outputs"] = {"y_channel": _results(x, w, form=Form.DEPTHWISE)}
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
            # No core is possible on the refused tensors either.
            frozenset({"matmul-realization", "matmul-tensor", "decision-no-viable-case"}),
        ),
        Probe(
            "a native depthwise realization no core of the platform reads",
            lambda: placed(depthwise()),
            frozenset({"matmul-native", "decision-no-viable-case"}),
            {"realization": "native"},
        ),
        # Its activations are then the window's image, which the channel does not carry.
        Probe(
            "a window whose pixels are not its rows",
            lambda: placed(matmul(), window=Window((2, 2), (1, 1))),
            frozenset({"matmul-window", "matmul-tensor", "decision-no-viable-case"}),
        ),
        Probe(
            "a window whose taps do not divide its reduction",
            lambda: placed(matmul(), window=Window((1, 5), (1, 4))),
            frozenset({"matmul-window", "matmul-tensor", "decision-no-viable-case"}),
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
