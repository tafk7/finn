# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MatMulKernel with and without collapsed forwarding: its decision keys are the same.

Collapse rewrites the compiled form's forwarding edges only. A MatMul, as a network binds
it, keeps its decision keys and their order whether its forwarding aliases are collapsed or
not (the compiled forms, read without evaluating either).
"""

from typing import Any

from core.space._collapse_support import compiled
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, matmul_point

from finn.core.space import Model, Space, inspection
from finn.kernels.matmul import MatMulKernel

FACTS = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    platform=FULL_DSP48E2,
)
# Stored (k, n), any values: collapse is checked on a memory, not on what it holds.
WEIGHTS = ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))


def _facts(case: str) -> dict[str, Any]:
    return {**FACTS, "weights": WEIGHTS} if case == "memstream-fifo" else FACTS


def _kernel(case: str, *, collapsed: bool) -> Model[Space]:
    """The MatMul's model as the network binds it, compiled afresh."""
    network = matmul_point(**_facts(case))
    point = network.member("matmul")
    facts = {
        name: getattr(point, name)
        for name in ("x_tensor", "w_tensor", "w_contents", "weights_dtype")
        if point.present(getattr(MatMulKernel, name))
    }
    return compiled(MatMulKernel(platform=FULL_DSP48E2, **facts), collapsed=collapsed)


def test_collapse_keeps_matmul_decision_keys() -> None:
    collapsed = _kernel("memstream-fifo", collapsed=True)
    plain = _kernel("memstream-fifo", collapsed=False)
    assert [item.key for item in inspection.decisions(collapsed)] == [
        item.key for item in inspection.decisions(plain)
    ]
