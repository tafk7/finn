# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""memstream: a stored operand streamed in its consumer's form. A source, reached by a
KernelOp only as its weights' source (MatMul's ``w.source``): its test-side reference is
its contents."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from qonnx.core.datatype import DataType

from finn.dataflow.traversal import tile, vector_major
from finn.kernels.memstream import MemStreamKernel
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, SweepCases, placed

STORED = (4, 6)
CONTENTS = tuple(tuple((7 * row + 5 * col) % 16 - 8 for col in range(6)) for row in range(4))


def memstream() -> dict[str, Any]:
    """The stored operand streamed in each consumer form: identity against its contents."""
    return dict(
        space_type=MemStreamKernel,
        inputs={},
        outputs={"output_channel": STORED},
        reference=lambda: {"output_channel": np.array(CONTENTS)},
        factors=(
            *({"form": vector_major(STORED, lanes)} for lanes in (1, 3, 6)),
            {"form": tile(*STORED, 2, 3)},
        ),
        choices={"ram_style": "auto", "pumped_memory": False},
        facts={"dtype": DataType["INT4"], "contents": CONTENTS, "platform": FULL_DSP48E2},
    )


def pumped_memstream() -> dict[str, Any]:
    """The stored operand from a pumped memory: half-width words at the doubled clock."""
    case = memstream()
    case["factors"] = ({"form": vector_major(STORED, 3)},)
    case["choices"] = {**case["choices"], "pumped_memory": True}
    return case


def one_bit() -> dict[str, Any]:
    """One-bit words, one element a beat: nothing for a pumped memory to split."""
    case = memstream()
    contents = tuple(tuple(value & 1 for value in row) for row in CONTENTS)
    case["factors"] = ({"form": vector_major(STORED, 1)},)
    case["facts"] = {**case["facts"], "dtype": DataType["BINARY"], "contents": contents}
    return case


SPEC = KernelSpec(
    kernel=MemStreamKernel,
    reference=TEST_SIDE,
    cases={"memstream": memstream, "memstream-pumped": pumped_memstream},
    space="memstream",
    probes=(
        Probe(
            "UltraRAM on a platform without it",
            lambda: placed(memstream(), platform=replace(FULL_DSP48E2, uram=False)),
            frozenset({"uram-absent"}),
            {"kernel.ram_style": "ultra"},
        ),
        Probe(
            "a pumped memory of one-bit words",
            lambda: placed(one_bit(), {"ram_style": "auto", "pumped_memory": True}),
            frozenset({"memstream-pumping"}),
        ),
    ),
    sweeps=(
        SweepCases(
            "kernels.sweeps.matmul_numeric",
            ("CASES",),
            ("sweep-memstream", "sweep-memstream-depthwise", "sweep-pumped-memory", "sweep-sets"),
        ),
    ),
    unit=("tests/kernels/test_memstream.py",),
)

__all__ = ["CONTENTS", "SPEC", "STORED", "memstream", "pumped_memstream"]
