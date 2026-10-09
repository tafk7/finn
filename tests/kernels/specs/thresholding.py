# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""thresholding_axi: Y = count(X >= T[c]) + bias, bound by the KernelOp Thresholding.

Its conformance cases state the count as their reference: a row a channel, one row
shared by every channel (C = 1), and runtime-writable rows written through AXI-Lite.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.core.space import Space, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.conformance import ControlPort
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import ORACLE, KernelSpec, Probe, SweepCases, refuses


def tensor(shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor(shape, ScalarEncoding(DataType[dtype]))


Array = npt.NDArray[Any]
PIXELS, CHANNELS = 3, 6
# Three thresholds a channel, two apart from the next channel's, so that a value
# thresholded against another channel's row often lands on another level: the
# planted-error tests (test_conformance.py) need their stimulus to tell channels apart.
THRESHOLDS = (tuple((-8 + 2 * c, -7 + 2 * c, -6 + 2 * c) for c in range(CHANNELS)),)


def levels(values: Array) -> Array:
    return np.asarray((values[..., None] >= np.array(THRESHOLDS[0])).sum(axis=-1))


FACTS = dict(
    input_dtype=DataType["INT4"],
    threshold_dtype=DataType["INT4"],
    thresholds=THRESHOLDS,
    bias=0,
    platform=FULL_DSP48E2,
)
# Its memories: of the two stages (N = 3), the deeper in block RAM, the other distributed.
CHOICES = {
    "use_axilite": False,
    "deep_pipeline": False,
    "ram_style": "distributed",
    "block_stages": 1,
    "ultra_stages": 0,
}
PE_FACTORS = ({"pe": 1}, {"pe": 3}, {"pe": CHANNELS})


def thresholding() -> dict[str, Any]:
    return dict(
        space_type=ThresholdingAxiKernel,
        inputs={"input_channel": tensor((PIXELS, CHANNELS), "INT4")},
        outputs={"output_channel": (PIXELS, CHANNELS)},
        reference=lambda input_channel: {"output_channel": levels(input_channel)},
        factors=PE_FACTORS,
        choices=CHOICES,
        facts=FACTS,
    )


# One row shared by every channel (C = 1), PE above it: each lane keeps the row; with
# AXI-Lite, the wrapper's configuration addresses N alone.
SHARED_CHANNELS = 8
SHARED_ROW = (-3, 0, 2)


def written_rows() -> dict[str, Any]:
    """A row a channel, runtime-writable: the testbench writes every threshold through
    AXI-Lite (two channel folds of three lanes), so a write to another lane's or
    fold's address changes some level."""
    case = thresholding()
    case["factors"] = ({"pe": 3},)
    case["choices"] = {**CHOICES, "use_axilite": True}
    case["facts"] = {**FACTS, "control": ControlPort("s_axilite")}
    return case


def shared_row(*, axilite: bool = False) -> dict[str, Any]:
    def reference(input_channel: Array) -> dict[str, Array]:
        return {"output_channel": (input_channel[..., None] >= np.array(SHARED_ROW)).sum(axis=-1)}

    facts: dict[str, object] = {**FACTS, "thresholds": ((SHARED_ROW,),)}
    choices = {**CHOICES, "use_axilite": axilite}
    if axilite:
        facts["control"] = ControlPort("s_axilite")
    return dict(
        space_type=ThresholdingAxiKernel,
        inputs={"input_channel": tensor((PIXELS, SHARED_CHANNELS), "INT4")},
        outputs={"output_channel": (PIXELS, SHARED_CHANNELS)},
        reference=reference,
        factors=({"pe": 4},) if axilite else ({"pe": 1}, {"pe": 4}, {"pe": SHARED_CHANNELS}),
        choices=choices,
        facts=facts,
    )


def alone(**facts: Any) -> Space:
    """The kernel alone on ``FACTS`` but ``facts``."""
    return design_space(ThresholdingAxiKernel(**{**FACTS, **facts}))


SPEC = KernelSpec(
    kernel=ThresholdingAxiKernel,
    reference=ORACLE.format(op="Thresholding"),
    cases={
        "thresholding": thresholding,
        "thresholding-shared-row": shared_row,
        "thresholding-shared-row-axilite": lambda: shared_row(axilite=True),
        "thresholding-written-rows": written_rows,
    },
    space="thresholding",
    # Placed with no control bus: nothing to write runtime thresholds through.
    refuses=refuses(("kernel.use_axilite", True, "threshold-control")),
    probes=(
        Probe(
            "a row out of order",
            lambda: alone(thresholds=(((1, 0, 2),),)),
            frozenset({"threshold-order"}),
        ),
        Probe(
            "a threshold its type does not hold",
            lambda: alone(thresholds=(((-9, 0, 2),),)),
            frozenset({"threshold-value"}),
        ),
        Probe(
            "input and thresholds of different signedness",
            lambda: alone(threshold_dtype=DataType["UINT4"], thresholds=(((1, 2, 3),),)),
            frozenset({"threshold-type"}),
        ),
        Probe(
            "a threshold at its type's minimum, which an input wider below it saturates to",
            lambda: alone(input_dtype=DataType["INT8"], thresholds=(((-8, 0, 2),),)),
            frozenset({"threshold-saturation"}),
        ),
        Probe(
            "a bias below -N-1, which the RTL does not sign-extend",
            lambda: alone(bias=-5),
            frozenset({"threshold-negative-range"}),
        ),
        Probe(
            "UltraRAM stages on a platform whose UltraRAM takes no initial contents",
            lambda: alone(platform=replace(FULL_DSP48E2, uram_init=False)),
            frozenset({"uram-init"}),
            {"ultra_stages": 1},
        ),
    ),
    sweeps=(
        SweepCases("kernels.sweeps.threshold_numeric", ("CASES",), ("sweep-thresholds",)),
        SweepCases("kernels.sweeps.adapter_numeric", ("ADAPTED",), ("sweep-adapters",)),
    ),
    unit=("tests/kernels/test_thresholding.py", "tests/kernels/test_threshold_memory.py"),
)

__all__ = [
    "CHANNELS",
    "CHOICES",
    "FACTS",
    "PE_FACTORS",
    "PIXELS",
    "SPEC",
    "THRESHOLDS",
    "levels",
    "shared_row",
    "tensor",
    "thresholding",
    "written_rows",
]
