# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition presenting AXI-Lite buses, declared by hand.

No KernelOp names a control port, so a partition of KernelOps presents no AXI-Lite bus:
the shell's rules, the core unit and its packaging are read on a partition built from the
kernels directly, AXI-Lite thresholdings each given its port (``thresholdings``)."""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.custom_op.kernels.shell import row_facts, row_offer
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.explore import CoreComposition
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.shells import ShellRow
from support.helpers import (
    by_hand,
    choose,
    with_adapter_memories,
    with_channel_defaults,
    with_direct_transports,
)
from support.tfc import ULTRA96

UINT2 = DataType["UINT2"]
TENSOR = Tensor((3, 2), ScalarEncoding(UINT2))
"""Each channel's tensor: three rows of two UINT2 channels."""
TABLE = IntegerTensorValue.of((((0, 1, 2), (0, 1, 2)),))
"""Each thresholding's table: two channels of three thresholds over UINT2, to UINT2."""


def thresholdings(count: int, row: ShellRow, presenting: int | None = None) -> CoreComposition:
    """A partition declared by hand in the shell of ``row``, of ``count`` AXI-Lite
    thresholdings in a row, the first ``presenting`` (every one by default) presenting
    its bus (``t<i>_s_axilite``) and the others not, configured: at its baseline otherwise,
    every transport direct."""
    platform = ULTRA96.platform
    # Each channel is on the ports of the thresholdings on either side of it; the first's
    # and the last's other side the boundary, offered the row's movers and stated by them.
    names = ["x", *(f"c{index}" for index in range(count - 1)), "y"]
    offer = row_offer(row)
    stated = offer.states() if offer is not None else None
    channels: dict[str, Any] = {}
    for index, name in enumerate(names):
        producer = (
            Port(f"t{index - 1}", "output") if index > 0 else Side.boundary("s_axis_0", stated)
        )
        consumer = (
            Port(f"t{index}", "input") if index < count else Side.boundary("m_axis_0", stated)
        )
        channels[name] = channel(producer, consumer, tensor=TENSOR, platform=platform)
    members: dict[str, Any] = {name: channels[name] for name in names[:-1]}
    for index in range(count):
        members[f"t{index}"] = Facts(
            ThresholdingKernel,
            {
                "threshold_dtype": UINT2,
                "thresholds": TABLE,
                "bias": 0,
                "input_tensor": TENSOR,
                "control": f"t{index}_s_axilite",
                "platform": platform,
            },
        )
    members["y"] = channels["y"]
    presented = count if presenting is None else presenting
    choices: dict[str, object] = {}
    for index in range(count):
        held = {"pe": 1, "deep_pipeline": False, "ram_style": "auto", "ultra_stages": 0}
        held["use_axilite"] = index < presented
        choices |= {f"t{index}.{key}": value for key, value in held.items()}
    point = by_hand(members, shell=row_facts(row), offer=offer)
    point = with_direct_transports(choose(point, choices))
    return with_channel_defaults(with_adapter_memories(point))
