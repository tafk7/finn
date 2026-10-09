# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pool: max pooling over FinnLib's HLS ``pooling_stream``, each window closed by its
channel's marker. No KernelOp reaches it yet (a Pool KernelOp from MaxPoolNHWC comes
later), so its reference is test-side (decision KT12 A1): ``max_pool``."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.pool import PoolKernel
from kernels.helpers import FULL_DSP48E2
from kernels.specs.base import TEST_SIDE, KernelSpec, Probe, placed

#: CNV's pooling, 2x2 windows at stride 2, on a small map: six channels, so PE 1, 3 and 6.
SHAPE, WINDOW = (4, 6, 6), (2, 2)


def max_pool(
    image: npt.NDArray[Any], window: tuple[int, int], stride: tuple[int, int]
) -> npt.NDArray[Any]:
    """Each channel's maximum over each ``window`` at ``stride`` of an HWC image, as many
    windows as fit whole."""
    (rows, columns), (row_step, column_step) = window, stride
    height, width, _ = image.shape
    return np.array(
        [
            [
                image[r : r + rows, c : c + columns].max(axis=(0, 1))
                for c in range(0, width - columns + 1, column_step)
            ]
            for r in range(0, height - rows + 1, row_step)
        ]
    )


def pool(
    shape: tuple[int, int, int] = SHAPE,
    window: tuple[int, int] = WINDOW,
    stride: tuple[int, int] = WINDOW,
    dtype: str = "INT4",
) -> dict[str, Any]:
    """The maximum of each window of a signed map."""
    height, width, channels = shape
    windows = ((height - window[0]) // stride[0] + 1, (width - window[1]) // stride[1] + 1)
    return dict(
        space_type=PoolKernel,
        inputs={"input_channel": Tensor(shape, ScalarEncoding(DataType[dtype]))},
        outputs={"output_channel": (*windows, channels)},
        reference=lambda input_channel: {"output_channel": max_pool(input_channel, window, stride)},
        facts={"window": window, "stride": stride, "platform": FULL_DSP48E2},
    )


SPEC = KernelSpec(
    kernel=PoolKernel,
    reference=TEST_SIDE,
    cases={"pool": pool},
    space="pool",
    probes=(
        Probe(
            "a float element: pooling_stream's maximum is over ap_int and ap_uint",
            lambda: placed(pool(dtype="FLOAT32")),
            frozenset({"dtype-family"}),
        ),
        Probe(
            "a window no wider than nothing",
            lambda: placed(pool(), window=(0, 2)),
            frozenset({"pool-geometry"}),
        ),
    ),
    unit=("tests/kernels/test_pool.py",),
)

__all__ = ["SPEC", "max_pool", "pool"]
