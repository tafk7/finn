# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Networks whose channels are adapted: a cyclic producer feeding a thresholding
(``adapted``, its module ``finn_adapted``), and ``inner_shuffle`` between two boundary
channels (``transposed``, ``finn_transposed``). ``test_adapters`` checks their plans and
adapters; the adapter XSI sweep simulates them."""

from __future__ import annotations

from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.transpose import TransposeKernel
from finn.platform.capabilities import Platform
from kernels.helpers import FULL_DSP48E2, choose, with_adapter_memories, with_direct_transports

ELEMENT = ScalarEncoding(DataType["INT4"])
ROWS, CHANNELS = 3, 12
# Fifteen thresholds -7..7: an INT4 value v becomes the level v + 8, so every
# output identifies the element that produced it.
LEVELS = tuple(range(-7, 8))


def values(rows: int = ROWS, channels: int = CHANNELS) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((5 * row + 3 * channel) % 16 - 8 for channel in range(channels))
        for row in range(rows)
    )


def adapted(
    source: Traversal, pe: int, *, adaptable: bool = True, commit_all: bool = True
) -> Network:
    """A cyclic producer presenting ``source``, thresholding ``pe`` channels a beat."""
    rows, channels = source.shape

    x_tensor = Tensor(source.shape, ELEMENT)
    y_tensor = Tensor(source.shape, ScalarEncoding(DataType["UINT4"]))

    network = Network(
        {
            "x": wire(
                tensor=x_tensor,
                adaptable=adaptable,
                producer=Interface("producer", "output_end"),
                consumer=Interface("activate", "input_end"),
                platform=FULL_DSP48E2,
            ),
            "y": wire(
                tensor=y_tensor,
                port="out0_V",
                producer=Interface("activate", "output_end"),
                platform=FULL_DSP48E2,
            ),
            "producer": MemStreamKernel(
                dtype=DataType["INT4"],
                form=source,
                contents=values(rows, channels),
                output_tensor=x_tensor,
                platform=FULL_DSP48E2,
            ),
            "activate": ThresholdingAxiKernel(
                input_dtype=DataType["INT4"],
                threshold_dtype=DataType["INT4"],
                thresholds=(tuple(LEVELS for _ in range(channels)),),
                bias=0,
                pe=pe,
                ram_style="auto",
                ultra_stages=0,
                input_tensor=x_tensor,
                output_tensor=y_tensor,
                platform=FULL_DSP48E2,
            ),
        }
    )
    point = with_direct_transports(
        choose(
            network,
            {
                "producer.ram_style": "distributed",
                "producer.pumped_memory": False,
                "activate.use_axilite": False,
                "activate.deep_pipeline": False,
            },
        )
    )
    return with_adapter_memories(point) if commit_all else point


def columns_first(rows: int, channels: int, lanes: int) -> Traversal:
    """Channel folds outer, rows inner: another beat order at the same lanes."""
    return Traversal.over(
        (rows, channels), ((1, channels // lanes, lanes), (0, rows, 1)), ((1, lanes, 1),)
    )


def transposed(
    rows: int,
    cols: int,
    simd: int,
    batches: int = 2,
    *,
    ram_style: str = "auto",
    platform: Platform = FULL_DSP48E2,
) -> Network:
    """``inner_shuffle`` placed between two boundary channels: rows in, columns out."""
    shape = (batches, rows, cols)

    tensor = Tensor(shape, ELEMENT)

    network = Network(
        {
            "a": wire(
                tensor=tensor,
                port="in0_V",
                consumer=Interface("shuffle", "input_end"),
                platform=platform,
            ),
            "b": wire(
                tensor=tensor,
                port="out0_V",
                producer=Interface("shuffle", "output_end"),
                platform=platform,
            ),
            "shuffle": TransposeKernel(
                input_tensor=tensor, output_tensor=tensor, platform=platform
            ),
        }
    )
    point = choose(network, {"shuffle.ram_style": ram_style, "shuffle.simd": simd})
    return with_direct_transports(point)
