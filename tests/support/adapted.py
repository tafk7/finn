# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Partitions built by hand whose channels are adapted: a cyclic producer feeding a thresholding
(``adapted``, its module ``finn_adapted``), and ``inner_shuffle`` between two boundary
channels (``transposed``, ``finn_transposed``). ``test_adapters`` checks their steps and
adapters; the adapter XSI sweep simulates them."""

from __future__ import annotations

from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.partition import Partition
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.transpose import TransposeKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import Platform
from support.helpers import (
    FULL_DSP48E2,
    by_hand,
    choose,
    keyed,
    with_channel_defaults,
    with_direct_transports,
)

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


def adapted(source: Traversal, pe: int, *, adaptable: bool = True) -> Partition:
    """A cyclic producer presenting ``source``, thresholding ``pe`` channels a beat, the
    kernels' choices stored as the partition is bound (a stored set breaking a channel's
    rule binds, the channel reading refused); each channel's open choices then by their
    first valid values, as the cut sets them (``with_channel_defaults``): its input
    adapter's memory and its transport (its adapters are picked by its steps); a channel
    refused on its own facts is left as it is."""
    rows, channels = source.shape

    x_tensor = Tensor(source.shape, ELEMENT)
    y_tensor = Tensor(source.shape, ScalarEncoding(DataType["UINT4"]))

    stored = keyed(
        {
            "producer.ram_style": "distributed",
            "producer.pumped_memory": False,
            "activate.pe": pe,
            "activate.ultra_stages": 0,
            "activate.ram_style": "auto",
            "activate.use_axilite": False,
            "activate.deep_pipeline": False,
        }
    )
    network = by_hand(
        {
            "x": channel(
                Port("producer", "output"),
                Port("activate", "input"),
                tensor=x_tensor,
                adaptable=adaptable,
                platform=FULL_DSP48E2,
            ),
            "y": channel(
                Port("activate", "output"),
                Side.boundary("out0_V"),
                tensor=y_tensor,
                platform=FULL_DSP48E2,
            ),
            "producer": Facts(
                MemStreamKernel,
                {
                    "dtype": DataType["INT4"],
                    "form": source,
                    "contents": IntegerTensorValue.of(values(rows, channels)),
                    "platform": FULL_DSP48E2,
                },
            ),
            "activate": Facts(
                ThresholdingAxiKernel,
                {
                    "threshold_dtype": DataType["INT4"],
                    "thresholds": IntegerTensorValue.of((tuple(LEVELS for _ in range(channels)),)),
                    "bias": 0,
                    "input_tensor": x_tensor,
                    "platform": FULL_DSP48E2,
                },
            ),
        },
        stored,
    )
    return with_channel_defaults(network)


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
) -> Partition:
    """``inner_shuffle`` placed between two boundary channels: rows in, columns out."""
    shape = (batches, rows, cols)

    tensor = Tensor(shape, ELEMENT)

    network = by_hand(
        {
            "a": channel(
                Side.boundary("in0_V"), Port("shuffle", "input"), tensor=tensor, platform=platform
            ),
            "b": channel(
                Port("shuffle", "output"), Side.boundary("out0_V"), tensor=tensor, platform=platform
            ),
            "shuffle": Facts(TransposeKernel, {"input_tensor": tensor, "platform": platform}),
        }
    )
    point = choose(network, {"shuffle.ram_style": ram_style, "shuffle.simd": simd})
    return with_channel_defaults(with_direct_transports(point))
