# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Roots whose channels are adapted: a cyclic producer feeding a thresholding, and
``inner_shuffle`` between two boundary channels. ``test_adapters`` checks their plans
and adapters; the adapter XSI sweep simulates them."""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import Traversal
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.transpose import TransposeKernel
from kernels.helpers import FULL_DSP48E2, Root, with_adapter_memories, with_direct_transports

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


def adapted(source: Traversal, pe: int, *, adaptable: bool = True, commit_all: bool = True) -> Any:
    """A cyclic producer presenting ``source``, thresholding ``pe`` channels a beat."""
    rows, channels = source.shape

    class Adapted(Root):
        x = Channel(
            tensor=Tensor(source.shape, ELEMENT), adaptable=adaptable, platform=FULL_DSP48E2
        )
        y = Channel(
            tensor=Tensor(source.shape, ScalarEncoding(DataType["UINT4"])),
            port="out0_V",
            platform=FULL_DSP48E2,
        )
        producer = MemStreamKernel(
            dtype=DataType["INT4"],
            form=source,
            contents=values(rows, channels),
            output_channel=x,
            platform=FULL_DSP48E2,
        )
        activate = ThresholdingAxiKernel(
            input_dtype=DataType["INT4"],
            threshold_dtype=DataType["INT4"],
            thresholds=(tuple(LEVELS for _ in range(channels)),),
            bias=0,
            pe=pe,
            ram_style="auto",
            ultra_stages=0,
            input_channel=x,
            output_channel=y,
            platform=FULL_DSP48E2,
        )

    point = commit(
        with_direct_transports(design_space(Adapted())),
        {
            "producer.ram_style": "distributed",
            "producer.pumped_memory": False,
            "activate.use_axilite": False,
            "activate.deep_pipeline": False,
        },
    )
    return with_adapter_memories(point) if commit_all else point


def columns_first(rows: int, channels: int, lanes: int) -> Traversal:
    """Channel folds outer, rows inner: another beat order at the same lanes."""
    return Traversal.over(
        (rows, channels), ((1, channels // lanes, lanes), (0, rows, 1)), ((1, lanes, 1),)
    )


def transposed(rows: int, cols: int, simd: int, batches: int = 2) -> Any:
    """``inner_shuffle`` placed between two boundary channels: rows in, columns out."""
    shape = (batches, rows, cols)

    class Transposed(Root):
        a = Channel(tensor=Tensor(shape, ELEMENT), port="in0_V", platform=FULL_DSP48E2)
        b = Channel(tensor=Tensor(shape, ELEMENT), port="out0_V", platform=FULL_DSP48E2)
        shuffle = TransposeKernel(input_channel=a, output_channel=b)

    point = with_direct_transports(design_space(Transposed()))
    return commit(point, {"shuffle.ram_style": "auto", "shuffle.simd": simd})
