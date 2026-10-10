# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conv's ONNX entry: a lowered convolution, qonnx ``Im2Col`` and the ONNX
``MatMul`` reading its patches, as graph preparation's P3 lowers an ONNX ``Conv``.

Positive: a window over the whole image (CNV's last 3 x 3 layer), 2 x 2 windows at
stride 2 (no overlap), 3 x 3 at stride 1 (overlapping, CNV's other layers), a strided
and dilated window that passes rows, and CNV's first layer (32 x 32 x 3, 64
outputs). Negative, by code: padding, a depthwise Im2Col, patches another node reads
too (conversion's ``match-interior-exposed``), patches no MatMul reads, batched
weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels import Conv
from finn.harness.reference import OpSpec
from kernel_ops.specs.base import GENERAL, source


def weights(rows: int, columns: int, low: int, high: int, seed: int = 0) -> Any:
    """``rows`` x ``columns`` integers in [low, high], seeded, both extremes present."""
    found = np.random.default_rng(seed).integers(low, high + 1, size=(rows, columns))
    found.flat[0], found.flat[-1] = low, high
    return found


def im2col(
    image: tuple[int, int, int, int],
    kernel: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    depthwise: int = 0,
    name: str = "im2col",
) -> NodeProto:
    """qonnx's Im2Col of x (``image``, NHWC) into the patches p."""
    return helper.make_node(
        "Im2Col",
        ["x"],
        ["p"],
        name=name,
        domain=GENERAL,
        stride=list(stride),
        kernel_size=list(kernel),
        dilations=list(dilation),
        input_shape=str(tuple(image)),
        pad_amount=list(pads),
        depthwise=depthwise,
    )


def lowered(
    image: tuple[int, int, int, int] = (1, 6, 6, 4),
    kernel: tuple[int, int] = (3, 3),
    outputs: int = 8,
    x: str | None = "INT4",
    w: str | None = "INT4",
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    depthwise: int = 0,
    w_shape: tuple[int, ...] | None = None,
) -> ModelWrapper:
    """x ``image`` -> Im2Col ``im2col`` -> p -> MatMul ``mm`` with w (stored, its rows the
    window's taps times the channels) -> y."""
    rows = kernel[0] * kernel[1] * image[3]
    low, high = (-8, 7) if w == "INT4" else (-1, 1)
    stored = weights(rows, outputs, low, high) if w_shape is None else np.ones(w_shape)
    nodes = [
        im2col(image, kernel, stride, dilation, pads, depthwise),
        helper.make_node("MatMul", ["p", "w"], ["y"], name="mm"),
    ]
    return source(nodes, {"x": (list(image), x)}, {"w": (stored, w)})


def fan_out() -> ModelWrapper:
    """The patches read by two MatMuls, their results added: the window's patches are no
    tensor of the op, so a second reader keeps the pair on the host."""
    taps = 9 * 4
    nodes = [
        im2col((1, 6, 6, 4), (3, 3)),
        helper.make_node("MatMul", ["p", "w"], ["y0"], name="mm"),
        helper.make_node("MatMul", ["p", "v"], ["y1"], name="mm_1"),
        helper.make_node("Add", ["y0", "y1"], ["y"], name="add"),
    ]
    stored = {"w": (weights(taps, 8, -8, 7), "INT4"), "v": (weights(taps, 8, -8, 7, 1), "INT4")}
    return source(nodes, {"x": ([1, 6, 6, 4], "INT4")}, stored)


def unread() -> ModelWrapper:
    """Patches no MatMul reads as A: a Relu reads them."""
    nodes = [im2col((1, 4, 4, 2), (2, 2)), helper.make_node("Relu", ["p"], ["y"], name="relu")]
    return source(nodes, {"x": ([1, 4, 4, 2], "INT4")}, {})


SPEC = OpSpec(
    Conv,
    positive={
        "whole-image": lambda: lowered((1, 3, 3, 8), outputs=4),
        "2x2-stride-2": lambda: lowered((1, 4, 4, 3), (2, 2), outputs=6, stride=(2, 2)),
        "3x3-stride-1": lambda: lowered(),
        "strided-dilated": lambda: lowered(
            (1, 8, 8, 2), outputs=4, x="UINT4", stride=(2, 2), dilation=(2, 2)
        ),
        # CNV_W2A2's first layer: an INT8 image (the input quantizer's), ternary weights.
        "cnv-conv0": lambda: lowered((1, 32, 32, 3), outputs=64, x="INT8", w="INT2"),
    },
    negative={
        "pads": (lambda: lowered(pads=(1, 1, 1, 1)), "window-pads"),
        "depthwise": (lambda: lowered(depthwise=1), "window-depthwise"),
        "fan-out": (fan_out, "match-interior-exposed"),
        "unread": (unread, "window-unread"),
        "batched": (lambda: lowered(w_shape=(4, 36, 8)), "matmul-batched"),
    },
)

__all__ = ["SPEC", "im2col", "lowered"]
