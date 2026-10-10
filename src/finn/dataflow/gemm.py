# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Matrix multiplication in canonical GEMM notation: ``Y[m, n] = sum_k X * W``.

The indices are ``m`` (rows), ``n`` (outputs) and ``k`` (the reduction), with
extents M, N and K. A ``Form`` says which indices each operand reads:

- ``DENSE``: ``X (m, k)``, ``W (k, n)``, ``Y (m, n)``; every output reads the
  whole activation row.
- ``DEPTHWISE``: ``X (m, k, n)``, ``W (k, n)``, ``Y (m, n)``; the activations
  also carry ``n``, so each output reads its own.

Weights are stored ``(k, n)``, as ONNX ``MatMul`` and FINN's MVAU initializer
store them. The stored axis order does not fix the order a port presents: every
port presents what its schedule derives.

A dense form may read its activations through a ``Window``: a convolution lowered
to a matrix multiplication (qonnx's ``Im2Col`` and ``MatMul``), whose rows are the
output pixels and whose reduction is a window of the image. Each row ``m`` is a
pixel ``(oh, ow)`` and the reduction ``k`` is ``(kh, kw, c)``, ``c`` innermost, the
order ``Im2Col`` and ``LowerConvsToMatMul``'s weights use. X is the image itself,
``(H, W, C)`` stored as its ``H * W`` rows of ``C``, read at
``(oh * SH + kh * DH, ow * SW + kw * DW, c)``; W is its stored ``(K, N)`` viewed
``(KH, KW, C, N)``; Y is ``(OH * OW, N)`` (``Window.signature``). No position is
padded: a window reads only the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.schedule import Affine, Index

m, n, k = Index("m"), Index("n"), Index("k")
oh, ow, kh, kw, c = (Index(name) for name in ("oh", "ow", "kh", "kw", "c"))


@dataclass(frozen=True)
class Signature:
    """The indices each operand reads, in its stored axis order."""

    x: tuple[Index | Affine, ...]
    w: tuple[Index | Affine, ...]
    y: tuple[Index | Affine, ...]


class Form(Enum):
    """Which indices each operand of ``Y[m, n] = sum_k X * W`` reads."""

    DENSE = Signature(x=(m, k), w=(k, n), y=(m, n))
    DEPTHWISE = Signature(x=(m, k, n), w=(k, n), y=(m, n))

    @property
    def x(self) -> tuple[Index | Affine, ...]:
        return self.value.x

    @property
    def w(self) -> tuple[Index | Affine, ...]:
        return self.value.w

    @property
    def y(self) -> tuple[Index | Affine, ...]:
        return self.value.y


def _pair(name: str, value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(each) is not int or each < 1 for each in value)
    ):
        raise ValueError(f"a window's {name} is two positive integers (rows, columns), not {value}")
    return value


@dataclass(frozen=True)
class Window:
    """A 2-D sliding window of ``kernel`` taps at ``stride`` and ``dilation`` (each rows,
    columns) over an ``image`` of (H, W) pixels: a dense form's rows and reduction (the
    module docstring). It reads the image only, no padding, so it fits the image; a
    malformed one raises ``ValueError``."""

    image: tuple[int, int]
    kernel: tuple[int, int]
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        for name in ("image", "kernel", "stride", "dilation"):
            _pair(name, getattr(self, name))
        for axis, name in enumerate(("rows", "columns")):
            span = self.dilation[axis] * (self.kernel[axis] - 1) + 1
            if span > self.image[axis]:
                raise ValueError(
                    f"a window spanning {span} {name} does not fit an image of {self.image[axis]}"
                )

    @property
    def output(self) -> tuple[int, int]:
        """(OH, OW): the windows that fit whole, a row and a column."""
        return (
            (self.image[0] - self.dilation[0] * (self.kernel[0] - 1) - 1) // self.stride[0] + 1,
            (self.image[1] - self.dilation[1] * (self.kernel[1] - 1) - 1) // self.stride[1] + 1,
        )

    @property
    def taps(self) -> int:
        """KH * KW: the pixels one window reads."""
        return self.kernel[0] * self.kernel[1]

    @property
    def extents(self) -> dict[Index, int]:
        """The window's own indices' extents; ``c`` and ``n`` are the operands'."""
        (rows, columns), (taps_h, taps_w) = self.output, self.kernel
        return {oh: rows, ow: columns, kh: taps_h, kw: taps_w}

    @property
    def signature(self) -> Signature:
        """X over the image's (H * W, C) rows, W over its (K, N) viewed (KH, KW, C, N), Y
        over (OH * OW, N)."""
        width = self.image[1]
        (row_step, column_step), (row_gap, column_gap) = self.stride, self.dilation
        pixel = oh * (row_step * width) + kh * (row_gap * width) + ow * column_step
        return Signature(
            x=(pixel + kw * column_gap, c),
            w=(kh, kw, c, n),
            y=(oh * self.output[1] + ow, n),
        )


__all__ = ["Form", "Signature", "Window", "c", "k", "kh", "kw", "m", "n", "oh", "ow"]
