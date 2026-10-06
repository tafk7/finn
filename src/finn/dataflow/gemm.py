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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.schedule import Index

m, n, k = Index("m"), Index("n"), Index("k")


@dataclass(frozen=True)
class Signature:
    """The indices each operand reads, in its stored axis order."""

    x: tuple[Index, ...]
    w: tuple[Index, ...]
    y: tuple[Index, ...]


class Form(Enum):
    """Which indices each operand of ``Y[m, n] = sum_k X * W`` reads."""

    DENSE = Signature(x=(m, k), w=(k, n), y=(m, n))
    DEPTHWISE = Signature(x=(m, k, n), w=(k, n), y=(m, n))

    @property
    def x(self) -> tuple[Index, ...]:
        return self.value.x

    @property
    def w(self) -> tuple[Index, ...]:
        return self.value.w

    @property
    def y(self) -> tuple[Index, ...]:
        return self.value.y


__all__ = ["Form", "Signature", "k", "m", "n"]
