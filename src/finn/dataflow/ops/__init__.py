# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source adapters and compiler-facing dataflow lifecycle.

``finn.dataflow.ops.base`` holds the generic contract -- one source node, one
frozen problem, one root occurrence, one persistence authority -- and
``source`` and ``mapping`` hold the two value families it reads and produces.
Concrete operations bind source facts and selected construction to reusable
implementations from ``finn.dataflow.kernels``; the Kernel library never
imports this package back.

This namespace performs no eager import. Importing the operation layer would
pull in QONNX and every Kernel behind it, and the generic Space
substrate must stay usable without any of that.
"""

__all__: list[str] = []
