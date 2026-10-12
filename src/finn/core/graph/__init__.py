# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN's view of an ONNX model.

- ``annotations``: the graph metadata FINN writes and reads, one module per namespace:
  the build target (``finn.platform``), a channel's choices on its tensor
  (``finn.channel``), a partition body's pins and objective (``finn.choices``), and what a
  build made of a partition (``finn.outputs``).
- ``facts``: what a graph states of a tensor, as the KernelOps read it: its datatype
  annotation, its shape as rows, an initializer of integers and its range.
- ``partition``: the kernel path's partition, its node and its body, and the digest of
  the design a body states.
- ``walk``: walks over a graph's dataflow, by its tensors.
- ``errors``: what the readers raise for a fact the graph does not state or contradicts.

The package imports nothing itself; each module is imported by name.
"""
