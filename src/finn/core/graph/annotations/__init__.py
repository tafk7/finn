# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph metadata FINN writes and reads: one module per namespace of qonnx's typed
metadata (``qonnx.core.metadata``), with its keys, its readers and its writers.

- ``platform``: ``finn.platform``, the build target, on the model;
- ``channel``: ``finn.channel``, a channel's choices, on the tensor it carries;
- ``choices``: ``finn.choices``, the pins a partition's body holds and the throughput it
  was explored to;
- ``outputs``: ``finn.outputs``, what a build made of a partition.

``outputs`` imports qonnx only, so the partition's readers and the builder read it
without the kernel stack. The package imports nothing itself.
"""
