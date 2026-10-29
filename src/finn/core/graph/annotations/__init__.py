# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph metadata FINN writes and reads, one module per namespace of qonnx's typed
metadata (``qonnx.core.metadata``): ``platform`` (``finn.platform``, the build target),
``boundary`` (``finn.boundary``, what the build states of a partition's boundary, by
tensor), ``channel`` (``finn.channel``, a channel's choices) and ``outputs``
(``finn.outputs``, what a build made of a partition, the search's record among it).
"""
