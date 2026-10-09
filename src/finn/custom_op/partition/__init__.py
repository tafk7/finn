# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.partition``: the kernel path's partition node.

The kernel path cuts its KernelOps once (``finn.transformation.kernels.cut``): the parent
graph holds one ``StreamingDataflowPartition`` of this domain, whose ``model``
attribute names its body, a model of KernelOps. The node runs its body, nothing more:
no placement, no estimates. What the flow reads of a partition and what a build made of
it are ``kernel_partitions``'s.

qonnx resolves this domain by importing it and reads its op classes from ``__all__``, so
this module exports op classes only. It imports qonnx only.
"""

from qonnx.custom_op.general.genericpartition import GenericPartition

opset_version = 1


class StreamingDataflowPartition(GenericPartition):
    """The parent graph's node of a partition of KernelOps: ``model``, the body's file,
    which the node executes on its inputs (qonnx's ``GenericPartition``, the inputs and
    outputs renamed to the body's)."""


__all__ = ["StreamingDataflowPartition"]
