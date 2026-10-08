# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's one cut: its KernelOps as one partition, the parent graph kept.

Partitioning is a graph operation, and the kernel path's first after inference: it
decides which nodes run in the fabric and which on the host. ``CutKernelPartition``
cuts a model of KernelOps and host nodes once (``CreateDataflowPartition``, whose
segmentation is the one rule of which KernelOps go together: all of them, contiguous):
the KernelOps become one StreamingDataflowPartition and the nodes before and after it
stay on the host. The result is the parent graph, which the build keeps as its model.
The partition has one name, ``PARTITION`` (``partition``), for its node, its body's file
(``<directory>/partition.onnx``), the IP it is packaged as (the shell root's module,
``finn_partition``) and the block design's instance; its body carries the parent's
target. Nothing of the space
is read or stored: everything after the cut, exploration first, opens the body through
the node (``kernel_partitions.partition_body``).

KernelOps that form more than one partition are refused: several partitions are to be
designed as one shell root, when CNV or Alveo needs them, not as neighbouring roots.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.shell import PARTITION
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.kernel_partitions import (
    PARTITION_OP,
    kernel_partition_nodes,
)


class CutKernelPartition(Transformation):
    """Cut the model's KernelOps once into the partition ``PARTITION``, its body saved in
    ``directory``; see the module docstring. A model whose KernelOps do not form one
    partition, or that holds a partition already, is refused."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if model.get_nodes_by_op_type(PARTITION_OP):
            raise ValueError("the model holds a partition already: the kernel path cuts once")
        self.directory.mkdir(parents=True, exist_ok=True)
        cut = Path(tempfile.mkdtemp(prefix="cut_", dir=self.directory))
        parent = model.transform(CreateDataflowPartition(partition_model_dir=str(cut)))
        found = kernel_partition_nodes(parent)
        partitions = parent.get_nodes_by_op_type(PARTITION_OP)
        if len(found) != 1 or len(partitions) != 1:
            raise ValueError(
                f"the KernelOps form {len(found)} partitions beside {len(partitions) - len(found)} "
                "others; the kernel path cuts them into one (a second partition of KernelOps "
                "waits for several partitions designed as one shell root)"
            )
        (node,) = found
        if any(each.name == PARTITION for each in parent.graph.node if each is not node):
            raise ValueError(f"a host node is named {PARTITION!r}, the partition's name")
        op = getCustomOp(node)
        cut_file = Path(op.get_nodeattr("model"))
        body = ModelWrapper(str(cut_file))
        body_file = self.directory / f"{PARTITION}.onnx"
        body.save(str(body_file))
        cut_file.unlink()
        cut.rmdir()
        node.name = PARTITION
        op.set_nodeattr("model", str(body_file))
        return parent, False


__all__ = ["CutKernelPartition"]
