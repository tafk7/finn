# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's one cut: its KernelOps as one partition, the parent graph kept.

Partitioning is a graph operation, and the kernel path's first after inference: it
decides which nodes run in the fabric and which on the host. ``CutKernelPartition``
cuts a model of KernelOps and host nodes once (``partition_kernel_ops``, the one rule of
which KernelOps go together: all of them, contiguous): the KernelOps become one
StreamingDataflowPartition (``finn.custom_op.partition``) and the nodes before and after
it stay on the host. The result is the parent graph, which the build keeps as its model.
The partition has one name, ``PARTITION`` (``partition``), for its node, its body's file
(``<directory>/partition.onnx``), the IP it is packaged as (the shell root's module,
``finn_partition``) and the block design's instance; its body carries the parent's
target. Nothing of the space is read or stored: everything after the cut, exploration
first, opens the body through the node (``kernel_partitions.partition_body``).

A channel's stated choices (``finn.channel``, its tensor's metadata) are a partition
body's only. A model with nodes on the host that states some is refused, by name: the
build's choices come in through ``kernel_choices.json`` (``Pinned``), which exploration
applies to the body. A model of KernelOps only is a body already, and the cut moves
its channels' choices into the body it makes: each follows its tensor there, and the
parent's copy of a boundary tensor's is cleared, so a channel has one home.

KernelOps that form more than one partition are refused: several partitions are to be
designed as one shell root, when CNV or Alveo needs them, not as neighbouring roots.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.create_generic_partitions import PartitionFromLambda

from finn.custom_op.kernels.base import (
    CHANNEL,
    KernelOpError,
    clear_channel_choices,
    refuse_outside_body,
)
from finn.custom_op.kernels.shell import PARTITION
from finn.custom_op.partition.kernel_partitions import (
    KERNEL_OPS_DOMAIN,
    PARTITION_DOMAIN,
    PARTITION_OP,
    kernel_partition_nodes,
)
from finn.transformation.kernels.convert import between_kernel_ops

#: The partition PartitionFromLambda puts the KernelOps in; every other node stays on
#: the host (-1).
_KERNEL_OPS = "kernels"


def _kernel_ops_together(node: NodeProto) -> str | int:
    return _KERNEL_OPS if node.domain == KERNEL_OPS_DOMAIN else -1


def partition_kernel_ops(model: ModelWrapper, directory: Path) -> ModelWrapper:
    """The parent graph of ``model``: its KernelOps as one StreamingDataflowPartition
    (``PARTITION_DOMAIN``, the domain imported), its body ``model`` attribute a file in
    ``directory``, every other node on the host, and the channel choices its tensors
    state in the body only (module docstring). KernelOps a host node stands between
    are refused (``KernelOpError``, naming the host nodes: the partition would depend on
    itself); a model without KernelOps is returned as it is."""
    between = between_kernel_ops(model)
    if between:
        raise KernelOpError(
            f"{len(between)} host nodes sit between KernelOps, so a partition of the "
            f"KernelOps would depend on itself: {', '.join(between)}"
        )
    refuse_outside_body(model, model.tensors_stating(CHANNEL))
    partitioning = PartitionFromLambda(  # type: ignore[no-untyped-call]
        partitioning=_kernel_ops_together, partition_dir=str(directory)
    )
    parent: ModelWrapper = model.transform(partitioning)
    clear_channel_choices(parent)
    nodes = parent.get_nodes_by_op_type("GenericPartition")
    imported = {opset.domain for opset in parent.model.opset_import}
    if nodes and PARTITION_DOMAIN not in imported:
        parent.model.opset_import.append(helper.make_opsetid(PARTITION_DOMAIN, 1))
    for node in nodes:
        node.op_type, node.domain = PARTITION_OP, PARTITION_DOMAIN
    return parent


class CutKernelPartition(Transformation):
    """Cut the model's KernelOps once into the partition ``PARTITION``, its body saved in
    ``directory``; see the module docstring. A model whose KernelOps do not form one
    partition, or that holds a partition already, is refused (``KernelOpError``)."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if model.get_nodes_by_op_type(PARTITION_OP):
            raise KernelOpError("the model holds a partition already: the kernel path cuts once")
        self.directory.mkdir(parents=True, exist_ok=True)
        cut = Path(tempfile.mkdtemp(prefix="cut_", dir=self.directory))
        parent = partition_kernel_ops(model, cut)
        found = kernel_partition_nodes(parent)
        if len(found) != 1:
            raise KernelOpError(
                f"the KernelOps form {len(found)} partitions; the kernel path cuts them into "
                "one (a second partition of KernelOps waits for several partitions designed "
                "as one shell root)"
            )
        (node,) = found
        if any(each.name == PARTITION for each in parent.graph.node if each is not node):
            raise KernelOpError(f"a host node is named {PARTITION!r}, the partition's name")
        op = getCustomOp(node)
        cut_file = Path(str(op.get_nodeattr("model")))
        body = ModelWrapper(str(cut_file))
        body_file = self.directory / f"{PARTITION}.onnx"
        body.save(str(body_file))
        cut_file.unlink()
        cut.rmdir()
        node.name = PARTITION
        op.set_nodeattr("model", str(body_file))
        return parent, False


__all__ = ["CutKernelPartition", "partition_kernel_ops"]
