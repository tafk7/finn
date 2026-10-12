# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's one cut: its KernelOps as one partition, the parent graph kept.

Partitioning is a graph operation, and the kernel path's first after inference: it
decides which nodes run in the fabric and which on the host. ``CutKernelPartition``
cuts a model of KernelOps and host nodes once, by one rule of which KernelOps go
together (all of them, contiguous): the KernelOps become one StreamingDataflowPartition
(``finn.custom_op.partition``) and the nodes before and after it stay on the host. The
result is the parent graph, which the build keeps as its model.
The partition has one name, ``PARTITION`` (``partition``), for its node, its body's file
(``<directory>/partition.onnx``), the IP it is packaged as (the shell root's module,
``finn_partition``) and the block design's instance; its body carries the parent's
target. Everything after the cut, exploration first, opens the body through the node
(``finn.core.graph.partition.partition_body``).

The cut creates the accelerator's design space, so it gives its body's choices their
baselines (``BaselineChoices``, ``shell.baseline_choices``): the body's shell root is
built, its nodes' choices (stored at conversion, ``ToKernelOps``) replayed in context,
and every choice still open with cases that is not required takes its baseline: the
channels' own, created here, where a channel first has both its ends, and a node's
choice replay drops as stale (``CutKernelPartition.dropped``, with why). Every value is
then persisted, a kernel's on its node, a channel's on its tensor (``finn.channel``). A
required choice (a FIFO's depth) stays open for a strategy.

A channel's stated choices (``finn.channel``, its tensor's metadata) are a partition
body's only. A model with nodes on the host that states some is refused, by name: the
build's choices come in through ``kernel_choices.json`` (``Pinned``), which exploration
applies to the body. A model of KernelOps only is a body already, and the cut moves
its channels' choices into the body it makes: each follows its tensor there, and the
parent's copy of a boundary tensor's is cleared, so a channel has one home.

A second partition of KernelOps cannot arise here: the cut puts every KernelOp in its
one partition, and a model that holds a partition already is refused (the shell root
refuses KernelOps outside its partition too). Several partitions are to be designed as
one shell root, when CNV or Alveo needs them, not as neighbouring roots.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.create_generic_partitions import PartitionFromLambda

from finn.core.graph.annotations.channel import CHANNEL, clear_channel_choices, refuse_outside_body
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, PARTITION_DOMAIN, PARTITION_OP
from finn.custom_op.kernels.shell import PARTITION, baseline_choices
from finn.transformation.kernels.convert import between_kernel_ops

#: The partition PartitionFromLambda puts the KernelOps in; every other node stays on
#: the host (-1).
_KERNEL_OPS = "kernels"


def _kernel_ops_together(node: NodeProto) -> str | int:
    return _KERNEL_OPS if node.domain == KERNEL_OPS_DOMAIN else -1


class BaselineChoices(Transformation):
    """Every choice of a partition's body at its baseline where the body leaves it open,
    persisted, and its nodes' stored choices re-checked in context, a stale one
    re-baselined (``shell.baseline_choices``): what the cut does to the body it makes,
    for a body made by hand (a test's, or one ``save_channels`` states choices on).
    ``dropped`` holds the choices replay dropped as stale, by root key, with why."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped: dict[str, str] = {}

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.dropped = baseline_choices(model)
        return model, False


class CutKernelPartition(Transformation):
    """Cut the model's KernelOps once into the partition ``PARTITION``, its body saved in
    ``directory`` with every choice at its baseline; see the module docstring. Refused
    (``KernelOpError``): a model that holds a partition already, one without KernelOps,
    one where host nodes stand between KernelOps (named: the partition would depend on
    itself), one whose host nodes state channel choices, and one with a host node named
    ``PARTITION``. ``dropped`` holds the nodes' choices the cut found stale in context
    and re-baselined, by root key, with why."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.dropped: dict[str, str] = {}

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if model.get_nodes_by_op_type(PARTITION_OP):
            raise KernelOpError("the model holds a partition already: the kernel path cuts once")
        host = [node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN]
        if len(host) == len(model.graph.node):
            raise KernelOpError("the model holds no KernelOps: there is nothing to cut")
        between = between_kernel_ops(model)
        if between:
            raise KernelOpError(
                f"{len(between)} host nodes sit between KernelOps, so a partition of the "
                f"KernelOps would depend on itself: {', '.join(between)}"
            )
        refuse_outside_body(model, model.tensors_stating(CHANNEL))
        if PARTITION in host:
            raise KernelOpError(f"a host node is named {PARTITION!r}, the partition's name")
        self.directory.mkdir(parents=True, exist_ok=True)
        cut = Path(tempfile.mkdtemp(prefix="cut_", dir=self.directory))
        partitioning = PartitionFromLambda(  # type: ignore[no-untyped-call]
            partitioning=_kernel_ops_together, partition_dir=str(cut)
        )
        parent: ModelWrapper = model.transform(partitioning)
        clear_channel_choices(parent)
        # PartitionFromLambda makes one node per partition id, named by it; the KernelOps
        # share one id, so this is the one partition node the cut makes.
        (node,) = [
            each
            for each in parent.get_nodes_by_op_type("GenericPartition")
            if each.name == f"GenericPartition_{_KERNEL_OPS}"
        ]
        node.op_type, node.domain, node.name = PARTITION_OP, PARTITION_DOMAIN, PARTITION
        if PARTITION_DOMAIN not in {opset.domain for opset in parent.model.opset_import}:
            parent.model.opset_import.append(helper.make_opsetid(PARTITION_DOMAIN, 1))
        op = getCustomOp(node)
        cut_file = Path(str(op.get_nodeattr("model")))
        body = ModelWrapper(str(cut_file))
        self.dropped = baseline_choices(body)
        body_file = self.directory / f"{PARTITION}.onnx"
        body.save(str(body_file))
        cut_file.unlink()
        cut.rmdir()
        op.set_nodeattr("model", str(body_file))
        return parent, False


__all__ = ["BaselineChoices", "CutKernelPartition"]
