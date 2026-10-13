# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The cut: all KernelOps into one partition, the other nodes left on the host.

``CutKernelPartition`` puts every KernelOp in one ``StreamingDataflowPartition`` and
returns the parent graph, which the build keeps as its model. The partition is named
``partition`` (``PARTITION``) for its node, its body's file
(``<directory>/partition.onnx``), its IP (``finn_partition``) and its block-design
instance. The body carries the parent's target. Everything after the cut opens the body
through the node (``finn.core.graph.partition.partition_body``).

The cut is where each channel first has both its ends, so it sets the channel choices
to their baselines, and checks the nodes' choices again in context: a node choice that
does not fit there is reset to its baseline and reported (``dropped``). A required choice (a
FIFO's depth) stays open for a strategy.

Channel choices may be stored only in a partition's body. A model with host nodes that
stores some is refused (a build's choices come in through ``kernel_choices.json``
instead); a model of KernelOps only keeps them, moved into the body, and the parent's
copy is cleared.
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
from finn.custom_op.kernels.network import PARTITION, baseline_choices
from finn.transformation.kernels.convert import between_kernel_ops

#: The partition PartitionFromLambda puts the KernelOps in; every other node stays on
#: the host (-1).
_KERNEL_OPS = "kernels"


def _kernel_ops_together(node: NodeProto) -> str | int:
    return _KERNEL_OPS if node.domain == KERNEL_OPS_DOMAIN else -1


class BaselineChoices(Transformation):
    """What the cut does to the body it makes, for a body made by hand: every open
    choice at its baseline, and the nodes' stored choices checked in context
    (``network.baseline_choices``). ``dropped`` lists those reset, by root key, with
    why."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped: dict[str, str] = {}

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.dropped = baseline_choices(model)
        return model, False


class CutKernelPartition(Transformation):
    """Cut the model's KernelOps into the partition ``PARTITION``, its body saved in
    ``directory`` at its baselines (see the module docstring). Refuses a model that
    holds a partition already or no KernelOps, one with host nodes between KernelOps or
    storing channel choices, and one with a host node named ``PARTITION``. ``dropped``
    lists the node choices reset at the cut, by root key, with why."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.dropped: dict[str, str] = {}

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if model.get_nodes_by_op_type(PARTITION_OP):
            raise KernelOpError("the model holds a partition already: a build cuts once")
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
