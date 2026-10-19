# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The cut: all KernelOps into one partition, the other nodes left on the host.

``CutKernelPartition`` puts every KernelOp in one ``StreamingDataflowPartition`` and
returns the parent graph, which the build keeps as its model. The partition is named
``partition`` (``PARTITION``) for its node, its body's file
(``<directory>/partition.onnx``), its IP (``finn_partition``) and its block-design
instance. The body carries the parent's target. Everything after the cut opens the body
through the node (``finn.core.graph.partition.partition_body``).

The cut is where each channel first has both its ends, so it makes the channel choices:
**interim**, until the ends pass, each takes its first valid value
(``finn.kernels.network.interim_channel_defaults``). It binds the nodes' choices again in
context: a stored value outside its domain there is dropped as stale, left open for a
strategy and reported (``dropped``); a node choice left open by a body that never went
through conversion is set by conversion's policies, reported (``late``); a choice no value
can be set for is left open and reported (``unset``). A required choice (a FIFO's depth)
stays open for a strategy.

Channel choices may be stored only in a partition's body. A model with host nodes that
stores some is refused (a build fixes them in a step after the cut, with ``Hold``); a
model of KernelOps only keeps them, moved into the body, and the parent's copy is cleared.
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
from finn.core.reporting.findings import Finding
from finn.custom_op.kernels.network import PARTITION
from finn.custom_op.kernels.shell import baseline_choices
from finn.transformation.kernels.convert import between_kernel_ops

#: The partition PartitionFromLambda puts the KernelOps in; every other node stays on
#: the host (-1).
_KERNEL_OPS = "kernels"


def _kernel_ops_together(node: NodeProto) -> str | int:
    return _KERNEL_OPS if node.domain == KERNEL_OPS_DOMAIN else -1


class BaselineChoices(Transformation):
    """What the cut does to the body it makes, for a body made by hand: the channels'
    choices made, the nodes' stored choices bound in context, and a node choice the body
    leaves open set by conversion's policies (``shell.baseline_choices``). ``dropped``
    lists the stored choices dropped as stale, ``late`` the node choices set late and
    ``unset`` the choices no value could be set for, by owner and key
    (``MatMul_0.compute.packed.pe``), each with its findings."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped: dict[str, tuple[Finding, ...]] = {}
        self.late: dict[str, tuple[Finding, ...]] = {}
        self.unset: dict[str, tuple[Finding, ...]] = {}

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        made = baseline_choices(model)
        self.dropped, self.late, self.unset = dict(made.stale), dict(made.late), dict(made.unset)
        return model, False


class CutKernelPartition(Transformation):
    """Cut the model's KernelOps into the partition ``PARTITION``, its body saved in
    ``directory`` with its channels' choices made (see the module docstring). Refuses a
    model that holds a partition already or no KernelOps, one with host nodes between
    KernelOps or storing channel choices, and one with a host node named ``PARTITION``.
    ``dropped``, ``late`` and ``unset``: as ``BaselineChoices``'."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.dropped: dict[str, tuple[Finding, ...]] = {}
        self.late: dict[str, tuple[Finding, ...]] = {}
        self.unset: dict[str, tuple[Finding, ...]] = {}

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if model.get_nodes_by_op_type(PARTITION_OP):
            raise KernelOpError.refused(
                "partition-present",
                CutKernelPartition.__name__,
                "the model holds a partition already: a build cuts once",
            )
        host = [node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN]
        if len(host) == len(model.graph.node):
            raise KernelOpError.refused(
                "kernel-ops-absent",
                CutKernelPartition.__name__,
                "the model holds no KernelOps: there is nothing to cut",
            )
        between = between_kernel_ops(model)
        if between:
            raise KernelOpError.refused(
                "host-between",
                CutKernelPartition.__name__,
                f"{len(between)} host nodes sit between KernelOps, so a partition of the "
                f"KernelOps would depend on itself: {', '.join(between)}",
            )
        refuse_outside_body(model, model.tensors_stating(CHANNEL))
        if PARTITION in host:
            raise KernelOpError.refused(
                "partition-name-taken",
                CutKernelPartition.__name__,
                f"a host node is named {PARTITION!r}, the partition's name",
                node=PARTITION,
            )
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
        made = baseline_choices(body)
        self.dropped, self.late, self.unset = dict(made.stale), dict(made.late), dict(made.unset)
        body_file = self.directory / f"{PARTITION}.onnx"
        body.save(str(body_file))
        cut_file.unlink()
        cut.rmdir()
        op.set_nodeattr("model", str(body_file))
        return parent, False


__all__ = ["BaselineChoices", "CutKernelPartition"]
