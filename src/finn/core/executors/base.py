# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The executor protocol: ONNX-shaped, a node of a model and its execution context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from finn.custom_op.partition.kernel_partitions import KERNEL_OPS_DOMAIN, is_partition

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

Context = dict[str, Any]
"""An execution context: every tensor of a model by name, as
``ModelWrapper.make_empty_exec_context`` makes it and qonnx's ``execute_node`` fills it."""


class Executor(Protocol):
    """Runs the nodes of a model it claims.

    ``claims`` answers for one node of ``model``, from the node and the model alone.
    ``run`` executes a claimed node: it reads the node's inputs from ``context`` and
    writes its outputs there, as qonnx's ``CustomOp.execute_node`` does.

    ``hardware`` states whether the executor runs what it claims as hardware does (a
    simulation of the hardware the node's kernels generate): a partition whole, from its
    inputs to its outputs. A run that requires hardware accepts only such an executor for
    a hardware node (``hardware_node``).
    """

    @property
    def hardware(self) -> bool: ...

    def claims(self, node: NodeProto, model: ModelWrapper) -> bool: ...

    def run(self, node: NodeProto, context: Context, model: ModelWrapper) -> None: ...


def hardware_node(node: NodeProto) -> bool:
    """Whether ``node`` stands for hardware: a KernelOp, or a partition node. The nodes
    ``Python`` runs as their reference, and a run that requires hardware refuses to."""
    return node.domain == KERNEL_OPS_DOMAIN or is_partition(node)


__all__ = ["Context", "Executor", "hardware_node", "is_partition"]
