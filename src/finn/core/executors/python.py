# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The default executor: the KernelOps' ``execute_node``, the reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from finn.core.executors.base import hardware_node

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.core.executors.base import Context


@dataclass(frozen=True)
class Python:
    """Claims KernelOps and partition nodes, and runs each through its op's
    ``execute_node``.

    A KernelOp's ``execute_node`` is the op's reference, the oracle its hardware is
    checked against. A partition node's ``execute_node`` executes its body with
    ``finn.core.onnx_exec.execute_onnx`` as the run that reached it asks
    (``finn.core.onnx_exec.running``): under its executors, so the caller's choice
    reaches the body's nodes, from or up to a node of the body where the run starts or
    ends there, and with the body's tensors in a full context. It is no hardware: a run
    that requires hardware refuses it.
    """

    hardware: ClassVar[bool] = False

    def claims(self, node: NodeProto, model: ModelWrapper) -> bool:
        return hardware_node(node)

    def run(self, node: NodeProto, context: Context, model: ModelWrapper) -> None:
        model.get_customop_wrapper(node).execute_node(context, model.graph)
