# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Executors: what runs a model's nodes under ``finn.core.onnx_exec.execute_onnx``.

The caller chooses them, as ONNX Runtime's execution providers are chosen:
``execute_onnx(model, inputs, executors=(...))``. For each node, the first executor
that claims it runs it; qonnx's execution runs the nodes none claims (the host's ONNX
ops and the custom ops no executor knows). The graph holds no execution state.

``Python`` is the default: it runs a KernelOp's ``execute_node``, the op's reference,
and a partition node's body under the executors its caller chose. ``XSim``
(``finn.core.rtlsim_exec``, not loaded here: it loads the kernel stack)
runs a partition of KernelOps as its hardware, in XSim. An executor states whether it
is ``hardware``; a run that requires hardware refuses to run a hardware node
(``hardware_node``: a KernelOp, a partition node) any other way.

The executor protocol is ONNX-shaped: a node of a model and its execution context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

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
