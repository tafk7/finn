# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.partition``: the kernel path's partition node.

The kernel path cuts its KernelOps once (``finn.transformation.kernels.cut``): the parent
graph holds one ``StreamingDataflowPartition`` of this domain, whose ``model``
attribute names its body, a model of KernelOps. The node runs its body, nothing more:
no placement, no estimates. What the flow reads of a partition and what a build made of
it are ``kernel_partitions``'s.

qonnx resolves this domain by importing it and reads its op classes from ``__all__``, so
this module exports op classes only. It imports qonnx and ``kernel_partitions`` only;
the executors it runs its body with (``finn.core.onnx_exec``), when it runs.
"""

from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.genericpartition import GenericPartition

from finn.custom_op.partition.kernel_partitions import body_file

opset_version = 1


class StreamingDataflowPartition(GenericPartition):
    """The parent graph's node of a partition of KernelOps: ``model``, the body's file,
    which the node executes on its inputs (qonnx's ``GenericPartition``, the inputs and
    outputs renamed to the body's) under the executors of the run that reached it
    (``finn.core.onnx_exec.executing``), so the caller's choice reaches the body's
    nodes."""

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        # Imported here: the executors read kernel_partitions, under this package.
        from finn.core.onnx_exec import execute_onnx, executing  # noqa: PLC0415

        node = self.onnx_node
        body = ModelWrapper(body_file(node))
        full = self.get_nodeattr("return_full_exec_context") == 1
        inputs = {
            body_input.name: context[name]
            for name, body_input in zip(node.input, body.graph.input, strict=True)
        }
        ran = execute_onnx(body, inputs, full, executors=executing())
        for name, body_output in zip(node.output, body.graph.output, strict=True):
            context[name] = ran[body_output.name]
        if full:
            outputs = {body_output.name for body_output in body.graph.output}
            for name, value in ran.items():
                if name not in outputs:
                    context[f"{node.name}_{name}"] = value


__all__ = ["StreamingDataflowPartition"]
