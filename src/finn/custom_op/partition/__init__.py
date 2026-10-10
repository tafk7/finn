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
    outputs renamed to the body's) as the run that reached it asks
    (``finn.core.onnx_exec.running``): under its executors, so the caller's choice
    reaches the body's nodes; from or up to the node of the body the run starts or ends
    at; and, when the run returns its full context, with the body's tensors in it under
    the node's name, ``<node>_<tensor>`` (its outputs under the parent's names).

    A body's tensor the context holds under that name (a caller's, from a full context)
    is given to the body's run as an input, so a run starts inside the body from it. Of
    the body's outputs, those its run reached are written to the context. Whether a run
    returns its full context is the run's, never the graph's: the node has no
    ``return_full_exec_context`` attribute."""

    def get_nodeattr_types(self) -> dict[str, tuple[str, bool, Any]]:
        return {"model": ("s", True, "")}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        # Imported here: the executors read kernel_partitions, under this package.
        from finn.core.onnx_exec import execute_onnx, running, window  # noqa: PLC0415

        node = self.onnx_node
        body = ModelWrapper(body_file(node))
        run = running()
        prefix = f"{node.name}_"
        given = {
            name: context[prefix + name]
            for name in body.get_all_tensor_names()
            if prefix + name in context
        }
        inputs = given | {
            body_input.name: context[name]
            for name, body_input in zip(node.input, body.graph.input, strict=True)
        }
        ran = execute_onnx(
            body,
            inputs,
            run.full_context,
            run.start_node,
            run.end_node,
            executors=run.executors,
        )
        whole = run.start_node is None and run.end_node is None
        reached = {
            tensor
            for inner in window(body, run.start_node, run.end_node)
            for tensor in inner.output
        }
        for name, body_output in zip(node.output, body.graph.output, strict=True):
            if whole or body_output.name in reached:
                context[name] = ran[body_output.name]
        if run.full_context:
            outputs = {body_output.name for body_output in body.graph.output}
            for name, value in ran.items():
                if name not in outputs:
                    context[prefix + name] = value


__all__ = ["StreamingDataflowPartition"]
