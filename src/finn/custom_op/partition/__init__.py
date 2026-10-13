# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.partition``: the parent graph's node for the
partition of KernelOps.

qonnx loads a domain by importing it and takes its op classes from ``__all__``, so this
module exports only op classes.
"""

from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.genericpartition import GenericPartition

from finn.core.graph.partition import body_file

opset_version = 1


class StreamingDataflowPartition(GenericPartition):
    """The parent graph's node for the partition of KernelOps; ``model`` is the body's
    file. Executing it runs the body with the caller's executors and start and end nodes
    (``finn.core.onnx_exec.running``), and writes the body's outputs the run reached.

    Body tensors the context already holds as ``<node>_<tensor>`` are fed to the body,
    so a run can start inside it. When the run returns its full context, the body's
    tensors are added to it under those names. The run decides that, not the node: it
    has no ``return_full_exec_context`` attribute."""

    def get_nodeattr_types(self) -> dict[str, tuple[str, bool, Any]]:
        return {"model": ("s", True, "")}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        # Imported when it runs: resolving the domain loads no executor.
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
                # "" is qonnx's placeholder for an unset optional input, no body tensor.
                if name and name not in outputs:
                    context[prefix + name] = value


__all__ = ["StreamingDataflowPartition"]
