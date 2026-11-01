# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.partition``: the parent graph's node for the
core of KernelOps.

qonnx loads a domain by importing it and takes its op classes from ``__all__``, so this
module exports only op classes. Resolving the domain loads neither the kernels nor the
executors: the node's readings import them when they run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.genericpartition import GenericPartition

from finn.core.graph.partition import body_file

if TYPE_CHECKING:
    from finn.core.search import Key
    from finn.custom_op.kernels.shell import Configured
    from finn.kernels.dataflow_core import Bound, DataflowCoreKernel

opset_version = 1


class DataflowCore(GenericPartition):
    """The parent graph's node for the core of KernelOps, the
    partition the cut makes; ``model`` is the body's
    file. Executing it runs the body with the caller's executors and start and end nodes
    (``finn.core.onnx_exec.running``), and writes the body's outputs the run reached.

    Body tensors the context already holds as ``<node>_<tensor>`` are fed to the body,
    so a run can start inside it. When the run returns its full context, the body's
    tensors are added to it under those names. The run decides that, not the node: it
    has no ``return_full_exec_context`` attribute.

    Its readings are a KernelOp's of its kernel, each a function of its body
    (``finn.custom_op.kernels.shell``: its core in the shell of its target, its module
    named by the node's name), so a step holding the body calls the same function:
    ``body``, ``bound`` (its design space, nothing chosen), ``point`` (its body's stored
    choices bound, a stale one dropped with why), ``persist`` (a design space's choices
    written onto its body) and ``configured`` (what emission reads)."""

    def get_nodeattr_types(self) -> dict[str, tuple[str, bool, Any]]:
        return {"model": ("s", True, "")}

    def body(self) -> ModelWrapper:
        """Its body, the model of KernelOps its ``model`` attribute names."""
        return ModelWrapper(body_file(self.onnx_node))

    def bound(self) -> DataflowCoreKernel:
        """Its core's design space, nothing chosen: its body's KernelOps and channels
        composed in the shell of its target (``shell.core_space``)."""
        from finn.custom_op.kernels.shell import core_space  # noqa: PLC0415

        return core_space(self.body(), self.onnx_node.name)

    def point(self, extra: Mapping[Key, object] | None = None) -> Bound:
        """Its core's design space with its body's stored choices bound (``shell.core_point``:
        a value outside its domain dropped as stale, with why, ``Bound.stale``), and
        ``extra``, by ``(member, key)``, set on it as one change; a change refused is a
        ``ValueError`` naming each refused value or rule
        (``finn.kernels.explore.CoreComposition.with_choices``)."""
        from finn.custom_op.kernels.shell import core_point  # noqa: PLC0415
        from finn.kernels.dataflow_core import Bound  # noqa: PLC0415
        from finn.kernels.explore import CoreComposition  # noqa: PLC0415

        found = core_point(self.body(), self.onnx_node.name)
        if not extra:
            return found
        changed = CoreComposition(found.point).with_choices(extra)
        return Bound(changed.point, found.dropped, found.stale)

    def persist(self, point: DataflowCoreKernel) -> None:
        """Write ``point``'s choices, a design space of its core, onto its body's KernelOps
        and tensors, each owner's whole (``network.persist``), and save its body."""
        from finn.custom_op.kernels.network import persist  # noqa: PLC0415

        body = self.body()
        persist(body, point)
        body.save(body_file(self.onnx_node))

    def configured(self) -> Configured:
        """Its core as emission reads it (``shell.configured_network``): its composition in
        its shell, its module and its cycles, refused where a stored value is stale, a rule
        broken or a choice open, named."""
        from finn.custom_op.kernels.shell import configured_network  # noqa: PLC0415

        return configured_network(self.body(), self.onnx_node.name)

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


__all__ = ["DataflowCore"]
