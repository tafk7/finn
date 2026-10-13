# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Node roots: the Space a KernelOp binds.

``node_root`` builds one Space class per op from the op's class attributes: its kernel,
with a channel on each ONNX input and output, and Params for the kernel arguments read
from the graph, each channel's tensor, and the values of an initializer the node holds.
Binding those Params to a node's facts gives the node's point.

Every channel is a boundary port (``in<i>_V``, ``out<j>_V`` by ONNX position): a kernel's
cores take their extents from the ports on its channels, so a node's choices can be
checked on the node alone. The output channels are optional: bound on its inputs only,
the root answers output inference. The platform is bound to the kernel and to every
channel, so the requirements of their cases read the device the model is built for.

A partition's shell root (``shell``) places the same kernels (``placed``), with fixed
arguments, on channels shared between nodes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, get_type_hints

from finn.core.space import UNSUPPLIED, Param, composite, default_semantics, supplied
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.values.semantics import INTEGER_TENSOR, IntegerTensor

if TYPE_CHECKING:
    from finn.custom_op.kernels.base import KernelOp


def placed(
    op: type[KernelOp], formals: Mapping[str, object], channels: Mapping[str, Channel]
) -> Kernel:
    """``op``'s kernel from ``formals`` (Params or literals) on ``channels`` (by port)."""
    references = {
        reference: channels[port] for port, reference in op.references.items() if port in channels
    }
    placement: Kernel = cast(Any, op.kernel)(**formals, **references)
    return placement


def _formal(declared: Any) -> Any:
    """A Param declared as the kernel declares ``declared``: required, optional, or with
    its default, and with its value semantics."""
    semantics = declared.explicit
    if declared.required:
        return Param(semantics=semantics)
    if declared.default is UNSUPPLIED:
        return Param(required=False, semantics=semantics)
    return Param(default=declared.default, semantics=semantics)


def node_root(op: type[KernelOp]) -> type[Kernel]:
    """``op``'s node root class; see the module docstring."""
    hints = get_type_hints(op.kernel)
    members: dict[str, Any] = {}
    annotations: dict[str, object] = {}
    for name in op.formals:
        members[name], annotations[name] = _formal(getattr(op.kernel, name)), hints[name]
    platform = members["platform"]
    channels: dict[str, Channel] = {}
    sides = (("in", op.ports), ("out", op.outputs))
    for side, ports in sides:
        for index, port in enumerate(ports):
            if port is None:
                continue
            output = side == "out"
            # An output's tensor is optional. Its semantics are given here, because its
            # guard reads them before the class is collected.
            tensor: Tensor = (
                Param(required=False, semantics=default_semantics(Tensor)) if output else Param()
            )
            members[f"{port}_tensor"] = tensor
            annotations[f"{port}_tensor"] = Tensor
            stated: dict[str, Any] = {}
            if output:
                members[f"{port}_stated"] = stated["when"] = supplied(tensor)
            if port in op.parameters:
                contents: IntegerTensor = Param(required=False, semantics=INTEGER_TENSOR)
                members[f"{port}_contents"] = stated["contents"] = contents
                annotations[f"{port}_contents"] = IntegerTensor
            channels[port] = Channel(
                tensor=tensor, port=f"{side}{index}_V", platform=platform, **stated
            )
    formals = {name: members[name] for name in op.formals}
    kernel = placed(op, formals, channels)
    identity = {"id": f"finn.custom_op.kernels.node.{op.op_type.lower()}", "version": op.op_version}
    return composite(
        f"{op.__name__}Node",
        {**identity, **members, **channels, op.member: kernel},
        base=Kernel,
        annotations=annotations,
    )


__all__ = ["node_root", "placed"]
