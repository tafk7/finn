# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An op's placement, and the node root generated from it: the Space a KernelOp binds.

A KernelOp states how its kernel sits in a graph once, as data
(``finn.custom_op.kernels.base.KernelOp``): the ``kernel`` class, the
``formals`` it reads from the graph (the kernel's own: facts stay facts), the
reference input each port's channel binds (``references``), and for a
parameter port the kernel's views of the tensor and of the value its channel
carries (``parameters``: MatMul's ``weight_tensor`` and ``weight_values``).
``placed`` is that placement: the kernel on channels by port, from formals
that are Params or literals. Two things are built with it:

- the **node root** (``node_root``), one class per op, compiled once: the
  formals are its Params, declared as the kernel declares them; each edge's
  channel, an input's or an output's, carries the tensor the graph states,
  a Param of its own (``x_tensor``, ``y_tensor``); a parameter port's channel
  carries the kernel's views, its ``contents`` supplied only when the kernel
  holds the value (the view's guard: an initializer the node owns, which the
  channel's ``source`` then stores; weights on a graph tensor arrive like any
  edge, with no source). Every channel is a boundary, ``in<i>_V`` and
  ``out<j>_V`` by ONNX position: a kernel's cores bind their extents from the
  ports on its channels (``kernel-extents``), so a node alone can commit its
  own choices only on channels;
- the **partition root** (``finn.custom_op.kernels.partition``): the same
  placement with literal formals, on channels the partition declares once per
  ONNX tensor and shares between nodes.

What the graph decides is a declaration here, never a choice. The platform is a
fact too: the target's capabilities and its clock period (``read_target``),
bound to the kernel and to every channel, so the requirements of their value
cases (``requires``) read the device the model is built for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast, get_type_hints

from finn.core.space import Param, composite
from finn.core.space.declarations import UNSUPPLIED
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel

if TYPE_CHECKING:
    from finn.custom_op.kernels.base import KernelOp


def placed(
    op: type[KernelOp],
    formals: Mapping[str, object],
    channels: Mapping[str, Channel],
    valued: Iterable[str],
) -> Kernel:
    """``op``'s kernel from ``formals`` (Params or literals) on ``channels`` (by port);
    each parameter port in ``valued`` carries the kernel's views of its tensor and value."""
    references = {
        reference: channels[port] for port, reference in op.references.items() if port in channels
    }
    kernel: Any = cast(Any, op.kernel)(**formals, **references)
    for port in valued:
        tensor, value = op.parameters[port]
        channels[port].tensor = getattr(kernel, tensor)
        channels[port].contents = getattr(kernel, value)
    placement: Kernel = kernel
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
            if port in op.parameters:
                channels[port] = Channel(port=f"{side}{index}_V", platform=platform)
                continue
            tensor: Tensor = Param()
            members[f"{port}_tensor"] = tensor
            annotations[f"{port}_tensor"] = Tensor
            channels[port] = Channel(tensor=tensor, port=f"{side}{index}_V", platform=platform)
    formals = {name: members[name] for name in op.formals}
    kernel = placed(op, formals, channels, op.parameters)
    identity = {"id": f"finn.custom_op.kernels.node.{op.op_type.lower()}", "version": op.op_version}
    return composite(
        f"{op.__name__}Node",
        {**identity, **members, **channels, op.member: kernel},
        base=Kernel,
        annotations=annotations,
    )


__all__ = ["node_root", "placed"]
