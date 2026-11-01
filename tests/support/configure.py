# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Choices set on a model, or on a network, the way the compiler sets them.

``composition(model)``: the core of ``model``'s KernelOps, its stored choices bound, read
by key. ``bound_in_shell(model)`` and ``core_in_shell(model)``: the core of ``model``'s KernelOps
in the shell of its target, as exploration binds a partition's body
(``finn.custom_op.kernels.shell.shell_point``, ``finn.kernels.shell``), its module named as
the cut names its node (``CUT_NAME``) unless a test names it: its stored choices bound
(``Bound``), or its composition, as the seam reads it.

``configured(model, {graph name: {key: value}})``: the KernelOps' core of ``model`` bound in
the shell of its target (``bound_in_shell``), the values set through the seam as a strategy
sets them (``Seam.set``, one batch, as ``Hold`` does), then each channel's open choices that
are not required set by the channel's policies, as the cut sets them
(``CoreComposition.defaulted``, once over the channels; a FIFO's depth stays open), and the
result persisted (``finn.custom_op.kernels.network.persist``, the one writer of choices).
What the
seam refuses is refused, by its findings (the seam's own codes, ``not-a-decision``,
``held``, or the engine's: a value outside its domain, ``domain-membership``, or each rule
the batch would break) at the top of the ``Refusal``, so a test asserts it by code
(``support.refusals.refused``); an owner that is no member of the network is ``Hold``'s own
refusal (``member-unknown``); a channel's choice on a graph with nodes on the host is
refused by ``persist``, the one writer, before anything is written
(``channel-choices-outside-body``); on such a graph, choices of kernels alone are written on
their nodes (as ``persist`` writes a node's), and no channel's are made. A set-up may name
any choice, a selector and the choices under its case together; a kernel's choice it leaves
open stays open.

The seam has no clearing: a value of ``None`` is no value of any Decision, so the engine
refuses it. A test of a stale or cleared choice writes the raw ONNX attribute instead.

``write_raw(model, node_name, key, value)`` writes a choice as the node's raw ONNX
attribute, as a graph FINN did not write holds it (or, with ``None``, removes it): no
network, no seam, no checking. It stands for the read side's tests and for a cleared
state, which ``configured`` cannot state.

A test helper, imported as ``support.configure``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import CUT_NAME, KERNEL_OPS_DOMAIN
from finn.core.reporting.findings import Refusal
from finn.core.search import Accepted, Key, Refused
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import bound_from, network, persist
from finn.custom_op.kernels.shell import body_row, row_facts, row_offer
from finn.kernels.composer import DataflowCoreDeclaration
from finn.kernels.dataflow_core import Bound
from finn.kernels.explore import CoreComposition, Seam
from finn.kernels.shell import ShellKernel, shell


def composition(
    model: ModelWrapper, nodes: Iterable[NodeProto] | None = None, *, name: str = CUT_NAME
) -> CoreComposition:
    """The core of ``nodes`` (every node of ``model`` where None), KernelOp nodes of
    ``model``, outside any shell (``finn.custom_op.kernels.network.network``), its module
    named ``name``, its stored choices bound, read by key."""
    selected = model.graph.node if nodes is None else nodes
    return CoreComposition(network(model, selected, name=name).point)


def bound_in_shell(
    model: ModelWrapper, nodes: Iterable[NodeProto] | None = None, *, name: str = CUT_NAME
) -> Bound[ShellKernel]:
    """The core of ``nodes`` (every node of ``model`` where None), KernelOp nodes of
    ``model``, in the shell of its target (``finn.custom_op.kernels.shell.shell_point``:
    its row offers the mover on each edge and states its budgets), its module named
    ``name``, its stored choices bound."""
    row = body_row(model)
    offer = row_offer(row)
    facts = row_facts(row)
    selected = model.graph.node if nodes is None else nodes

    def made(declared: DataflowCoreDeclaration) -> ShellKernel:
        return shell(declared, row=facts, offer=offer)

    return bound_from(model, selected, name=name, made=made, offer=offer)


def core_in_shell(
    model: ModelWrapper, nodes: Iterable[NodeProto] | None = None, *, name: str = CUT_NAME
) -> CoreComposition:
    """``bound_in_shell``'s design space, read as the seam reads it."""
    return CoreComposition(bound_in_shell(model, nodes, name=name).point)


def configured(model: ModelWrapper, choices: Mapping[str, Mapping[str, object]]) -> ModelWrapper:
    """``model`` with ``choices`` (by graph name: a kernel's under its node, a channel's
    under its tensor) set the way the compiler sets them: the network of the model's
    KernelOps bound, the values set through the seam as a strategy sets them, one batch,
    each channel's open choices then set by the channel's policies, as the cut sets them,
    the result persisted (``persist``); a value the seam refuses is refused, by its code.
    ``model`` is changed in place and returned."""
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    bound = core_in_shell(model, nodes)
    seam = Seam(read_target(model).platform)
    batch = seam.keyed(bound, choices, "configured")
    outcome = seam.set(bound, batch)
    if isinstance(outcome, Refused):
        raise Refusal(dict.fromkeys(finding for found in outcome.why.values() for finding in found))
    core = outcome.network
    if len(nodes) < len(model.graph.node) and all(owner in core.kernels for owner, _ in batch):
        # Not a partition's body (nodes on the host): no channel's choices are made, and
        # only the kernels' are written, each on its node.
        for node in nodes:
            op = kernel_op(model, node)
            held = {key: value for (_, key), value in core.chosen([node.name]).items()}
            op._write(op._saved(held))
        return model
    done = core.defaulted(members=core.channels)  # one pass: what emission names stays open
    before = core.chosen()
    made: dict[str, dict[Key, object]] = {}
    for key, value in core.made(done.point).chosen(core.channels).items():
        if key not in before:
            made.setdefault(key[0], {})[key] = value
    for values in made.values():
        step = seam.set(core, values)
        if isinstance(step, Accepted):
            core = step.network
    persist(model, core.point)
    return model


def write_raw(model: ModelWrapper, node_name: str, key: str, value: int | str | None) -> None:
    """``key`` on the node ``node_name`` as a raw ONNX attribute holding ``value`` (one
    already there is replaced), or no attribute for ``None``. Nothing is checked: the
    graph is as one FINN did not write, or whose choice was cleared."""
    (node,) = [node for node in model.graph.node if node.name == node_name]
    kept = [attribute for attribute in node.attribute if attribute.name != key]
    if value is not None:
        kept.append(helper.make_attribute(key, value))
    del node.attribute[:]
    node.attribute.extend(kept)


__all__ = ["bound_in_shell", "composition", "configured", "core_in_shell", "write_raw"]
