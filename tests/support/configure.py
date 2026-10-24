# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Choices set on a model, or on a network, the way the compiler sets them.

``configured(model, {graph name: {key: value}})``: the KernelOps' partition of ``model`` bound
in the shell of its target (as exploration binds a partition's body), the values set through
the seam as a strategy sets them (``Seam.set``, one batch, as ``Hold`` does), then each
channel's open choices that are not required set by the cut's policies, as the cut sets
them (``Partition.defaulted``, once over the channels; a FIFO's depth stays open), and the result
persisted (``finn.custom_op.kernels.network.persist``, the one writer of choices). What the
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

from collections.abc import Mapping

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.annotations.platform import read_target
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.reporting.findings import Refusal
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import persist
from finn.custom_op.kernels.shell import in_shell
from finn.kernels.composer import Key
from finn.kernels.explore import Accepted, Refused, Seam


def configured(model: ModelWrapper, choices: Mapping[str, Mapping[str, object]]) -> ModelWrapper:
    """``model`` with ``choices`` (by graph name: a kernel's under its node, a channel's
    under its tensor) set the way the compiler sets them: the network of the model's
    KernelOps bound, the values set through the seam as a strategy sets them, one batch,
    each channel's open choices then set by the cut's policies, as the cut sets them,
    the result persisted (``persist``); a value the seam refuses is refused, by its code.
    ``model`` is changed in place and returned."""
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    bound = in_shell(model, nodes)
    seam = Seam(read_target(model).platform)
    batch = seam.keyed(bound, choices, "configured")
    outcome = seam.set(bound, batch)
    if isinstance(outcome, Refused):
        raise Refusal(dict.fromkeys(finding for found in outcome.why.values() for finding in found))
    network = outcome.network
    if len(nodes) < len(model.graph.node) and all(owner in network.kernels for owner, _ in batch):
        # Not a partition's body (nodes on the host): no channel's choices are made, and
        # only the kernels' are written, each on its node.
        for node in nodes:
            op = kernel_op(model, node)
            held = {key: value for (_, key), value in network.chosen([node.name]).items()}
            op._write(op._saved(held))
        return model
    done = network.defaulted(members=network.channels)  # one pass: what emission names stays open
    before = network.chosen()
    made: dict[str, dict[Key, object]] = {}
    for key, value in network.at(done.point).chosen(network.channels).items():
        if key not in before:
            made.setdefault(key[0], {})[key] = value
    for values in made.values():
        step = seam.set(network, values)
        if isinstance(step, Accepted):
            network = step.network
    persist(model, network)
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


__all__ = ["configured", "write_raw"]
