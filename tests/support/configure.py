# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Choices set on a model the way the compiler sets them.

``configured(model, {graph name: {key: value}})``: the KernelOps' network of ``model`` bound
in the shell of its target (as exploration binds a partition's body), the values set through
the seam as a strategy sets them (``Seam.set``, one batch, as ``Hold`` does), the result
persisted (``finn.custom_op.kernels.network.persist``, the one writer of choices). What the
seam refuses is refused, by the seam's own findings (its codes: ``not-a-decision``,
``case-not-viable``, ``forced``, ``inapplicable``, the engine's, ...) at the top of the
``Refusal``, so a test asserts it by code (``support.refusals.refused``); an owner that is
no member of the network is ``Hold``'s own refusal (``member-unknown``); a channel's
choice on a graph with nodes on the host is refused by ``persist``, the one writer, before
anything is written (``channel-choices-outside-body``).

A key the network reports forced to the very value given is held back from the batch, not
refused: the compiler never stores a forced Decision (``persist`` writes only the
committed ones), so leaving it out states exactly what the compiler would, and a set-up may
list a kernel's whole folding whatever the target forces. What the batch then sets may open
a key that was forced before it (pumping reads SIMD): as a strategy's later set sees what
the earlier one opened, the keys held back that the new network no longer forces are set in
a further batch, and so on while something opens; a key still forced to the given value
stays out. A key forced to another value is passed on, and the seam refuses it (``forced``).
The seam itself, and so ``Hold``, is unchanged: it refuses a forced key whatever its value.

The seam has no clearing: a value of ``None`` is no value of any Decision, so the seam
refuses it (``case-not-viable``). A test of a stale or cleared choice writes the raw ONNX
attribute instead.

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
from finn.custom_op.kernels.network import persist
from finn.custom_op.kernels.shell import in_shell
from finn.kernels.explore import Refused, Seam

_ABSENT = object()


def configured(model: ModelWrapper, choices: Mapping[str, Mapping[str, object]]) -> ModelWrapper:
    """``model`` with ``choices`` (by graph name: a kernel's under its node, a channel's
    under its tensor) set the way the compiler sets them: the network of the model's
    KernelOps bound, the values set through the seam as a strategy sets them, the result
    persisted (``persist``). A key forced to the value given is held back, and set once
    the batch opens it (the module docstring); a value the seam refuses is refused, by its
    code. ``model`` is changed in place and
    returned."""
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    bound = in_shell(model, nodes)
    seam = Seam(read_target(model).platform)
    waiting = seam.keyed(bound, choices, "configured")
    while waiting:
        forced = bound.forced({owner for owner, _ in waiting})
        batch = {key: value for key, value in waiting.items() if forced.get(key, _ABSENT) != value}
        if not batch:
            break
        outcome = seam.set(bound, batch)
        if isinstance(outcome, Refused):
            raise Refusal(
                dict.fromkeys(finding for found in outcome.why.values() for finding in found)
            )
        bound = outcome.network
        waiting = {key: value for key, value in waiting.items() if key not in batch}
    persist(model, bound)
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
