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
choice on a graph with nodes on the host is refused as the readers refuse it
(``channel-choices-outside-body``), since ``persist`` writes it without asking.

A key the network reports forced to the very value given is dropped from the batch, not
refused: the compiler never stores a forced Decision (``persist`` writes only the
committed ones), so dropping it states exactly what the compiler would, and a set-up may
list a kernel's whole folding whatever the target forces. A key forced to another value is
passed on, and the seam refuses it (``forced``). The seam itself, and so ``Hold``, is
unchanged: it refuses a forced key whatever its value.

The seam has no clearing: a value of ``None`` is no value of any Decision, so the seam
refuses it (``case-not-viable``). A test of a stale or cleared choice writes the raw ONNX
attribute instead.

A test helper, imported as ``support.configure``.
"""

from __future__ import annotations

from collections.abc import Mapping

from qonnx.core.modelwrapper import ModelWrapper

from finn.core.graph.annotations.channel import refuse_outside_body
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
    persisted (``persist``). A key forced to the value given is dropped (the module docstring);
    a value the seam refuses is refused, by its code. ``model`` is changed in place and
    returned."""
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    # Channel choices are a partition's body's only; nothing below checks it before a write.
    names = {node.name for node in model.graph.node}
    refuse_outside_body(
        model, [owner for owner, given in choices.items() if given and owner not in names]
    )
    bound = in_shell(model, nodes)
    seam = Seam(read_target(model).platform)
    batch = seam.keyed(bound, choices, "configured")
    forced = bound.forced({owner for owner, _ in batch})
    batch = {key: value for key, value in batch.items() if forced.get(key, _ABSENT) != value}
    if batch:
        outcome = seam.set(bound, batch)
        if isinstance(outcome, Refused):
            raise Refusal(
                dict.fromkeys(finding for found in outcome.why.values() for finding in found)
            )
        bound = outcome.network
    persist(model, bound)
    return model


__all__ = ["configured"]
