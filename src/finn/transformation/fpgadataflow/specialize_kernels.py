############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam B — the SPECIALIZE seam: commit a backend selection onto kernel nodes.

Classic FINN selects an impl in build step 6 (:class:`SpecializeLayers`), a per-op
``if op_type ==`` RTL-feasibility god-switch — the base-knows-subclasses inversion this
effort deletes. In ``finn.kernels`` the engine RESOLVES a node once its ``backend`` axis is
committed, and each backend carries its own feasibility as data (predicates on the pool
member). :class:`SpecializeKernels` is the transform that SELECTS and commits ``backend``
across a graph — the kernel-side parallel of ``SpecializeLayers`` (hence the name), run
first at :func:`step_specialize_layers` so classic nodes see only the remainder.

The selection DRIVER is deliberately abstracted as a :class:`Policy` passed as DATA (the
same shape as :class:`~finn.transformation.fpgadataflow.infer_kernels.InferKernels`'s pool):
HOW resolution is driven — config file, cost-ranked optimizer, interactive DSE — is NOT
settled, so the transform must not hardcode "per-node first-feasible is the only driver".
This module ships ONE trivial reference policy (:func:`first_feasible` via
:class:`PerNodePolicy`); a ``ConfigPolicy`` / ``CostRankedPolicy`` drops in later with zero
change to :class:`SpecializeKernels`.

STAGING is INCREMENTAL: the reference policy commits ONLY ``backend`` here; the folding axes
(SIMD/PE) stay unpinned for ``SetFolding`` (steps 7-8) to fill as a growing assignment —
partial Points, which the guarded-axis engine supports natively. This mirrors classic FINN
(impl at 6, folding at 7-8) and is FORCED: ``SetFolding`` and the estimate analyses skip any
node failing ``is_specialized_node``, so an uncommitted node is invisible to folding.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.kernels.engine.device import DeviceFacts
from finn.kernels.ir.routing import KERNEL_DOMAIN, is_specialized
from finn.kernels.model.backend import BACKEND_AXIS

logger = logging.getLogger(__name__)


class Policy(ABC):
    """The SELECTION driver, passed to :class:`SpecializeKernels` as DATA — the seam that
    keeps the DSE interaction model OPEN.

    A policy maps a whole graph to a per-node axis assignment. GLOBAL-primary by design: the
    signature sees the whole model so a cross-node driver (a global optimizer, a config file
    keyed by node name) is expressible without changing the transform; the per-node
    first-feasible MVP is just one adapter (:class:`PerNodePolicy`).
    """

    @abstractmethod
    def assign(self, model: ModelWrapper, constraints=None, device=None) -> list:
        """Return a list of ``(node, {axis_name: value})`` pairs — the axes to commit on each
        node. Only ``finn.kernels`` nodes should appear (classic nodes are SpecializeLayers'
        job); a node absent from the list is left untouched. A LIST of pairs rather than a
        ``{node: ...}`` dict because ``NodeProto`` is unhashable; the pair keeps precise node
        identity with no node-name-collision risk.

        ``constraints`` is threaded for future drivers (a target-fps budget, a device pin) but
        is UNUSED by the reference policy — do not build a cost model or ranking behind it;
        that is explicitly out of this seam's scope.

        ``device`` is the build's :class:`~finn.kernels.engine.device.DeviceFacts`. Distinct
        from ``constraints``, and not folded into it: it is not a driver preference but a
        GIVEN that feasibility is evaluated against. Every policy needs it; only some will
        ever want constraints."""
        raise NotImplementedError


class PerNodePolicy(Policy):
    """A :class:`Policy` adapter that decides each node INDEPENDENTLY via a local function.

    ``local_fn(node, model, device) -> dict | None`` returns the axes to commit for one node,
    or ``None`` to leave it unspecialized. This adapter owns the two graph-walk concerns — the
    ``finn.kernels`` domain gate and skipping already-specialized nodes — so a local_fn sees
    only fresh kernel nodes and returns pure per-node data."""

    def __init__(self, local_fn):
        self.local_fn = local_fn

    def assign(self, model: ModelWrapper, constraints=None, device=None) -> list:
        out: list = []
        for node in model.graph.node:
            if node.domain != KERNEL_DOMAIN or is_specialized(node):
                continue
            axes = self.local_fn(node, model, device)
            if axes:
                out.append((node, axes))
        return out


def first_feasible(
    node: NodeProto, model: ModelWrapper, device: DeviceFacts | None = None
) -> dict | None:
    """Reference local_fn: commit the FIRST feasible backend (pool order = precedence).

    Delegates to the model-aware op bridge (``KernelOp.first_feasible_backend``), which
    reuses the SAME per-backend trial that infer's ``has_feasible_point`` claim check runs —
    so a node infer claimed as buildable specializes here to a concrete member. Returns
    ``None`` when no backend is feasible (should not happen for an infer-claimed node; logged
    as a lost kernel, INV5).

    ``device`` carries the target part/clock. Without it every device-gated backend refuses
    to answer and the selection silently collapses to the first part-independent member —
    F11, which is why this is a parameter rather than something recovered from the graph."""
    inst = model.get_customop_wrapper(node)
    inst.attach_device(device)
    name = inst.first_feasible_backend()
    if name is None:
        logger.warning(
            "SpecializeKernels: %s node %s has no feasible backend — leaving it "
            "unspecialized (it will be invisible to folding + estimate).",
            node.op_type,
            node.name,
        )
        return None
    return {BACKEND_AXIS: name}


class SpecializeKernels(Transformation):
    """Commit a backend selection onto every ``finn.kernels`` node, driven by a :class:`Policy`.

    The kernel-side parallel of :class:`~finn.transformation.fpgadataflow.specialize_layers.SpecializeLayers`:
    run first in :func:`step_specialize_layers`, it touches ONLY ``finn.kernels`` nodes
    (via the policy's domain gate), leaving classic HW nodes for ``SpecializeLayers`` — the
    two are disjoint by domain, so a mixed graph is safe.

    Args:
        policy: the selection driver (:class:`Policy`). For the reference build this is
            ``PerNodePolicy(first_feasible)``; a config/optimizer/interactive driver swaps in
            here with no change to this transform.
        device: the build's :class:`~finn.kernels.engine.device.DeviceFacts` — target part,
            clock, toolchain. Feasibility is evaluated AGAINST these, so omitting them makes
            every device-gated backend refuse to answer and the selection collapse to the
            first part-independent member (F11). Mirrors the incumbent
            ``SpecializeLayers(cfg._resolve_fpga_part())``: same fact, same owner, same seam.
            Optional so a bare-node test can still construct the transform, but a real build
            always passes it.
    """

    def __init__(self, policy: Policy, device: DeviceFacts | None = None):
        super().__init__()
        self.policy = policy
        self.device = device

    def apply(self, model: ModelWrapper):
        assignment = self.policy.assign(model, device=self.device)
        for node, axes in assignment:
            inst = model.get_customop_wrapper(node)
            inst.attach_device(self.device)
            for axis_name, value in axes.items():
                inst.set_nodeattr(axis_name, value)

        if assignment:
            # Specialization refines the backend-derived output dtype (e.g. MVAU's
            # accumulator under noActivation), so re-propagate datatypes across the graph —
            # the same InferDataTypes the classic step runs after SpecializeLayers.
            model = model.transform(InferDataTypes())

        # No re-run loop: the policy already skips specialized nodes, so a second apply is a
        # no-op. graph_modified stays False (node insertion/removal never happens — only
        # nodeattr commits + a datatype re-propagation the transform runs itself).
        return (model, False)
