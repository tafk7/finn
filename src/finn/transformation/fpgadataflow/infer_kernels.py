############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam A — the INFER seam: frontend graph pattern → unresolved kernel node.

FINN converts frontend ONNX ops (MatMul, MultiThreshold, …) into hardware-layer nodes
inside ``step_convert_to_hw``. The classic path bakes the kernel's design-space logic as
hidden imperative asserts inside a 270-line god-method
(``InferQuantizedMatrixVectorActivation``), and freezes graph-derived facts
(MW/MH/actval/numInputVectors) onto the node. The kernel system claims its frontend
pattern FIRST, at infer, and never touches FINN's classic MVAU path.

This module carries the generic DRIVER of that seam:

  * :class:`InferKernels` — one generic qonnx ``Transformation`` driven by a POOL of
    DataflowOp classes. For each graph node, the first pool op whose ``can_infer_from``
    returns True wins; its ``infer_from`` produces the replacement node(s). The concrete
    match/build logic lives ON each DataflowOp (``compute/mvau/op.py``,
    ``compute/thresholding/op.py``) — this driver is oblivious to how the pool was assembled,
    so a future registry-driven pool drops in with no change here.

The per-node return of a kernel's ``infer_from`` — :class:`~finn.kernels.ir.kernel_op.TransformationResult`
— is the kernel INFER CONTRACT, co-located with the ``DataflowOp`` base in ``kernels/ir/``
so a kernel op names its own return type without importing this transformation.
"""

from __future__ import annotations

import logging

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

logger = logging.getLogger(__name__)

# The taxonomy domain of every kernel node (handoff Seam C). The package IS the domain
# module — ``finn.kernels`` imports directly, no alias.
KERNEL_DOMAIN = "finn.kernels"


class InferKernels(Transformation):
    """Drive a POOL of DataflowOp classes over the graph, claiming frontend patterns.

    Constructed with an ordered pool (``list`` order == precedence): for each node, the
    FIRST pool op whose ``can_infer_from`` returns True wins, its ``infer_from`` builds
    the replacement, and the graph edit is applied. Before committing, each freshly-built
    kernel node is instantiated model-aware and validated (``infer_node_datatype``) — so
    the seam produces ONLY nodes that legally instantiate. InferShapes + InferDataTypes
    re-run once at the end if anything changed (as every FINN infer does).

    Args:
        pool: ordered ``list[type[DataflowOp]]``. For the vertical slice this is the
            hardcoded ``[MvauDataflowOp, ThresholdingDataflowOp]`` supplied at the call site;
            the future registry-driven version swaps that literal for ``registry.kernels()``
            with no change to this driver.
    """

    def __init__(self, pool: list):
        super().__init__()
        self.pool = list(pool)

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False

        consumed: set[int] = set()
        for node_ind, node in enumerate(list(graph.node)):
            # A node an earlier iteration already CONSUMED (e.g. a MultiThreshold absorbed by
            # a fused MVAU claim) is stale — skip it rather than re-claiming a node no longer
            # in the graph (which would fail at the remove() below).
            if id(node) in consumed:
                continue
            kernel_cls = self._match(node, model)
            if kernel_cls is None:
                continue

            # infer_from is only reached AFTER can_infer_from returned True, so a raise here
            # is a BUILD BUG (the claim promised a legal build the builder could not deliver),
            # not a "no match". Let it PROPAGATE — a broken builder must fail loudly rather
            # than silently leave a classic node on FINN's path (INV5). A future "I matched
            # but on closer inspection decline" is an explicit return sentinel, not an
            # exception.
            result = kernel_cls.infer_from(node, model, node_ind + 1)

            # Model-aware validation guard (brainsmith infer_kernel.py:128-143): a kernel
            # node that cannot instantiate + publish its output dtype is not committed.
            if not self._validate(result, model, kernel_cls, node):
                continue

            for i, new_node in enumerate(result.nodes_to_insert):
                graph.node.insert(node_ind + 1 + i, new_node)
            for old_node in result.nodes_to_remove:
                graph.node.remove(old_node)
                consumed.add(id(old_node))  # don't re-claim a node this pass already removed
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)

    def _match(self, node: NodeProto, model: ModelWrapper):
        """The first pool op that claims ``node``, or None. List order is precedence.

        ``can_infer_from`` is a PREDICATE — it must be total. A raise here is a KERNEL BUG
        (a broken claim check), not a "no match", so it PROPAGATES rather than being swallowed
        into a silent skip (INV5)."""
        for kernel_cls in self.pool:
            if kernel_cls.can_infer_from(node, model):
                return kernel_cls
        return None

    def _validate(self, result, model, kernel_cls, src_node) -> bool:
        """Instantiate + validate each new kernel node before commit. Only kernel-domain
        nodes are validated (an infer might also emit layout/helper nodes). Returns False
        (skip this inference) if any kernel node fails to instantiate.

        This IS a legitimate filter — its purpose is "only commit nodes that legally
        instantiate". But the catch is NARROW: the expected failure is the
        ``Illegal``→``ValueError`` (or ``KeyError``) surface of ``_point``/
        ``infer_node_datatype`` — a node that cannot resolve. An UNEXPECTED error type (e.g.
        an ``AttributeError`` from a typo in the kernel code) is a real bug and PROPAGATES
        (INV5), rather than masquerading as a legitimate skip."""
        for new_node in result.nodes_to_insert:
            if new_node.domain != KERNEL_DOMAIN:
                continue
            try:
                kernel_op = model.get_customop_wrapper(new_node)
                kernel_op.infer_node_datatype(model)
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "InferKernels: skipping %s inference from %s node %s — validation "
                    "failed: %s",
                    kernel_cls.__name__,
                    src_node.op_type,
                    src_node.name,
                    exc,
                )
                return False
        return True
