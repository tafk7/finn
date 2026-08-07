############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``MinimizeAccumulatorWidth`` must re-propagate after EVERY dataflow node.

The pass narrows a node, then runs ``InferDataTypes`` so the next node reads the
narrowed dtype. Its loop invariant is *"once node i has been visited, the graph
reflects node i's narrowing"* — and the re-propagation used to be nested under
``hasattr(inst, "minimize_accumulator_width")``, so it only ran for nodes carrying
that hook.

A ``finn.kernels`` node breaks that. It IS a dataflow member (``is_fpgadataflow_node``
is True on domain alone) but narrows during datatype INFERENCE rather than through the
hook, so ``hasattr`` is False. Nested, a kernel node therefore both skipped narrowing
AND suppressed the re-propagation — leaving a downstream CLASSIC node to size its
accumulator from a STALE input dtype. That under-sizes the accumulator, which is silent
overflow in hardware, not merely wasted resources.

The default build flow masked this incidentally: ``step_specialize_layers`` ends with
``InferDataTypes()``. But ``step_target_fps_parallelization`` and
``step_apply_folding_config`` both sit between that and ``step_minimize_bit_width``, and
both write ``runtime_writeable_weights`` — which flips ``ParamDatatype.values_visible``
and moves ``accDataType`` from value-narrowed to worst-case envelope. So the staleness is
reachable through the shipped flow, and ``step_minimize_bit_width`` runs exactly ONE round.

The gate is a MIXED graph measured against an ALL-CLASSIC control with the identical
configuration: one round must reach the same fixpoint for both.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.kernels.compute.mvau.op import MvauKernelOp, WEIGHTS
from finn.kernels.compute.thresholding.op import ThresholdingKernelOp
from finn.kernels.dataflow.parameters.names import DECOUPLED
from finn.kernels.model.param_names import runtime_writeable_key, topology_key
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferQuantizedMatrixVectorActivation,
)
from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.specialize_kernels import (
    PerNodePolicy,
    SpecializeKernels,
    first_feasible,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

pytestmark = pytest.mark.integration

PART = "xcvc1902-vsvd1760-2MP-e-S"
MW, MH = 1024, 16

_rng = np.random.RandomState(1)
# Ternary first matrix, wide second: the first node's narrowed output must be WIDE
# enough that the second node's accumulator genuinely depends on getting it right.
W1 = _rng.randint(-1, 2, size=(MW, MH)).astype(np.float32)
W2 = _rng.randint(-128, 128, size=(MH, MH)).astype(np.float32)


class _OnlyFirstMatMul(MvauKernelOp):
    """Claims ONLY ``mm0``, so ``mm1`` falls through to the classic HLS/RTL path —
    which is what makes the graph mixed at a kernel→classic boundary."""

    @classmethod
    def can_infer_from(cls, node, model):
        return node.name == "mm0" and super().can_infer_from(node, model)


def _matmul_chain():
    n0 = helper.make_node("MatMul", ["t0", "w1"], ["t1"], name="mm0")
    n1 = helper.make_node("MatMul", ["t1", "w2"], ["t2"], name="mm1")
    shapes = {"t0": [1, MW], "w1": [MW, MH], "t1": [1, MH], "w2": [MH, MH], "t2": [1, MH]}
    vis = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
        for n, s in shapes.items()
        if n not in ("t0", "t2")
    ]
    graph = helper.make_graph(
        [n0, n1],
        "g",
        [helper.make_tensor_value_info("t0", TensorProto.FLOAT, shapes["t0"])],
        [helper.make_tensor_value_info("t2", TensorProto.FLOAT, shapes["t2"])],
        value_info=vis,
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w1", W1)
    model.set_initializer("w2", W2)
    for name, dt in (("t0", "INT8"), ("w1", "INT8"), ("w2", "INT8"), ("t1", "INT8")):
        model.set_tensor_datatype(name, DataType[dt])
    return model


def _prepared(mixed: bool):
    """A specialized two-MVAU chain, then the folding-config mutation that invalidates
    the first node's published output dtype WITHOUT a following ``InferDataTypes`` —
    reproducing the gap between ``step_apply_folding_config`` and
    ``step_minimize_bit_width``."""
    model = _matmul_chain()
    pool = [_OnlyFirstMatMul, ThresholdingKernelOp] if mixed else [ThresholdingKernelOp]
    model = model.transform(InferKernels(pool))
    model = model.transform(InferQuantizedMatrixVectorActivation())
    model = model.transform(InferShapes()).transform(InferDataTypes())
    model = model.transform(SpecializeKernels(PerNodePolicy(first_feasible)))
    model = model.transform(SpecializeLayers(PART))
    model = model.transform(InferShapes()).transform(InferDataTypes())

    for node in model.graph.node:
        inst = model.get_customop_wrapper(node)
        inst.set_nodeattr("SIMD", 4)
        inst.set_nodeattr("PE", 4)

    # Runtime-writeable weights blind the owner to its weight VALUES, so the accumulator
    # falls back to the worst-case envelope — a strictly WIDER dtype than the one already
    # published on t1. This is the staleness the pass must repair.
    first = model.graph.node[0]
    inst = model.get_customop_wrapper(first)
    if first.domain == "finn.kernels":
        inst.set_nodeattr(topology_key(WEIGHTS), DECOUPLED)
        inst.set_nodeattr(runtime_writeable_key(WEIGHTS), 1)
    else:
        inst.set_nodeattr("mem_mode", "internal_decoupled")
        inst.set_nodeattr("runtime_writeable_weights", 1)
    return model


def _downstream_acc(model):
    """The CLASSIC consumer's accumulator width, and the dtype it read to size it."""
    inst = model.get_customop_wrapper(model.graph.node[1])
    return (
        DataType[inst.get_nodeattr("accDataType")],
        DataType[inst.get_nodeattr("inputDataType")],
    )


def test_mixed_graph_matches_all_classic_in_one_round():
    """One round of the pass must reach the same accumulator on a mixed graph as on the
    all-classic control. Nesting the re-propagation gave INT32 vs the control's INT40."""
    mixed = _prepared(mixed=True).transform(MinimizeAccumulatorWidth())
    classic = _prepared(mixed=False).transform(MinimizeAccumulatorWidth())

    mixed_acc, _ = _downstream_acc(mixed)
    classic_acc, _ = _downstream_acc(classic)
    assert mixed_acc == classic_acc, (
        f"mixed graph sized the downstream accumulator {mixed_acc} but the identically "
        f"configured all-classic graph sized it {classic_acc} — the kernel node's "
        f"narrowing was not propagated before its classic consumer was visited"
    )


def test_downstream_accumulator_holds_its_own_input():
    """The consumer's accumulator must be able to hold products of the input dtype it
    declares. The nested form produced accDataType NARROWER than its own inputDataType —
    an accumulator that cannot represent a single one of its input elements."""
    model = _prepared(mixed=True).transform(MinimizeAccumulatorWidth())
    acc, idt = _downstream_acc(model)
    assert acc.bitwidth() >= idt.bitwidth(), (
        f"accDataType {acc} is narrower than inputDataType {idt} — the node is "
        f"internally contradictory (accumulator cannot hold one input element)"
    )


def test_pass_is_idempotent_on_a_mixed_graph():
    """A second round must change nothing. Non-idempotence means the single round the
    build flow actually runs (``step_minimize_bit_width``) ships a non-fixpoint."""
    once = _prepared(mixed=True).transform(MinimizeAccumulatorWidth())
    twice = once.transform(MinimizeAccumulatorWidth())
    assert _downstream_acc(once) == _downstream_acc(twice), (
        "MinimizeAccumulatorWidth is not idempotent on a mixed graph: the build flow "
        "runs it once, so a second-round change is a shipped miscompile"
    )
