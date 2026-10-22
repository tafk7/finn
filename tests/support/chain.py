# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain: a network of two MatMuls and a thresholding on their channels.

It computes ``thresholds(x @ W1) @ W2``. Each MatMul reads its tensors and presents
its ports' contracts on the network's channels; each channel takes them from its
kernels (``Interface``), and the tensor of a MatMul's results is the MatMul's statement
of them (``result_tensor``), read before it is placed. The weights are given to each MatMul
(``w_contents``): each weight channel carries what its MatMul states for it, its
contents (``stored_weights``) and its tensor over their range (``carried_weights``). ``chain()``
is the Chain configured as the kernel tests, the KernelOp partition tests and the
XSim harness use it; its module is named ``finn_chain`` (``STEM``).
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import Space, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.matmul import MatMulKernel, column_range
from finn.kernels.network import Fact, Interface, Network, Wire, wire
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.domains import range_dtype
from support.helpers import (
    FULL_DSP48E2,
    choose,
    with_adapter_memories,
    with_direct_transports,
)

ROWS, INPUTS, HIDDEN, OUTPUTS, PE, SIMD = 3, 4, 4, 4, 2, 2
A = W = DataType["INT3"]
T = DataType["UINT2"]
THRESHOLDS = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(HIDDEN)),)
# The smallest type of H's signedness holding THRESHOLDS: what the ordered pass annotates.
THRESHOLD_DTYPE = DataType["INT5"]
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))
# Each MatMul's result: its weights' columns over its activations' datatype (K7), in
# the range's smallest encoding: INT6 over [-28, 28] and INT5 over [-15, 12].
H = range_dtype(*column_range(ScalarEncoding(A), W1))
Y = range_dtype(*column_range(ScalarEncoding(T), W2))
X_TENSOR = Tensor((ROWS, INPUTS), ScalarEncoding(A))
LEVELS = Tensor((ROWS, HIDDEN), ScalarEncoding(T))
STEM = "chain"
"""The Chain's module's stem: ``finn_chain``."""

MEMBERS = ("first", "activate", "second", "x", "w1", "hidden", "levels", "w2", "y")
"""The Chain's members, ranked: its kernels, then its channels."""


def matmul(weights: Any, x_tensor: Any) -> MatMulKernel:
    """A MatMul of known ``weights`` (``W``) on the activations ``x_tensor``: its extents
    and the activations' datatype are its tensors'."""
    return MatMulKernel(
        weights_dtype=W,
        platform=FULL_DSP48E2,
        x_tensor=x_tensor,
        w_contents=weights,
    )


def result(weights: Any, x_tensor: Any) -> Tensor:
    """The tensor of a MatMul's results: its statement, which reads its facts and weights."""
    found: Tensor = design_space(matmul(weights, x_tensor)).result_tensor
    return found


HIDDEN_TENSOR = result(W1, X_TENSOR)
Y_TENSOR = result(W2, LEVELS)


def network(hidden: Tensor = HIDDEN_TENSOR) -> Network:
    """The Chain (``members``), nothing chosen but the thresholding's fixed ``PE``."""
    return choose(Network(members(hidden)), {"activate.pe": PE})


def members(hidden: Tensor = HIDDEN_TENSOR) -> dict[str, Space | Wire]:
    """The Chain's members, as declared. The input is the network's to state, and the
    levels' channel's tensor too, as a graph states the thresholding's output. Each
    MatMul's results are its statement, which reads only its facts and its weights, and
    each weight channel carries what its MatMul states for it. The hidden channel's
    tensor, which the thresholding reads, is ``hidden``: the first MatMul's results by
    default."""
    platform = FULL_DSP48E2
    return dict(
        {
            "x": wire(
                platform=platform,
                tensor=X_TENSOR,
                port="s_axis_0",
                consumer=Interface("first", "x_contract"),
            ),
            "w1": wire(
                tensor=Fact("first", "carried_weights"),
                contents=Fact("first", "stored_weights"),
                consumer=Interface("first", "w_contract"),
                platform=platform,
            ),
            "hidden": wire(
                tensor=hidden,
                producer=Interface("first", "y_contract"),
                consumer=Interface("activate", "input_contract"),
                platform=platform,
            ),
            "levels": wire(
                tensor=LEVELS,
                producer=Interface("activate", "output_contract"),
                consumer=Interface("second", "x_contract"),
                platform=platform,
            ),
            "w2": wire(
                tensor=Fact("second", "carried_weights"),
                contents=Fact("second", "stored_weights"),
                consumer=Interface("second", "w_contract"),
                platform=platform,
            ),
            "y": wire(
                tensor=Y_TENSOR,
                port="m_axis_0",
                producer=Interface("second", "y_contract"),
                platform=platform,
            ),
            "first": matmul(W1, X_TENSOR),
            "activate": ThresholdingAxiKernel(
                threshold_dtype=THRESHOLD_DTYPE,
                thresholds=THRESHOLDS,
                bias=0,
                input_tensor=hidden,
                platform=platform,
            ),
            "second": matmul(W2, LEVELS),
        }
    )


HELD = {("activate", "pe"): PE}
"""The thresholding's fixed PE, which a test's seam holds: no strategy folds it."""

LAYERS = (("first", "w1"), ("second", "w2"))
"""Each MatMul of the Chain and its weight channel."""


def configure_chain(chain: Network, **choices: object) -> Network:
    """``chain`` (the Chain) with ``choices`` and each MatMul's packed core configured,
    then its channels, which bind as their contracts become known and take their first valid
    choices (the interim channel rule): each weight's memory, each transport direct."""
    folding: dict[str, object] = dict(choices)
    weights: dict[str, object] = {}
    for layer, stream in LAYERS:
        folding |= {
            f"{layer}.compute": "packed",
            f"{layer}.pe": PE,
            f"{layer}.simd": SIMD,
            f"{layer}.compute.packed.compute_pumping": False,
            f"{layer}.compute.packed.reducer": "tree",
        }
        weights |= {
            f"{stream}.source.memstream.ram_style": "distributed",
            f"{stream}.source.memstream.pumped_memory": False,
        }
    configured = with_direct_transports(choose(chain, folding))
    return with_adapter_memories(choose(configured, weights))


# The thresholding's choices: no AXI-Lite, no deep pipeline, its memories in their
# baseline, LUTRAM (as the weights' and the adapters', each the first case its size
# orders).
ACTIVATE = {
    "activate.use_axilite": False,
    "activate.deep_pipeline": False,
    "activate.ram_style": "distributed",
    "activate.block_stages": 0,
    "activate.ultra_stages": 0,
}


def chain() -> Network:
    """The Chain, configured."""
    return configure_chain(network(), **ACTIVATE)
