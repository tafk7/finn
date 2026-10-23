# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain: a partition of two MatMuls and a thresholding on their channels, declared by
hand (``support.helpers.by_hand``).

It computes ``thresholds(x @ W1) @ W2``. Each MatMul reads its tensors and presents its
ports on the partition's channels; each channel's sides are its kernels' ports
(``Port``), and the tensor of a MatMul's results is the MatMul's statement of them
(``result_tensor``), read before it is placed. The weights are given to each MatMul
(``w_contents``): each weight channel carries what its MatMul states for it (``Stated``),
its contents (``stored_weights``) and its tensor over their range (``carried_weights``), its
source their memory. ``chain()`` is the Chain configured as the kernel tests, the KernelOp
partition tests and the XSim harness use it; its module is named ``finn_chain``
(``STEM``).
"""

from __future__ import annotations

from typing import Any, cast

from qonnx.core.datatype import DataType

from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.composer import ChannelDeclaration, Facts, Port, Stated, channel
from finn.kernels.matmul import MatMulKernel, column_range
from finn.kernels.partition import Partition
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.domains import range_dtype
from support.helpers import FULL_DSP48E2, by_hand, choose, with_channel_defaults

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


def matmul(weights: Any, x_tensor: Any) -> Facts:
    """A MatMul of known ``weights`` (``W``) on the activations ``x_tensor``: its extents
    and the activations' datatype are its tensors'."""
    return Facts(
        MatMulKernel,
        {"weights_dtype": W, "platform": FULL_DSP48E2, "x_tensor": x_tensor, "w_contents": weights},
    )


def result(weights: Any, x_tensor: Any) -> Tensor:
    """The tensor of a MatMul's results: its statement, which reads its facts and weights."""
    found: Tensor = cast(Any, matmul(weights, x_tensor).bind()).result_tensor
    return found


HIDDEN_TENSOR = result(W1, X_TENSOR)
Y_TENSOR = result(W2, LEVELS)


def network(hidden: Tensor = HIDDEN_TENSOR) -> Partition:
    """The Chain (``members``), nothing chosen but the thresholding's fixed ``PE``."""
    return choose(by_hand(members(hidden)), {"activate.pe": PE})


def members(hidden: Tensor = HIDDEN_TENSOR) -> dict[str, Facts | ChannelDeclaration]:
    """The Chain's members, as declared, in its module's order. The input is the
    partition's to state, and the levels' channel's tensor too, as a graph states the
    thresholding's output. Each MatMul's results are its statement, which reads only its
    facts and its weights, and each weight channel carries what its MatMul states for it.
    The hidden channel's tensor, which the thresholding reads, is ``hidden``: the first
    MatMul's results by default."""
    platform = FULL_DSP48E2
    return {
        "x": channel(
            Side.boundary("s_axis_0"), Port("first", "x"), tensor=X_TENSOR, platform=platform
        ),
        "w1": channel(
            Side.source(),
            Port("first", "w"),
            tensor=Stated("first", "carried_weights"),
            contents=Stated("first", "stored_weights"),
            platform=platform,
        ),
        "hidden": channel(
            Port("first", "y"), Port("activate", "input"), tensor=hidden, platform=platform
        ),
        "levels": channel(
            Port("activate", "output"), Port("second", "x"), tensor=LEVELS, platform=platform
        ),
        "w2": channel(
            Side.source(),
            Port("second", "w"),
            tensor=Stated("second", "carried_weights"),
            contents=Stated("second", "stored_weights"),
            platform=platform,
        ),
        "y": channel(
            Port("second", "y"), Side.boundary("m_axis_0"), tensor=Y_TENSOR, platform=platform
        ),
        "first": matmul(W1, X_TENSOR),
        "activate": Facts(
            ThresholdingAxiKernel,
            {
                "threshold_dtype": THRESHOLD_DTYPE,
                "thresholds": THRESHOLDS,
                "bias": 0,
                "input_tensor": hidden,
                "platform": platform,
            },
        ),
        "second": matmul(W2, LEVELS),
    }


HELD = {("activate", "pe"): PE}
"""The thresholding's fixed PE, which a test's seam holds: no strategy folds it."""

LAYERS = (("first", "w1"), ("second", "w2"))
"""Each MatMul of the Chain and its weight channel."""


def configure_chain(chain: Partition, **choices: object) -> Partition:
    """``chain`` (the Chain) with ``choices`` and each MatMul's packed core configured,
    then its channels' open choices by their first valid values, as the cut sets them
    (``with_channel_defaults``: each transport direct, each adapter's memory), and each
    weight's memory as given."""
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
            f"{stream}.source.ram_style": "distributed",
            f"{stream}.source.pumped_memory": False,
        }
    return choose(with_channel_defaults(choose(chain, folding)), weights)


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


def chain() -> Partition:
    """The Chain, configured."""
    return configure_chain(network(), **ACTIVATE)
