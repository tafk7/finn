# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain: a root of two MatMuls and a thresholding on its channels.

It computes ``thresholds(x @ W1) @ W2``. Each MatMul reads the root's tensors and
presents its ends on the root's channels; the root declares every channel, taking
each end from its kernel (``end_of``) and the tensor of a MatMul's results from the
MatMul's view (``result_tensor``). The weights are the root's, given to each MatMul
(``w_contents``): each weight channel carries what its MatMul states for it, its
contents (``w_stored``) and its tensor over their range (``w_carried``). ``chain()``
is the Chain configured as the kernel tests, the KernelOp partition tests and the
XSim harness use it.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel, end_of
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, column_range
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.domains import range_dtype
from finn.kernels.values.semantics import INTEGER_TENSOR, IntegerTensor
from kernels.helpers import (
    FULL_DSP48E2,
    Root,
    view_of,
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
H = range_dtype(*column_range(A, W1))
Y = range_dtype(*column_range(T, W2))
X_TENSOR = Tensor((ROWS, INPUTS), ScalarEncoding(A))
LEVELS = Tensor((ROWS, HIDDEN), ScalarEncoding(T))


def matmul(k: int, n: int, dtype: Any, weights: Any, x_tensor: Any, y_tensor: Any) -> MatMulKernel:
    """A MatMul of ``k`` by ``n`` known ``weights`` on activations of ``dtype``."""
    return MatMulKernel(
        m=ROWS,
        n=n,
        k=k,
        activation_dtype=dtype,
        weights_dtype=W,
        platform=FULL_DSP48E2,
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        w_contents=weights,
    )


class Chain(Root):
    # The input is the root's to state, and the thresholding's output too: a leaf
    # binds its extents from its own ports, so its output tensor is not read from it.
    # Each MatMul's results are its view, which reads only its facts and its weights,
    # and each weight channel carries what its MatMul states for it.
    hidden_tensor = view_of("first", "result_tensor")
    y_tensor = view_of("second", "result_tensor")
    w1_tensor = view_of("first", "w_carried")
    w1_contents: IntegerTensor = view_of("first", "w_stored", INTEGER_TENSOR)
    w2_tensor = view_of("second", "w_carried")
    w2_contents: IntegerTensor = view_of("second", "w_stored", INTEGER_TENSOR)
    x_consumer = end_of("first", "x_end")
    w1_consumer = end_of("first", "w_end")
    hidden_producer = end_of("first", "y_end")
    hidden_consumer = end_of("activate", "input_end")
    levels_producer = end_of("activate", "output_end")
    levels_consumer = end_of("second", "x_end")
    w2_consumer = end_of("second", "w_end")
    y_producer = end_of("second", "y_end")

    x = Channel(platform=FULL_DSP48E2, tensor=X_TENSOR, port="s_axis_0", consumer=x_consumer)
    w1 = Channel(
        tensor=w1_tensor, contents=w1_contents, consumer=w1_consumer, platform=FULL_DSP48E2
    )
    hidden = Channel(
        tensor=hidden_tensor,
        producer=hidden_producer,
        consumer=hidden_consumer,
        platform=FULL_DSP48E2,
    )
    levels = Channel(
        tensor=LEVELS, producer=levels_producer, consumer=levels_consumer, platform=FULL_DSP48E2
    )
    w2 = Channel(
        tensor=w2_tensor, contents=w2_contents, consumer=w2_consumer, platform=FULL_DSP48E2
    )
    y = Channel(tensor=y_tensor, port="m_axis_0", producer=y_producer, platform=FULL_DSP48E2)
    first = matmul(INPUTS, HIDDEN, A, W1, X_TENSOR, hidden_tensor)
    activate = ThresholdingAxiKernel(
        input_dtype=H,
        threshold_dtype=THRESHOLD_DTYPE,
        thresholds=THRESHOLDS,
        bias=0,
        pe=PE,
        input_tensor=hidden_tensor,
        output_tensor=LEVELS,
        platform=FULL_DSP48E2,
    )
    second = matmul(HIDDEN, OUTPUTS, T, W2, LEVELS, y_tensor)


LAYERS = (("first", "w1"), ("second", "w2"))
"""Each MatMul of the Chain and its weight channel."""


def configure_chain(root: Root, **choices: object) -> Any:
    """``root`` (a Chain) with ``choices`` and each MatMul's packed core configured."""
    choices = choices | {f"{stream}.transport": "direct" for _, stream in LAYERS}
    point = with_adapter_memories(commit(with_direct_transports(design_space(root)), choices))
    # The Decisions inside the candidates just selected, each keyed by its owner.
    nested: dict[str, object] = {}
    for layer, stream in LAYERS:
        nested |= {
            f"{layer}.compute.packed.pe": PE,
            f"{layer}.compute.packed.simd": SIMD,
            f"{layer}.compute.packed.compute_pumping": False,
            f"{layer}.compute.packed.reducer": "tree",
            f"{stream}.source.memstream.ram_style": "distributed",
            f"{stream}.source.memstream.pumped_memory": False,
        }
    return with_adapter_memories(commit(point, nested))


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


def chain() -> Any:
    """The Chain, configured."""
    return configure_chain(Chain(), **ACTIVATE)
