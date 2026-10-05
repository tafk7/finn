# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's ends traverse its tensor: each end its shape, the elements in order.

The plan between the ends is ``test_plan``'s; a channel's refusals
(``channel-tensor``, ``channel-plan``) are ``tests/kernels``'.
"""

from __future__ import annotations

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.ends import End, Ends, misfit
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major

INT4 = ScalarEncoding(resolve_qonnx_datatype_name("INT4"))
NARROW = ScalarEncoding(resolve_qonnx_datatype_name("INT4"), (-7, 7))
ROWS = BeatSequence(vector_major((3, 8), 2))


def ends(produced: ScalarEncoding, consumed: ScalarEncoding, sink: BeatSequence = ROWS) -> Ends:
    return Ends(End("producer", produced, ROWS), End("consumer", consumed, sink))


def test_every_end_traverses_the_channel_s_tensor() -> None:
    other = BeatSequence(vector_major((4, 6), 2))
    why = misfit(Tensor((3, 8), INT4), ends(INT4, INT4, other))
    assert why is not None and why.startswith("consumer traverses a (4, 6) tensor")


def test_a_producer_s_values_fit_the_tensor_and_the_tensor_s_the_consumer() -> None:
    # A producer tighter than a plainly stated tensor fits.
    assert misfit(Tensor((3, 8), INT4), ends(NARROW, INT4)) is None
    # A tensor tighter than its producer does not, the range printed only when tightened.
    assert misfit(Tensor((3, 8), NARROW), ends(INT4, INT4)) == (
        "producer carries INT4; the channel carries INT4 over [-7, 7]"
    )
    # The consumer's side is the same rule: a tighter tensor fits a plain consumer.
    assert misfit(Tensor((3, 8), NARROW), ends(NARROW, INT4)) is None
