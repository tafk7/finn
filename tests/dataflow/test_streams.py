# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's streams traverse its tensor: each stream its shape, the elements in order.

The steps between the streams are ``test_steps``'; a channel's refusals
(``channel-tensor``, ``channel-steps``) are ``tests/kernels``'.
"""

from __future__ import annotations

from qonnx.core.ranges import TensorRange

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.streams import Stream, Streams, misfit
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major

INT4 = ScalarEncoding(resolve_qonnx_datatype_name("INT4"))
NARROW = ScalarEncoding(resolve_qonnx_datatype_name("INT4"), TensorRange(-7, 7))
ROWS = BeatSequence(vector_major((3, 8), 2))
OWNERS = ("producer", "consumer")


def streams(
    produced: ScalarEncoding, consumed: ScalarEncoding, sink: BeatSequence = ROWS
) -> Streams:
    return Streams(Stream(produced, ROWS), Stream(consumed, sink))


def test_every_stream_traverses_the_channel_s_tensor() -> None:
    other = BeatSequence(vector_major((4, 6), 2))
    why = misfit(Tensor((3, 8), INT4), streams(INT4, INT4, other), OWNERS)
    assert why is not None and why.startswith("consumer traverses a (4, 6) tensor")


def test_a_stream_without_an_owner_is_the_boundary() -> None:
    other = BeatSequence(vector_major((4, 6), 2))
    why = misfit(Tensor((3, 8), INT4), streams(INT4, INT4, other))
    assert why is not None and why.startswith("the boundary traverses a (4, 6) tensor")


def test_a_producer_s_values_fit_the_tensor_and_the_tensor_s_the_consumer() -> None:
    # A producer tighter than a plainly stated tensor fits.
    assert misfit(Tensor((3, 8), INT4), streams(NARROW, INT4), OWNERS) is None
    # A tensor tighter than its producer does not, the range printed only when tightened.
    assert misfit(Tensor((3, 8), NARROW), streams(INT4, INT4), OWNERS) == (
        "producer carries INT4; the channel carries INT4 over [-7, 7]"
    )
    # The consumer's side is the same rule: a tighter tensor fits a plain consumer.
    assert misfit(Tensor((3, 8), NARROW), streams(NARROW, INT4), OWNERS) is None
    # The same stream on both sides is read as each side.
    same = Stream(INT4, ROWS)
    assert misfit(Tensor((3, 8), NARROW), Streams(same, same), OWNERS) == (
        "producer carries INT4; the channel carries INT4 over [-7, 7]"
    )
