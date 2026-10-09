# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Words that differ, decoded as the order a stream was walked in (``finn.harness.orders``).

Synthetic: a stream walked in another order than declared, its words formed here
as the hardware would form them; the wrong-order conformance cases decode the
same way (``test_conformance.py``), offline and in XSim.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from finn.dataflow.traversal import Traversal, offsets, pack, vector_major
from finn.harness.orders import Integers, Stream, decode, describe, orders

B, C, S = 2, 3, 4
SHAPE = (B, C, S)
DECLARED = Traversal.over(SHAPE, ((0, B, 1), (2, S, 1), (1, C, 1)), ())  # (b, s, c)
WALKED = vector_major(SHAPE, 1)  # (b, c, s)
VALUES = np.arange(B * C * S, dtype=np.int64).reshape(SHAPE) * 7 % 50


def stream(form: Traversal, bits: int = 8) -> Stream:
    return Stream(form, bits, ("b", "c", "s"))


def test_the_orders_are_the_declared_then_its_loops_permuted_split_at_the_axes() -> None:
    found = orders(WALKED)  # one merged loop, split into b, c and s
    assert found[0] == WALKED and len(found) == 6
    assert DECLARED in found
    assert describe(stream(WALKED), WALKED) == "beats [b:2, c:3, s:4] lanes []"
    tiled = vector_major((6, 4), 2)
    assert describe(Stream(tiled, 4), tiled) == "beats [d0:6, d1/2:2] lanes [d1:2]"


def test_a_permuted_stream_names_the_beat_that_carries_another_index_tuple() -> None:
    received = pack(WALKED, VALUES.ravel().tolist(), 8)
    decoded = decode({"y": stream(DECLARED)}, {"y": received}, expected={"y": VALUES})
    assert decoded is not None and dict(decoded.walked) == {"y": WALKED}
    assert decoded.message == (
        "the words are another order's: y: beat 1 carries (b=0, c=0, s=1); declared "
        "(b=0, c=1, s=0): the hardware walks beats [b:2, c:3, s:4] lanes [], declared "
        "beats [b:2, s:4, c:3] lanes []"
    )


def test_lanes_in_another_order_decode_as_a_lane_order() -> None:
    declared = Traversal.over((2, 6), ((0, 2, 1),), ((1, 3, 1), (1, 2, 3)))  # lane 2i + o
    walked = vector_major((2, 6), 6)
    values = np.arange(12).reshape(2, 6)
    received = pack(walked, values.ravel().tolist(), 4)
    decoded = decode({"y": Stream(declared, 4)}, {"y": received}, expected={"y": values})
    assert decoded is not None and dict(decoded.walked) == {"y": walked}
    assert "beat 0 carries (d0=0, d1=0..5); declared (d0=0, d1=[0 3 1 4 2 5])" in decoded.message


def scaled(read: Mapping[str, Integers]) -> dict[str, Integers]:
    """A reference that reads the position: each value times one more than its c."""
    return {"y": read["x"] * (np.arange(C)[None, :, None] + 1)}


def test_an_input_read_in_another_order_decodes_through_the_reference() -> None:
    """The hardware reads x's beats as its own order's positions and computes there:
    the outputs are no permutation of the expected ones, but the reference on what it
    read, in its order, is what arrived."""
    seen = np.zeros(VALUES.size, dtype=np.int64)
    seen[offsets(WALKED)] = VALUES.ravel()[offsets(DECLARED)]
    received = pack(WALKED, scaled({"x": seen.reshape(SHAPE)})["y"].ravel().tolist(), 8)
    expected = scaled({"x": VALUES})["y"]
    assert sorted(received) != sorted(pack(DECLARED, expected.ravel().tolist(), 8))
    assert decode({"y": stream(DECLARED)}, {"y": received}, expected={"y": expected}) is None
    decoded = decode(
        {"y": stream(DECLARED)},
        {"y": received},
        inputs={"x": stream(DECLARED)},
        values={"x": VALUES},
        reference=scaled,
    )
    assert decoded is not None and dict(decoded.walked) == {"x": WALKED, "y": WALKED}
    assert decoded.message.startswith(
        "the words are another order's: x: beat 1 carries (b=0, c=0, s=1); declared "
    )


def test_words_no_order_explains_are_not_decoded() -> None:
    received = list(pack(DECLARED, VALUES.ravel().tolist(), 8))
    assert decode({"y": stream(DECLARED)}, {"y": received}, expected={"y": VALUES}) is None
    received[3] ^= 1
    assert decode({"y": stream(DECLARED)}, {"y": received}, expected={"y": VALUES}) is None


def test_decoding_needs_the_expected_words_or_a_reference() -> None:
    with pytest.raises(ValueError, match="expected outputs or a reference"):
        decode({"y": stream(DECLARED)}, {"y": []})
    with pytest.raises(ValueError, match="words received on"):
        decode({"y": stream(DECLARED)}, {"z": []}, expected={"y": VALUES})
