# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The cycle arithmetic of a measurement (``finn.harness.rtl.Measured``), on recorded handshakes."""

from __future__ import annotations

import pytest

from finn.harness.rtl import Measured

# Three frames of a design taking two beats in and presenting one beat out per frame:
# frames enter at cycles 10, 14 and 18, back to back at one frame per four cycles after
# the first, and leave five cycles after their last input beat.
BEATS = {
    "s_axis_0": (10, 11, 14, 15, 18, 19),
    "core.s_axis_tdata": (10, 11, 14, 15, 18, 19),
    "core_out.idat": (13, 17, 21),
    "m_axis_0": (16, 20, 24),
}


def measured() -> Measured:
    return Measured(frames=3, inputs=("s_axis_0",), outputs=("m_axis_0",), beats=BEATS)


def test_the_root_counts_each_frame_from_its_first_input_to_its_last_output_beat() -> None:
    result = measured()
    assert result.latencies == (7, 7, 7)
    assert result.intervals() == (4, 4)
    assert result.interval() == 4
    assert result.total == 15  # cycles 10 to 24, both counted


def test_each_stream_splits_into_frames() -> None:
    result = measured()
    assert result.of("core.s_axis_tdata") == ((10, 11), (14, 15), (18, 19))
    assert result.per_frame("core.s_axis_tdata") == 2
    assert result.busy("core.s_axis_tdata") == (2, 2, 2)
    assert result.span("s_axis_0", "core_out.idat") == (4, 4, 4)
    assert result.interval("core_out.idat") == 4


def test_the_table_names_every_stream() -> None:
    text = measured().table()
    assert "latency per frame [7, 7, 7]; intervals [4, 4]; total 15" in text
    assert all(name in text for name in BEATS)


def test_beats_that_do_not_split_into_the_frames_are_refused() -> None:
    with pytest.raises(ValueError, match="m_axis_0: 2 beats do not split into 3 frames"):
        Measured(
            frames=3,
            inputs=("s_axis_0",),
            outputs=("m_axis_0",),
            beats={**BEATS, "m_axis_0": (16, 20)},
        )


def test_a_root_stream_without_beats_is_refused() -> None:
    with pytest.raises(ValueError, match="no beats recorded on the root's stream m_axis_1"):
        Measured(frames=3, inputs=("s_axis_0",), outputs=("m_axis_1",), beats=BEATS)


def test_an_interval_needs_two_frames() -> None:
    one = Measured(frames=1, inputs=("a",), outputs=("b",), beats={"a": (1,), "b": (3,)})
    assert one.latencies == (3,) and one.total == 3
    with pytest.raises(ValueError, match="at least two frames"):
        one.interval()
