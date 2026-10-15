# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The cycle arithmetic of a measurement (``finn.core.rtlsim_bench.Measured``), on
recorded handshakes: its cycles, each stream's stalls and starves, each FIFO's occupancy,
and the FIFO it names for holding its producer back."""

from __future__ import annotations

import pytest

from finn.core.reporting.findings import FindingKind
from finn.core.rtlsim_bench import FIFO_HELD_PRODUCER, FifoStreams, Measured

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


# Two frames of three words through a FIFO of depth 2 (``buffer``) between ``first`` and
# its consumer: the first two words of a frame enter back to back and the third waits
# two cycles at the full FIFO (its producer's stalls); the consumer waits for the first
# word (the fill), and for the third (its starves).
FIFO = FifoStreams(into="buffer.idat", out="second.idat", producer="first", depth=2)
FIFO_BEATS = {
    "s_axis_0": (8, 9, 12, 18, 19, 22),
    "buffer.idat": (10, 11, 14, 20, 21, 24),
    "second.idat": (12, 13, 18, 22, 23, 28),
    "m_axis_0": (15, 16, 21, 25, 26, 31),
}
NONE = (0, 0, 0, 0, 0, 0)


def through_fifo(stalls: tuple[int, ...], starves: tuple[int, ...]) -> Measured:
    """The FIFO's frames, its producer stalled and its consumer starved as given, beat by
    beat; no other stream waits."""
    return Measured(
        frames=2,
        inputs=("s_axis_0",),
        outputs=("m_axis_0",),
        beats=FIFO_BEATS,
        stalls={**dict.fromkeys(FIFO_BEATS, NONE), "buffer.idat": stalls},
        starves={**dict.fromkeys(FIFO_BEATS, NONE), "second.idat": starves},
        fifos={"buffer": FIFO},
    )


def test_waits_are_counted_by_frame() -> None:
    result = through_fifo((0, 0, 2, 0, 0, 2), (2, 0, 3, 0, 0, 3))
    assert result.stalled("buffer.idat") == (2, 2)
    assert result.starved("second.idat") == (5, 3)  # the table's count keeps the fill
    assert result.stalled("s_axis_0") == (0, 0)


def test_a_fifo_holds_its_words_in_minus_its_words_out() -> None:
    result = through_fifo(NONE, NONE)
    assert result.occupancy("buffer") == (
        (10, 1), (11, 2), (12, 1), (13, 0), (14, 1), (18, 0),
        (20, 1), (21, 2), (22, 1), (23, 0), (24, 1), (28, 0),
    )  # fmt: skip
    assert result.peaks("buffer") == (2, 2)


def test_a_word_in_and_a_word_out_in_one_cycle_leave_the_count() -> None:
    result = Measured(
        frames=1,
        inputs=("a",),
        outputs=("b",),
        beats={"a": (1, 2, 3), "b": (2, 3, 4)},
        fifos={"q": FifoStreams(into="a", out="b", producer="a")},
    )
    assert result.occupancy("q") == ((1, 1), (2, 1), (3, 1), (4, 0))
    assert result.peaks("q") == (1,)


def test_a_fifo_that_stalls_its_producer_and_starves_its_consumer_is_named() -> None:
    (found,) = through_fifo((0, 0, 2, 0, 0, 2), (2, 0, 3, 0, 0, 3)).findings
    assert (found.kind, found.code, found.owner) == (
        FindingKind.LIMITATION,
        FIFO_HELD_PRODUCER,
        "measure",
    )
    assert dict(found.details) == {
        "instance": "buffer",
        "producer": "first",
        "depth": 2,
        "frames": (0, 1),
        "stalls": (2, 2),
        "starves": (3, 3),  # the fill left out
        "peaks": (2, 2),
    }
    assert found.message.startswith(
        "buffer: in frames [0, 1] it held its producer first back 4 cycles and left its "
        "consumer waiting 6 cycles, holding at most 2 words of its depth 2"
    )


def test_a_fifo_that_only_fills_is_not_named() -> None:
    # Its consumer waits only for the first word, the pipeline's fill.
    assert through_fifo((0, 0, 2, 0, 0, 2), (2, 0, 0, 0, 0, 0)).findings == ()


def test_a_stall_and_a_starve_in_different_frames_are_not_named() -> None:
    assert through_fifo((0, 0, 2, 0, 0, 0), (2, 0, 0, 0, 0, 3)).findings == ()


def test_a_stream_without_a_fifo_that_stalls_is_not_named() -> None:
    result = Measured(
        frames=2,
        inputs=("s_axis_0",),
        outputs=("m_axis_0",),
        beats=FIFO_BEATS,
        stalls={**dict.fromkeys(FIFO_BEATS, NONE), "second.idat": (0, 0, 2, 0, 0, 2)},
        starves={**dict.fromkeys(FIFO_BEATS, NONE), "m_axis_0": (2, 0, 3, 0, 0, 3)},
    )
    assert result.findings == ()
    assert result.stalled("second.idat") == (2, 2)


def test_the_table_states_waits_peaks_and_findings() -> None:
    lines = through_fifo((0, 0, 2, 0, 0, 2), (2, 0, 3, 0, 0, 3)).table().splitlines()
    assert lines[1].split() == ["stream", "beats", "busy", "II", "first", "stall", "starve"]
    rows = {line.split()[0]: line.split()[1:] for line in lines[2:6]}
    assert rows["buffer.idat"] == ["3", "5", "10", "2", "2", "0"]
    assert rows["second.idat"] == ["3", "7", "10", "4", "0", "3"]
    assert lines[7] == f"{'buffer':<56} {2:>6}  [2, 2], [2, 2], [5, 3]"
    assert lines[8].startswith("measure: fifo-held-producer: buffer: in frames [0, 1]")


def test_a_measurement_without_waits_shows_none() -> None:
    (row,) = (line for line in measured().table().splitlines() if line.startswith("m_axis_0"))
    assert row.split()[-2:] == ["-", "-"]
    assert measured().findings == ()


def test_waits_that_do_not_match_the_beats_are_refused() -> None:
    with pytest.raises(ValueError, match="m_axis_0: 2 counts of waits for 3 beats"):
        Measured(
            frames=3,
            inputs=("s_axis_0",),
            outputs=("m_axis_0",),
            beats=BEATS,
            stalls={"m_axis_0": (0, 0)},
        )


def test_a_fifo_that_does_not_pass_every_word_is_refused() -> None:
    with pytest.raises(ValueError, match="q: 6 words in and 3 out"):
        Measured(
            frames=3,
            inputs=("s_axis_0",),
            outputs=("m_axis_0",),
            beats=BEATS,
            fifos={"q": FifoStreams(into="core.s_axis_tdata", out="core_out.idat", producer="p")},
        )
