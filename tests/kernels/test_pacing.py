# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one pacing spec (``finn.harness.pacing``) both RTL drivers take."""

from __future__ import annotations

import pytest

from finn.harness.pacing import FREE, STALLED, Pace, Pacing


@pytest.mark.parametrize(("burst", "pause"), [(0, 0), (1, -1), (True, 0)])
def test_a_pace_bursts_at_least_once_and_pauses_no_less_than_zero(burst: int, pause: int) -> None:
    with pytest.raises(ValueError, match="burst|pause"):
        Pace(burst, pause)


def test_a_pace_s_cycles_are_its_beats_and_a_pause_after_each_full_burst() -> None:
    assert Pace().cycles(10) == 10
    assert Pace(2, 1).cycles(5) == 5 + 2
    assert Pace(1, 5).cycles(3) == 3 + 15


def test_streams_take_their_paces_by_position_cycling() -> None:
    pacing = Pacing((Pace(2, 1), Pace(3, 2)), (Pace(1, 5),))
    assert [pacing.input(index) for index in range(3)] == [Pace(2, 1), Pace(3, 2), Pace(2, 1)]
    assert pacing.output(4) == Pace(1, 5)
    assert Pacing.from_json(pacing.as_json()) == pacing
    assert not FREE.stalls and STALLED.stalls
    with pytest.raises(ValueError, match="at least one"):
        Pacing(inputs=())
