# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The schedule's names: a core's iterations, its folds and the beats a port takes are
kept apart, and the retired names are not back.

The line-independent guard of the retired ``tests/dataflow_next/test_vocabulary.py`` at
the immutable baseline 1ac1616, on the canonical package; its translation tests stay
retired with the line.
"""

from __future__ import annotations

from typing import Any

import pytest

import finn.dataflow.schedule as schedule
import finn.dataflow.traversal as traversal
from finn.dataflow.timing import Kernel


def test_schedule_names_keep_iterations_separate_from_port_beats() -> None:
    a, b = schedule.Index("a"), schedule.Index("b")
    core = schedule.Schedule({a: 3, b: 4}, {b: 2})
    assert core.fold(b) == 2 and core.iterations == 6
    assert core.beat_times(reduces=(b,)) == (1, 3, 5)
    assert not hasattr(core, "beat_count") and not hasattr(core, "steps")
    assert not hasattr(traversal, "period") and not hasattr(traversal, "AxisStep")
    old_keyword: dict[str, Any] = {"view": (3, 4)}
    with pytest.raises(TypeError, match="view"):
        core.present((3, 4), (a, b), lanes=(b,), **old_keyword)
    assert Kernel("k", iterations=2, reads=((0, 1),), writes=()).iterations == 2
    marker = traversal.level_done(beats=6, inner=2)
    assert marker == traversal.LevelDone(beats=6, inner=2)
    assert repr(marker) == "LevelDone(beats=6, inner=2)"
