# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The grouper (``finn.kernels.grouper``): a core's control buses placed behind its unit's
one port, each at its offset in the unit (``map``, each aligned to its own window, from 0),
the unit's window the least power of two holding them and the row's least window at least
(``window``), and the decoder's resources by its slaves, none for one bus or none.
"""

from __future__ import annotations

from typing import Any

import pytest

from finn.core.space import design_space
from finn.kernels.grouper import DECODER, GrouperKernel
from finn.platform.capabilities import Resources


def grouper(windows: tuple[int, ...], least_window: int | None = 4096) -> Any:
    return design_space(GrouperKernel(windows=windows, least_window=least_window))


@pytest.mark.parametrize(
    ("windows", "least", "offsets", "window"),
    [
        ((), 4096, (), ()),
        ((64,), 4096, (0,), (4096,)),
        ((32,) * 8, 4096, tuple(range(0, 256, 32)), (4096,)),
        ((32,) * 8, 128, tuple(range(0, 256, 32)), (256,)),
        ((32, 64, 32), 128, (0, 64, 128), (256,)),
        ((4096, 65536, 4096), 4096, (0, 65536, 131072), (262144,)),
    ],
    ids=[
        "none",
        "one bus",
        "eight equal buses within the least window",
        "eight equal buses beyond it",
        "a hole by alignment",
        "a 64 KiB bus between two of 4 KiB",
    ],
)
def test_each_bus_is_placed_aligned_to_itself_and_the_unit_holds_them(
    windows: tuple[int, ...], least: int, offsets: tuple[int, ...], window: tuple[int, ...]
) -> None:
    found = grouper(windows, least)
    assert (found.map, found.window) == (offsets, window)


def test_a_decoder_is_counted_from_two_buses_on() -> None:
    """One bus is the unit's port, renamed, and none is no unit: neither takes a decoder."""
    assert grouper(()).resources == grouper((64,)).resources == Resources()
    for count in (2, 16):
        found = grouper((64,) * count).resources
        assert found == DECODER.at(count) and found.lut > 0 and found.ff > 0
