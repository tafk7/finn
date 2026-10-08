# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The boards a build may name: each board's part and the Vivado board preset the
Zynq shell's block design selects for it.

The boards are FINN's PYNQ boards (``finn.util.basic.pynq_part_map``), less the retired
Zynq 7000 boards (Pynq-Z1, Pynq-Z2). A board is a FINN name for one board revision; its
part is the one part it carries. The Zynq template has a branch for each board with a
preset; a board without one (ZCU111) is named for its part, on the ``ip`` shell, and
the ``pynq`` shell is not built for it (``finn.platform.shells``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Board:
    """A board: its FINN ``name``, its ``part``, and the ``preset`` (Vivado's
    ``board_part``) the Zynq template's branch for it selects (``None``: the template
    has no branch for it, so it cannot build the block design there, and the ``pynq``
    shell has no row for it)."""

    name: str
    part: str
    preset: str | None


BOARDS: dict[str, Board] = {
    board.name: board
    for board in (
        Board(name="Ultra96", part="xczu3eg-sbva484-1-e", preset="avnet.com:ultra96v1:part0:1.2"),
        Board(
            name="Ultra96-V2", part="xczu3eg-sbva484-1-i", preset="avnet.com:ultra96v2:part0:1.2"
        ),
        Board(name="ZCU102", part="xczu9eg-ffvb1156-2-e", preset="xilinx.com:zcu102:part0:3.3"),
        Board(name="ZCU104", part="xczu7ev-ffvc1156-2-e", preset="xilinx.com:zcu104:part0:1.1"),
        Board(name="ZCU111", part="xczu28dr-ffvg1517-2-e", preset=None),
        Board(
            name="RFSoC2x2", part="xczu28dr-ffvg1517-2-e", preset="xilinx.com:rfsoc2x2:part0:1.1"
        ),
        Board(
            name="RFSoC4x2",
            part="xczu48dr-ffvg1517-2-e",
            preset="realdigital.org:rfsoc4x2:part0:1.0",
        ),
        Board(
            name="KV260_SOM", part="xck26-sfvc784-2LV-c", preset="xilinx.com:kv260_som:part0:1.3"
        ),
        Board(
            name="AUP-ZU3_8GB",
            part="xczu3eg-sfvc784-2-e",
            preset="realdigital.org:aup-zu3-8gb:part0:1.0",
        ),
    )
}


__all__ = ["BOARDS", "Board"]
