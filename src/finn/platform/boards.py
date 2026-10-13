# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The boards a build may name: each board's part, the Vivado board preset the
Zynq shell's block design selects for it, and any Vivado release it needs.

FINN's PYNQ boards and the V80, each with its part. A board is a FINN name for one board
revision. A board without a Vivado preset (ZCU111, the V80) can only be named for its
part, on the ``ip`` shell.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Board:
    """A board: its FINN ``name``, its ``part``, and the Vivado ``board_part`` the Zynq
    template selects for it (``preset``). A ``None`` preset means the template has no
    branch for the board, so the ``pynq`` shell has no row for it. ``vivado`` is the
    earliest Vivado release, (year, minor), a build for it needs; ``None`` means any."""

    name: str
    part: str
    preset: str | None
    vivado: tuple[int, int] | None = None


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
            vivado=(2024, 1),
        ),
        Board(name="V80", part="xcv80-lsva4737-2MHP-e-S", preset=None),
    )
}


__all__ = ["BOARDS", "Board"]
