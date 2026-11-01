# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A made-up platform outside FINN (``platform.toml`` beside this file): the code its
declarations name. Its linker (``ExampleLinker``) checks that the board states its slot
and links a description into a stub image, placing each unit at a made-up base and
delivering the main clock as asked; its runtime (``RecordingRuntime``) writes the runtime
file and the operations its driver would run; its moved mover kind makes an offer as
FINN's own does (``finn.kernels.iodma.IodmaHls.offer``). Its shell offers FINN's own mover
kind, the one a shell places yet (``finn.kernels.movers.PLACED_KINDS``). A test loads the
declarations with ``extra=``, or as an installed package's entry point would name this
package.

It shows a platform declared, checked and resolved, its code found, and a core placed in
its shell, built to an image, given a driver and deployed with no change to FINN. No tool
runs: the image is a stub."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

from finn.kernels.movers import MoverOffer
from finn.shells.integration import MAIN, Integration, IntegrationError, Linked, runtime_text
from finn.toolchain.launcher import Toolchain

DIRECTORY = Path(__file__).resolve().parent
"""This package's directory, holding its ``platform.toml``."""

IMAGE = "image.example"
"""The stub image the linker writes."""

MAP = "image.map"
"""The file the image loads with: each placed unit's base."""

BASE = 0x4000_0000
"""The first unit's base: each unit with a control window is placed at the next multiple of
its window from there, in the description's order."""

DESCRIPTION = "example/description.json"
"""The linker's report: the description it linked, by its path under a build's report/."""

OPERATIONS = "operations.json"
"""What the runtime writes beside the runtime file: the operations its driver would run."""


class ExampleLinker:
    """The ``example-link`` linker (the module docstring); ``debug`` is recorded in the
    description it writes."""

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def check(self, integration: Integration, board: Mapping[str, str]) -> None:
        """Refuse a board that states no slot (``slot-unstated``)."""
        if "slot" not in board:
            raise IntegrationError(
                "slot-unstated", "example-link", f"the board states no slot: {dict(board)}"
            )

    def link(
        self,
        integration: Integration,
        board: Mapping[str, str],
        directory: Path,
        tools: Toolchain,
    ) -> Linked:
        """Write the description, each unit's IP given, a stub image and its map into a new
        directory under ``directory``; no tool runs."""
        self.check(integration, board)
        directory.mkdir(parents=True, exist_ok=True)
        linked = Path(tempfile.mkdtemp(prefix="example_link_", dir=directory))
        placement: dict[str, int] = {}
        base = BASE
        for unit in integration.units:
            if unit.window is not None:
                base = -(-base // unit.window) * unit.window
                placement[unit.name] = base
                base += unit.window
        description = {
            "slot": board["slot"],
            "debug": self.debug,
            "ips": {
                unit.name: None if unit.ip is None else unit.ip.vlnv for unit in integration.units
            },
            **integration.report(),
        }
        (linked / "description.json").write_text(json.dumps(description, indent=2))
        (linked / IMAGE).write_bytes(b"example image\n")
        (linked / MAP).write_text(json.dumps(placement, indent=2))
        return Linked(
            image=linked / IMAGE,
            placement=placement,
            clocks_mhz={MAIN: round(1000 / integration.period_ns, 3)},
            reports={DESCRIPTION: linked / "description.json"},
            beside=(linked / MAP,),
        )


class RecordingRuntime:
    """The ``example-run`` runtime: into a driver's directory, the runtime file
    (``integration.json``) and the operations its driver would run (``OPERATIONS``): the
    image loaded, the main clock set, and each core's movers started by unit and base."""

    def emit(self, integration: Integration, linked: Linked, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "integration.json").write_text(runtime_text(integration, linked))
        operations: list[list[object]] = [
            ["load", linked.image.name],
            ["clock", linked.clocks_mhz.get(MAIN)],
        ]
        for core in integration.cores:
            for edge in (*core.outputs, *core.inputs):
                operations.append(["start", edge.unit, linked.placement[edge.unit]])
        (directory / OPERATIONS).write_text(json.dumps(operations, indent=2))


EXAMPLE_DMA = "example_dma"
"""The name its mover kind is declared under."""


class ExampleDma:
    """A moved mover kind of the platform's own."""

    @staticmethod
    def offer(width_cap: int) -> MoverOffer:
        """Its offer on a shell whose memory port is at most ``width_cap`` bits wide: a
        frame a call, and the call costs of FINN's own mover."""
        return MoverOffer(
            kind=EXAMPLE_DMA,
            width_cap=width_cap,
            frames_per_call=1,
            call_converted=4,
            call_direct=13,
        )


__all__ = [
    "BASE",
    "DESCRIPTION",
    "DIRECTORY",
    "EXAMPLE_DMA",
    "IMAGE",
    "MAP",
    "OPERATIONS",
    "ExampleDma",
    "ExampleLinker",
    "RecordingRuntime",
]
