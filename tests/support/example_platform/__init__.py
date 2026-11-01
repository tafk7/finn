# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A made-up platform outside FINN (``platform.toml`` beside this file): the code its
declarations name. Its linker writes the description it is given and a stub image, its
runtime records the operations it is asked for, and its moved mover kind makes an offer
as FINN's own does (``finn.kernels.movers.IodmaMover.offer``). Its shell offers FINN's
own mover kind, the one a core places yet (``finn.kernels.movers.PLACED_KINDS``). A test
loads the declarations with ``extra=``, or as an installed package's entry point would
name this package.

It shows a platform declared, checked and resolved, its code found, and a core placed in
its shell with no change to FINN. It does not show a core linked or run through it: FINN
does not call a linker or a runtime through its declarations yet."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finn.kernels.movers import MoverOffer

DIRECTORY = Path(__file__).resolve().parent
"""This package's directory, holding its ``platform.toml``."""

IMAGE = "image.example"
"""The stub image the linker writes."""


def link(description: Mapping[str, Any], directory: Path) -> Path:
    """Write ``description`` (``description.json``) and a stub image into ``directory``;
    the image's path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "description.json").write_text(json.dumps(description, indent=2))
    image = directory / IMAGE
    image.write_bytes(b"example image\n")
    return image


@dataclass
class RecordingRuntime:
    """A runtime that records each operation it is asked for, in order."""

    operations: list[tuple[str, Any]] = field(default_factory=list)

    def load(self, image: Path) -> None:
        self.operations.append(("load", image.name))

    def write(self, endpoint: str, value: int) -> None:
        self.operations.append(("write", (endpoint, value)))


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


__all__ = ["DIRECTORY", "EXAMPLE_DMA", "IMAGE", "ExampleDma", "RecordingRuntime", "link"]
