# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A made-up platform outside FINN (``platform.toml`` beside this file): the code its
declarations name. Its linker writes the description it is given and a stub image, its
runtime records the operations it is asked for, and its pushed mover kind is a plain
class. A test loads the declarations with ``extra=``, or as an installed package's entry
point would name this package."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class ExamplePush:
    """A pushed adapter: the host writes each frame to its stream, ``width`` bits a beat."""

    width: int = 32


__all__ = ["DIRECTORY", "IMAGE", "ExamplePush", "RecordingRuntime", "link"]
