# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a partition's body records about its choices beyond their values: the
``finn.choices`` namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qonnx.core.metadata import JSON, Namespace

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

CHOICES = Namespace("finn.choices", version=1)
"""The namespace, on a partition's body: ``HELD`` and ``OBJECTIVE_FPS``."""


def _by_owner_values(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(held, dict) for held in value.values())


HELD = CHOICES.key(
    "held", JSON, check=_by_owner_values, expect="{graph name: {key: value}}, the pinned choices"
)
"""The choices pinned in a partition's body, by owner and key (``{graph name: {key:
value}}``, the form ``kernel_choices.json`` takes). Every exploration of the body keeps
them (``Seam.held``), so a pin holds for the whole build. Absent where nothing is
pinned."""

OBJECTIVE_FPS = CHOICES.key("objective_fps", float)
"""The frames a second a target throughput explored the body to (``TargetThroughput``),
which a shell's build reports its delivered clock against. Absent where none did, or
where another strategy that may change the folding (``FOLDING``) explored it since."""


def held_choices(model: ModelWrapper) -> dict[str, dict[str, object]]:
    """The choices pinned in a partition's body, ``model``, by owner and key there
    (``HELD``); none where nothing is pinned."""
    held: dict[str, dict[str, object]] | None = model.get(HELD)
    return {} if held is None else held


__all__ = [
    "CHOICES",
    "HELD",
    "OBJECTIVE_FPS",
    "held_choices",
]
