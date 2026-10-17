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


def _keys_by_owner(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(keys, list) and all(isinstance(key, str) for key in keys)
        for keys in value.values()
    )


HELD = CHOICES.key(
    "held", JSON, check=_keys_by_owner, expect="{graph name: [key, ...]}, the held choices' keys"
)
"""The keys of the choices held in a partition's body, by owner (``{graph name: [key,
...]}``, each owner's keys sorted). Strategies leave a held choice alone: every
exploration of the body keeps it (``Seam.held``), so a hold holds for the whole build.
The mark names keys only; a held choice's value is the one its node or tensor states.
Absent where nothing is held. It is no part of the design (``design_digest``): a hold
changes no hardware."""

OBJECTIVE_FPS = CHOICES.key("objective_fps", float)
"""The frames a second a target throughput explored the body to (``TargetThroughput``),
which a shell's build reports its delivered clock against. Absent where none did, or
where an exploration whose strategy changed any kernel's choice saved it since."""


def held_keys(model: ModelWrapper) -> dict[str, list[str]]:
    """The keys of the choices held in a partition's body, ``model``, by owner (``HELD``):
    which choices strategies leave alone, their values being the ones the nodes and tensors
    state; none where nothing is held."""
    held: dict[str, list[str]] | None = model.get(HELD)
    return {} if held is None else held


__all__ = [
    "CHOICES",
    "HELD",
    "OBJECTIVE_FPS",
    "held_keys",
]
