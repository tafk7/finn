# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-neutral computation contracts for semantic and physical binding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputationContract:
    """What the traffic crossing a Region boundary is required to mean."""

    id: str
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a computation contract must be named")


DOT_PRODUCT_COMPUTATION = ComputationContract("mvau.dot_product")
ACTIVATION_REPLAY_COMPUTATION = ComputationContract("mvau.activation_replay")

__all__ = [
    "ACTIVATION_REPLAY_COMPUTATION",
    "ComputationContract",
    "DOT_PRODUCT_COMPUTATION",
]
