# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed capability vocabulary for implementation Spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from finn.dataflow.space.declarations import Projection, Space


View: TypeAlias = Projection


@dataclass(frozen=True, slots=True)
class ImplementationIdentity:
    family: str
    version: str

    def __post_init__(self) -> None:
        if not self.family or not self.version:
            raise ValueError("implementation family and version must be non-empty")


def implementation_identity(implementation: Space | type[Space]) -> ImplementationIdentity:
    implementation_type = (
        implementation if isinstance(implementation, type) else type(implementation)
    )
    family = getattr(implementation_type, "id", "")
    version = getattr(implementation_type, "version", "")
    if not isinstance(family, str) or not isinstance(version, str):
        raise TypeError("implementation id and version must be strings")
    return ImplementationIdentity(family, version)


__all__ = ["ImplementationIdentity", "View", "implementation_identity"]
