# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical result values, independent of hierarchical composition algorithms."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import TypeAlias
from finn.dataflow.model.logical.network import DataflowNetwork
from finn.dataflow.model.logical.region import DataflowRegion

_ATOM = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


class CompositionError(ValueError):
    """A hierarchy cannot be lowered to one canonical flat Network."""


@dataclass(frozen=True, slots=True)
class ImplementationPath:
    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments or any(not _ATOM.fullmatch(segment) for segment in segments):
            raise ValueError("an implementation path needs non-empty ASCII identifier segments")
        object.__setattr__(self, "segments", segments)

    def child(self, segment: str) -> ImplementationPath:
        return ImplementationPath((*self.segments, segment))

    @property
    def value(self) -> str:
        return "/".join(self.segments)


@dataclass(frozen=True, slots=True)
class RegionResult:
    region: DataflowRegion

    def __post_init__(self) -> None:
        if not isinstance(self.region, DataflowRegion):
            raise TypeError("RegionResult contains one DataflowRegion")


@dataclass(frozen=True, slots=True)
class QualifiedChildResult:
    use_path: ImplementationPath
    result: RegionResult | NetworkResult


@dataclass(frozen=True, slots=True)
class NetworkResult:
    network: DataflowNetwork
    children: tuple[QualifiedChildResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.network, DataflowNetwork):
            raise TypeError("NetworkResult contains one DataflowNetwork")
        children = tuple(self.children)
        if any(not isinstance(child, QualifiedChildResult) for child in children):
            raise TypeError("NetworkResult children must be QualifiedChildResult values")
        object.__setattr__(self, "children", children)


LogicalResult: TypeAlias = RegionResult | NetworkResult


# Historical identities remain stable for stored values and fingerprints.
for _type in (
    CompositionError,
    ImplementationPath,
    RegionResult,
    QualifiedChildResult,
    NetworkResult,
):
    _type.__module__ = "finn.dataflow.model.logical.composition"

__all__ = [
    "CompositionError",
    "ImplementationPath",
    "RegionResult",
    "QualifiedChildResult",
    "NetworkResult",
    "LogicalResult",
]
