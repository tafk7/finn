# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source contribution values, independent of resolution and rendering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from finn.kernels.artifacts.sources import DEFAULT_LIBRARY, CompileOptions, Language, Role


class ContributionError(Exception):
    """A contribution could not be resolved into a source file."""


@dataclass(frozen=True)
class CopiedSource:
    """A file taken verbatim, keyed by its content.

    ``root`` is the *named* root -- ``kernels``, ``finnlib`` -- and ``path`` is
    relative beneath it, so the contribution does not move with the checkout.
    """

    root: str
    path: str
    library: str = DEFAULT_LIBRARY
    language: Language = Language.SYSTEMVERILOG
    role: Role = Role.SOURCE
    standard: str = ""
    options: CompileOptions = CompileOptions()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.root or not self.path:
            raise ContributionError("a copied source names a root and a path beneath it")
        if self.path.startswith("/") or ".." in self.path.split("/"):
            raise ContributionError(f"{self.path!r} is not relative beneath its source root")
        object.__setattr__(self, "provides", tuple(sorted(set(self.provides))))
        object.__setattr__(self, "requires", tuple(sorted(set(self.requires))))


@dataclass(frozen=True)
class RenderedSource:
    """A file generated from a template file and a flat scalar context.

    The template is named rather than inlined so its content digest can enter
    the plan key: two template revisions of one configuration must produce two
    names, or two unequal components claim one repository coordinate.
    """

    name: str
    template: str
    language: Language = Language.SYSTEMVERILOG
    library: str = DEFAULT_LIBRARY
    role: Role = Role.SOURCE
    standard: str = ""
    options: CompileOptions = CompileOptions()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.template:
            raise ContributionError("a rendered source names its output and its template")


@dataclass(frozen=True)
class DataSlotSpec:
    """The shape of a hole a parameter image will fill."""

    width: int
    depth: int
    packing: str
    #: The filename the RTL reads it under, which is build ABI and not meaning.
    referenced_as: str

    def __post_init__(self) -> None:
        if self.width < 1 or self.depth < 1:
            raise ContributionError("a data slot has a positive width and depth")
        if not self.referenced_as:
            raise ContributionError("a data slot names the file the RTL reads")


@dataclass(frozen=True)
class DataSlot:
    """A declared hole.  It carries no contents, and that is the point."""

    name: str
    spec: DataSlotSpec

    def __post_init__(self) -> None:
        if not self.name:
            raise ContributionError("a data slot needs a name")


Contribution = Union[CopiedSource, RenderedSource, DataSlot]


# Declaration and processing modules share one facade identity in this package.
for _type in (ContributionError, CopiedSource, RenderedSource, DataSlotSpec, DataSlot):
    _type.__module__ = "finn.kernels.artifacts.contributions"

__all__ = [
    "ContributionError",
    "CopiedSource",
    "RenderedSource",
    "DataSlotSpec",
    "DataSlot",
    "Contribution",
]
