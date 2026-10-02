# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a kernel contributes to its module's sources: files it copies and data it generates.

``provides`` and ``requires`` name module symbols (``module:dotp``), from which a
module's sources are ordered (``sources.ordered``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


class ContributionError(Exception):
    """A contribution names no file it could stand for."""


def _relative(path: str, label: str) -> None:
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise ContributionError(f"{path!r} is not a name relative to {label}")


@dataclass(frozen=True)
class CopiedSource:
    """A file taken verbatim from a named root (``finnlib``), so it does not move
    with the checkout."""

    root: str
    path: str
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.root:
            raise ContributionError("a copied source names its root")
        _relative(self.path, "its source root")
        object.__setattr__(self, "provides", tuple(sorted(set(self.provides))))
        object.__setattr__(self, "requires", tuple(sorted(set(self.requires))))


@dataclass(frozen=True)
class GeneratedData:
    """A data file whose contents the kernel generates, such as a memory image.

    The contents are part of the requirements, so they enter its fingerprint.
    ``path`` is the name the RTL reads it under (an ``INIT_FILE``), beside the
    module's sources.
    """

    path: str
    data: bytes

    def __post_init__(self) -> None:
        _relative(self.path, "the module's directory")
        if type(self.data) is not bytes:
            raise ContributionError("generated data is bytes")


Contribution = Union[CopiedSource, GeneratedData]
"""What a kernel contributes: a file it copies, or data it generates."""


__all__ = ["Contribution", "ContributionError", "CopiedSource", "GeneratedData"]
