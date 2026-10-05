# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's sources, in an order a tool compiles: each once, providers first.

A composed module gathers its children's sources, so one file (``add_multi.sv``,
``fifo.sv``) arrives from several kernels; ``ordered`` stages it once. Two
different files claiming one path or one module symbol are refused rather than
left to whichever the tool reads first. Declared order is kept wherever the
symbol relations (``provides``, ``requires``) do not contradict it: a file
that requires nothing keeps its place.

A header (``.svh``, ``.vh``) is staged like any source but never compiled on
its own: a tool reads it where a source `includes it, from its directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath

HEADER_SUFFIXES = (".svh", ".vh")


def is_header(path: str | PurePath) -> bool:
    """Whether a staged source is a header: included, never compiled on its own."""
    return PurePath(path).suffix in HEADER_SUFFIXES


def include_directories(paths: Sequence[str | PurePath]) -> tuple[PurePath, ...]:
    """The directories of the headers among ``paths``, each once, in order."""
    return tuple(dict.fromkeys(PurePath(path).parent for path in paths if is_header(path)))


class SourceError(Exception):
    """A module's sources cannot be ordered."""


@dataclass(frozen=True)
class SourceFile:
    """One file of a module: its staged name, a digest of its bytes, its symbols."""

    path: str
    content: str
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()


def ordered(files: Sequence[SourceFile]) -> tuple[SourceFile, ...]:
    """``files`` with duplicates staged once and every provider before its requirers."""

    unique: list[SourceFile] = []
    by_path: dict[str, SourceFile] = {}
    for source in files:
        previous = by_path.get(source.path)
        if previous is None:
            by_path[source.path] = source
            unique.append(source)
        elif previous != source:
            raise SourceError(f"two different sources are staged as {source.path}")

    providers: dict[str, int] = {}
    for index, source in enumerate(unique):
        for symbol in source.provides:
            other = providers.setdefault(symbol, index)
            if other != index:
                raise SourceError(f"{unique[other].path} and {source.path} both provide {symbol!r}")
    missing = sorted(
        f"{source.path} requires {symbol!r}"
        for source in unique
        for symbol in source.requires
        if symbol not in providers
    )
    if missing:
        raise SourceError("unresolved symbols: " + ", ".join(missing))

    # Kahn's algorithm, the earliest declared ready file first.
    waiting = {
        index: {providers[symbol] for symbol in source.requires} - {index}
        for index, source in enumerate(unique)
    }
    result: list[SourceFile] = []
    while waiting:
        ready = [index for index, needed in waiting.items() if not needed]
        if not ready:
            cycle = ", ".join(unique[index].path for index in sorted(waiting))
            raise SourceError(f"the sources' symbol relations are cyclic: {cycle}")
        chosen = min(ready)
        del waiting[chosen]
        for needed in waiting.values():
            needed.discard(chosen)
        result.append(unique[chosen])
    return tuple(result)


__all__ = [
    "HEADER_SUFFIXES",
    "SourceError",
    "SourceFile",
    "include_directories",
    "is_header",
    "ordered",
]
