# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source sets: declared order, composable closures, and refused collisions.

Compile order is **contributor-declared and identity-bearing**.  A topological
sort over module symbols does not capture textual includes and macro state,
VHDL libraries and contexts, tool-specific ordering rules, files deliberately
compiled twice under different defines, or ordering among otherwise
independent files -- and ``provides``/``requires`` is *itself* a declaration, a
harder one than an ordered list, so replacing the list with it trades a simple
authority for a complex one and discharges nothing.

So the list stays authoritative.  Symbol relations earn their place by doing
two things a list cannot:

**Closure merge.**  A total order per Kernel does not compose.  Merging two
Kernels' closures means merging two total orders, which is ill-defined -- and
that merge is exactly where the current duplicate-staging behaviour comes from,
with ``add_multi.sv`` numbered independently by each placement and staged
twice.  Symbol relations are *local per file*, so they compose.

**Collision detection.**  Two Kernels staging different revisions of a file
that both define ``add_multi`` is otherwise silent, and whichever the tool
reads first wins.  Two unequal definitions of one symbol in one library are an
error; the acceptable resolutions are library isolation or refusal, and
guessing is not one of them.

Two deduplications live here and they are not the same one:

============  =========================================  ==========================
level         key                                        effect
============  =========================================  ==========================
blob          content digest                             one copy in the store
unit          content, library, defines, options         one entry in a compile list
============  =========================================  ==========================

Identical bytes compiled under two logical libraries, or with different
defines, are **two compilation units and one blob**.  Collapsing them is a
wrong hit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from finn.dataflow.artifacts.derivation import ContentRef


class SourceError(Exception):
    """A source set is not well formed, so no closure exists."""


class Language(Enum):
    SYSTEMVERILOG = "systemverilog"
    VERILOG = "verilog"
    VHDL = "vhdl"
    CPP = "cpp"
    DATA = "data"
    CONSTRAINT = "constraint"


class Role(Enum):
    SOURCE = "source"
    DATA = "data"
    CONSTRAINT = "constraint"
    HEADER = "header"


#: The logical HDL library a file is compiled into when nothing says otherwise.
DEFAULT_LIBRARY = "work"


@dataclass(frozen=True)
class CompileOptions:
    """Per-file compilation inputs, which are part of a unit's identity.

    ``defines`` is sorted, because a define table written in another order is
    the same table.  ``includes`` and ``flags`` are not: an include search path
    is searched in order, and tool flags are order-sensitive often enough that
    reordering them is a change rather than a rewrite.
    """

    defines: tuple[tuple[str, str], ...] = ()
    includes: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.defines)
        if len(names) != len(set(names)):
            raise SourceError("a define is named twice")
        object.__setattr__(self, "defines", tuple(sorted(self.defines, key=lambda item: item[0])))


NO_OPTIONS = CompileOptions()


@dataclass(frozen=True, init=False)
class SourceFile:
    """One file: what is in it, how it is compiled, and what it exports.

    ``provides`` and ``requires`` are supplied as any iterable and stored
    **sorted**, because a ``frozenset`` cannot enter a key -- iterating one is
    exactly the unordered iteration the projection refuses.  Declaring them is
    optional; where they are absent, the closure merge falls back on declared
    order alone and says so.
    """

    content: ContentRef
    #: The name this file is staged as, relative to the unit directory.
    path: str
    language: Language
    library: str = DEFAULT_LIBRARY
    role: Role = Role.SOURCE
    #: Dialect or revision, where it matters: ``2012``, ``c++17``.
    standard: str = ""
    options: CompileOptions = NO_OPTIONS
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __init__(
        self,
        content: ContentRef,
        path: str,
        language: Language,
        library: str = DEFAULT_LIBRARY,
        role: Role = Role.SOURCE,
        standard: str = "",
        options: CompileOptions = NO_OPTIONS,
        provides: Iterable[str] = (),
        requires: Iterable[str] = (),
    ) -> None:
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise SourceError(f"{path!r} is not a name relative to the unit directory")
        if not library:
            raise SourceError(f"{path} must name a library, even if it is {DEFAULT_LIBRARY!r}")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "library", library)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "standard", standard)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "provides", tuple(sorted(set(provides))))
        object.__setattr__(self, "requires", tuple(sorted(set(requires))))

    @property
    def unit(self) -> CompilationUnit:
        """What makes this one entry in a compile list rather than another."""

        return CompilationUnit(self.content, self.library, self.options)


@dataclass(frozen=True)
class CompilationUnit:
    """Coarser than a blob and finer than a file.

    The same bytes under two libraries are two units and one blob.  The first
    revision of the design had only the content-digest form, which would have
    collapsed them.
    """

    content: ContentRef
    library: str
    options: CompileOptions


@dataclass(frozen=True)
class SourceDefinition:
    """One contributor's ordered files.

    ``origin`` is a diagnostic label and is deliberately **not** part of any
    identity taken over this definition: a shared artifact that carried one
    occurrence's owner would acquire a different manifest depending on who
    built it first.  It exists so a refusal can name who asked for what.
    """

    files: tuple[SourceFile, ...]
    origin: str = ""

    def __post_init__(self) -> None:
        seen: set[tuple[str, str]] = set()
        for source in self.files:
            coordinate = (source.library, source.path)
            if coordinate in seen:
                raise SourceError(
                    f"{self.origin or 'a definition'} stages {source.path} twice "
                    f"into library {source.library}"
                )
            seen.add(coordinate)


@dataclass(frozen=True)
class SymbolCollision:
    """Two unequal definitions of one symbol in one library."""

    symbol: str
    library: str
    origins: tuple[str, ...]
    contents: tuple[str, ...]

    def __str__(self) -> str:
        origins = " and ".join(self.origins) or "two definitions"
        return (
            f"{origins} define {self.symbol!r} in library {self.library!r} with "
            f"different contents ({', '.join(digest[:12] for digest in self.contents)}); "
            "isolate them by library or refuse"
        )


@dataclass(frozen=True)
class Closure:
    """The merged, ordered compile list of several definitions."""

    files: tuple[SourceFile, ...] = field(default_factory=tuple)

    @property
    def units(self) -> tuple[CompilationUnit, ...]:
        return tuple(source.unit for source in self.files)

    @property
    def blobs(self) -> tuple[ContentRef, ...]:
        """Distinct content, in first-appearance order."""

        seen: list[ContentRef] = []
        for source in self.files:
            if source.content not in seen:
                seen.append(source.content)
        return tuple(seen)


def _collisions(definitions: Sequence[SourceDefinition]) -> tuple[SymbolCollision, ...]:
    """Every symbol claimed twice in one library by unequal content."""

    claims: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for definition in definitions:
        for source in definition.files:
            for symbol in source.provides:
                claims.setdefault((source.library, symbol), []).append(
                    (definition.origin, source.content.digest)
                )

    found: list[SymbolCollision] = []
    for (library, symbol), claimed in claims.items():
        digests = {digest for _, digest in claimed}
        if len(digests) > 1:
            found.append(
                SymbolCollision(
                    symbol,
                    library,
                    tuple(dict.fromkeys(origin for origin, _ in claimed if origin)),
                    tuple(sorted(digests)),
                )
            )
    return tuple(sorted(found, key=lambda item: (item.library, item.symbol)))


def merge_closures(definitions: Sequence[SourceDefinition]) -> Closure:
    """Merge several declared orders into one, deterministically.

    The rule, in order of strength:

    1. **A unit appears once.**  A file two Kernels both want is staged once,
       which is the duplicate-staging defect this exists to fix.
    2. **Declared order is respected** wherever it does not contradict a symbol
       relation.  Each unit keeps the position of its first appearance across
       the definitions, taken in the order they were given.
    3. **A requirement is met before its requirer**, where both are declared.
       This is a *check on* and a *repair of* the declared order, not a
       replacement for it: a file that requires nothing keeps its declared slot.
    4. **The tie-break is first appearance**, so the result depends on the
       inputs and not on a set iteration.

    A symbol collision refuses before any of this, because a merged order over
    contradictory definitions would be a well-formed answer to a broken
    question.
    """

    collisions = _collisions(definitions)
    if collisions:
        raise SourceError("; ".join(str(collision) for collision in collisions))

    ordered: list[SourceFile] = []
    position: dict[CompilationUnit, int] = {}
    for definition in definitions:
        for source in definition.files:
            unit = source.unit
            if unit in position:
                continue
            position[unit] = len(ordered)
            ordered.append(source)

    providers: dict[tuple[str, str], int] = {}
    for index, source in enumerate(ordered):
        for symbol in source.provides:
            providers[(source.library, symbol)] = index

    # Edges point from provider to requirer, so a topological order compiles a
    # definition before whatever instantiates it.
    dependencies: dict[int, set[int]] = {index: set() for index in range(len(ordered))}
    for index, source in enumerate(ordered):
        for symbol in source.requires:
            provider = providers.get((source.library, symbol))
            if provider is not None and provider != index:
                dependencies[index].add(provider)

    return Closure(tuple(_topological(ordered, dependencies)))


def _topological(
    ordered: Sequence[SourceFile], dependencies: dict[int, set[int]]
) -> list[SourceFile]:
    """Kahn's algorithm with first-appearance as the tie-break.

    ``min()`` over the ready set rather than a ``pop()``: the whole point is
    that the result cannot depend on iteration order, and a set has none worth
    depending on.
    """

    remaining = {index: set(needed) for index, needed in dependencies.items()}
    result: list[SourceFile] = []
    while remaining:
        ready = [index for index, needed in remaining.items() if not needed]
        if not ready:
            cycle = ", ".join(ordered[index].path for index in sorted(remaining))
            raise SourceError(f"the declared symbol relations are cyclic: {cycle}")
        chosen = min(ready)
        del remaining[chosen]
        for needed in remaining.values():
            needed.discard(chosen)
        result.append(ordered[chosen])
    return result


def declared_order_is_consistent(definition: SourceDefinition) -> tuple[str, ...]:
    """Where a definition's own order contradicts its own symbol relations.

    Returned rather than raised.  Derivation from symbols is permitted only as
    a *check* on the declared order, and a check that refuses would quietly
    become the authority.
    """

    provided: dict[tuple[str, str], int] = {}
    for index, source in enumerate(definition.files):
        for symbol in source.provides:
            provided[(source.library, symbol)] = index

    complaints: list[str] = []
    for index, source in enumerate(definition.files):
        for symbol in source.requires:
            at = provided.get((source.library, symbol))
            if at is not None and at > index:
                complaints.append(
                    f"{source.path} requires {symbol!r}, which "
                    f"{definition.files[at].path} provides later in the declared order"
                )
    return tuple(complaints)


def with_library(definition: SourceDefinition, library: str) -> SourceDefinition:
    """Isolate a whole definition into one library.

    One of the two acceptable resolutions for a symbol collision, and the one
    that keeps both revisions compilable.  The other is refusal.
    """

    return replace(
        definition, files=tuple(replace(source, library=library) for source in definition.files)
    )


__all__ = [
    "DEFAULT_LIBRARY",
    "NO_OPTIONS",
    "Closure",
    "CompilationUnit",
    "CompileOptions",
    "Language",
    "Role",
    "SourceDefinition",
    "SourceError",
    "SourceFile",
    "SymbolCollision",
    "declared_order_is_consistent",
    "merge_closures",
    "with_library",
]
