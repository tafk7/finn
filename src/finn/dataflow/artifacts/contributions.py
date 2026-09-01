# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed source contributions: copied, rendered, and a hole for data.

A Kernel's source declaration is an ordered list of *contributions*, not a list
of files.  The difference is what deletes the filename-suffix special case in
the tree today, where a Kernel declares a Verilog template as an ordinary
source file and an operation module removes it from the manifest by testing
``path.endswith(...)``.  Filename-based composition is what the architecture
forbids, and it exists only because the manifest has no way to say "this entry
is rendered".

``DataSlot`` is a **hole**, not contents.  A definition declares the slot; a
``DataBinding`` supplies an image, and a stage includes the binding in its key
only if it consumes the contents.  Putting the image in the structural closure
and then asserting the structural key is independent of it cannot both be true
-- anything in a closure is in that closure's key -- so the independence is
made structural instead of conventional.

The reuse this buys is not marginal.  Every layer in a network with equal
folding shares one structural component instead of holding N, and a
runtime-writable memory shares the complete structural component across every
parameter value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from finn.dataflow.artifacts.derivation import ContentRef
from finn.dataflow.artifacts.projection import content_digest
from finn.dataflow.artifacts.render import RenderError, render_template
from finn.dataflow.artifacts.sources import (
    DEFAULT_LIBRARY,
    CompileOptions,
    Language,
    Role,
    SourceDefinition,
    SourceError,
    SourceFile,
)


class ContributionError(Exception):
    """A contribution could not be resolved into a source file."""


@dataclass(frozen=True)
class CopiedSource:
    """A file taken verbatim, keyed by its content.

    ``root`` is the *named* root -- ``finn``, ``finnlib`` -- and ``path`` is
    relative beneath it, so the contribution does not move with the checkout.
    """

    root: str
    path: str
    library: str = DEFAULT_LIBRARY
    language: Language = Language.SYSTEMVERILOG
    role: Role = Role.SOURCE
    options: CompileOptions = CompileOptions()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.root or not self.path:
            raise ContributionError("a copied source names a root and a path beneath it")


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


@dataclass(frozen=True)
class ParameterImageRef:
    """Contents, as a separate artifact with its own key."""

    content: ContentRef
    datatype: str
    shape: tuple[int, ...]
    packing: str


@dataclass(frozen=True)
class DataBinding:
    """Slot name to image, supplied at the stage that consumes the contents."""

    bindings: tuple[tuple[str, ParameterImageRef], ...] = ()

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.bindings)
        if len(names) != len(set(names)):
            raise ContributionError("a data slot is bound twice")
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda item: item[0])))


@dataclass(frozen=True)
class ResolvedContributions:
    """What a manifest becomes once its templates have been rendered.

    ``slots`` stay separate from ``definition``: the structural closure is
    complete without any image, which is what makes the structural key
    independent of the contents rather than merely documented as independent.
    """

    definition: SourceDefinition
    slots: tuple[DataSlot, ...] = ()
    #: Digest of every template read, for the pre-render plan key (§9.2).
    template_digests: tuple[tuple[str, ContentRef], ...] = ()


def resolve(
    contributions: Sequence[Contribution],
    *,
    roots: Mapping[str, Path],
    template_roots: Sequence[Path],
    context: Mapping[str, object],
    origin: str = "",
) -> ResolvedContributions:
    """Turn an ordered manifest into an ordered source definition.

    Order is preserved exactly.  Nothing here sorts, filters by suffix, or
    decides that an entry is "really" a template -- the manifest already said
    which entries are rendered, which is the whole reason it is typed.
    """

    files: list[SourceFile] = []
    slots: list[DataSlot] = []
    digests: list[tuple[str, ContentRef]] = []

    for contribution in contributions:
        if isinstance(contribution, DataSlot):
            slots.append(contribution)
            continue
        if isinstance(contribution, CopiedSource):
            root = roots.get(contribution.root)
            if root is None:
                raise ContributionError(
                    f"{origin or 'a manifest'} declares sources under "
                    f"{contribution.root!r}, which this checkout does not resolve"
                )
            located = Path(root) / contribution.path
            try:
                data = located.read_bytes()
            except OSError as error:
                raise ContributionError(f"{located} cannot be read") from error
            text_path = contribution.path
        else:
            found = _locate(template_roots, contribution.template)
            digests.append((contribution.template, ContentRef(content_digest(found.read_bytes()))))
            try:
                rendered = render_template(template_roots, contribution.template, context)
            except RenderError as error:
                raise ContributionError(f"{contribution.name}: {error}") from error
            data = rendered.encode()
            text_path = contribution.name

        try:
            files.append(
                SourceFile(
                    ContentRef(content_digest(data)),
                    text_path,
                    contribution.language,
                    library=contribution.library,
                    role=contribution.role,
                    options=contribution.options,
                    provides=contribution.provides,
                    requires=contribution.requires,
                )
            )
        except SourceError as error:
            raise ContributionError(str(error)) from error

    return ResolvedContributions(
        SourceDefinition(tuple(files), origin=origin), tuple(slots), tuple(digests)
    )


def _locate(roots: Sequence[Path], name: str) -> Path:
    for root in roots:
        candidate = Path(root) / name
        if candidate.is_file():
            return candidate
    raise ContributionError(f"template {name!r} is not under any declared template root")


__all__ = [
    "Contribution",
    "ContributionError",
    "CopiedSource",
    "DataBinding",
    "DataSlot",
    "DataSlotSpec",
    "ParameterImageRef",
    "RenderedSource",
    "ResolvedContributions",
    "resolve",
]
