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

from finn.kernels.artifacts.derivation import ContentRef
from finn.kernels.artifacts.projection import content_digest
from finn.kernels.artifacts.render import RenderError, render_template
from finn.kernels.artifacts.sources import (
    SourceDefinition,
    SourceError,
    SourceFile,
)


from finn.kernels.artifacts.contribution_types import (
    ContributionError,
    CopiedSource,
    RenderedSource,
    DataSlotSpec,
    DataSlot,
    Contribution,
)


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
                    standard=contribution.standard,
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
