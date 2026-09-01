# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaging as a plugin, and the refusal that keeps one honest.

Two axes, not one stage list.  The single most common structural error is
treating "generated source, packaged unit, OOC synthesis, IP-XACT, simulation
object" as one *list*, then noticing empirically that some of them are
siblings.  They are siblings because they are on different axes:

*realization* lowers a component -- source, netlist, checkpoint -- and
*distribution* projects it -- RTL module directory, IP-XACT, FuseSoC, simulator
library, tarball.  A packager declares the realization it **requires**, which
is why a checkpoint-backed IP package does not impose its ordering on any
other format.  Adding a distribution costs one ``Packager`` and touches
nothing; adding a realization level touches no packager that does not require
it.  In a single stage list both are edits to the list and to everything that
walks it.

``supports()`` is the load-bearing method.  **A format must refuse rather than
degrade.**  The cautionary case is a template that assumes one AXI-Stream in
and one out: given a component with two inputs and an extra clock it emits
something that looks fine and is missing a port.  Silent degradation at the
packaging boundary is the same wrong-hit class as a bad cache key, one layer
out, and this is where it is caught.

A packager reads **only** ``(PortableComponent, Target, Options)``.  It cannot
reach a Kernel, a Region, a design point, an Operation or a graph, so it cannot
be operation-specific -- and a second Operation gets every format for free.

``PortableComponent`` is deliberately **not** a stage: it is a realization
artifact reference plus an ABI, consuming no new input and materializing
nothing separately.  Giving it a key would distinguish nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Union

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.derivation import ArtifactRef, ContentRef, Derivation


class PackagingError(Exception):
    """A package could not be planned or validated."""


class Realization(Enum):
    """Which representation of the hardware exists.  A lowering, not a stage."""

    SOURCE = "source"
    NETLIST = "netlist"
    CHECKPOINT = "checkpoint"
    PLACED_ROUTED = "placed-routed"


#: Increasing order, so "at least this lowered" is a comparison.
_ORDER = {
    Realization.SOURCE: 0,
    Realization.NETLIST: 1,
    Realization.CHECKPOINT: 2,
    Realization.PLACED_ROUTED: 3,
}


def at_least(available: Realization, required: Realization) -> bool:
    return _ORDER[available] >= _ORDER[required]


@dataclass(frozen=True)
class Supported:
    """The format can express this ABI without losing anything."""


@dataclass(frozen=True)
class Refused:
    """The format cannot express this ABI, and says which part.

    Naming the part is the difference between a refusal somebody can act on
    and one they work around.
    """

    reason: str
    ports: tuple[str, ...] = ()

    def __str__(self) -> str:
        named = f" ({', '.join(self.ports)})" if self.ports else ""
        return f"{self.reason}{named}"


Support = Union[Supported, Refused]


@dataclass(frozen=True)
class Target:
    """An exact part, and the capabilities that decide coverage and sharing.

    **Capabilities decide coverage and source sharing; synthesis identity uses
    the exact part.**  An out-of-context result is tied to the device, package,
    speed grade, timing database, primitive mapping and tool version, so equal
    capability records do not make two parts one synthesis result -- and
    inferring a compatibility envelope from them would assert something nothing
    checked.
    """

    part: str
    capabilities: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.part:
            raise PackagingError("a target needs an exact part")
        object.__setattr__(
            self, "capabilities", tuple(sorted(self.capabilities, key=lambda item: item[0]))
        )


@dataclass(frozen=True)
class PackageOptions:
    """Per-format options.  Formats subclass this; the base carries nothing."""

    def as_options(self) -> tuple[tuple[str, str], ...]:
        """Flattened for the derivation key, sorted, since it is a table."""

        return ()


@dataclass(frozen=True)
class PortableComponent:
    """A realization artifact plus its ABI.  A view, not a stage."""

    artifact: ArtifactRef
    abi: ComponentABI
    realization: Realization
    #: Relative name to content, in declared compile order.
    files: tuple[tuple[str, ContentRef], ...] = ()
    entry_point: str = ""

    def __post_init__(self) -> None:
        names = [name for name, _ in self.files]
        if len(names) != len(set(names)):
            raise PackagingError("a component stages one name twice")


@dataclass(frozen=True)
class PackagePlan:
    """What a packager decided, before anything is written.

    ``contents`` **is** the declared layout, name for name and in order.  The
    two used to be independent, and the first format written got it wrong in
    the least visible way available: it declared the sources and emitted only
    the descriptor, so the package that shipped had no RTL in it.  A round trip
    that reads the descriptor back does not notice, because the descriptor is
    the one file that *was* there.

    So the plan checks itself.  A layout is a promise about a tree, and a plan
    that cannot fill its own promise is not a plan.
    """

    derivation: Derivation
    #: Relative name to bytes, in declared-layout order.
    contents: tuple[tuple[str, bytes], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        layout = self.derivation.outputs
        if layout is None:
            raise PackagingError(
                f"{self.derivation.kind} declares no output layout, so nothing says what "
                "this package is supposed to contain"
            )
        emitted = tuple(name for name, _ in self.contents)
        if emitted != layout.entries:
            missing = [name for name in layout.entries if name not in set(emitted)]
            extra = [name for name in emitted if name not in set(layout.entries)]
            detail = (
                f"declares {list(layout.entries)} and emits {list(emitted)}"
                if not missing and not extra
                else f"declares {list(layout.entries)}"
                + (f", does not emit {missing}" if missing else "")
                + (f", emits undeclared {extra}" if extra else "")
            )
            raise PackagingError(f"{self.derivation.kind} {detail}")


class ContentSource(Protocol):
    """Somewhere a ``ContentRef`` can be turned back into bytes.

    ``ArtifactStore`` satisfies this structurally, which is what gives
    ``get_blob`` a caller other than a test: a packager holds references, and
    the only way to put a source file into a package is to resolve one.
    """

    def get_blob(self, reference: ContentRef) -> bytes: ...


class Packager(Protocol):
    """A distribution format.  It reads a component, a target, and options.

    ``contents`` is a resolver and not a fourth input: it turns the references
    the component already carries into the bytes they name, and a packager can
    reach nothing through it that the component did not declare.
    """

    format_id: str
    contract_version: str
    required_realization: Realization
    options_schema: type[PackageOptions]

    def supports(self, abi: ComponentABI) -> Support: ...

    def plan(
        self,
        component: PortableComponent,
        target: Target,
        options: PackageOptions,
        contents: ContentSource,
    ) -> PackagePlan: ...

    def parse(self, contents: Mapping[str, bytes]) -> ComponentABI: ...


def staged_sources(
    component: PortableComponent, contents: ContentSource
) -> tuple[tuple[str, bytes], ...]:
    """The component's files as bytes, in declared compile order.

    Shared by both in-tree formats, because "resolve what the component
    declared, in the order it declared it" is not a per-format decision and
    writing it twice is how the two would eventually disagree.
    """

    return tuple((name, contents.get_blob(reference)) for name, reference in component.files)


def check_realization(packager: Packager, component: PortableComponent) -> Refused | None:
    """A declared requirement, not a hardcoded chain."""

    if not at_least(component.realization, packager.required_realization):
        return Refused(
            f"{packager.format_id} requires a {packager.required_realization.value} "
            f"realization and the component is {component.realization.value}"
        )
    return None


def plan_package(
    packager: Packager,
    component: PortableComponent,
    target: Target,
    options: PackageOptions,
    contents: ContentSource,
) -> PackagePlan:
    """Refuse before planning, so a degraded package is never produced."""

    refusal = check_realization(packager, component)
    if refusal is not None:
        raise PackagingError(str(refusal))
    support = packager.supports(component.abi)
    if isinstance(support, Refused):
        raise PackagingError(f"{packager.format_id} cannot express this component: {support}")
    return packager.plan(component, target, options, contents)


def round_trip(packager: Packager, plan: PackagePlan) -> ComponentABI:
    """Package an ABI, parse it back out of the format, and return what survived.

    The conformance kit, and the reason it exists: without the parse half, a
    round-trip test checks the emitter against itself.  ``component.xml`` is
    parseable and a ``.core`` file is YAML, so this is cheap for every format
    we plan to add -- and it is what keeps ``supports()`` truthful rather than
    optimistic.
    """

    return packager.parse(dict(plan.contents))


def registry(packagers: Sequence[Packager]) -> Mapping[str, Packager]:
    """Format id to packager, refusing a duplicate registration."""

    found: dict[str, Packager] = {}
    for packager in packagers:
        if packager.format_id in found:
            raise PackagingError(f"{packager.format_id} is registered twice")
        found[packager.format_id] = packager
    return found


__all__ = [
    "ContentSource",
    "PackageOptions",
    "PackagePlan",
    "Packager",
    "PackagingError",
    "PortableComponent",
    "Realization",
    "Refused",
    "Support",
    "Supported",
    "Target",
    "at_least",
    "check_realization",
    "plan_package",
    "registry",
    "round_trip",
    "staged_sources",
]
