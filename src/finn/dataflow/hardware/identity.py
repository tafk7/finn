# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Artifact identity: what makes two builds of the same hardware one build.

An identity is computed from the *build inputs alone*, and it is computed
per **stage**, because the stages do not consume the same inputs:

```text
generated source   Kernel family, its sources and their content, the physical
                   parameters driven into them, its own committed choices
packaged unit      the generated source, plus the staged layout and the shape
                   of the command that instantiates it
OOC synthesis      the packaged unit, plus the part, the clock, and the builder
IP-XACT package    the packaged unit, plus the VLNV, the part, and the builder
```

The last two are *siblings*, not a chain.  Both consume the packaged unit and
neither reads the other's output: packaging an IP does not need a utilization
report, and synthesizing does not need a repository coordinate.  Making one
depend on the other would put an input in a key that the stage never reads,
which is the collapse the Phase 5 review rejected on the same grounds.

Those boundaries are not a matter of taste.  ``render_decomposed_wrapper``
never invokes Vivado, so the builder version cannot change one byte of
generated source; and every way a target reaches the RTL is already a declared
parameter (``VERSION``, ``SEGMENTLEN``), so two parts admitting the same
parameters admit the same source.  Putting either into the source key would
make identical text key differently and stop it being shared -- the mirror of
the wrong-hit this design is against, and just as wrong.

What is excluded from every stage is as load-bearing as what is included.
ONNX node names, the dataflow scope id, source tensor names, graph position,
and physical instance names never reach a key.  Neither does a materialized
absolute path: an artifact does not become a different artifact because it was
staged under another directory.
:class:`~finn.dataflow.hardware.KernelOrigin` carries the placement facts --
it is binding provenance, and this is the other value it points at.

Each identity is a typed frozen value rather than a bare string.  ``key`` is a
digest for addressing; ``serialization`` is what the digest was taken over,
and it is the only one of the two that a failed lookup can be explained from.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from string import Formatter
from pathlib import Path

from finn.dataflow.hardware.kernel import HardwareKernel, scalar_parameters

#: Bumped when what an identity *contains* changes.  Without it, adding a field
#: makes every prior identity look like a different design rather than like one
#: this version cannot read.
#:
#: ``v2`` moved the target and the builder out to the stages that consume them.
KERNEL_ARTIFACT_SCHEMA_VERSION = "kernel-artifact-identity-v2"

#: The composed reading has its own schema, because it is its own value.
COMPOSED_ARTIFACT_SCHEMA_VERSION = "composed-artifact-identity-v2"

#: Stage two: what was generated, plus how it is laid out and instantiated.
PACKAGED_ARTIFACT_SCHEMA_VERSION = "packaged-artifact-identity-v1"

#: Stage three: the packaged unit, plus the device it was synthesized for.
SYNTHESIS_ARTIFACT_SCHEMA_VERSION = "synthesis-artifact-identity-v1"

#: Stage three, the other one: the packaged unit as an IP-XACT component.
#:
#: A *sibling* of synthesis rather than a stage after it.  Both consume the
#: packaged unit and neither consumes the other -- packaging an IP does not
#: need a utilization report, and synthesizing does not need a VLNV.  Stacking
#: them would make the IP key depend on a synthesis run it never reads, which
#: is the "moving upper-stage inputs downward" the Phase 5 review rejected.
IP_PACKAGE_ARTIFACT_SCHEMA_VERSION = "ip-package-artifact-identity-v1"

Scalar = bool | int | float | str


class ArtifactIdentityError(Exception):
    """A build input could not be read or represented, so no identity exists."""


def content_hash(data: bytes) -> str:
    """The one hash function used everywhere in this module."""

    return sha256(data).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _encode(name: str, value: object) -> Scalar:
    """One committed choice, as something a key can be taken over.

    Scalars pass through.  An ``Enum`` becomes
    ``module.QualName.MEMBER`` -- its *name* and not its value, because two
    members sharing a value are two choices and would otherwise key alike.
    Anything else is refused rather than stringified: ``str()`` on an arbitrary
    object is ``repr()`` by another route, and ``repr()`` has no stability
    contract.

    Fully qualified, and that is not tidiness.  The bare class name collides:
    two unrelated ``Mode`` enums in different modules both encode as
    ``Mode.X``, so two Kernels choosing genuinely different things would key
    alike -- a wrong hit reached through the one field added specifically to
    stop Kernel-local choices colliding.
    """

    if isinstance(value, Enum):
        kind = type(value)
        return f"{kind.__module__}.{kind.__qualname__}.{value.name}"
    if type(value) in (bool, int, float, str):
        return value  # type: ignore[return-value]
    raise ArtifactIdentityError(
        f"{name} is a {type(value).__name__}, which has no stable serialization; "
        "a build-affecting choice must be a scalar or an Enum"
    )


#: A word that is an absolute path.  Crude on purpose: a command *shape* has no
#: business containing one at all, so anything that looks like one is refused
#: rather than parsed.
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s{(\[=])[/~]\S")


def _check_command_shape(recipe: str) -> None:
    """A recipe must be a template, not a command someone already rendered.

    Checked by *parsing*, because the earlier "does it contain a brace or a
    newline" test passed anything multi-line -- including a fully rendered
    script with absolute paths in it, which is precisely the root-dependence
    the shape exists to keep out of the key.

    Two conditions.  It has to name at least one substitution, since a rendered
    command names none; and no literal part of it may contain an absolute path,
    since a template that hard-codes one is root-dependent wherever its
    placeholders are.
    """

    try:
        parsed = tuple(Formatter().parse(recipe))
    except ValueError as error:
        raise ArtifactIdentityError(f"{recipe!r} is not a well-formed command shape") from error
    if not any(field is not None for _, field, _, _ in parsed):
        raise ArtifactIdentityError(
            f"{recipe!r} names no substitution, so it is a rendered command rather "
            "than a command shape; a rendered command carries materialized paths"
        )
    for literal, _, _, _ in parsed:
        found = _ABSOLUTE_PATH.search(literal)
        if found:
            raise ArtifactIdentityError(
                f"a command shape must not hard-code an absolute path, and "
                f"{recipe!r} contains {found.group().strip()!r}"
            )


def _local_assignments(kernel: HardwareKernel) -> tuple[tuple[str, Scalar], ...]:
    """A Kernel's own committed choices, under placement-independent names.

    A Kernel placed twice has two namespaces over one authored design, so the
    *qualified* path is a placement fact and the trailing name is not.  The
    prefix is therefore stripped against the declaration's own namespace rather
    than by splitting on the last dot, which would silently accept a path
    belonging to something else.

    These belong in the key because ``elaborate()`` is allowed to read them.
    The MVAU happens to route its one local choice into an RTL parameter as
    well, so this slice would be safe without them -- but the contract is that
    a Kernel *may* let a local choice change its elaboration without that choice
    becoming a parameter, and two such configurations must not collide.
    """

    namespace = kernel.declaration.namespace
    prefix = f"{namespace}."
    encoded: list[tuple[str, Scalar]] = []
    for path, value in kernel.assignments.items():
        text = str(path)
        if not text.startswith(prefix):
            raise ArtifactIdentityError(
                f"{kernel.kernel_id} committed {text}, which is not under its own "
                f"namespace {namespace}; a Kernel owns only its local choices"
            )
        name = text[len(prefix) :]
        encoded.append((name, _encode(f"{kernel.kernel_id}.{name}", value)))
    return tuple(sorted(encoded))


@dataclass(frozen=True)
class BuilderIdentity:
    """Which tool built it, as a declared label rather than a probe.

    Deliberately *not* read from the installed Vivado.  Reading it would make
    identity construction depend on the tool being present, so every test of
    the value would need a stub and the identity would stop being a pure
    function of its inputs.  A caller that knows its tool version passes it.

    This is also not ``BuildBackend`` from the vocabulary note, which is a
    tool-execution adapter.  This is a label; if it grows a ``run()`` method,
    the separation has failed.

    The cost of the choice, stated: nothing checks the supplied version against
    the tool that actually runs, so a caller that lies gets a wrong cache hit.
    It reaches only :class:`SynthesisArtifactIdentity`, which is the stage a
    tool is actually invoked at, so the exposure is exactly as wide as the
    claim.
    """

    backend_id: str
    tool_version: str

    def __post_init__(self) -> None:
        if not self.backend_id or not self.tool_version:
            raise ValueError("a builder identity needs a backend id and a tool version")


#: What a caller that says nothing gets.
#:
#: The version is the literal string ``unspecified`` and not a probe of
#: ``XILINX_VIVADO``.  Reading the environment here would look more honest and
#: be less so: it would make the identity depend on ambient state, so the same
#: inputs would key differently in a shell that happened to have the tool on
#: its path.  A caller that cares passes it, and
#: ``finn.util.basic.get_vivado_version`` is where it reads one from.
DEFAULT_BUILDER = BuilderIdentity("vivado", "unspecified")


@dataclass(frozen=True)
class TargetIdentity:
    """The device and timing an artifact was *synthesized* for.

    Not a generated-source input.  Every way a target reaches the RTL is
    already a declared parameter, so two parts admitting the same parameters
    admit the same text, and keying the text by part would stop them sharing it.
    """

    fpga_part: str
    clock_period_ns: float

    def __post_init__(self) -> None:
        if not self.fpga_part:
            raise ValueError("a target identity needs an FPGA part")


@dataclass(frozen=True)
class SourceIdentity:
    """One compiled file: where it was declared, and what was in it.

    The root is the *named* root -- ``finn``, ``finnlib`` -- and the path is
    relative beneath it, so the identity does not move with the checkout.  The
    content hash is not decoration: with ``FINNLIB_ROOT`` pointed at a working
    clone, editing ``dotp.sv`` in place is routine, and a manifest of paths
    would keep the identity equal across a change to the hardware itself.
    """

    root: str
    path: str
    digest: str


@dataclass(frozen=True)
class KernelArtifactIdentity:
    """What one bound Kernel generates, keyed by its source inputs alone.

    Canonical by construction.  The class is public and directly
    constructible, so ordering cannot be left to whoever calls it: a table
    written in another order is the same table, and two keys for it would be
    the wrong-miss to match every wrong-hit elsewhere in this module.
    """

    kernel_id: str
    kernel_version: str
    #: In compile order, never sorted: ``dotp_axi`` instantiates ``dotp``, which
    #: instantiates the DSP core.  A reordered manifest names the same files and
    #: is not the same build.
    sources: tuple[SourceIdentity, ...]
    #: Sorted on construction -- a parameter table is a mapping, so its order is
    #: not a fact about the build the way compile order is.
    parameters: tuple[tuple[str, Scalar], ...]
    #: The Kernel's own committed choices, likewise sorted, under names local to
    #: the Kernel rather than to its placement.
    assignments: tuple[tuple[str, Scalar], ...] = ()
    schema_version: str = KERNEL_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.kernel_id or not self.kernel_version:
            raise ValueError("an artifact identity needs a Kernel id and version")
        # Encode *and store*, not merely validate: an ``Enum`` left in the tuple
        # would pass a check and then fail at serialization, which is the same
        # defect one call later and somewhere less obvious.
        for attribute, label in (("parameters", "parameter"), ("assignments", "choice")):
            table: tuple[tuple[str, object], ...] = getattr(self, attribute)
            names = tuple(name for name, _ in table)
            if len(names) != len(set(names)):
                raise ArtifactIdentityError(f"a {label} is named twice")
            object.__setattr__(
                self, attribute, tuple(sorted((name, _encode(name, v)) for name, v in table))
            )

    @property
    def serialization(self) -> str:
        """The canonical text the key is taken over.

        Explicitly labelled and explicitly ordered.  Nothing here iterates a
        set or a dict, which is what makes the key equal across processes.
        """

        return _canonical(
            [
                ["schema", self.schema_version],
                ["kernel", [self.kernel_id, self.kernel_version]],
                ["sources", [[item.root, item.path, item.digest] for item in self.sources]],
                ["parameters", [[name, value] for name, value in self.parameters]],
                ["choices", [[name, value] for name, value in self.assignments]],
            ]
        )

    @property
    def key(self) -> str:
        """A lookup key: the hash of the serialization."""

        return content_hash(self.serialization.encode())


@dataclass(frozen=True)
class ComposedArtifactIdentity:
    """Several bound Kernels plus a generated top, as one generated source.

    The wrapper hash is not redundant with the Kernel identities.  The wrapper
    is *generated*, so a change to its generator changes the built hardware
    while every Kernel identity stays equal -- exactly the wrong-cache-hit a
    naive composition of sub-identities misses.

    (The Phase 5 plan named this ``DecomposedArtifactIdentity``.  Composition
    is not MVAU vocabulary and this layer knows nothing about the decomposed
    MVAU, so the generic name is used here.)
    """

    #: In compile order, for the same reason the source manifest is.
    kernels: tuple[KernelArtifactIdentity, ...]
    wrapper_digest: str
    schema_version: str = COMPOSED_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ArtifactIdentityError("a composed artifact needs at least one Kernel")

    @property
    def serialization(self) -> str:
        return _canonical(
            [
                ["schema", self.schema_version],
                ["kernels", [item.serialization for item in self.kernels]],
                ["wrapper", self.wrapper_digest],
            ]
        )

    @property
    def key(self) -> str:
        return content_hash(self.serialization.encode())


@dataclass(frozen=True)
class PackagedArtifactIdentity:
    """Stage two: the generated source, laid out and made instantiable.

    Its inputs are the *shape* of the packaging and never its materialization.
    ``layout`` is the staged file names relative to the unit's own directory,
    in compile order; ``command_schema`` is the instantiation command with the
    module and instance held out.  Neither depends on where the unit was
    written, so packaging the same requirements under two repository roots is
    one artifact -- which it plainly is.

    Computable before anything is staged, which is what lets it name the
    directory and be looked up.  An identity that could only be formed after
    the build would be a receipt, not a key.
    """

    upstream: str
    layout: tuple[str, ...]
    command_schema: str
    schema_version: str = PACKAGED_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.upstream or not self.command_schema:
            raise ArtifactIdentityError("a packaged artifact needs an upstream key and a command")
        for name in self.layout:
            if "/" in name or "\\" in name or name in ("", ".", ".."):
                raise ArtifactIdentityError(
                    f"{name!r} is not a name relative to the unit directory; a packaged "
                    "identity must not carry a materialized path"
                )

    @property
    def serialization(self) -> str:
        return _canonical(
            [
                ["schema", self.schema_version],
                ["upstream", self.upstream],
                ["layout", list(self.layout)],
                ["command", self.command_schema],
            ]
        )

    @property
    def key(self) -> str:
        return content_hash(self.serialization.encode())


@dataclass(frozen=True)
class SynthesisArtifactIdentity:
    """Stage three: the packaged unit, built for a device by a tool.

    This is where the target and the builder enter, because this is the stage
    that consumes them.  One packaged unit synthesized for two parts is two
    results over one source -- which is the reuse that keeping the part out of
    the lower stages buys.

    ``constraints`` and ``recipe`` are the other two inputs, and leaving them
    out was the same mistake the wrapper digest exists to prevent one stage up.
    The clock period reproduces the constraint text under *today's* generator;
    a change to that generator, or to the synthesis command's options, changes
    what is built while every other field stays equal.  ``recipe`` is the
    command *shape* and never a rendered command, for the same reason the
    packaged layout is names and not paths.
    """

    upstream: str
    target: TargetIdentity
    builder: BuilderIdentity = DEFAULT_BUILDER
    #: Content hash of the constraints the run is synthesized against.
    constraints_digest: str = ""
    #: The synthesis command shape, with paths and names held out.
    recipe: str = ""
    schema_version: str = SYNTHESIS_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.upstream:
            raise ArtifactIdentityError("a synthesis artifact needs an upstream key")
        if not self.constraints_digest or not self.recipe:
            raise ArtifactIdentityError(
                "a synthesis artifact needs its constraints and its command shape; "
                "the target alone reproduces them only under one generator"
            )
        _check_command_shape(self.recipe)

    @property
    def serialization(self) -> str:
        return _canonical(
            [
                ["schema", self.schema_version],
                ["upstream", self.upstream],
                ["target", [self.target.fpga_part, self.target.clock_period_ns]],
                ["builder", [self.builder.backend_id, self.builder.tool_version]],
                ["constraints", self.constraints_digest],
                ["recipe", self.recipe],
            ]
        )

    @property
    def key(self) -> str:
        return content_hash(self.serialization.encode())


@dataclass(frozen=True)
class VlnvIdentity:
    """Vendor, library, name and version: what an IP repository indexes by.

    A caller-supplied label, exactly like :class:`BuilderIdentity`.  It is a
    *naming* decision -- which repository coordinate this unit occupies -- and
    not something derivable from the RTL, so it is declared rather than probed.

    ``name`` is deliberately absent: it is the packaged unit's top module, which
    the stage already has.  A separately supplied name would be a second
    authority for one fact, and the one that reached the ``.tcl`` would win.
    """

    vendor: str
    library: str
    version: str

    def __post_init__(self) -> None:
        for label, value in (
            ("vendor", self.vendor),
            ("library", self.library),
            ("version", self.version),
        ):
            if not value or any(character in value for character in ":/\\ "):
                raise ArtifactIdentityError(
                    f"an IP {label} must be a non-empty word without ':', '/' or spaces; "
                    f"got {value!r}"
                )


#: What FINN's own stitcher would look this unit up as, if nothing said
#: otherwise.  ``ipx::package_project`` defaults to these too.
DEFAULT_VLNV = VlnvIdentity("amd", "finn", "1.0")


@dataclass(frozen=True)
class IpPackageArtifactIdentity:
    """Stage three, sibling: the packaged unit as an IP-XACT component.

    The same four terms as every other stage, which is the point of it looking
    like the synthesis identity.  Its *key* is what a repository would resolve;
    its *inputs* are the packaged unit, the coordinate it is filed under, the
    part the packaging project is opened on, the tool, and the command shape;
    its *place* is a directory named from that key; its *states* are prepared
    and packaged.

    The part is here and not merely on the caller because ``create_project``
    takes one and an IP-XACT component records the families it was inferred
    for.  Two parts therefore produce two component descriptions over one set
    of sources -- which is exactly the relationship synthesis has to the same
    unit, and the reason both live at this level rather than under it.

    No clock period, though: nothing in packaging reads one.  Reusing
    :class:`TargetIdentity` here would have put a value in the key that cannot
    change what is produced, and a key that moves without the artifact moving
    is a wrong *miss*.
    """

    upstream: str
    vlnv: VlnvIdentity = DEFAULT_VLNV
    fpga_part: str = ""
    builder: BuilderIdentity = DEFAULT_BUILDER
    #: The packaging command shape, with paths and names held out.
    recipe: str = ""
    schema_version: str = IP_PACKAGE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.upstream:
            raise ArtifactIdentityError("an IP package artifact needs an upstream key")
        if not self.fpga_part:
            raise ArtifactIdentityError(
                "an IP package artifact needs the part its project is opened on; "
                "the component records what it was inferred for"
            )
        if not self.recipe:
            raise ArtifactIdentityError(
                "an IP package artifact needs its command shape; the VLNV alone "
                "reproduces it only under one generator"
            )
        _check_command_shape(self.recipe)

    @property
    def serialization(self) -> str:
        return _canonical(
            [
                ["schema", self.schema_version],
                ["upstream", self.upstream],
                ["vlnv", [self.vlnv.vendor, self.vlnv.library, self.vlnv.version]],
                ["part", self.fpga_part],
                ["builder", [self.builder.backend_id, self.builder.tool_version]],
                ["recipe", self.recipe],
            ]
        )

    @property
    def key(self) -> str:
        return content_hash(self.serialization.encode())


def source_identities(
    kernel: HardwareKernel, roots: Mapping[str, Path]
) -> tuple[SourceIdentity, ...]:
    """Hash every file the Kernel declares, in the order it declares them."""

    entries: list[SourceIdentity] = []
    for source in kernel.sources:
        root = roots.get(source.root)
        if root is None:
            raise ArtifactIdentityError(
                f"{kernel.kernel_id} declares sources under {source.root!r}, "
                "which this checkout does not resolve"
            )
        located = Path(root) / source.path
        try:
            data = located.read_bytes()
        except OSError as error:
            raise ArtifactIdentityError(
                f"{kernel.kernel_id} declares {located}, which cannot be read"
            ) from error
        entries.append(SourceIdentity(source.root, source.path, content_hash(data)))
    return tuple(entries)


def kernel_artifact_identity(
    kernel: HardwareKernel, roots: Mapping[str, Path]
) -> KernelArtifactIdentity:
    """The generated-source identity of one bound Kernel.

    Everything it reads is either declared by the Kernel or committed to it.
    It never touches the binding's Regions, node ids, or edges -- those are the
    instance facts the key exists to exclude.
    """

    return KernelArtifactIdentity(
        kernel.kernel_id,
        kernel.kernel_version,
        source_identities(kernel, roots),
        scalar_parameters(dict(kernel.parameters)),
        _local_assignments(kernel),
    )


def composed_artifact_identity(
    kernels: tuple[KernelArtifactIdentity, ...], *generated: str
) -> ComposedArtifactIdentity:
    """The generated-source identity of several Kernels under one top.

    Every generated text, not only the top.  Each is produced by a generator
    that can change while the Kernel identities stay equal, so each one left
    out is a way for the built hardware to move without the key moving.
    """

    if not generated:
        raise ArtifactIdentityError("a composed artifact needs its generated text")
    return ComposedArtifactIdentity(kernels, content_hash("\0".join(generated).encode()))


__all__ = [
    "COMPOSED_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_BUILDER",
    "DEFAULT_VLNV",
    "IP_PACKAGE_ARTIFACT_SCHEMA_VERSION",
    "KERNEL_ARTIFACT_SCHEMA_VERSION",
    "PACKAGED_ARTIFACT_SCHEMA_VERSION",
    "SYNTHESIS_ARTIFACT_SCHEMA_VERSION",
    "ArtifactIdentityError",
    "BuilderIdentity",
    "ComposedArtifactIdentity",
    "IpPackageArtifactIdentity",
    "KernelArtifactIdentity",
    "PackagedArtifactIdentity",
    "Scalar",
    "SourceIdentity",
    "SynthesisArtifactIdentity",
    "TargetIdentity",
    "VlnvIdentity",
    "composed_artifact_identity",
    "content_hash",
    "kernel_artifact_identity",
    "source_identities",
]
