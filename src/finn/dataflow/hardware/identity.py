# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Artifact identity: what makes two builds of the same hardware one build.

An identity is computed from the *physical build inputs alone* -- the Kernel
family, the source files it compiles and their content, the parameters driven
into them, the target, and the builder.  Two equal configured Kernels at
different source nodes therefore key the same, which is the entire point:
today ``top_module_name`` bakes the source node in, so identical hardware at
two graph positions is built twice.

What is excluded is as load-bearing as what is included.  ONNX node names, the
dataflow scope id, source tensor names, graph position, and physical instance
names never reach the key.  :class:`~finn.dataflow.hardware.KernelOrigin`
carries exactly those -- it is provenance, and its docstring already forbids it
becoming a key.  This is the other value it points at.

The identity is a typed frozen value rather than a bare string.  A string is a
fine *key*, and :attr:`KernelArtifactIdentity.key` is one, but "these two
hashes differ" is not something a failing test can explain; the value is.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from finn.dataflow.hardware.kernel import KernelBinding, scalar_parameters

#: Bumped when what the identity *contains* changes.  Without it, adding a
#: field makes every prior identity look like a different design rather than
#: like one this version cannot read.
KERNEL_ARTIFACT_SCHEMA_VERSION = "kernel-artifact-identity-v1"

#: The composed reading has its own schema, because it is its own value.
COMPOSED_ARTIFACT_SCHEMA_VERSION = "composed-artifact-identity-v1"

Scalar = bool | int | float | str


class ArtifactIdentityError(Exception):
    """A build input could not be read, so no identity can be computed."""


def content_hash(data: bytes) -> str:
    """The one hash function used everywhere in this module."""

    return sha256(data).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


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
    That is acceptable while nothing stores artifacts by this key, and it
    becomes a real obligation for whoever adds a store.
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
#: its path.  A caller that cares about the tool version passes it, and
#: ``finn.util.basic.get_vivado_version`` is where it reads one from.
#:
#: ``unspecified`` is therefore a claim in its own right -- "nobody said" -- and
#: two builds under different real Vivado versions share it.  That is the §7.4
#: cost made visible rather than hidden behind a plausible-looking number.
DEFAULT_BUILDER = BuilderIdentity("vivado", "unspecified")


@dataclass(frozen=True)
class TargetIdentity:
    """The device and timing the artifact was built for."""

    fpga_part: str
    clock_period_ns: float

    def __post_init__(self) -> None:
        if not self.fpga_part:
            raise ValueError("a target identity needs an FPGA part")


@dataclass(frozen=True)
class SourceIdentity:
    """One compiled file: where it was declared, and what was in it.

    The content hash is not decoration.  A path is a location, and with
    ``FINNLIB_ROOT`` pointed at a working clone, editing ``dotp.sv`` in place is
    routine -- so a manifest of paths would keep the identity equal across a
    change to the hardware itself.
    """

    root: str
    path: str
    digest: str


@dataclass(frozen=True)
class KernelArtifactIdentity:
    """What one bound Kernel builds, keyed by its physical inputs alone."""

    kernel_id: str
    kernel_version: str
    #: In compile order, never sorted: ``dotp_axi`` instantiates ``dotp``, which
    #: instantiates the DSP core.  A reordered manifest names the same files and
    #: is not the same build.
    sources: tuple[SourceIdentity, ...]
    #: Sorted by name -- a parameter table is a mapping, so its order is not a
    #: fact about the build the way compile order is.
    parameters: tuple[tuple[str, Scalar], ...]
    target: TargetIdentity
    builder: BuilderIdentity
    schema_version: str = KERNEL_ARTIFACT_SCHEMA_VERSION

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
                ["target", [self.target.fpga_part, self.target.clock_period_ns]],
                ["builder", [self.builder.backend_id, self.builder.tool_version]],
            ]
        )

    @property
    def key(self) -> str:
        """A directory name or lookup key: the hash of the serialization."""

        return content_hash(self.serialization.encode())


@dataclass(frozen=True)
class ComposedArtifactIdentity:
    """Several bound Kernels plus a generated top, as one build.

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


def source_identities(
    binding: KernelBinding, roots: Mapping[str, Path]
) -> tuple[SourceIdentity, ...]:
    """Hash every file the Kernel declares, in the order it declares them."""

    entries: list[SourceIdentity] = []
    for source in binding.kernel.sources:
        root = roots.get(source.root)
        if root is None:
            raise ArtifactIdentityError(
                f"{binding.kernel_id} declares sources under {source.root!r}, "
                "which this checkout does not resolve"
            )
        located = Path(root) / source.path
        try:
            data = located.read_bytes()
        except OSError as error:
            raise ArtifactIdentityError(
                f"{binding.kernel_id} declares {located}, which cannot be read"
            ) from error
        entries.append(SourceIdentity(source.root, source.path, content_hash(data)))
    return tuple(entries)


def kernel_artifact_identity(
    binding: KernelBinding,
    roots: Mapping[str, Path],
    *,
    target: TargetIdentity,
    builder: BuilderIdentity = DEFAULT_BUILDER,
) -> KernelArtifactIdentity:
    """The artifact identity of one bound Kernel.

    Everything it reads is either declared by the Kernel or supplied here.  It
    never touches the binding's Regions, node ids, or edges -- those are the
    instance facts the key exists to exclude.
    """

    return KernelArtifactIdentity(
        binding.kernel_id,
        binding.kernel_version,
        source_identities(binding, roots),
        scalar_parameters(dict(binding.parameters)),
        target,
        builder,
    )


def composed_artifact_identity(
    kernels: tuple[KernelArtifactIdentity, ...], wrapper_source: str
) -> ComposedArtifactIdentity:
    """The artifact identity of several Kernels under one generated top."""

    if not kernels:
        raise ArtifactIdentityError("a composed artifact needs at least one Kernel")
    return ComposedArtifactIdentity(kernels, content_hash(wrapper_source.encode()))


__all__ = [
    "COMPOSED_ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_BUILDER",
    "KERNEL_ARTIFACT_SCHEMA_VERSION",
    "ArtifactIdentityError",
    "BuilderIdentity",
    "ComposedArtifactIdentity",
    "KernelArtifactIdentity",
    "Scalar",
    "SourceIdentity",
    "TargetIdentity",
    "composed_artifact_identity",
    "content_hash",
    "kernel_artifact_identity",
    "source_identities",
]
