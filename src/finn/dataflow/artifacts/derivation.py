# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``Derivation``: one identity mechanism for every producer.

Rendering a Kernel's sources, composing a wrapper, exporting HLS, synthesizing
out of context, packaging in any format, compiling a simulation object -- each
is a ``Derivation``, and each gets its key the same way.  Two digests, and they
answer different questions:

``build_key``
    hashed from the declared inputs, known **before** anything runs, and the
    only one a lookup uses.

``tree_digest``
    hashed from the completed output tree, known **after**, and never consulted
    for lookup.  Vendor tools embed timestamps and paths in ``component.xml``
    and ``.dcp``, so demanding byte-reproducible output would be a fight with
    the tools for no benefit.  Hashing it anyway is what makes store corruption
    and residual nondeterminism visible instead of invisible.

**Producer identity is declared, not computed from the code.**  Hashing a
Python callable is not well defined -- hashing its source file misses imported
helpers, package versions, interpreter behaviour, dynamically loaded templates
and native dependencies, and hashing the environment destroys the portability
this package exists for.  What is honest is a stable id, a versioned public
contract, digests of the templates the producer reads, and an explicit request
schema.  All four are here; none of them pretends to be the fifth.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from string import Formatter
from typing import Union

from finn.dataflow.artifacts.projection import ProjectionError, digest

#: A value an option may take.  ``Enum`` is included by the projection, which
#: qualifies it; anything else is refused rather than stringified, because
#: ``str()`` on an arbitrary object is ``repr()`` by another route and
#: ``repr()`` has no stability contract.
Scalar = Union[bool, int, float, str, Enum]

_DIGEST = re.compile(r"^[0-9a-f]{64}$")

#: A word that is an absolute path.  Crude on purpose: a request schema has no
#: business containing one at all, so anything that looks like one is refused
#: rather than parsed.
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s{(\[=])[/~]\S")


class DerivationError(Exception):
    """A derivation could not be formed, so it has no key."""


@dataclass(frozen=True)
class ContentRef:
    """A blob, by content alone.

    The store deduplicates on this, across every artifact, stage and
    derivation.  Compilation-unit identity is separate and coarser: identical
    bytes under two logical libraries are two compilation units and one blob.
    """

    digest: str

    def __post_init__(self) -> None:
        if not _DIGEST.match(self.digest):
            raise DerivationError(f"{self.digest!r} is not a sha256 content digest")


@dataclass(frozen=True)
class ArtifactRef:
    """An upstream artifact, **by key**.

    By key and never by embedding the upstream value.  Embedding makes a
    downstream preimage grow with everything above it, and it hides the
    question of what actually changed behind a wall of nested text -- which is
    the diagnostic this design is typed to avoid.
    """

    kind: str
    key: str

    def __post_init__(self) -> None:
        if not self.kind:
            raise DerivationError("an artifact reference needs a stage kind")
        if not _DIGEST.match(self.key):
            raise DerivationError(f"{self.key!r} is not a build key")


@dataclass(frozen=True)
class ProducerIdentity:
    """Who produced it, and which public contract they were speaking.

    The contract is versioned because *other code reads it*: the output layout,
    the entry-point rule, the compile-metadata shape, and -- decisively -- the
    completion criteria.  An artifact stored as complete under a contract that
    required three files is not complete under one that requires four, and no
    hash of the files determines that.  Without the version a store answers
    "hit" for an artifact validated by rules that no longer apply; with it, the
    store can say *unsupported contract* instead.

    The implementation is deliberately **not** separately versioned.  Its
    effect is the bytes it produces, and those are already in the key.
    """

    producer_id: str
    contract_version: str

    def __post_init__(self) -> None:
        if not self.producer_id or not self.contract_version:
            raise DerivationError("a producer needs an id and a contract version")


@dataclass(frozen=True)
class ToolRequirement:
    """What the stage needs installed, as a constraint rather than a probe.

    Naming a tool is a declared input.  Invoking one happens outside
    ``finn.dataflow`` entirely, which is why this type has no ``run``.
    """

    tool: str
    constraint: str = ""

    def __post_init__(self) -> None:
        if not self.tool:
            raise DerivationError("a tool requirement needs a tool")


@dataclass(frozen=True)
class RequestSchema:
    """The shape of the command a tool stage is asked to run.

    A declared input, not a guard.  It names substitutions and hard-codes no
    absolute path; paths are injected at run time from the sealed mount map.
    That invariant is a property of this type rather than a heuristic
    inspecting a rendered string, which is the difference between a rule and a
    hope.
    """

    command: str

    def __post_init__(self) -> None:
        try:
            parsed = tuple(Formatter().parse(self.command))
        except ValueError as error:
            raise DerivationError(f"{self.command!r} is not a well-formed command shape") from error
        if not any(name is not None for _, name, _, _ in parsed):
            raise DerivationError(
                f"{self.command!r} names no substitution, so it is a rendered command "
                "rather than a command shape; a rendered command carries materialized paths"
            )
        for literal, _, _, _ in parsed:
            found = _ABSOLUTE_PATH.search(literal)
            if found:
                raise DerivationError(
                    "a request schema must not hard-code an absolute path, and "
                    f"{self.command!r} contains {found.group().strip()!r}"
                )

    @property
    def substitutions(self) -> tuple[str, ...]:
        """The names the schema expects, in declared order."""

        return tuple(name for _, name, _, _ in Formatter().parse(self.command) if name is not None)


@dataclass(frozen=True)
class OutputLayout:
    """What the stage promises to produce, as names relative to its own tree.

    Declared rather than discovered.  The legacy HLS path globs its outputs,
    which makes a partial run indistinguishable from a complete one -- there is
    no file whose absence says "this failed".
    """

    entries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise DerivationError("a derivation must declare what it produces")
        for name in self.entries:
            if name.startswith("/") or ".." in name.split("/") or name in ("", "."):
                raise DerivationError(
                    f"{name!r} is not a path relative to the output tree; a declared "
                    "layout must not carry a materialized path"
                )
        if len(set(self.entries)) != len(self.entries):
            raise DerivationError("a declared output layout names a file twice")


@dataclass(frozen=True)
class Derivation:
    """One producer's declared inputs, and nothing it does not consume.

    ``options`` is sorted on construction, since a table written in another
    order is the same table and two keys for it would be a wrong *miss*.
    ``inputs`` is deliberately not, for the reason given on the field.
    """

    #: Stage kind and the schema version of this derivation's own shape.
    kind: str
    schema_version: str
    producer: ProducerIdentity
    #: Content of every template and asset the producer reads.  Two source
    #: revisions of one configuration must not claim one key.
    templates: tuple[ContentRef, ...] = ()
    request: RequestSchema | None = None
    tool: ToolRequirement | None = None
    #: Named inputs **in declared order**, not a mapping.
    #:
    #: The design's §6 sketch writes this as ``{name -> ref}``, and a mapping
    #: cannot express the one ordering that §5.1 says is identity-bearing:
    #: compile order.  ``dotp_axi`` instantiates ``dotp``, which instantiates
    #: the DSP core, and a manifest sorted by name names the same files and is
    #: not the same build.  A producer for which order is *not* a fact sorts
    #: before constructing, which is a declaration it can make and a mapping
    #: cannot.
    inputs: tuple[tuple[str, ContentRef | ArtifactRef], ...] = ()
    options: tuple[tuple[str, Scalar], ...] = ()
    outputs: OutputLayout | None = None

    def __post_init__(self) -> None:
        if not self.kind or not self.schema_version:
            raise DerivationError("a derivation needs a stage kind and a schema version")
        for table, label in ((self.options, "option"), (self.inputs, "input")):
            names = tuple(name for name, _ in table)
            if len(names) != len(set(names)):
                raise DerivationError(f"an {label} is named twice")
        object.__setattr__(self, "options", tuple(sorted(self.options, key=lambda item: item[0])))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        # Project once, here, so a value that cannot be hashed is refused where
        # it was supplied rather than at the lookup that needed the key.
        try:
            digest(self)
        except ProjectionError as error:
            raise DerivationError(
                f"{self.kind} has an input that cannot enter a key: {error}"
            ) from error


def build_key(derivation: Derivation) -> str:
    """The lookup key: known before execution, over declared inputs alone."""

    return digest(derivation)


def tree_digest(entries: Mapping[str, str]) -> str:
    """Integrity over a completed output tree: relative name to content digest.

    Never used for lookup.  It is how a corrupt store and a nondeterministic
    tool become detectable rather than silent.
    """

    if not entries:
        raise DerivationError("a completed tree has at least one file")
    return digest(tuple(sorted(entries.items())))


__all__ = [
    "ArtifactRef",
    "ContentRef",
    "Derivation",
    "DerivationError",
    "OutputLayout",
    "ProducerIdentity",
    "RequestSchema",
    "Scalar",
    "ToolRequirement",
    "build_key",
    "tree_digest",
]
