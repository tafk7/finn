# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# msgspec is declared in requirements.txt but is not installed in the
# interpreter the mypy gate runs under, so `Struct` resolves to Any and every
# subclass and keyword here reads as an error.  The same situation qonnx is in
# throughout this tree.  Waived for this module only; the runtime behaviour --
# strict decode, forbidden unknown fields -- is what the tests next door check.
# mypy: disable-error-code="call-arg, misc, no-any-return"

"""``artifact.json``: intrinsic derivation only, and a strict decode.

**A manifest is a function of its key, and of nothing else.**  If it carried
one occurrence's source owner or one run's timestamp, the same artifact would
acquire a different manifest depending on who produced it first -- so the
manifest would not be a function of the key, and two stores would disagree
about one artifact.  Provenance therefore has three homes and this is only the
first:

===================  ==================================================
``artifact.json``    intrinsic derivation: kind, keys, the canonical
                     derivation text, producer and contract version,
                     upstream refs, ordered file entries, entry points,
                     the ABI, the actual toolchain where there was one
association index     source owners, declaration paths, Kernel origins,
                     graph occurrences -- many-to-many
attempt receipt       timestamps, executor, resource usage, logs
===================  ==================================================

Storing the canonical derivation text is what lets a miss be *explained*.
"These two hashes differ" is the diagnostic this design is typed to avoid.

The decode is strict on purpose.  A manifest read from a store is untrusted
input -- it may have been written by another version, or corrupted -- and
``forbid_unknown_fields`` is **not** msgspec's default, so it is set on every
struct here.  Without it a newer manifest decodes as though it were understood,
which is the same wrong-hit class as a bad key one layer out.
"""

from __future__ import annotations

from collections.abc import Sequence

import msgspec  # type: ignore[import-not-found]

#: Bumped when the manifest's *shape* changes.  A store can then say
#: *unsupported schema* rather than reporting an unexplained miss.
MANIFEST_SCHEMA_VERSION = "artifact-manifest-v1"


class ManifestError(Exception):
    """A manifest is absent, unreadable, or not about what was asked."""


class FileEntry(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One file in the artifact tree, by relative path.

    Relative always.  An artifact does not become a different artifact because
    it was staged under another directory, and the legacy
    ``code_gen_dir_ipgen`` family -- where identity *is* a path -- is what the
    alternative turns into.
    """

    path: str
    digest: str
    size: int
    language: str = ""
    library: str = ""
    #: Rendered compile options, flattened, so a reader needs no second schema.
    options: tuple[tuple[str, str], ...] = ()


class UpstreamRef(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """An artifact this one was built from, by key."""

    name: str
    kind: str
    key: str


class ToolchainRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The toolchain that actually ran, where one did."""

    tool: str
    version: str
    install_id: str = ""
    image_digest: str = ""


class Manifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Everything intrinsic to one completed artifact."""

    schema_version: str
    kind: str
    build_key: str
    tree_digest: str
    #: The canonical derivation text, so a miss can be explained rather than
    #: merely reported.
    derivation: str
    producer_id: str
    contract_version: str
    #: In declared compile order, never sorted.
    files: tuple[FileEntry, ...]
    upstream: tuple[UpstreamRef, ...] = ()
    entry_points: tuple[str, ...] = ()
    #: Serialized ``ComponentABI``, where the stage publishes one.
    abi: str = ""
    toolchain: ToolchainRecord | None = None


_ENCODER = msgspec.json.Encoder()
_DECODER = msgspec.json.Decoder(Manifest)


def encode(manifest: Manifest) -> bytes:
    """The manifest as bytes, with a trailing newline so it reads as a file."""

    return _ENCODER.encode(manifest) + b"\n"


def decode(data: bytes) -> Manifest:
    """Parse a manifest, refusing anything that is not exactly one.

    Refuses rather than repairs.  A manifest that decoded "mostly" would be a
    store answering with something it does not understand.
    """

    try:
        manifest = _DECODER.decode(data)
    except msgspec.ValidationError as error:
        raise ManifestError(f"not a valid artifact manifest: {error}") from error
    except msgspec.DecodeError as error:
        raise ManifestError(f"artifact manifest is not readable JSON: {error}") from error
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema {manifest.schema_version!r} is not "
            f"{MANIFEST_SCHEMA_VERSION!r}; this reader cannot say what it means"
        )
    return manifest


def relative_path_issues(files: Sequence[FileEntry]) -> tuple[str, ...]:
    """Names that would escape the artifact directory, or repeat inside it."""

    issues: list[str] = []
    seen: set[str] = set()
    for entry in files:
        name = entry.path
        if name.startswith("/") or ".." in name.split("/") or name in ("", "."):
            issues.append(f"{name!r} is not a path relative to the artifact directory")
        if name in seen:
            issues.append(f"{name!r} appears twice in one manifest")
        seen.add(name)
    return tuple(issues)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "FileEntry",
    "Manifest",
    "ManifestError",
    "ToolchainRecord",
    "UpstreamRef",
    "decode",
    "encode",
    "relative_path_issues",
]
