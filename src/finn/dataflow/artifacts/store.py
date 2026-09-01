# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A content-addressed store: blobs, objects, attempts, and atomic publish.

::

    artifact-store/
    |-- objects/<kind>/<key-prefix>/<build-key>/
    |       `-- artifact.json          references blobs; relative paths only
    |-- blobs/sha256/<digest-prefix>/<content-digest>
    |-- attempts/<request-key>/<attempt-id>/
    `-- incoming/                      private workspaces, inside the root

Two properties do most of the work.

**Blobs deduplicate at file level across every artifact, stage and
derivation**, including bytes reached by different routes.  Compilation-unit
identity is separate and coarser (§5.3), and lives in ``sources``.

**A failed or abandoned run is an attempt, not an artifact.**  Failures never
reach ``objects/``, so the store cannot hold a partial tree, and the material
needed to diagnose them stays addressable instead of being discarded.

The publish step is hand-written and tested, because no library gets it right
generically:

``the workspace lives inside the store root, never in /tmp``
    ``os.replace`` is atomic only *within one filesystem*.  The FINN container
    bind-mounts the repository, so a temp directory on the container overlay
    and a final path on the bind mount are different filesystems, and the
    rename silently degrades to copy-then-unlink -- losing exactly the
    atomicity the step exists for.

``the parent directory is fsynced after the rename``
    required for the same reason on NFS, where ``O_EXCL`` create is the
    reliable primitive and client attribute caching can otherwise expose
    partial state.

A wrong or corrupt hit is an **error**, never a miss followed by a rebuild.  A
broken store hidden behind a slow build stays broken.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from finn.dataflow.artifacts.derivation import ArtifactRef, ContentRef, Derivation, build_key
from finn.dataflow.artifacts.lifecycle import (
    Materialization,
    check_layout,
    materialize,
    ordered_digests,
    verify_contents,
)
from finn.dataflow.artifacts.manifest import (
    MANIFEST_SCHEMA_VERSION,
    FileEntry,
    Manifest,
    ManifestError,
    ToolchainRecord,
    UpstreamRef,
    decode,
    encode,
    relative_path_issues,
)
from finn.dataflow.artifacts.projection import content_digest, digest, project
from finn.dataflow.artifacts.request import ExecutionReceipt, refuse_publication

MANIFEST_NAME = "artifact.json"

#: How many hex characters of a digest become a fan-out directory.  Two keeps
#: any single directory under 256 children, which is what filesystems and
#: humans both cope with.
_PREFIX = 2


class StoreError(Exception):
    """The store was asked for something it cannot honestly answer."""


@dataclass(frozen=True)
class StoredArtifact:
    """A previously built artifact, as this store reports it.

    Structurally what ``hardware.store.StoredArtifact`` is -- a key, a
    directory and an ordered file list -- so this store drops into the existing
    ``ArtifactStore`` seam without either side importing the other.  That is
    the seam being real rather than declared.
    """

    key: str
    directory: str
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.directory:
            raise StoreError("a stored artifact needs a key and a directory")


class ArtifactStore:
    """A directory of immutable artifacts, and the rules for entering it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._objects = self.root / "objects"
        self._blobs = self.root / "blobs" / "sha256"
        self._attempts = self.root / "attempts"
        self._incoming = self.root / "incoming"
        for directory in (self._objects, self._blobs, self._attempts, self._incoming):
            directory.mkdir(parents=True, exist_ok=True)

    # -- placement -------------------------------------------------------------

    def object_directory(self, kind: str, key: str) -> Path:
        return self._objects / kind / key[:_PREFIX] / key

    def blob_path(self, content: ContentRef) -> Path:
        return self._blobs / content.digest[:_PREFIX] / content.digest

    # -- blobs -----------------------------------------------------------------

    def put_blob(self, data: bytes) -> ContentRef:
        """Write bytes once.  Writing them again is a no-op, by construction."""

        reference = ContentRef(content_digest(data))
        target = self.blob_path(reference)
        if target.exists():
            return reference
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, data)
        return reference

    def get_blob(self, reference: ContentRef) -> bytes:
        target = self.blob_path(reference)
        try:
            data = target.read_bytes()
        except OSError as error:
            raise StoreError(f"blob {reference.digest[:12]} is not in this store") from error
        actual = content_digest(data)
        if actual != reference.digest:
            raise StoreError(
                f"blob {reference.digest[:12]} hashes to {actual[:12]}; the store is corrupt"
            )
        return data

    # -- publication -----------------------------------------------------------

    def workspace(self, derivation: Derivation) -> Path:
        """A private directory **inside the store root**, never in ``/tmp``.

        See the module docstring: a workspace on another filesystem makes the
        publish step silently non-atomic, and the container bind-mount makes
        that the normal case rather than an exotic one.
        """

        key = build_key(derivation)
        path = self._incoming / f"{derivation.kind}-{key[:16]}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        return path

    def publish(
        self,
        derivation: Derivation,
        workspace: Path,
        *,
        receipt: ExecutionReceipt | None = None,
        entry_points: Sequence[str] = (),
        abi: str = "",
    ) -> StoredArtifact:
        """Validate, hash, write the manifest, and rename into place.

        In that order, and the order is the point.  Nothing is visible under
        ``objects/`` until everything about it has already been checked, so a
        reader can never observe a tree mid-construction.
        """

        if derivation.outputs is None:
            raise StoreError(
                f"{derivation.kind} declares no output layout, so there is nothing "
                "to validate a completed tree against"
            )
        if receipt is not None:
            refusal = refuse_publication(receipt)
            if refusal is not None:
                raise StoreError(refusal)

        found = materialize(workspace, derivation.outputs)
        tree = digest(ordered_digests(found.entries))
        if receipt is not None and receipt.tree_digest != tree:
            raise StoreError(
                f"the receipt records tree digest {receipt.tree_digest[:12]} and the "
                f"workspace hashes to {tree[:12]}"
            )

        for name, content in found.entries:
            self.put_blob((workspace / name).read_bytes())
            del content

        manifest = Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            kind=derivation.kind,
            build_key=build_key(derivation),
            tree_digest=tree,
            derivation=_canonical_text(derivation),
            producer_id=derivation.producer.producer_id,
            contract_version=derivation.producer.contract_version,
            files=tuple(
                FileEntry(path=name, digest=content, size=(workspace / name).stat().st_size)
                for name, content in found.entries
            ),
            upstream=tuple(
                UpstreamRef(name=name, kind=reference.kind, key=reference.key)
                for name, reference in derivation.inputs
                if isinstance(reference, ArtifactRef)
            ),
            entry_points=tuple(entry_points),
            abi=abi,
            toolchain=(
                ToolchainRecord(
                    tool=receipt.actual_toolchain.tool,
                    version=receipt.actual_toolchain.version,
                    install_id=receipt.actual_toolchain.install_id,
                    image_digest=receipt.actual_toolchain.image_digest,
                )
                if receipt is not None
                else None
            ),
        )
        issues = relative_path_issues(manifest.files)
        if issues:
            raise StoreError("; ".join(issues))
        (workspace / MANIFEST_NAME).write_bytes(encode(manifest))

        target = self.object_directory(derivation.kind, manifest.build_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Already published.  Two builders racing on one key is a real
            # problem and a different one; what matters here is that the
            # second does not half-replace the first.
            shutil.rmtree(workspace)
            return self._stored(manifest, target)
        os.replace(workspace, target)
        _fsync_directory(target.parent)
        return self._stored(manifest, target)

    def record_attempt(self, receipt: ExecutionReceipt, workspace: Path) -> Path:
        """Keep a failed run where it can be diagnosed, and out of ``objects/``.

        The workspace is *moved*, not copied: whatever the run left behind is
        the evidence, and a copy invites the original to be cleaned up.
        """

        attempt = self._attempts / receipt.build_key[:_PREFIX] / receipt.build_key
        attempt.mkdir(parents=True, exist_ok=True)
        destination = attempt / f"attempt-{len(list(attempt.iterdir())):04d}"
        os.replace(workspace, destination)
        _fsync_directory(attempt)
        return destination

    # -- lookup ----------------------------------------------------------------

    def lookup(self, identity: Derivation) -> StoredArtifact | None:
        """Answer for this derivation, or refuse.  Never a quiet miss.

        The validation list is long because every entry on it is a way a store
        has been wrong before: the requested key against the returned one, the
        schema, the producer contract, relative-path safety, the declared file
        set and its order, sizes and digests, and the tree digest.

        A miss returns ``None``.  A *corrupt* or *wrong* hit raises -- the two
        are different answers, and collapsing them hides a broken store behind
        a slow build.
        """

        key = build_key(identity)
        directory = self.object_directory(identity.kind, key)
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            return None

        try:
            manifest = decode(manifest_path.read_bytes())
        except ManifestError as error:
            raise StoreError(f"{directory}: {error}") from error

        if manifest.build_key != key:
            raise StoreError(
                f"{directory} is filed under {key} and its manifest says "
                f"{manifest.build_key}; the store is inconsistent"
            )
        if manifest.kind != identity.kind:
            raise StoreError(
                f"{directory} holds a {manifest.kind!r} artifact and a {identity.kind!r} "
                "one was asked for"
            )
        if manifest.contract_version != identity.producer.contract_version:
            raise StoreError(
                f"{directory} was validated under producer contract "
                f"{manifest.contract_version!r}; this build speaks "
                f"{identity.producer.contract_version!r}, so completion means something else"
            )
        issues = list(relative_path_issues(manifest.files))

        materialization = Materialization(
            directory, tuple((entry.path, entry.digest) for entry in manifest.files)
        )
        if identity.outputs is not None:
            issues.extend(check_layout(materialization, identity.outputs))
        issues.extend(verify_contents(materialization))
        for entry in manifest.files:
            located = directory / entry.path
            if located.is_file() and located.stat().st_size != entry.size:
                issues.append(
                    f"{entry.path!r} is {located.stat().st_size} bytes and the manifest "
                    f"records {entry.size}"
                )
        recomputed = digest(ordered_digests(materialization.entries))
        if recomputed != manifest.tree_digest:
            issues.append(
                f"the tree hashes to {recomputed[:12]} and the manifest records "
                f"{manifest.tree_digest[:12]}"
            )
        if issues:
            raise StoreError(f"{directory} is a corrupt hit: " + "; ".join(issues))

        return self._stored(manifest, directory)

    def _stored(self, manifest: Manifest, directory: Path) -> StoredArtifact:
        return StoredArtifact(
            manifest.build_key, str(directory), tuple(entry.path for entry in manifest.files)
        )


def _canonical_text(derivation: Derivation) -> str:
    """The derivation as the text its key was taken over.

    Stored so a miss can be *explained*.  Reconstructed from the projection
    rather than from a second serializer, because a second serializer is a
    second authority for what the key covers.
    """

    return "\n".join(f"{path}\t{tag}\t{text}" for path, tag, text in project(derivation))


def _atomic_write(target: Path, data: bytes) -> None:
    """Write beside the target and rename, so a reader never sees half a file."""

    scratch = target.with_name(target.name + ".incoming")
    with open(scratch, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(scratch, target)
    _fsync_directory(target.parent)


def _fsync_directory(directory: Path) -> None:
    """Durably record a rename.

    Skipped silently where the platform has no directory descriptor to sync;
    on Linux, which is where this runs, it is required.
    """

    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def blob_layout(root: Path) -> Mapping[str, Path]:
    """Where the store keeps its three kinds of thing.  For diagnostics."""

    return {
        "objects": root / "objects",
        "blobs": root / "blobs" / "sha256",
        "attempts": root / "attempts",
        "incoming": root / "incoming",
    }


__all__ = [
    "MANIFEST_NAME",
    "ArtifactStore",
    "StoreError",
    "StoredArtifact",
    "blob_layout",
]
