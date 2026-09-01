# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A4: a real store, and every way one has been wrong before.

The negatives are the point.  A store that answers correctly on the happy path
and quietly wrongly on a corrupt one is worse than no store, because the wrong
answer is plausible RTL for the wrong design and nothing downstream notices.
So there is one test per entry on the lookup validation list, and each asserts
a **distinct** diagnostic -- a single "invalid artifact" would tell whoever hits
it nothing about which of eight things went wrong.

``a corrupt hit is an error, never a miss followed by a rebuild``: falling back
to building hides a broken store behind a slow build, and it stays broken.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    ContentRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    build_key,
)
from finn.dataflow.artifacts.lifecycle import LifecycleError, materialize
from finn.dataflow.artifacts.manifest import MANIFEST_SCHEMA_VERSION, decode, encode
from finn.dataflow.artifacts.projection import content_digest, digest
from finn.dataflow.artifacts.request import (
    ExecutionReceipt,
    FailureCategory,
    ToolchainIdentity,
)
from finn.dataflow.artifacts.store import MANIFEST_NAME, ArtifactStore, StoreError

UPSTREAM = "c" * 64
VIVADO = ToolchainIdentity("vivado", "2024.2", "xilinx-2024.2", "sha256:" + "f" * 64)
OLDER = ToolchainIdentity("vivado", "2023.2", "xilinx-2023.2", "sha256:" + "f" * 64)


def _derivation(**overrides: object) -> Derivation:
    defaults: dict[str, object] = {
        "kind": "kernel-source",
        "schema_version": "kernel-source-v1",
        "producer": ProducerIdentity("finn.kernel-source", "1"),
        "inputs": (("upstream", ArtifactRef("composed-source", UPSTREAM)),),
        "options": (("PE", 2), ("SIMD", 2)),
        "outputs": OutputLayout(("dotp_axi.sv", "replay_buffer.sv")),
    }
    defaults.update(overrides)
    return Derivation(**defaults)  # type: ignore[arg-type]


def _populate(workspace: Path, *, contents: str = "module a; endmodule\n") -> None:
    (workspace / "dotp_axi.sv").write_text(contents)
    (workspace / "replay_buffer.sv").write_text("module b; endmodule\n")


@pytest.fixture(name="store")
def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifact-store")


def _publish(store: ArtifactStore, derivation: Derivation | None = None) -> object:
    built = derivation or _derivation()
    workspace = store.workspace(built)
    _populate(workspace)
    return store.publish(built, workspace)


# -- the happy path, so the negatives mean something ---------------------------


def test_a_published_artifact_is_found_again_by_its_derivation(store: ArtifactStore) -> None:
    published = _publish(store)
    found = store.lookup(_derivation())
    assert found is not None
    assert found.key == published.key  # type: ignore[attr-defined]
    assert found.files == ("dotp_axi.sv", "replay_buffer.sv")


def test_an_unbuilt_derivation_is_a_miss_and_not_an_error(store: ArtifactStore) -> None:
    """A miss and a corrupt hit are different answers."""

    assert store.lookup(_derivation(options=(("PE", 8),))) is None


def test_the_returned_file_list_is_in_declared_order(store: ArtifactStore) -> None:
    """Compile order is a fact, and a store that sorted it would destroy one."""

    _publish(store)
    found = store.lookup(_derivation())
    assert found is not None
    assert found.files == ("dotp_axi.sv", "replay_buffer.sv")


# -- blobs deduplicate by content ----------------------------------------------


def test_identical_bytes_are_stored_once_however_they_were_reached(
    store: ArtifactStore,
) -> None:
    first = store.put_blob(b"shared")
    second = store.put_blob(b"shared")
    assert first == second
    assert store.blob_path(first).is_file()


def test_a_blob_whose_bytes_changed_underneath_the_store_is_refused(
    store: ArtifactStore,
) -> None:
    reference = store.put_blob(b"shared")
    store.blob_path(reference).write_bytes(b"tampered")
    with pytest.raises(StoreError, match="the store is corrupt"):
        store.get_blob(reference)


def test_a_blob_that_was_never_written_is_refused_rather_than_returned_empty(
    store: ArtifactStore,
) -> None:
    with pytest.raises(StoreError, match="not in this store"):
        store.get_blob(ContentRef("d" * 64))


# -- one test per entry on the validation list, each with its own diagnostic ---


def test_a_manifest_filed_under_the_wrong_key_is_refused(store: ArtifactStore) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    manifest = decode((directory / MANIFEST_NAME).read_bytes())
    tampered = decode(encode(manifest).replace(manifest.build_key.encode(), b"e" * 64))
    (directory / MANIFEST_NAME).write_bytes(encode(tampered))
    with pytest.raises(StoreError, match="the store is inconsistent"):
        store.lookup(derivation)


def test_an_artifact_of_another_kind_is_refused(store: ArtifactStore) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    manifest = decode((directory / MANIFEST_NAME).read_bytes())
    (directory / MANIFEST_NAME).write_bytes(
        encode(decode(encode(manifest).replace(b'"kernel-source"', b'"ipxact-package"', 1)))
    )
    with pytest.raises(StoreError, match="was asked for"):
        store.lookup(derivation)


def test_an_artifact_validated_under_another_producer_contract_is_refused(
    store: ArtifactStore,
) -> None:
    """Completion means something else, and no hash of the files determines it."""

    _publish(store)
    newer = _derivation(producer=ProducerIdentity("finn.kernel-source", "2"))
    # Same key material apart from the contract, so it lands in the same place.
    directory = store.object_directory("kernel-source", build_key(newer))
    shutil.copytree(store.object_directory("kernel-source", build_key(_derivation())), directory)
    manifest = decode((directory / MANIFEST_NAME).read_bytes())
    (directory / MANIFEST_NAME).write_bytes(
        encode(
            decode(encode(manifest).replace(manifest.build_key.encode(), build_key(newer).encode()))
        )
    )
    with pytest.raises(StoreError, match="completion means something else"):
        store.lookup(newer)


def test_a_missing_file_is_refused_and_named(store: ArtifactStore) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    (directory / "dotp_axi.sv").unlink()
    with pytest.raises(StoreError, match="dotp_axi.sv.*recorded and missing"):
        store.lookup(derivation)


def test_corrupt_content_is_refused_with_both_digests(store: ArtifactStore) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    (directory / "dotp_axi.sv").write_text("module tampered; endmodule\n")
    with pytest.raises(StoreError) as raised:
        store.lookup(derivation)
    assert "hashes to" in str(raised.value)
    assert "the manifest records" in str(raised.value)


def test_a_manifest_that_is_not_a_manifest_is_refused(store: ArtifactStore) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    (directory / MANIFEST_NAME).write_bytes(b"{}")
    with pytest.raises(StoreError, match="not a valid artifact manifest"):
        store.lookup(derivation)


def test_a_manifest_from_another_schema_is_refused_by_name(store: ArtifactStore) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    text = (directory / MANIFEST_NAME).read_bytes()
    (directory / MANIFEST_NAME).write_bytes(
        text.replace(MANIFEST_SCHEMA_VERSION.encode(), b"artifact-manifest-v99")
    )
    with pytest.raises(StoreError, match="cannot say what it means"):
        store.lookup(derivation)


def test_an_unknown_field_is_refused_rather_than_ignored(store: ArtifactStore) -> None:
    """msgspec does *not* do this by default; every struct sets it explicitly."""

    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    text = (directory / MANIFEST_NAME).read_bytes()
    (directory / MANIFEST_NAME).write_bytes(
        text.replace(b'{"schema_version"', b'{"surprise":1,"schema_version"', 1)
    )
    with pytest.raises(StoreError, match="not a valid artifact manifest"):
        store.lookup(derivation)


# -- publication ----------------------------------------------------------------


def test_the_workspace_lives_inside_the_store_root(store: ArtifactStore) -> None:
    """Not in /tmp.  ``os.replace`` is atomic only within one filesystem, and
    the container bind-mounts the repository, so a /tmp workspace makes the
    publish step silently non-atomic."""

    workspace = store.workspace(_derivation())
    assert workspace.is_relative_to(store.root)


def test_a_cross_filesystem_rename_is_not_atomic_which_is_why_the_rule_exists(
    finn_root: Path,
) -> None:
    """The failure mode, demonstrated rather than asserted from memory.

    ``/tmp`` against the repository checkout is the exact pairing the rule is
    about: in the container one is the overlay and the other is the bind
    mount.  Skipped where this machine happens to put both on one device,
    since then there is nothing to demonstrate -- and the skip is itself the
    honest answer about what was checked here.
    """

    scratch = Path("/tmp") / f"a4-cross-fs-{os.getpid()}"
    scratch.mkdir(exist_ok=True)
    inside_repo = finn_root / f".a4-cross-fs-{os.getpid()}"
    try:
        if os.stat(scratch).st_dev == os.stat(finn_root).st_dev:
            pytest.skip("/tmp and the checkout share a filesystem on this machine")
        with pytest.raises(OSError) as raised:
            os.replace(scratch, inside_repo)
        assert raised.value.errno == 18  # EXDEV
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(inside_repo, ignore_errors=True)


def test_nothing_appears_under_objects_until_it_has_been_validated(
    store: ArtifactStore,
) -> None:
    """An incomplete tree never becomes an artifact, not even briefly."""

    derivation = _derivation()
    workspace = store.workspace(derivation)
    (workspace / "dotp_axi.sv").write_text("module a; endmodule\n")  # only one of two
    with pytest.raises(LifecycleError, match="incomplete tree is an attempt"):
        store.publish(derivation, workspace)
    assert not store.object_directory(derivation.kind, build_key(derivation)).exists()


def test_a_producer_that_writes_what_it_did_not_declare_is_refused(
    store: ArtifactStore,
) -> None:
    derivation = _derivation()
    workspace = store.workspace(derivation)
    _populate(workspace)
    (workspace / "surprise.sv").write_text("module c; endmodule\n")
    with pytest.raises(LifecycleError, match="does not.*mention"):
        store.publish(derivation, workspace)


def test_a_stage_with_no_declared_layout_cannot_publish(store: ArtifactStore) -> None:
    derivation = _derivation(outputs=None)
    workspace = store.workspace(derivation)
    _populate(workspace)
    with pytest.raises(StoreError, match="declares no output layout"):
        store.publish(derivation, workspace)


def test_publishing_twice_leaves_the_first_artifact_intact(store: ArtifactStore) -> None:
    """Two builders racing on one key is a different problem; what matters here
    is that the second does not half-replace the first."""

    first = _publish(store)
    second = _publish(store)
    assert first.directory == second.directory  # type: ignore[attr-defined]
    assert store.lookup(_derivation()) is not None


def test_the_manifest_stores_the_derivation_text_so_a_miss_can_be_explained(
    store: ArtifactStore,
) -> None:
    """ "These two hashes differ" is the diagnostic this design is typed to avoid."""

    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    manifest = decode((directory / MANIFEST_NAME).read_bytes())
    assert "options[0][0]" in manifest.derivation
    assert "PE" in manifest.derivation


def test_the_manifest_records_upstream_by_key_and_not_by_value(
    store: ArtifactStore,
) -> None:
    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    manifest = decode((directory / MANIFEST_NAME).read_bytes())
    assert manifest.upstream[0].key == UPSTREAM
    assert manifest.upstream[0].kind == "composed-source"


def test_the_manifest_carries_no_timestamp_or_owner(store: ArtifactStore) -> None:
    """It is a function of the key.  Occurrence facts live in the association
    index and run facts in the attempt receipt, or two stores would disagree
    about one artifact."""

    derivation = _derivation()
    _publish(store, derivation)
    directory = store.object_directory(derivation.kind, build_key(derivation))
    text = (directory / MANIFEST_NAME).read_text()
    for forbidden in ("timestamp", "created", "owner", "hostname", "user"):
        assert forbidden not in text


# -- attempts -------------------------------------------------------------------


def test_a_failed_run_lands_in_attempts_and_never_in_objects(
    store: ArtifactStore,
) -> None:
    derivation = _derivation()
    workspace = store.workspace(derivation)
    _populate(workspace)
    receipt = ExecutionReceipt(
        build_key=build_key(derivation),
        stage_kind=derivation.kind,
        executor_id="test",
        declared_toolchain=VIVADO,
        actual_toolchain=VIVADO,
        exit_status=1,
        failure=FailureCategory.SYNTHESIS,
    )
    with pytest.raises(StoreError, match="a failed run is an attempt"):
        store.publish(derivation, workspace, receipt=receipt)
    landed = store.record_attempt(receipt, workspace)
    assert landed.is_dir()
    assert (landed / "dotp_axi.sv").is_file()
    assert not store.object_directory(derivation.kind, build_key(derivation)).exists()


def test_a_toolchain_mismatch_refuses_publication(store: ArtifactStore) -> None:
    derivation = _derivation()
    workspace = store.workspace(derivation)
    _populate(workspace)
    found = materialize(workspace, OutputLayout(("dotp_axi.sv", "replay_buffer.sv")))
    receipt = ExecutionReceipt(
        build_key=build_key(derivation),
        stage_kind=derivation.kind,
        executor_id="test",
        declared_toolchain=VIVADO,
        actual_toolchain=OLDER,
        exit_status=0,
        tree_digest=digest(tuple(sorted(found.entries))),
    )
    with pytest.raises(StoreError, match="publishing it would make the key a lie"):
        store.publish(derivation, workspace, receipt=receipt)


def test_a_receipt_whose_tree_digest_disagrees_with_the_workspace_is_refused(
    store: ArtifactStore,
) -> None:
    derivation = _derivation()
    workspace = store.workspace(derivation)
    _populate(workspace)
    receipt = ExecutionReceipt(
        build_key=build_key(derivation),
        stage_kind=derivation.kind,
        executor_id="test",
        declared_toolchain=VIVADO,
        actual_toolchain=VIVADO,
        exit_status=0,
        tree_digest=content_digest(b"something else"),
    )
    with pytest.raises(StoreError, match="the workspace hashes to"):
        store.publish(derivation, workspace, receipt=receipt)


def test_two_attempts_on_one_key_are_both_kept(store: ArtifactStore) -> None:
    """The material to diagnose a failure is kept, not discarded."""

    derivation = _derivation()
    receipt = ExecutionReceipt(
        build_key=build_key(derivation),
        stage_kind=derivation.kind,
        executor_id="test",
        declared_toolchain=VIVADO,
        actual_toolchain=VIVADO,
        exit_status=1,
        failure=FailureCategory.LICENCE,
    )
    first = store.workspace(derivation)
    _populate(first)
    landed = store.record_attempt(receipt, first)
    second = store.workspace(derivation)
    _populate(second)
    other = store.record_attempt(receipt, second)
    assert landed != other
    assert landed.is_dir() and other.is_dir()
