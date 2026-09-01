# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A4's validation move: a real store, in the seam that already has call sites.

``hardware/store.py`` has defined ``ArtifactKey``, ``ArtifactStore``,
``StoredArtifact`` and ``checked_lookup`` since Phase 5, and
``package_decomposed_artifact(..., store=)`` has accepted one all along.  It
has answered *no* to everything the whole time.  This is the first time
something answers *yes*, and it is done against **real MVAU artifacts** with no
production file edited.

That matters more than it sounds.  Phase 5's packaging increment was "the
increment with no baseline oracle"; a generic store exercised only by its own
fixtures would be the same thing again.  Here the artifact is one the rtlsim
fixtures already build, and the seam is one the build path already calls.

Neither side imports the other.  ``artifacts`` may not import ``hardware``, and
``hardware`` does not know ``artifacts`` exists -- they meet through a
``Protocol``, which is what makes this a seam rather than a coupling.  The
adapter that joins them lives here, and it is small on purpose: if joining the
two layers needed more, the seam would not be one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from finn.dataflow.artifacts.derivation import (
    ArtifactRef,
    Derivation,
    OutputLayout,
    ProducerIdentity,
    RequestSchema,
)
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.hardware.identity import PackagedArtifactIdentity

# The seam's own ``StoredArtifact``, not the one ``artifacts`` defines.  Two
# types with one shape, and keeping them distinct is the boundary working: the
# adapter is on the hardware side, and converting is its whole job.
from finn.dataflow.hardware.store import ArtifactKey, StoredArtifact, checked_lookup
from finn.dataflow.mvau.hardware.composition import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    package_decomposed_artifact,
    packaged_artifact_identity,
)
from finn.dataflow.mvau.providers import elaborate_mvau

FINN_ROOT = Path(__file__).resolve().parents[3]


def _requirements() -> MVAUDecomposedArtifactRequirements:
    """The same fixture ``test_decomposed_packaging`` builds, built here.

    Spelled out rather than imported.  The private helper next door belongs to
    the ``DataflowDesign`` migration, and reaching into it makes this the one
    place a branch that touches nothing outside two trees can be broken from
    outside them.  Four lines is a cheaper price than that coupling.
    """

    resolved = _committed(_model(), pe=2).resolve_dataflow(_context())
    return build_decomposed_artifact_requirements(resolved, elaborate_mvau(resolved), FINN_ROOT)


def _derivation(identity: PackagedArtifactIdentity) -> Derivation:
    """The packaged stage, as a derivation the new store can key on.

    The upstream composed key enters as an ``ArtifactRef`` rather than as the
    embedded text the current identity carries -- §6.2, and the same correction
    the A1 adapters make.
    """

    return Derivation(
        kind="rtl-module-package",
        schema_version=identity.schema_version,
        producer=ProducerIdentity("finn.rtl-module-package", "1"),
        inputs=(("upstream", ArtifactRef("composed-source", identity.upstream)),),
        request=RequestSchema(identity.command_schema),
        options=tuple((f"layout.{index:03d}", name) for index, name in enumerate(identity.layout)),
        outputs=OutputLayout(tuple(identity.layout)),
    )


class _Adapter:
    """Translate the identity, ask the real store, convert the answer back."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.asked: list[str] = []

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        self.asked.append(identity.key)
        assert isinstance(identity, PackagedArtifactIdentity)
        found = self.store.lookup(_derivation(identity))
        if found is None:
            return None
        # Two things the seam's contract requires, and it checks both.  The
        # answer carries the key it was asked about, so ``checked_lookup`` can
        # refuse an answer about something else.  And the file list is paths
        # *inside* the reported directory -- the store keeps relative names,
        # because an artifact does not change identity when it is staged
        # elsewhere, so rejoining them to a location happens here.
        directory = Path(found.directory)
        return StoredArtifact(
            identity.key, found.directory, tuple(str(directory / name) for name in found.files)
        )


@pytest.fixture(name="requirements")
def _requirements_fixture() -> MVAUDecomposedArtifactRequirements:
    built = _requirements()
    if any(not Path(path).is_file() for path in built.finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    return built


def _publish_existing(
    store: ArtifactStore, directory: Path, identity: PackagedArtifactIdentity
) -> tuple[str, ...]:
    """Take a unit the build path already wrote and put it in the new store.

    The declared layout is the identity's own, not a listing of the directory.
    Reading it off disk would make the store agree with whatever happened to be
    there, which is the check not being a check.
    """

    layout = tuple(identity.layout)
    derivation = _derivation(identity)
    workspace = store.workspace(derivation)
    for name in layout:
        shutil.copy2(directory / name, workspace / name)
    store.publish(derivation, workspace)
    return layout


# -- the gate ------------------------------------------------------------------


def test_a_real_store_answers_through_the_existing_seam(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Miss, build, publish, hit -- with no production file edited.

    The second call is handed a *different* repository root on purpose.  A
    store that answered with a directory derived from the caller's root would
    look right here and be wrong the moment two callers disagree, so the test
    asserts the answer comes from the store's own tree and that the second
    root was never written to at all.
    """

    identity = packaged_artifact_identity(requirements)
    built = package_decomposed_artifact(requirements, tmp_path / "first")
    assert built.reused is False

    store = ArtifactStore(tmp_path / "artifact-store")
    layout = _publish_existing(store, Path(built.directory), identity)
    adapter = _Adapter(store)

    reused = package_decomposed_artifact(requirements, tmp_path / "second", store=adapter)
    assert reused.reused is True
    assert adapter.asked == [identity.key]
    assert Path(reused.directory).is_relative_to(store.root)
    assert {Path(name).name for name in reused.files} == set(layout)
    assert not (tmp_path / "second").exists()


def test_the_store_is_consulted_with_this_stage_s_own_key(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Not the composed key, and not one computed from where it was staged."""

    identity = packaged_artifact_identity(requirements)
    adapter = _Adapter(ArtifactStore(tmp_path / "artifact-store"))
    package_decomposed_artifact(requirements, tmp_path / "out", store=adapter)
    assert adapter.asked == [identity.key]


def test_an_empty_answer_still_builds(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """The seam changes what is askable, not what happens on a miss."""

    store = ArtifactStore(tmp_path / "artifact-store")
    built = package_decomposed_artifact(requirements, tmp_path / "out", store=_Adapter(store))
    assert built.reused is False
    assert Path(built.directory).is_relative_to(tmp_path / "out")


def test_a_store_that_answers_about_another_artifact_is_refused_by_the_seam(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """``checked_lookup`` already guards this; the new store must not defeat it."""

    identity = packaged_artifact_identity(requirements)

    class _Liar:
        def lookup(self, asked: ArtifactKey) -> StoredArtifact:
            return StoredArtifact("e" * 64, str(tmp_path), ("a.sv",))

    with pytest.raises(Exception, match="answered with"):
        checked_lookup(_Liar(), identity)
