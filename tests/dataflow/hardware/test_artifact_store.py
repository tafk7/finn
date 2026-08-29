# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5d: the lookup seam, with no store behind it.

The obligation is narrow and worth stating narrowly: the build path *consults*
the seam, and *skips* on a hit.  Both are properties of the call path, so a
question and a recorded answer demonstrate them; a store would add nothing to
the evidence and a great deal to the surface.

What a store must not be able to do quietly is answer for the wrong artifact.
So the tests below are as much about what the seam is asked -- the identity,
not a name, a path, or a source node -- as about what it replies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from finn.dataflow.hardware import (
    NO_ARTIFACT_STORE,
    ArtifactKey,
    ArtifactStore,
    EmptyArtifactStore,
    StoredArtifact,
)
from finn.dataflow.mvau.hardware.composition import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    write_decomposed_artifact,
)
from finn.dataflow.mvau.providers import elaborate_mvau

FINN_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _RecordingStore:
    """A test double: it records what it was asked and answers as told."""

    answer: StoredArtifact | None = None
    asked: list[ArtifactKey] = field(default_factory=list)

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        self.asked.append(identity)
        return self.answer


def _requirements(*, pe: int = 2) -> MVAUDecomposedArtifactRequirements:
    resolved = _committed(_model(), pe=pe).resolve_dataflow(_context())
    return build_decomposed_artifact_requirements(resolved, elaborate_mvau(resolved), FINN_ROOT)


def _skip_without_finnlib(requirements: MVAUDecomposedArtifactRequirements) -> None:
    if any(not Path(path).is_file() for path in requirements.finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")


# -- the default -------------------------------------------------------------


def test_the_default_store_has_nothing_and_changes_nothing(tmp_path: Path) -> None:
    """Adding the seam must not alter a single existing build."""

    requirements = _requirements()
    _skip_without_finnlib(requirements)

    assert NO_ARTIFACT_STORE.lookup(requirements.identity) is None
    with_default = write_decomposed_artifact(requirements, tmp_path / "a")
    explicit = write_decomposed_artifact(requirements, tmp_path / "b", store=EmptyArtifactStore())

    assert [Path(item).name for item in with_default] == [Path(item).name for item in explicit]
    assert all(Path(item).is_file() for item in with_default)


# -- the two properties that matter ------------------------------------------


def test_the_build_path_asks_the_store_before_it_writes(tmp_path: Path) -> None:
    """Consulting it is the first half, and asking with the *identity* matters.

    A seam keyed on the output directory, the module name, or the source node
    would be just as easy to write and would answer for the wrong artifact.
    """

    requirements = _requirements()
    _skip_without_finnlib(requirements)
    store = _RecordingStore()

    write_decomposed_artifact(requirements, tmp_path, store=store)

    assert len(store.asked) == 1
    assert store.asked[0] == requirements.identity
    assert store.asked[0].key == requirements.identity.key


def test_a_hit_returns_the_stored_files_and_writes_nothing(tmp_path: Path) -> None:
    """Skipping is the other half, and "wrote nothing" is the assertion.

    Returning the right list while still staging every file would pass a
    weaker test and buy nothing at all.
    """

    requirements = _requirements()
    output = tmp_path / "output"
    previous = ("/previously/built/dotp.sv", "/previously/built/top.sv")
    store = _RecordingStore(
        StoredArtifact(requirements.identity.key, "/previously/built", previous)
    )

    assert write_decomposed_artifact(requirements, output, store=store) == previous
    assert not output.exists()


def test_a_miss_builds_exactly_as_it_did(tmp_path: Path) -> None:
    requirements = _requirements()
    _skip_without_finnlib(requirements)

    built = write_decomposed_artifact(requirements, tmp_path / "cold", store=_RecordingStore())
    unstored = write_decomposed_artifact(requirements, tmp_path / "plain")

    assert [Path(item).name for item in built] == [Path(item).name for item in unstored]


def test_two_configurations_are_two_questions(tmp_path: Path) -> None:
    """The seam distinguishes what the identity distinguishes, and no less.

    A store that answered on the module name would conflate nothing here today,
    but a store that answered on the *source node* would -- which is the whole
    reason the identity is what is passed.
    """

    store = _RecordingStore()
    for index, pe in enumerate((2, 4)):
        requirements = _requirements(pe=pe)
        _skip_without_finnlib(requirements)
        write_decomposed_artifact(requirements, tmp_path / str(index), store=store)

    assert len({item.key for item in store.asked}) == 2


# -- what the seam is, and is not --------------------------------------------


def test_the_seam_asks_for_a_key_that_can_explain_itself() -> None:
    """``serialization`` is part of the protocol on purpose.

    A store holding only digests can report a miss and say nothing about why,
    and "these two hashes differ" is the diagnostic this design was typed to
    avoid.  The requirement is on the *protocol*, so a store cannot be written
    against a bare string by accident.
    """

    requirements = _requirements()
    identity = requirements.identity

    assert isinstance(identity, ArtifactKey)
    # The key is a digest; the serialization is what the digest was taken over,
    # and it is the only one of the two a miss can be explained from.
    assert identity.key not in identity.serialization
    for kernel in requirements.identity.kernels:
        assert kernel.kernel_id in identity.serialization


def test_the_default_store_is_a_seam_and_not_a_store() -> None:
    """The drift this increment exists to prevent, as a failing test.

    §7.3 keeps a production cache out of scope, and the value of that line is
    that 5d stays testable without one.  A default that grew somewhere to put
    things would have quietly taken the scope with it.
    """

    for forbidden in ("store", "put", "insert", "save", "evict", "clear", "path"):
        assert not hasattr(NO_ARTIFACT_STORE, forbidden)
    assert not vars(EmptyArtifactStore())


def test_a_store_is_structural_and_needs_no_base_class() -> None:
    """Anything with ``lookup`` is one, which is what keeps the seam a seam."""

    store: ArtifactStore = _RecordingStore()
    assert store.lookup(_requirements().identity) is None


def test_a_stored_artifact_must_say_where_it_is() -> None:
    with pytest.raises(ValueError):
        StoredArtifact("", "/somewhere", ())
    with pytest.raises(ValueError):
        StoredArtifact("abc", "", ())
