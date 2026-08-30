# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5d: the lookup seam, with no store behind it.

The obligation is narrow: something can be asked "is there an artifact for
this identity", the default answers no, and an answer about a *different*
artifact is refused rather than used.

That last part is the whole reason this file is not three lines.  A store is
the one component in the chain this layer does not control, and a hit is the
one path where nothing else would notice a wrong answer -- so the check
belongs here, not in a comment asking stores to behave.

The build path's use of the seam is tested where the build path lives, in
``dataflow/mvau/test_decomposed_packaging.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from finn.dataflow.hardware import (
    NO_ARTIFACT_STORE,
    ArtifactKey,
    ArtifactStore,
    ArtifactStoreError,
    EmptyArtifactStore,
    PackagedArtifactIdentity,
    StoredArtifact,
    checked_lookup,
)


def _identity(upstream: str = "upstream-key") -> PackagedArtifactIdentity:
    return PackagedArtifactIdentity(upstream, ("a.sv", "top.sv"), "instantiate {module} {instance}")


@dataclass
class _RecordingStore:
    """A test double: it records what it was asked and answers as told."""

    answer: StoredArtifact | None = None
    asked: list[ArtifactKey] = field(default_factory=list)

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        self.asked.append(identity)
        return self.answer


# -- the default -------------------------------------------------------------


def test_the_default_store_has_nothing() -> None:
    """So adding the seam alters no existing build."""

    assert NO_ARTIFACT_STORE.lookup(_identity()) is None
    assert checked_lookup(NO_ARTIFACT_STORE, _identity()) is None


def test_the_default_store_is_a_seam_and_not_a_store() -> None:
    """The drift this increment exists to prevent, as a failing test.

    §7.3 keeps a production cache out of scope, and the value of that line is
    that the seam stays testable without one.  A default that grew somewhere to
    put things would have quietly taken the scope with it.
    """

    for forbidden in ("store", "put", "insert", "save", "evict", "clear", "path"):
        assert not hasattr(NO_ARTIFACT_STORE, forbidden)
    assert not vars(EmptyArtifactStore())


def test_a_store_is_structural_and_needs_no_base_class() -> None:
    """Anything with ``lookup`` is one, which is what keeps the seam a seam."""

    double = _RecordingStore()
    store: ArtifactStore = double
    assert checked_lookup(store, _identity()) is None
    assert double.asked == [_identity()]


# -- the guarantee -----------------------------------------------------------


def test_an_answer_about_another_artifact_is_refused() -> None:
    """The finding this check was added for.

    ``StoredArtifact`` has always carried its own key and nothing compared it
    with the key that was asked for.  A store returning the wrong entry was
    accepted in silence, and the build proceeded with somebody else's RTL --
    the exact wrong hit the identity was introduced to prevent.
    """

    wrong = _RecordingStore(StoredArtifact("some-other-key", "/cached/wrong", ("/w/top.sv",)))
    with pytest.raises(ArtifactStoreError) as raised:
        checked_lookup(wrong, _identity())

    # The message names both keys and where the bad entry was staged: a store
    # that answers wrongly has to be findable, not just stopped.
    assert "some-other-key" in str(raised.value)
    assert _identity().key in str(raised.value)
    assert "/cached/wrong" in str(raised.value)


def test_a_mismatch_is_loud_rather_than_a_miss() -> None:
    """Falling back to building would hide a broken store behind a slow build.

    Tempting, because it is "safe" -- the output would be correct.  But the
    store stays broken and every subsequent consumer keeps asking it, so the
    one observable symptom of the defect is that the cache never helps.
    """

    wrong = _RecordingStore(StoredArtifact("elsewhere", "/cached", ("/cached/top.sv",)))
    with pytest.raises(ArtifactStoreError):
        checked_lookup(wrong, _identity())


def test_a_hit_with_no_files_is_refused() -> None:
    """An empty hit would skip the build and hand a synthesizer nothing."""

    empty = _RecordingStore(StoredArtifact(_identity().key, "/cached", ()))
    with pytest.raises(ArtifactStoreError, match="no files"):
        checked_lookup(empty, _identity())


def test_a_correct_hit_passes_through_unchanged() -> None:
    """The check must not be so strict that a right answer fails it."""

    entry = StoredArtifact(_identity().key, "/cached", ("/cached/a.sv", "/cached/top.sv"))
    assert checked_lookup(_RecordingStore(entry), _identity()) is entry


def test_two_identities_are_two_questions() -> None:
    """The seam distinguishes what the identity distinguishes, and no less."""

    store = _RecordingStore()
    for upstream in ("one", "two"):
        checked_lookup(store, _identity(upstream))

    assert len({item.key for item in store.asked}) == 2


# -- what the seam asks with -------------------------------------------------


def test_the_seam_asks_for_a_key_that_can_explain_itself() -> None:
    """``serialization`` is part of the protocol on purpose.

    A store holding only digests can report a miss and say nothing about why,
    and "these two hashes differ" is the diagnostic this design was typed to
    avoid.  The requirement is on the *protocol*, so a store cannot be written
    against a bare string by accident.
    """

    identity = _identity()

    assert isinstance(identity, ArtifactKey)
    assert identity.key not in identity.serialization
    assert identity.upstream in identity.serialization


def test_a_stored_artifact_must_say_where_it_is() -> None:
    with pytest.raises(ValueError):
        StoredArtifact("", "/somewhere", ())
    with pytest.raises(ValueError):
        StoredArtifact("abc", "", ())
