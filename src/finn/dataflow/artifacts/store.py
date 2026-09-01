# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The lookup seam: a place to ask "have we built this before".

Deliberately a seam and not a store.  The default answers *no* to everything,
so nothing behaves differently today; what the seam buys is that the build path
can be shown to *consult* it and to *skip* on a hit, and that property is what
matters.  Demonstrating it needs a question and an answer, not a database.

Explicitly out of scope, and each for its own reason:

- **Eviction.** Needs a policy, and a policy needs usage data nothing collects.
- **Concurrency.** Two builders racing on one key is a real problem and a
  different one; solving it here would fix it for a store that does not exist.
- **Persistence and cross-machine sharing.** Both require the builder version
  in the identity to be trustworthy, and it is a caller-supplied label
  (:data:`~finn.dataflow.artifacts.identity.DEFAULT_BUILDER`).  A caller that
  lies gets a wrong hit -- harmless while nothing is stored, and the first
  obligation on whoever stores anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class ArtifactKey(Protocol):
    """What a store is allowed to index on: a key, and why it is that key.

    ``serialization`` is not decoration.  A store that held only the digest
    could report a miss and have nothing to say about it, and "these two hashes
    differ" is the diagnostic this whole design was typed to avoid.
    """

    @property
    def key(self) -> str: ...

    @property
    def serialization(self) -> str: ...


@dataclass(frozen=True)
class StoredArtifact:
    """A previously built artifact, as a store reports it."""

    key: str
    directory: str
    #: The file list a simulator or synthesizer should read, in compile order.
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.key or not self.directory:
            raise ValueError("a stored artifact needs a key and a directory")


class ArtifactStore(Protocol):
    """Somewhere a built artifact might already be."""

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        """The artifact for this identity, or ``None`` if there is not one."""


class EmptyArtifactStore:
    """The default: nothing has ever been built.

    Every build therefore runs, which is exactly today's behaviour -- the seam
    changes what is *askable*, not what happens.
    """

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        return None

    def __repr__(self) -> str:
        return "EmptyArtifactStore()"


#: What a caller that says nothing gets.
NO_ARTIFACT_STORE: ArtifactStore = EmptyArtifactStore()


class ArtifactStoreError(Exception):
    """A store answered for something other than what it was asked about."""


def checked_lookup(store: ArtifactStore, identity: ArtifactKey) -> StoredArtifact | None:
    """Ask a store, and refuse an answer that is not about the question.

    ``StoredArtifact`` carries its own key, and until this existed nothing
    compared it with the key that was asked for -- so a store returning the
    wrong entry was accepted silently and the build used somebody else's RTL.
    That is precisely the wrong-hit the identity was introduced to prevent, so
    leaving it to a store's good behaviour put the guarantee in the one place
    this module does not control.

    Loud rather than a miss.  A mismatch is a broken store, and falling back to
    building would hide a defect behind a slow build.
    """

    found = store.lookup(identity)
    if found is None:
        return None
    if found.key != identity.key:
        raise ArtifactStoreError(
            f"{type(store).__name__} was asked for {identity.key} and answered with "
            f"{found.key}, staged at {found.directory}"
        )
    if not found.files:
        raise ArtifactStoreError(
            f"{type(store).__name__} reported a hit for {identity.key} with no files"
        )
    return found


__all__ = [
    "NO_ARTIFACT_STORE",
    "ArtifactKey",
    "ArtifactStore",
    "ArtifactStoreError",
    "EmptyArtifactStore",
    "StoredArtifact",
    "checked_lookup",
]
