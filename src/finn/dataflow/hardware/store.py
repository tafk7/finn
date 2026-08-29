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
  (:data:`~finn.dataflow.hardware.identity.DEFAULT_BUILDER`).  A caller that
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


__all__ = [
    "NO_ARTIFACT_STORE",
    "ArtifactKey",
    "ArtifactStore",
    "EmptyArtifactStore",
    "StoredArtifact",
]
