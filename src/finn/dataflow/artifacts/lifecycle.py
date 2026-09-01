# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle states and the declared-layout check.

Written fresh as generic rather than refactored out of the three hand-copied
implementations in the MVAU tree.  Those three agree on the shape and disagree
on the details, and lifting the intersection of three accidents is how an
abstraction ends up with the union of their assumptions.

Three states, and the distinction that matters is between the last two:

``Required``
    the derivation exists and its key is known.  Nothing has run.

``Prepared``
    a tool request has been formed.  Stages with no tool skip this entirely --
    a state that means "nothing happened" is not a state.

``Completed``
    a validated, hashed tree exists.  **Only this one may enter the store.**

Each of the three has a type, and only ``Required`` and ``Completed`` are
declared here.  ``Prepared`` is ``request.PreparedToolRun``: it lives beside
the receipt it will be checked against, and it carries the mounts, the
substitutions and the toolchain that make "a request has been formed" mean
something.  A second, emptier ``Prepared`` here would be a state with two
authorities, so the enum member points at that one instead.

A materialization is checked against the *declared* layout, never against what
happens to be on disk.  Discovering outputs makes a partial run
indistinguishable from a complete one: there is no file whose absence says
"this failed".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from finn.dataflow.artifacts.derivation import Derivation, OutputLayout, build_key
from finn.dataflow.artifacts.projection import content_digest


class LifecycleError(Exception):
    """A materialization does not match what its derivation declared."""


class State(Enum):
    REQUIRED = "required"
    PREPARED = "prepared"
    COMPLETED = "completed"


@dataclass(frozen=True)
class Materialization:
    """A tree on disk, with the digests that were taken over it.

    ``root`` is where it is *now*.  It is deliberately absent from every
    digest: an artifact does not become a different artifact because it was
    staged somewhere else.
    """

    root: Path
    #: Relative name to content digest, in declared order.
    entries: tuple[tuple[str, str], ...]

    def digests(self) -> Mapping[str, str]:
        return dict(self.entries)

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.entries)


@dataclass(frozen=True)
class Required:
    """A derivation exists and its key is known.  Nothing has run.

    Thin because the state is: what is true here is exactly that the inputs
    are declared, and anything else this carried would be a fact from a later
    state smuggled into an earlier one.
    """

    derivation: Derivation

    @property
    def state(self) -> State:
        return State.REQUIRED

    @property
    def key(self) -> str:
        return build_key(self.derivation)


@dataclass(frozen=True)
class Completed:
    """A validated tree, its derivation, and the digest over its contents."""

    derivation: Derivation
    materialization: Materialization
    tree_digest: str

    @property
    def state(self) -> State:
        return State.COMPLETED


def materialize(root: Path, layout: OutputLayout) -> Materialization:
    """Read exactly the declared files, and refuse anything else.

    Two failures are distinguished because they mean different things: a
    missing file is an incomplete run, and an extra file is a producer that
    did something it did not declare.
    """

    missing: list[str] = []
    entries: list[tuple[str, str]] = []
    for name in layout.entries:
        located = root / name
        if not located.is_file():
            missing.append(name)
            continue
        entries.append((name, content_digest(located.read_bytes())))
    if missing:
        raise LifecycleError(
            f"{root} is missing {missing}, which the declared layout requires; "
            "an incomplete tree is an attempt, not an artifact"
        )

    declared = set(layout.entries)
    found = {str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()}
    undeclared = sorted(found - declared)
    if undeclared:
        raise LifecycleError(
            f"{root} also contains {undeclared}, which the declared layout does not "
            "mention; a producer that writes what it did not declare has an output "
            "nothing checks"
        )
    return Materialization(root, tuple(entries))


def check_layout(materialization: Materialization, layout: OutputLayout) -> tuple[str, ...]:
    """Where a materialization and a declared layout disagree.

    Returned rather than raised, because a store's lookup wants to report every
    reason at once rather than the first one it met.
    """

    issues: list[str] = []
    present = materialization.digests()
    for name in layout.entries:
        if name not in present:
            issues.append(f"the declared layout requires {name!r}, which is absent")
    for name in materialization.files:
        if name not in layout.entries:
            issues.append(f"{name!r} is present and not declared")
    if materialization.files != tuple(name for name in layout.entries if name in present):
        issues.append("the file order does not match the declared layout")
    return tuple(issues)


def verify_contents(materialization: Materialization) -> tuple[str, ...]:
    """Re-hash the tree and report every file whose bytes have changed.

    A corrupt hit is an error, never a silent rebuild: a broken store hidden
    behind a slow build stays broken.
    """

    issues: list[str] = []
    for name, expected in materialization.entries:
        located = materialization.root / name
        if not located.is_file():
            issues.append(f"{name!r} is recorded and missing from {materialization.root}")
            continue
        actual = content_digest(located.read_bytes())
        if actual != expected:
            issues.append(
                f"{name!r} hashes to {actual[:12]} and the manifest records {expected[:12]}"
            )
    return tuple(issues)


def ordered_digests(entries: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """The name/digest pairs a ``tree_digest`` is taken over, sorted.

    Sorted rather than in declared order: the tree digest is an integrity check
    over a *set* of files, and the order they were declared in is already in
    the build key.  Putting it in both would make one fact move two digests.
    """

    return tuple(sorted(entries))


__all__ = [
    "Completed",
    "LifecycleError",
    "Materialization",
    "Required",
    "State",
    "check_layout",
    "materialize",
    "ordered_digests",
    "verify_contents",
]
