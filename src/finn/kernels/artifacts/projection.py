# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The canonical projection: a value, as ordered pairs of tagged scalars.

Everything hashed in this package is hashed through here, and the shape is
chosen rather than inherited.  A preimage of ordered ``(path, tag, text)``
triples has **no mapping to order and no number whose type is ambiguous**, so
the two problems a canonical-encoding library would solve do not arise:

* There is no map-ordering rule to obey.  RFC 8949 §4.2.1 orders map keys
  bytewise-lexicographically and §4.2.3 orders them length-first; at least one
  widely used library's "canonical" mode implements the second.  A projection
  that emits no maps cannot pick the wrong one.
* ``True``, ``1``, ``1.0`` and ``"1"`` differ by construction, because the tag
  is part of the preimage.  RFC 8785 JCS could not have given us that.

So the projection is ours and versioned, and it moves under a contract version
rather than under somebody's dependency upgrade.

Two properties are worth stating because the tests hold them rather than a
docstring: nothing here iterates anything unordered, and every scalar carries
its type.  A float is written as ``float.hex()`` -- exact, round-trippable, and
free of the decimal-repr question entirely.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
from typing import Union

#: Bumped when the projection's *shape* changes -- a new tag, a different path
#: syntax, a changed float encoding.  Every key in the store is taken over a
#: preimage that begins with this, so a change is visible as an unsupported
#: projection rather than as an unexplained miss.
PROJECTION_VERSION = "projection-v1"

#: One projected leaf: where it sits, what type it is, and its text.
Pair = tuple[str, str, str]

#: What the projection accepts.  Deliberately narrow: a type that has no
#: obvious canonical text is refused rather than guessed at.
Projectable = Union[
    None, bool, int, float, str, bytes, Enum, Sequence[object], Mapping[str, object], object
]


class ProjectionError(Exception):
    """A value has no canonical projection, so it cannot enter a key."""


def _tagged(value: object, path: str) -> Pair:
    """One scalar, with the tag that keeps it distinct from its look-alikes.

    ``bool`` is tested before ``int`` because it *is* an ``int`` in Python, and
    a ``True`` that projects as ``1`` is the exact collision the tags exist to
    prevent.
    """

    if value is None:
        return (path, "none", "")
    if isinstance(value, bool):
        return (path, "bool", "true" if value else "false")
    if isinstance(value, int):
        return (path, "int", str(value))
    if isinstance(value, float):
        # ``hex()`` and not ``repr()``: exact, and it has no locale, no
        # precision policy, and no distinction between 1.0 and 1 to lose.
        if value != value or value in (float("inf"), float("-inf")):
            raise ProjectionError(f"{path} is {value!r}, which is not a value a build depends on")
        return (path, "float", value.hex())
    if isinstance(value, str):
        return (path, "str", value)
    if isinstance(value, bytes):
        return (path, "bytes", sha256(value).hexdigest())
    raise ProjectionError(f"{path} is a {type(value).__name__}, which has no canonical text")


def _enum(value: Enum, path: str) -> Pair:
    """An enum member, fully qualified.

    ``module.QualName.MEMBER``, and the member's *name* rather than its value:
    two members sharing a value are two different choices, and the bare class
    name collides -- two unrelated ``Mode`` enums in different modules would
    both project as ``Mode.X``.
    """

    kind = type(value)
    return (path, "enum", f"{kind.__module__}.{kind.__qualname__}.{value.name}")


def project(value: Projectable, path: str = "") -> tuple[Pair, ...]:
    """Flatten a value into ordered pairs of tagged scalars.

    Sequence order is preserved because it is usually a fact -- compile order
    is the standing example, and a reordered source manifest names the same
    files and is not the same build.  Mapping order is *not* a fact, so a
    mapping is emitted as its items sorted by key: the result is a sequence,
    which is what keeps the preimage free of maps.
    """

    if isinstance(value, Enum):
        return (_enum(value, path),)
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, Mapping)):
        if is_dataclass(value) and not isinstance(value, type):
            projected: list[Pair] = []
            for field in fields(value):
                attribute = getattr(value, field.name)
                projected.extend(project(attribute, f"{path}.{field.name}" if path else field.name))
            if not projected:
                raise ProjectionError(f"{path or '<root>'} projects to nothing")
            return tuple(projected)
        return (_tagged(value, path),)

    if isinstance(value, Mapping):
        items: list[Pair] = []
        for name in sorted(value):
            if not isinstance(name, str):
                raise ProjectionError(f"{path}: a mapping key must be a string, got {name!r}")
            items.extend(project(value[name], f"{path}[{name}]" if path else f"[{name}]"))
        return tuple(items)

    elements: list[Pair] = []
    for index, element in enumerate(value):
        elements.extend(project(element, f"{path}[{index}]" if path else f"[{index}]"))
    return tuple(elements)


def preimage(pairs: Sequence[Pair]) -> bytes:
    """The exact bytes a key is taken over.

    Every part is length-prefixed.  Without that, ``("ab", "c")`` and
    ``("a", "bc")`` concatenate to the same bytes, and two different builds
    would share a key through nothing more than where a boundary fell.
    """

    chunks = [f"{len(PROJECTION_VERSION)}:{PROJECTION_VERSION}".encode()]
    for path, tag, text in pairs:
        for part in (path, tag, text):
            encoded = part.encode()
            chunks.append(f"{len(encoded)}:".encode() + encoded)
    return b"".join(chunks)


def digest(value: Projectable) -> str:
    """Project a value and hash it.  The one hash function in this package."""

    return sha256(preimage(project(value))).hexdigest()


def content_digest(data: bytes) -> str:
    """The digest of a file's bytes, as stored in a ``ContentRef``."""

    return sha256(data).hexdigest()


__all__ = [
    "PROJECTION_VERSION",
    "Pair",
    "Projectable",
    "ProjectionError",
    "content_digest",
    "digest",
    "preimage",
    "project",
]
