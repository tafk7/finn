# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resource declarations: parsing, validation and merging.

A declaration is a TOML table naming one resource and its source. FINN's own
declarations ship in ``finn/resources.toml``. Installed packages add theirs through
the ``finn.resources`` entry-point group: each entry point names a module that
contains a ``resources.toml``.
"""
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

FINN_FILE = Path(__file__).resolve().parent.parent / "resources.toml"
PREFIX = "FINN_RESOURCES_"

_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
# Names whose override variable would be one of the settings.
_RESERVED = {"dir", "system-cache", "offline"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MODULE = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
_FIELDS = {"kind", "redistributable", "git", "commit", "digest", "package", "subdir", "into"}


class ResourceError(RuntimeError):
    """A resource cannot be declared, found or fetched. ``code`` names the reason
    (kebab-case); the message is the error's own."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DeclarationError(ResourceError):
    """A declaration is invalid or conflicts with another."""


@dataclass(frozen=True)
class Resource:
    name: str
    origin: str
    kind: tuple = ()
    redistributable: bool = False
    git: Optional[str] = None
    commit: Optional[str] = None
    digest: Optional[str] = None
    package: Optional[str] = None
    subdir: Optional[str] = None
    into: Optional[str] = None

    @property
    def env(self):
        """The variable that replaces this resource with a local directory."""
        return PREFIX + self.name.upper().replace("-", "_")

    @property
    def entry(self):
        """The cache entry name: a new pin gets a new directory."""
        return f"{self.name}-{self.digest.removeprefix('sha256:')[:16]}"

    @property
    def source(self):
        if self.package:
            return f"package {self.package}" + (f" ({self.subdir})" if self.subdir else "")
        where = f"{self.git}@{self.commit[:12]}"
        return f"{where} ({self.subdir})" if self.subdir else where


def parse(table, origin) -> list[Resource]:
    """Validate a table of declarations, returning resources in declaration order."""
    if not isinstance(table, dict):
        raise DeclarationError(
            "declarations-not-table", f"{origin}: resource declarations must be a table"
        )
    return [_resource(name, fields, origin) for name, fields in table.items()]


def _resource(name, fields, origin):
    def fail(code, message):
        raise DeclarationError(code, f"{origin}: resource {name!r}: {message}")

    if not _NAME.fullmatch(name) or name in _RESERVED:
        fail(
            "name-invalid",
            "use lower-case letters, digits and '-' (not dir, system-cache, offline)",
        )
    if not isinstance(fields, dict):
        fail("resource-not-table", "must be a table")
    unknown = set(fields) - _FIELDS
    if unknown:
        fail("field-unknown", f"unknown field(s) {', '.join(sorted(unknown))}")
    for key, value in fields.items():
        expected = bool if key == "redistributable" else list if key == "kind" else str
        items = value if key == "kind" and isinstance(value, list) else []
        if not isinstance(value, expected) or not all(isinstance(i, str) and i for i in items):
            fail("field-type-invalid", f"{key} must be a {expected.__name__}")
    for key in ("subdir", "into"):
        if key in fields:
            path = PurePosixPath(fields[key])
            if not fields[key] or path.is_absolute() or ".." in path.parts:
                fail("path-not-relative", f"{key} must be a relative path inside the source")
    if "package" in fields:
        if not _MODULE.fullmatch(fields["package"]):
            fail("package-invalid", "package is an importable module; a directory in it is subdir")
        if set(fields) & {"git", "commit", "digest", "into"}:
            fail("package-fields-invalid", "a package source takes only subdir")
    else:
        if not ("git" in fields and _COMMIT.fullmatch(fields.get("commit", ""))):
            fail(
                "source-invalid",
                "give a package, or a git repository with a full 40-character commit",
            )
        if not _DIGEST.fullmatch(fields.get("digest", "")):
            fail(
                "digest-invalid",
                "digest must be 'sha256:' and 64 hex digits "
                "(scripts/update-resource.py computes it)",
            )
    return Resource(
        name=name,
        origin=str(origin),
        **{key: tuple(value) if key == "kind" else value for key, value in fields.items()},
    )


def load(path) -> list[Resource]:
    """Read the [resources] table of a TOML file."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DeclarationError(
            "declarations-unreadable", f"Cannot read resource declarations from {path}: {error}"
        ) from None
    return parse(data.get("resources", {}), path)


def package_files():
    """Declaration files of installed packages, from the finn.resources entry points."""
    from importlib.metadata import entry_points  # noqa: PLC0415
    from importlib.resources import files  # noqa: PLC0415

    found = []
    for point in sorted(
        entry_points(group="finn.resources"),
        key=lambda p: ((p.dist.name if p.dist else ""), p.name),
    ):
        where = f"entry point {point.name!r} of {point.dist.name if point.dist else '?'}"
        try:
            file = files(point.value).joinpath("resources.toml")
        except (ImportError, TypeError) as error:
            raise DeclarationError(
                "entry-point-unimportable", f"{where}: cannot import {point.value}: {error}"
            ) from None
        if not file.is_file() or not isinstance(file, Path):
            raise DeclarationError(
                "entry-point-no-declarations", f"{where}: {point.value} contains no resources.toml"
            )
        found.append(file)
    return found


def merge(finn, packages) -> dict[str, Resource]:
    """FINN's declarations, then the packages'; a package may only add."""
    merged = {r.name: r for r in finn}
    for resource in packages:
        if resource.name in merged:
            raise DeclarationError(
                "resource-redeclared",
                f"Resource {resource.name!r} from {resource.origin} is already declared by "
                f"{merged[resource.name].origin}",
            )
        merged[resource.name] = resource
    return merged
