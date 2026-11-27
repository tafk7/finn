# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resources: directory trees FINN uses that are not Python code.

A resource is declared in a ``resources.toml`` table with one source:

- ``package`` (and ``subdir``): a directory inside an installed module, used in
  place. FINN's own trees are package data under ``finn/_sources``, in a checkout
  (symbolic links to finn-rtllib, finn_xsi, custom_hls) as in a wheel.
- ``git``, ``commit`` and ``digest``: fetched on first use, verified against the
  digest and cached by it (finn-hlslib, Vivado board files).

Consumers ask by name or kind, never by location::

    from finn import resources

    resources.path("hlslib")               # one resource, fetched if needed
    resources.join("rtllib", "mvu")        # a path inside a resource
    resources.paths("vivado-boards")       # every resource of a kind

FINN declares its own in ``finn/resources.toml``; an installed package adds more
through the ``finn.resources`` entry-point group, naming a module that contains a
``resources.toml``. ``FINN_RESOURCES_<NAME>=/dir`` replaces one resource with a
local directory, unverified.

Fetched resources go to FINN_RESOURCES_DIR (default ``$FINN_HOME/resources``); the
read-only FINN_RESOURCES_SYSTEM_CACHE (default /opt/finn/resources, if it exists)
is searched too. FINN_RESOURCES_OFFLINE=1 turns a fetch into an error.

Standard library only, so an image build can fill a cache before FINN is installed:
``PYTHONPATH=src python -m finn.resources fetch --all``.
"""
import os
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

from . import _declare, _store
from ._declare import PREFIX, DeclarationError, Resource, ResourceError

__all__ = [
    "DeclarationError",
    "Resource",
    "ResourceError",
    "Status",
    "declarations",
    "fetch",
    "join",
    "path",
    "paths",
    "status",
]

_declared: Optional[dict[str, Resource]] = None


@dataclass(frozen=True)
class Status:
    """Where a resource is: 'override', 'package', 'cached' or 'missing'."""

    name: str
    state: str
    path: Optional[str] = None


def declarations() -> dict[str, Resource]:
    """FINN's declarations, then the installed packages', by name in order."""
    global _declared
    if _declared is None:
        packages = [r for f in _declare.package_files() for r in _declare.load(f)]
        _declared = _declare.merge(_declare.load(_declare.FINN_FILE), packages)
    return dict(_declared)


def _get(name):
    try:
        return declarations()[name]
    except KeyError:
        raise ResourceError(
            "resource-undeclared", f"No resource named {name!r} is declared"
        ) from None


def _package_path(resource):
    """A directory inside an installed module. FINN is a namespace package, which may
    have several portions; the directory is in the first portion that holds it."""
    try:
        spec = find_spec(resource.package)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        raise ResourceError(
            "package-not-installed",
            f"Resource {resource.name}: {resource.package} is not installed",
        )
    portions = spec.submodule_search_locations or [os.path.dirname(spec.origin)]
    for portion in portions:
        location = os.path.join(portion, resource.subdir or "")
        if os.path.isdir(location):
            return os.path.realpath(location)
    raise ResourceError(
        "package-not-directory",
        f"Resource {resource.name}: {resource.source} is not a directory in an unpacked "
        "installation",
    )


def status(name) -> Status:
    """Where a resource is, without fetching it."""
    resource = _get(name)
    override = os.environ.get(resource.env)
    if override:
        if not os.path.isdir(override):
            raise ResourceError(
                "override-not-directory", f"{resource.env}={override} is not a directory"
            )
        return Status(name, "override", os.path.realpath(override))
    if resource.package:
        return Status(name, "package", _package_path(resource))
    entry = _store.lookup(resource)
    return Status(name, "cached", str(entry)) if entry else Status(name, "missing")


def path(name: str) -> str:
    """Return a resource's directory, fetching it into the cache if needed."""
    current = status(name)
    if current.path:
        return current.path
    if _store.offline():
        searched = ", ".join(str(root) for root, _ in _store.roots())
        raise ResourceError(
            "resource-not-cached",
            f"Resource {name} is not in any cache ({searched}) and FINN is offline. Where the "
            f"network is available, run `finn resources fetch {name} --dest DIR`, then set "
            f"{PREFIX}DIR=DIR (or {PREFIX}SYSTEM_CACHE=DIR to use it read-only).",
        )
    return str(_store.fetch(_get(name), _store.fetch_root()))


def join(name: str, *parts: str) -> str:
    """Return an existing path inside a resource, fetching it if needed.

    The parts stay inside the resource: absolute parts and ``..`` are refused.
    """
    for part in parts:
        if Path(part).is_absolute() or ".." in Path(part).parts:
            raise ValueError(f"{part!r} leaves resource {name}")
    location = os.path.join(path(name), *parts)
    if not os.path.exists(location):
        raise ResourceError(
            "resource-file-missing",
            f"{location} does not exist in resource {name} ({_get(name).source})",
        )
    return location


def paths(kind: str) -> list[str]:
    """Return the directories of every resource of a kind, in declaration order."""
    return [path(r.name) for r in declarations().values() if kind in r.kind]


def fetch(names, dest=None) -> list[str]:
    """Fetch git resources into a cache root (default: the first writable one).

    Unlike path(), this ignores overrides and copies elsewhere: it fills one cache,
    such as an image's system cache or a directory to carry offline. Package
    resources need no fetching and are skipped. Returns the entries.
    """
    root = Path(dest) if dest else _store.fetch_root()
    return [str(_store.fetch(r, root)) for r in map(_get, names) if not r.package]
