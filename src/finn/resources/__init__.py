"""External resources: directory trees FINN uses but does not contain.

Each resource (finn-hlslib, Vivado board files, a user's RTL library...) is
declared with a pinned source and a content digest, fetched on first use, and
cached by digest. Consumers ask for resources by kind, so they never need to
know names::

    from finn import resources

    resources.path("hlslib")               # one resource, fetched if needed
    resources.paths("vivado-boards")       # every resource of a kind

Declarations come from FINN (``finn/_data/resources.toml``), then the nearest
``pyproject.toml`` (``[tool.finn.resources]``) and files listed in
FINN_RESOURCES_FILES. ``FINN_RESOURCES_<NAME>=/dir`` replaces a resource with a
local directory, unverified; FINN_HLSLIB_PATH is an alias for the hlslib one.

Caches are searched in order: FINN_RESOURCES_CACHE, the read-only system cache
FINN_RESOURCES_SYSTEM_CACHE (default /opt/finn/resources, if it exists), then
``${XDG_CACHE_HOME:-~/.cache}/finn/resources``. FINN_RESOURCES_OFFLINE=1 turns a
fetch into an error.

This package uses only the standard library, so an image build can run it
before FINN is installed: ``PYTHONPATH=src python -m finn.resources``.
"""
import os
import warnings
from dataclasses import dataclass

from . import _declare, _store
from ._declare import PREFIX, DeclarationError, Resource, ResourceError
from ._store import tree_digest

__all__ = [
    "DeclarationError",
    "Resource",
    "ResourceError",
    "Status",
    "declarations",
    "path",
    "paths",
    "status",
    "tree_digest",
]

# Replaced variables, still honoured: variable -> resource name.
_ALIASES = {"FINN_HLSLIB_PATH": "hlslib"}
# Removed variables, reported when set so they are not silently ignored.
_REMOVED = {
    "FINN_BOARD_FILES_PATH": "board files are now separate resources of kind "
    "'vivado-boards'; override each with its FINN_RESOURCES_<NAME> variable",
}
_cache = {}


@dataclass(frozen=True)
class Status:
    """Where a resource comes from: 'override', 'package', 'cached' or 'missing'."""

    name: str
    state: str
    path: str = None
    detail: str = ""


def declarations():
    """The merged declarations, by name, in declaration order.

    The project is found from the working directory on first use and kept for
    the rest of the process, so a later change of directory changes nothing.
    """
    key = os.environ.get(PREFIX + "FILES", "")
    if key not in _cache:
        for variable, advice in _REMOVED.items():
            if os.environ.get(variable):
                warnings.warn(f"{variable} is no longer used: {advice}", stacklevel=2)
        finn = _declare.load(_declare.FINN_FILE, ("resources",))
        project = [r for f, keys in _declare.project_files() for r in _declare.load(f, keys)]
        _cache[key] = _declare.merge(finn, [], project)
    return dict(_cache[key])


def _get(name):
    try:
        return declarations()[name]
    except KeyError:
        raise ResourceError(f"No resource named {name!r} is declared") from None


def _override(resource):
    variables = [resource.env] + [v for v, n in _ALIASES.items() if n == resource.name]
    for variable in variables:
        value = os.environ.get(variable)
        if value:
            if not os.path.isdir(value):
                raise ResourceError(f"{variable}={value} is not a directory")
            return variable, os.path.realpath(value)
    return None


def _package_path(resource):
    from importlib.resources import files  # noqa: PLC0415

    module, _, subdir = resource.package.partition(":")
    try:
        location = files(module).joinpath(subdir) if subdir else files(module)
    except ModuleNotFoundError:
        raise ResourceError(f"Resource {resource.name}: module {module} is not installed") from None
    if not os.path.isdir(str(location)):
        raise ResourceError(
            f"Resource {resource.name}: {resource.package} is not a directory in an "
            "unpacked installation"
        )
    return os.path.realpath(str(location))


def status(name):
    """Report whether a resource is overridden, cached or missing, and where."""
    resource = _get(name)
    override = _override(resource)
    if override:
        return Status(name, "override", override[1], override[0])
    if resource.package:
        return Status(name, "package", _package_path(resource), resource.package)
    entry = _store.lookup(resource)
    if entry:
        return Status(name, "cached", str(entry), str(entry.parent))
    return Status(name, "missing", None, str(_store.fetch_root()))


def path(name, fetch=True):
    """Return a resource's directory, fetching it into the cache if needed."""
    current = status(name)
    if current.path:
        return current.path
    resource = _get(name)
    if not fetch or os.environ.get(PREFIX + "OFFLINE", "") not in ("", "0"):
        searched = ", ".join(str(root) for root, _ in _store.roots())
        raise ResourceError(
            f"Resource {name} is not in any cache ({searched}) and fetching is disabled. "
            f"Where the network is available, run `finn-resources fetch {name} --dest DIR`, "
            f"then use DIR as {PREFIX}CACHE."
        )
    return str(_store.fetch(resource, _store.fetch_root()))


def paths(kind, fetch=True):
    """Return the directories of every resource of a kind, in declaration order."""
    return [path(r.name, fetch) for r in declarations().values() if kind in r.kind]
