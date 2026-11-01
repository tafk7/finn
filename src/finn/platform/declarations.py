# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform's declarations: the boards, shells, linkers, runtimes and mover kinds a
build may name, stated as data in ``platform.toml`` files.

Five kinds of thing are declared, each stating only its own facts:

- a **board** (``[boards.<name>]``): its ``part``, nothing about any flow;
- a **linker** (``[linkers.<name>]``): its ``code``, the ``tool`` it runs and the
  releases it runs on (``versions``, optional), what it needs of each board a shell
  offers it on (``board_keys``), the ``image`` it produces, and the least address range
  it assigns a control bus (``least_range``, bytes, a power of two);
- a **runtime** (``[runtimes.<name>]``): its ``code``, the ``images`` it loads, the
  mover kinds it drives (``movers``), the supply and delivery kinds it serves
  (``kinds``) and the description schemas it reads (``schemas``);
- a **mover kind** (``[movers.<name>]``): its ``code`` and the kinds it serves;
- a **shell** (``[shells.<name>]``): its ``linker`` and the ``runtimes`` it offers (the
  first is the default), its typed facts (``finn.platform.shells``), and the boards it
  supports (``boards``: a table of entries, each giving what its linker needs of that
  board and any fact that differs there; or ``"any"``, any part). ``based_on`` names a
  shell whose facts it starts from: each key it states replaces the base's whole, and
  nothing is inherited at run time.

They are found in three places, in this order:

1. FINN's own, ``finn/platform/data/platform.toml``;
2. each entry point of the group ``finn.platform``, which names a package holding a
   ``platform.toml``, found through its distribution's metadata without importing it;
3. each directory on ``FINN_PLATFORM_PATH`` (``os.pathsep`` between them) holding a
   ``platform.toml``.

``load(extra=...)`` adds files after them, for a test. Every source only adds: a name
declared twice is refused (``platform-name-taken``), naming both origins, so a project
cannot change what a shell FINN declares means; a variant takes a new name. A source
other than FINN's may also hold a parts overlay beside its ``platform.toml``
(``parts.json``, ``finn.platform.catalog``).

Code is named by ``module:attr`` and imported only when it is used (``code``), so reading,
showing and resolving declarations runs none of it. A malformed declaration is refused
when it is read (``declaration-invalid``), owned by its file and table; whether the
declarations agree with each other is ``finn.platform.check``'s.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import Distribution, EntryPoint, entry_points
from pathlib import Path
from typing import Any

from finn.platform import _stated as valid
from finn.platform.boards import Board
from finn.platform.refusal import TargetRefused

FINN_FILE = Path(__file__).resolve().parent / "data" / "platform.toml"
"""FINN's own declarations."""
FILE = "platform.toml"
"""The file a source directory declares in."""
PARTS_OVERLAY = "parts.json"
"""The parts overlay a source directory other than FINN's may hold beside its declarations."""
GROUP = "finn.platform"
"""The entry-point group naming the installed packages that declare."""
PATH_SETTING = "FINN_PLATFORM_PATH"
"""The setting listing the user's declaration directories."""

FINN, PACKAGE, PATH, EXTRA = "finn", "package", "path", "extra"
"""Where a declaration comes from (``Origin.kind``)."""

ANY = "any"
"""A shell's ``boards`` when it supports any part, with no board (``ip``)."""

KINDS = ("boards", "linkers", "runtimes", "movers", "shells")
"""The kinds of declaration, as the tables of a ``platform.toml`` name them."""

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PACKAGE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_SHELL_OWN = {"based_on", "linker", "runtimes", "boards"}


@dataclass(frozen=True)
class Origin:
    """Where a declaration comes from: FINN's file (``kind`` ``finn``), an installed
    package's (``package``, with its distribution and version), a directory on
    ``FINN_PLATFORM_PATH`` (``path``), or a file a caller added (``extra``); ``file``
    is the ``platform.toml`` itself."""

    kind: str
    file: Path
    package: str | None = None
    version: str | None = None

    def __str__(self) -> str:
        if self.kind == FINN:
            return "FINN"
        if self.kind == PACKAGE:
            return f"package {self.package} {self.version}"
        if self.kind == PATH:
            return f"{PATH_SETTING} {self.file.parent}"
        return str(self.file)

    def record(self) -> dict[str, str | None]:
        """It as a build records it: its ``kind``, with the package and its version for a
        package's (``None`` where the distribution states none), the file for a
        directory's or a caller's, and nothing more for FINN's (the build records FINN's
        revision)."""
        if self.kind == FINN:
            return {"kind": FINN}
        if self.kind == PACKAGE:
            return {"kind": PACKAGE, "package": self.package, "version": self.version}
        return {"kind": self.kind, "file": str(self.file)}


@dataclass(frozen=True, kw_only=True)
class Linker:
    """A linker's declaration: its ``code`` (``module:attr``), the ``tool`` it runs and
    its releases (``versions``: a release range, ``None``: any), the keys it needs of each
    board a shell offers it on (``board_keys``), the ``image`` it produces, and the least
    address range it assigns a control bus (``least_range``, bytes): a shell it links
    takes that window of a unit at least (``finn.platform.shells.Control.least_window``)."""

    name: str
    code: str
    tool: str
    versions: str | None
    board_keys: tuple[str, ...]
    image: str
    least_range: int


@dataclass(frozen=True, kw_only=True)
class Runtime:
    """A runtime's declaration: its ``code``, the ``images`` it loads, the mover kinds it
    drives (``movers``), the supply and delivery kinds it serves (``kinds``) and the
    description schemas it reads (``schemas``)."""

    name: str
    code: str
    images: tuple[str, ...]
    movers: tuple[str, ...]
    kinds: tuple[str, ...]
    schemas: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class Mover:
    """A mover kind's declaration: its ``code`` and the supply or delivery kinds it serves
    (``kinds``)."""

    name: str
    code: str
    kinds: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Shell:
    """A shell's declaration, its ``based_on`` already applied: its ``linker`` (``None``:
    its user links), the ``runtimes`` it offers, the boards it supports (``boards``:
    each board's entry, a table; ``ANY``: any part, with no board), and its typed facts
    as stated (``facts``, read by ``finn.platform.shells``)."""

    name: str
    linker: str | None
    runtimes: tuple[str, ...]
    boards: str | Mapping[str, Mapping[str, Any]]
    facts: Mapping[str, Any]
    based_on: str | None

    @property
    def any_board(self) -> bool:
        """It supports any part, with no board."""
        return self.boards == ANY

    def supports(self, board: str) -> bool:
        """It names ``board`` among its boards."""
        return not isinstance(self.boards, str) and board in self.boards


@dataclass(frozen=True)
class Declarations:
    """Every declaration of one load, by kind and name, where each is declared
    (``origins``, by ``kind.name``), and the directories it read (``directories``: each
    source other than FINN's, with its origin), where a parts overlay may sit."""

    boards: Mapping[str, Board]
    linkers: Mapping[str, Linker]
    runtimes: Mapping[str, Runtime]
    movers: Mapping[str, Mover]
    shells: Mapping[str, Shell]
    origins: Mapping[str, Origin]
    directories: tuple[tuple[Origin, Path], ...] = ()

    def origin(self, kind: str, name: str) -> Origin:
        """Where ``kind.name`` (``shells.pynq``) is declared."""
        return self.origins[f"{kind}.{name}"]

    def board(self, name: str) -> Board:
        """The board ``name``; refused if none is declared (``unknown-board``)."""
        if name not in self.boards:
            raise TargetRefused(
                "unknown-board",
                "configuration.target.board",
                f"{name!r} is not a board (one of {sorted(self.boards)})",
            )
        return self.boards[name]

    def shell(self, name: str) -> Shell:
        """The shell ``name``; refused if none is declared (``unknown-shell``), listing
        the declared shells."""
        if name not in self.shells:
            raise TargetRefused(
                "unknown-shell",
                "configuration.target.shell",
                f"{name!r} is not a declared shell (one of {sorted(self.shells)})",
            )
        return self.shells[name]

    def supporting(self, board: str) -> tuple[str, ...]:
        """The shells that support ``board``: those naming it, then those of any part."""
        named = [name for name, shell in self.shells.items() if shell.supports(board)]
        anywhere = [name for name, shell in self.shells.items() if shell.any_board]
        return (*sorted(named), *sorted(anywhere))


# -- reading -----------------------------------------------------------------------------


def _refused(origin: Origin, where: str, why: str) -> TargetRefused:
    return TargetRefused("declaration-invalid", f"{origin.file}: {where}", why)


def _board(name: str, table: dict[str, Any]) -> Board:
    valid.keys(table, f"boards.{name}", {"part"}, set())
    return Board(name=name, part=valid.text(table["part"], f"boards.{name}.part"))


def _linker(name: str, table: dict[str, Any]) -> Linker:
    where = f"linkers.{name}"
    valid.keys(table, where, {"code", "tool", "board_keys", "image", "least_range"}, {"versions"})
    versions = table.get("versions")
    return Linker(
        name=name,
        code=valid.code(table["code"], f"{where}.code"),
        tool=valid.text(table["tool"], f"{where}.tool"),
        versions=None if versions is None else valid.releases(versions, f"{where}.versions"),
        board_keys=valid.texts(table["board_keys"], f"{where}.board_keys"),
        image=valid.text(table["image"], f"{where}.image"),
        least_range=valid.power_of_two(table["least_range"], f"{where}.least_range"),
    )


def _runtime(name: str, table: dict[str, Any]) -> Runtime:
    where = f"runtimes.{name}"
    valid.keys(table, where, {"code", "images", "movers", "kinds", "schemas"}, set())
    schemas = table["schemas"]
    if not isinstance(schemas, list) or not all(type(each) is int and each > 0 for each in schemas):
        raise valid.Invalid(f"{where}.schemas is a list of schema numbers, not {schemas!r}")
    return Runtime(
        name=name,
        code=valid.code(table["code"], f"{where}.code"),
        images=valid.texts(table["images"], f"{where}.images"),
        movers=valid.texts(table["movers"], f"{where}.movers"),
        kinds=valid.texts(table["kinds"], f"{where}.kinds"),
        schemas=tuple(schemas),
    )


def _mover(name: str, table: dict[str, Any]) -> Mover:
    where = f"movers.{name}"
    valid.keys(table, where, {"code", "kinds"}, set())
    return Mover(
        name=name,
        code=valid.code(table["code"], f"{where}.code"),
        kinds=valid.texts(table["kinds"], f"{where}.kinds"),
    )


def _shell_table(name: str, table: dict[str, Any]) -> dict[str, Any]:
    """A shell's table, its own keys checked; its facts are ``finn.platform.shells``'."""
    where = f"shells.{name}"
    if "based_on" in table:
        valid.text(table["based_on"], f"{where}.based_on")
    if "linker" in table:
        valid.text(table["linker"], f"{where}.linker")
    if "runtimes" in table:
        valid.texts(table["runtimes"], f"{where}.runtimes")
    boards = table.get("boards")
    if boards is not None and boards != ANY:
        for board, entry in valid.table(boards, f"{where}.boards").items():
            valid.table(entry, f"{where}.boards.{board}")
    return table


@dataclass
class _Found:
    """The declarations read so far, by kind and name, and each one's origin."""

    boards: dict[str, Board] = field(default_factory=dict)
    linkers: dict[str, Linker] = field(default_factory=dict)
    runtimes: dict[str, Runtime] = field(default_factory=dict)
    movers: dict[str, Mover] = field(default_factory=dict)
    shells: dict[str, tuple[dict[str, Any], Origin]] = field(default_factory=dict)
    origins: dict[str, Origin] = field(default_factory=dict)

    def add(self, kind: str, name: str, origin: Origin) -> None:
        """Note ``kind.name`` as declared by ``origin``; refused if it already is."""
        key = f"{kind}.{name}"
        if key in self.origins:
            raise TargetRefused(
                "platform-name-taken",
                f"{origin.file}: {key}",
                f"{key} is declared by {self.origins[key]} and again by {origin}: a source "
                "only adds, so a variant takes a new name",
            )
        self.origins[key] = origin


def _read(path: Path, origin: Origin, found: _Found) -> None:
    """Add the declarations in the file at ``path`` to ``found``; the shells as their
    tables, ``based_on`` not yet applied."""
    try:
        with open(path, "rb") as handle:
            stated = tomllib.load(handle)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise TargetRefused(
            "declaration-invalid", str(path), f"cannot read declarations: {error}"
        ) from None
    unknown = sorted(set(stated) - set(KINDS))
    if unknown:
        raise _refused(origin, ", ".join(unknown), f"a declaration is one of {', '.join(KINDS)}")
    for kind in KINDS:
        try:
            tables = valid.table(stated.get(kind, {}), kind)
        except valid.Invalid as error:
            raise _refused(origin, kind, str(error)) from None
        for name, table in tables.items():
            where = f"{kind}.{name}"
            try:
                if not _NAME.fullmatch(name):
                    raise valid.Invalid(f"{name!r} is no name (letters, digits, '_', '.', '-')")
                table = valid.table(table, where)
                if kind == "boards":
                    board = _board(name, table)
                elif kind == "linkers":
                    linker = _linker(name, table)
                elif kind == "runtimes":
                    runtime = _runtime(name, table)
                elif kind == "movers":
                    mover = _mover(name, table)
                else:
                    shell = _shell_table(name, table)
            except valid.Invalid as error:
                raise _refused(origin, where, str(error)) from None
            found.add(kind, name, origin)
            if kind == "boards":
                found.boards[name] = board
            elif kind == "linkers":
                found.linkers[name] = linker
            elif kind == "runtimes":
                found.runtimes[name] = runtime
            elif kind == "movers":
                found.movers[name] = mover
            else:
                found.shells[name] = (shell, origin)


def _resolve_shell(
    name: str, tables: Mapping[str, tuple[dict[str, Any], Origin]], seen: tuple[str, ...] = ()
) -> dict[str, Any]:
    """A shell's table with its ``based_on`` applied: the base's, resolved in turn, with
    each key the shell states in its place."""
    table, origin = tables[name]
    base = table.get("based_on")
    if base is None:
        return dict(table)
    if base in seen or base == name:
        raise _refused(origin, f"shells.{name}.based_on", f"{base!r} is based on itself")
    if base not in tables:
        raise _refused(origin, f"shells.{name}.based_on", f"{base!r} is not a declared shell")
    resolved = _resolve_shell(base, tables, (*seen, name))
    resolved.update(table)
    return resolved


def _shell(name: str, table: dict[str, Any], based_on: str | None, origin: Origin) -> Shell:
    where = f"shells.{name}"
    if "boards" not in table:
        raise _refused(origin, where, "states no boards (a table of entries, or 'any')")
    if table.get("linker") is None and table["boards"] != ANY:
        raise _refused(origin, where, "a shell with no linker supports any part: boards = 'any'")
    if not table["boards"]:
        raise _refused(origin, where, "supports no board: its boards are a table of entries")
    return Shell(
        name=name,
        linker=table.get("linker"),
        runtimes=tuple(table.get("runtimes", ())),
        boards=table["boards"],
        facts={key: value for key, value in table.items() if key not in _SHELL_OWN},
        based_on=based_on,
    )


# -- the sources -------------------------------------------------------------------------


def _package_file(package: str, dist: Distribution | None) -> Path | None:
    """The ``platform.toml`` of ``package``, found without importing it: among its
    distribution's recorded files, else in the search locations of its top-level package
    (an editable install's), which finding runs none of its code."""
    names = package.split(".")
    if dist is not None:
        for recorded in dist.files or ():
            if tuple(recorded.parts) == (*names, FILE):
                return Path(str(dist.locate_file(recorded)))
    try:
        spec = importlib.util.find_spec(names[0])
    except (ImportError, ValueError):
        return None
    for location in (spec and spec.submodule_search_locations) or ():
        file = Path(location, *names[1:], FILE)
        if file.is_file():
            return file
    return None


def package_sources() -> list[tuple[Origin, Path]]:
    """Each installed package's declarations, from the ``finn.platform`` entry points,
    in order of distribution and name: its origin and its directory. The package an entry
    point names is not imported; one that holds no ``platform.toml`` is refused."""
    found = []
    for point in sorted(entry_points(group=GROUP), key=lambda p: (_distribution(p), p.name)):
        dist, version = _distribution(point), point.dist.version if point.dist else None
        where = f"package {dist} {version}: entry point {point.name!r}"
        if not _PACKAGE.fullmatch(point.value):
            raise TargetRefused(
                "declaration-invalid", where, f"names a package, not {point.value!r}"
            )
        file = _package_file(point.value, point.dist)
        if file is None or not file.is_file():
            raise TargetRefused("declaration-invalid", where, f"{point.value} holds no {FILE}")
        found.append((Origin(PACKAGE, file, package=dist, version=version), file.parent))
    return found


def _distribution(point: EntryPoint) -> str:
    return point.dist.name if point.dist else "?"


def path_sources(environ: Mapping[str, str] | None = None) -> list[tuple[Origin, Path]]:
    """Each directory on ``FINN_PLATFORM_PATH`` holding declarations, in order: its origin
    and its path. A directory without a ``platform.toml`` that holds a parts overlay
    adds parts only; one that holds neither is refused."""
    environ = os.environ if environ is None else environ
    found = []
    for entry in (environ.get(PATH_SETTING) or "").split(os.pathsep):
        if not entry:
            continue
        directory = Path(entry).expanduser()
        if not directory.is_dir() or not (
            (directory / FILE).is_file() or (directory / PARTS_OVERLAY).is_file()
        ):
            raise TargetRefused(
                "declaration-invalid",
                PATH_SETTING,
                f"{entry}: a directory holding {FILE} or {PARTS_OVERLAY}",
            )
        found.append((Origin(PATH, directory / FILE), directory))
    return found


def load(
    extra: Iterable[Path] = (), *, environ: Mapping[str, str] | None = None, finn: bool = True
) -> Declarations:
    """Every declaration: FINN's own (unless ``finn`` is false), the installed packages',
    the directories on ``FINN_PLATFORM_PATH`` (of ``environ``, the process's by default),
    then each of ``extra`` (a ``platform.toml``, or a directory holding one). A name
    declared twice is refused (``platform-name-taken``)."""
    sources: list[tuple[Origin, Path | None]] = []
    if finn:
        sources.append((Origin(FINN, FINN_FILE), None))
    sources += [*package_sources(), *path_sources(environ)]
    for each in extra:
        path = Path(each)
        file = path / FILE if path.is_dir() else path
        if not (file.is_file() or (path.is_dir() and (path / PARTS_OVERLAY).is_file())):
            raise TargetRefused(
                "declaration-invalid",
                str(path),
                f"{path}: a {FILE}, or a directory holding {FILE} or {PARTS_OVERLAY}",
            )
        sources.append((Origin(EXTRA, file), file.parent))
    found = _Found()
    for origin, _ in sources:
        if origin.file.is_file():  # else a path directory with a parts overlay only
            _read(origin.file, origin, found)
    shells = {
        name: _shell(name, _resolve_shell(name, found.shells), table.get("based_on"), origin)
        for name, (table, origin) in found.shells.items()
    }
    return Declarations(
        boards=found.boards,
        linkers=found.linkers,
        runtimes=found.runtimes,
        movers=found.movers,
        shells=shells,
        origins=found.origins,
        directories=tuple(
            (origin, directory) for origin, directory in sources if directory is not None
        ),
    )


@functools.cache
def declarations() -> Declarations:
    """The declarations this process resolves by (``load``), read once."""
    return load()


def code(reference: str) -> object:
    """The object a declaration's ``code`` names (``module:attr``), imported now;
    refused if it does not import, whatever its module raises (``code-unimportable``)."""
    module, _, attribute = reference.partition(":")
    try:
        found: object = importlib.import_module(module)
        for part in attribute.split("."):
            found = getattr(found, part)
    except Exception as error:  # importing runs the module, which may raise anything
        raise TargetRefused(
            "code-unimportable", reference, f"{reference} does not import: {error}"
        ) from None
    return found


def overlays(declared: Declarations) -> tuple[Path, ...]:
    """The parts overlays the directories of ``declared`` hold, in order."""
    return tuple(
        directory / PARTS_OVERLAY
        for _, directory in declared.directories
        if (directory / PARTS_OVERLAY).is_file()
    )


__all__ = [
    "ANY",
    "FILE",
    "FINN_FILE",
    "GROUP",
    "KINDS",
    "PARTS_OVERLAY",
    "PATH_SETTING",
    "Declarations",
    "Linker",
    "Mover",
    "Origin",
    "Runtime",
    "Shell",
    "code",
    "declarations",
    "load",
    "overlays",
]
