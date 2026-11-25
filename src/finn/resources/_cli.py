# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""finn resources: list, fetch, locate and clean FINN's resources."""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import finn.resources as api

from . import _store
from ._declare import ResourceError

_ENTRY = re.compile(r"(?P<name>[a-z0-9][a-z0-9-]*)-[0-9a-f]{16}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="finn resources",
        description="FINN's resources: its own trees, finn-hlslib, board files and packages'.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("list", help="declared resources and where they are")
    command = commands.add_parser("path", help="print a resource's directory, fetching it")
    command.add_argument("name")
    command = commands.add_parser("fetch", help="fetch resources into a cache, e.g. to go offline")
    command.add_argument("names", nargs="*", metavar="NAME")
    command.add_argument("--kind", action="append", help="every resource of this kind")
    command.add_argument("--all", action="store_true", help="every declared resource")
    command.add_argument("--dest", help="cache root to fill (default: the first writable one)")
    command.add_argument(
        "--redistributable-only",
        action="store_true",
        help="skip resources that may not be redistributed",
    )
    commands.add_parser("golden", help="build the VCK190 golden reference with the sourced Vivado")
    command = commands.add_parser(
        "clean", help="remove cached resources, XSI builds and golden references"
    )
    command.add_argument(
        "--unused",
        action="store_true",
        help="keep what this installation and toolchain use (others may use the rest)",
    )
    args = parser.parse_args(argv)
    try:
        return globals()["_" + args.command](args) or 0
    except ResourceError as error:
        print(f"finn resources: error: {error}", file=sys.stderr)
        return 1


def _list(args):
    rows = [("NAME", "STATE", "KIND", "LOCATION")]
    for resource in api.declarations().values():
        current = api.status(resource.name)
        where = current.path or resource.source
        rows.append((resource.name, current.state, ",".join(resource.kind) or "-", where))
    _table(rows)


def _table(rows):
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths)] + [row[-1]]
        print("  ".join(cells).rstrip())


def _path(args):
    print(api.path(args.name))


def _fetch(args):
    declared = api.declarations()
    unknown = [name for name in args.names if name not in declared]
    if unknown:
        raise ResourceError(
            "resource-undeclared",
            f"No resource named {', '.join(map(repr, unknown))} is declared",
        )
    if not (args.names or args.kind or args.all):
        raise ResourceError(
            "fetch-nothing-named", "Name resources to fetch, or use --kind or --all"
        )
    names = [
        r.name
        for r in declared.values()
        if (args.all or r.name in args.names or set(args.kind or ()) & set(r.kind))
        and (r.redistributable or not args.redistributable_only)
    ]
    for entry in api.fetch(names, args.dest):
        print(entry)


def _golden(args):
    from finn.resources import prepare  # noqa: PLC0415

    try:
        print(prepare.golden())
    except RuntimeError as error:
        print(f"finn resources: error: {error}", file=sys.stderr)
        return 1


def _clean(args):
    """Remove derived and cached state from FINN_HOME and the writable caches.

    Each entry is removed under its own lock, as it is published. A site
    FINN_GOLDEN_DIR is supplied rather than built, so it is never cleaned.
    """
    from finn.resources import prepare  # noqa: PLC0415
    from finn.xsi import paths as xsi_paths  # noqa: PLC0415

    keep = set()
    if args.unused:
        declared = [r for r in api.declarations().values() if not r.package]
        keep.update(root / r.entry for root, _ in _store.roots() for r in declared)
        for current in (_current_boards, xsi_paths.build_dir, prepare.golden_dir):
            try:
                keep.add(Path(current()))
            except (ResourceError, OSError):
                pass
    roots = [(root, _ENTRY) for root, writable in _store.roots() if writable]
    roots.append((xsi_paths.root(), re.compile(r"[0-9a-f]{20}")))
    roots.append((prepare.golden_root(), re.compile(r"[a-z0-9_-]+-[0-9a-f]{16}")))
    for root, pattern in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not pattern.fullmatch(entry.name) or not entry.is_dir() or entry in keep:
                continue
            with _store.locked(root, entry.name):
                shutil.rmtree(entry)
            print(f"removed {entry}")


def _current_boards():
    """The assembled board files of the current pins, if their sources are at hand."""
    from finn.resources import prepare  # noqa: PLC0415

    if os.environ.get("FINN_BOARD_FILES_PATH"):
        raise ResourceError("board-path-supplied", "site board files are not cached")
    return prepare.board_entry()
