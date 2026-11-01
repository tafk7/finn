# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""finn-platform: list, show and check the parts, boards and shells a build may name.

    finn-platform list [boards|shells|linkers|runtimes|movers]
    finn-platform show part PART
    finn-platform show board BOARD
    finn-platform show shell SHELL [--board BOARD]
    finn-platform check

Reads the process's declarations (``finn.platform.declarations``: FINN's, the installed
packages', the directories on ``FINN_PLATFORM_PATH``) and its part catalog. ``list`` and
``show`` import no declared code; ``check`` imports each one, and exits 1 on any finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from finn.platform.catalog import part, part_report
from finn.platform.check import check
from finn.platform.declarations import KINDS, Declarations, declarations
from finn.platform.refusal import TargetRefused
from finn.platform.shells import ShellRow, shell_row


def _write(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _plain(value: object) -> Any:
    """``value`` as JSON's: a dataclass by its fields, a tuple as a list."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(each) for each in value]
    if isinstance(value, dict):
        return {str(key): _plain(each) for key, each in value.items()}
    return value


def _list(declared: Declarations, kind: str | None) -> None:
    for each in KINDS if kind is None else (kind,):
        entries: Any = getattr(declared, each)
        _write(f"{each}:")
        for name in sorted(entries):
            _write(f"  {name:<24} {declared.origin(each, name)}")


def _row(found: ShellRow) -> dict[str, Any]:
    stated = _plain(found)
    # A row's tables print as JSON, its windows and bases in hex.
    if found.control is not None:
        stated["control"]["aperture"] = hex(found.control.aperture)
    if found.fixed is not None:
        stated["fixed"]["control_base"] = hex(found.fixed.control_base)
    return dict(stated)


def _show(declared: Declarations, what: str, name: str, board: str | None) -> None:
    if what == "part":
        part(name)  # an unknown part is refused, as an unknown board or shell is
        _write(json.dumps(part_report(name), indent=2))
        return
    if what == "board":
        found = declared.board(name)
        _write(f"board {found.name}: part {found.part} ({declared.origin('boards', name)})")
        for shell in declared.supporting(name):
            entry = declared.shells[shell].boards
            needs = {} if isinstance(entry, str) else dict(entry[name])
            _write(f"  shell {shell}" + (f": {json.dumps(needs)}" if needs else " (any part)"))
        return
    found_shell = declared.shell(name)
    _write(f"shell {name} ({declared.origin('shells', name)})")
    if found_shell.based_on is not None:
        _write(f"  based on {found_shell.based_on}")
    boards: list[str | None]
    if found_shell.any_board:
        boards = [None]
    else:
        boards = [board] if board is not None else list(sorted(found_shell.boards))
    for each in boards:
        _write(f"  board {each or 'any'}:")
        stated = json.dumps(_row(shell_row(name, each, declared)), indent=2)
        _write("\n".join("    " + line for line in stated.splitlines()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finn-platform",
        description="The parts, boards and shells a FINN build may name.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    command = commands.add_parser("list", help="the declarations, each with its origin")
    command.add_argument("kind", nargs="?", choices=KINDS)
    command = commands.add_parser("show", help="a part, a board, or a shell's rows")
    command.add_argument("what", choices=("part", "board", "shell"))
    command.add_argument("name")
    command.add_argument("--board", help="a shell's row on this board only")
    commands.add_parser("check", help="check every declaration, by code")
    args = parser.parse_args(argv)
    try:
        declared = declarations()
        if args.command == "list":
            _list(declared, args.kind)
        elif args.command == "show":
            if args.board is not None and args.what != "shell":
                parser.error("--board is a shell's")
            _show(declared, args.what, args.name, args.board)
        else:
            findings = check(declared)
            for finding in findings:
                _write(str(finding))
            if findings:
                return 1
            counts = ", ".join(f"{len(getattr(declared, kind))} {kind}" for kind in KINDS)
            _write(f"finn-platform: every declaration checks ({counts})")
    except TargetRefused as refused:
        for finding in refused.findings:
            sys.stderr.write(f"finn-platform: {finding}\n")
        return 1
    return 0


__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
