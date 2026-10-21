# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle's tool, run from ``tests`` with ``finn`` importable:

    python -m support.space_oracle dump TARGET OUT.json [--implementation M:A] [--permuted]
    python -m support.space_oracle diff A.json B.json
    python -m support.space_oracle walk TARGET --seeds A-B --steps N --against M:A
        [--implementation M:A] [--permuted]

``dump`` writes every station of ``TARGET`` (``targets.TARGETS``: toys, chain-8,
chain-64, tfc) in its canonical form by key (``forms``), as one implementation answers it
(today's by default): each design space, and its change's report and successor. The
file's first line is its header: the commit of the tree ``finn`` is imported from
(``git rev-parse HEAD``, and whether its ``src`` has changes), the implementation, the
target and the numbering. ``--permuted`` sets ``FINN_SPACE_PERMUTE=1`` before any model
is made.

``diff`` prints every disagreement of two dumps by key, their headers apart, and exits 1
on any. ``walk`` walks every station of ``TARGET`` with each seed, ``--implementation``
(today's by default) against ``--against``, and exits 1 on any disagreement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PERMUTE = "FINN_SPACE_PERMUTE"
"""The engine's permuted numbering, read when a model is made (``_structure.PERMUTE``)."""


def _tree() -> dict[str, object]:
    """The tree ``finn`` is imported from: its root, ``HEAD``, and whether ``src`` has
    changes."""
    import finn.core.space  # noqa: PLC0415 - imported after the numbering is set

    root = Path(finn.core.space.__file__).resolve().parents[4]

    def git(*args: str) -> str:
        found = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return (
            found.stdout.strip() if found.returncode == 0 else f"unknown ({found.stderr.strip()})"
        )

    return {
        "commit": git("rev-parse", "HEAD"),
        "src changed": bool(git("status", "--porcelain", "--", "src")),
        "tree": str(root),
    }


def dump(target: str, out: Path, implementation: str, permuted: bool) -> int:
    from support.space_oracle import load  # noqa: PLC0415 - after the numbering is set
    from support.space_oracle.targets import dumped, stations  # noqa: PLC0415

    chosen = load(implementation)
    header = {
        "finn": _tree(),
        "implementation": chosen.name,
        "target": target,
        "permuted": permuted,
    }
    body: dict[str, Any] = {}
    for station in stations(target):
        body[station.label] = found = dumped(station, chosen)
        print(f"{station.label}: {len(found['space']['members'])} members", flush=True)
    with out.open("w") as file:
        file.write('{"header": ' + json.dumps(header, sort_keys=True) + ",\n")
        file.write(' "stations": ' + json.dumps(body, sort_keys=True, indent=1) + "}\n")
    print(f"wrote {out}: {header}")
    return 0


def diff(first: Path, second: Path) -> int:
    from support.space_oracle import forms  # noqa: PLC0415

    left, right = json.loads(first.read_text()), json.loads(second.read_text())
    for key in sorted(left["header"].keys() | right["header"].keys()):
        if left["header"].get(key) != right["header"].get(key):
            print(f"header {key}: {left['header'].get(key)} vs {right['header'].get(key)}")
    found = forms.differences(left["stations"], right["stations"])
    for difference in found:
        print(difference)
    kinds = Counter(difference.kind for difference in found)
    print(f"disagreements: {len(found)} {dict(sorted(kinds.items()))}")
    return 1 if found else 0


def _seeds(text: str) -> range:
    first, dash, last = text.partition("-")
    return range(int(first), int(last if dash else first) + 1)


def walk(target: str, seeds: range, steps: int, base: str, against: str) -> int:
    from support.space_oracle import compare, load  # noqa: PLC0415
    from support.space_oracle.targets import stations  # noqa: PLC0415

    left, right = load(base), load(against)
    print(f"finn: {_tree()}; {left.name} against {right.name}")
    failed = 0
    for station in stations(target):
        for seed in seeds:
            found = compare.walk(
                station.make, station.facts, seed, steps, left, right, station.viable
            )
            failed += bool(found.disagreements)
            kinds, tally = dict(sorted(found.by_kind.items())), dict(sorted(found.tally.items()))
            print(
                f"{station.label} seed {seed}: {len(found.disagreements)} disagreements "
                f"{kinds}; compared {tally}"
            )
            for line in found.lines(10):
                print("    " + line)
    print(f"walks with disagreements: {failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m support.space_oracle")
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("dump", help="a target's answers by key, to a file")
    made.add_argument("target")
    made.add_argument("out", type=Path)
    made.add_argument("--implementation", default="today")
    made.add_argument("--permuted", action="store_true")
    compared = commands.add_parser("diff", help="two dumps' disagreements by key")
    compared.add_argument("first", type=Path)
    compared.add_argument("second", type=Path)
    walked = commands.add_parser("walk", help="seeded walks of a target, two implementations")
    walked.add_argument("target")
    walked.add_argument("--seeds", type=_seeds, default=_seeds("0-1"))
    walked.add_argument("--steps", type=int, default=20)
    walked.add_argument("--against", required=True)
    walked.add_argument("--implementation", default="today")
    walked.add_argument("--permuted", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "permuted", False):
        os.environ[PERMUTE] = "1"  # before any model is made
    if args.command == "dump":
        return dump(args.target, args.out, args.implementation, args.permuted)
    if args.command == "diff":
        return diff(args.first, args.second)
    return walk(args.target, args.seeds, args.steps, args.implementation, args.against)


if __name__ == "__main__":
    sys.exit(main())
