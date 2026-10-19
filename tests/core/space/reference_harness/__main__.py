# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""From ``tests``: ``python -m core.space.reference_harness`` prints the acceptance matrix as a
markdown table; ``python -m core.space.reference_harness walks --seeds 0-99 --steps 40`` runs
the seeded walks (both index paths unless ``--paths`` says otherwise) and prints their
summary, then each failed walk whole. While a milestone with rows is pending, the engine's
disagreements are counted; ``--without "engine core"`` takes that milestone's rows as removed,
so every disagreement fails its walk (the engine core's engine walked as if it had removed
them). It exits 1 when a walk failed. The walks are imported by their command alone, so the
matrix prints whatever state the walks are in."""

import argparse
import sys

from .matrix import markdown


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.space.reference_harness")
    commands = parser.add_subparsers(dest="command")
    walks = commands.add_parser("walks", help="the seeded walks against the reference model")
    walks.add_argument("--seeds", default="0-9", help="A-B, inclusive, or one seed")
    walks.add_argument("--steps", type=int, default=30)
    walks.add_argument("--paths", default="index,scan", help="index, scan or both")
    walks.add_argument(
        "--without",
        default="",
        help='milestones whose rows are taken as removed, comma-separated ("engine core")',
    )
    arguments = parser.parse_args(argv)
    if arguments.command != "walks":
        print(markdown())
        return 0
    from .walks import seeds, table, walk  # noqa: PLC0415

    paths = [path for path in arguments.paths.split(",") if path]
    without = [item.strip() for item in arguments.without.split(",") if item.strip()]
    tallies = [
        walk(seed, arguments.steps, path, without)
        for seed in seeds(arguments.seeds)
        for path in paths
    ]
    print(table(tallies, without))
    failed = [tally for tally in tallies if tally.failure]
    for tally in failed:
        print(f"\n{tally.failure}")
    return 1 if failed else 0


sys.exit(main(sys.argv[1:]))
