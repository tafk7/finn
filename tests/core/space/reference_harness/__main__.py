# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""From ``tests``: ``python -m core.space.reference_harness`` prints the acceptance matrix as
a markdown table; ``python -m core.space.reference_harness walks --seeds 0-99 --steps 40``
runs the seeded walks (both index paths unless ``--paths`` says otherwise) and prints their
summary, then each failed walk whole. It exits 1 when a walk failed."""

import argparse
import sys

from .matrix import markdown
from .walks import seeds, table, walk


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.space.reference_harness")
    commands = parser.add_subparsers(dest="command")
    walks = commands.add_parser("walks", help="the seeded walks against the reference model")
    walks.add_argument("--seeds", default="0-9", help="A-B, inclusive, or one seed")
    walks.add_argument("--steps", type=int, default=30)
    walks.add_argument("--paths", default="index,scan", help="index, scan or both")
    arguments = parser.parse_args(argv)
    if arguments.command != "walks":
        print(markdown())
        return 0
    paths = [path for path in arguments.paths.split(",") if path]
    tallies = [
        walk(seed, arguments.steps, path) for seed in seeds(arguments.seeds) for path in paths
    ]
    print(table(tallies))
    failed = [tally for tally in tallies if tally.failure]
    for tally in failed:
        print(f"\n{tally.failure}")
    return 1 if failed else 0


sys.exit(main(sys.argv[1:]))
