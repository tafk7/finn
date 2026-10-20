# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""From ``tests``: ``python -m core.space.reference_harness`` prints the acceptance matrix as a
markdown table; ``python -m core.space.reference_harness walks --seeds 0-99 --steps 40`` runs
the seeded walks and prints their summary, then each failed walk whole. With no milestone
pending (none is: the engine core's and the facts-instance milestones are accepted), every
disagreement with the engine fails its walk. While a later milestone with rows is pending,
its disagreements are counted; ``--without MILESTONE`` takes its rows as removed, so every
disagreement fails its walk (its engine walked as if it had removed them). It exits 1 when a
walk failed. The walks are imported by their command alone, so the matrix prints whatever
state the walks are in."""

import argparse
import sys

from .matrix import markdown


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.space.reference_harness")
    commands = parser.add_subparsers(dest="command")
    walks = commands.add_parser("walks", help="the seeded walks against the reference model")
    walks.add_argument("--seeds", default="0-9", help="A-B, inclusive, or one seed")
    walks.add_argument("--steps", type=int, default=30)
    walks.add_argument(
        "--without",
        default="",
        help="pending milestones whose rows are taken as removed, comma-separated",
    )
    arguments = parser.parse_args(argv)
    if arguments.command != "walks":
        print(markdown())
        return 0
    from .walks import seeds, table, walk  # noqa: PLC0415

    without = [item.strip() for item in arguments.without.split(",") if item.strip()]
    tallies = [walk(seed, arguments.steps, without) for seed in seeds(arguments.seeds)]
    print(table(tallies, without))
    failed = [tally for tally in tallies if tally.failure]
    for tally in failed:
        print(f"\n{tally.failure}")
    return 1 if failed else 0


sys.exit(main(sys.argv[1:]))
