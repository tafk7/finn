# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""From ``tests``: ``python -m core.space.reference_harness`` prints the acceptance matrix as a
markdown table; ``python -m core.space.reference_harness walks --seeds 0-99 --steps 40`` runs
the seeded walks and prints their summary, then each failed walk whole.

The full run (the default) draws both of the rulings' causes the engine has not reached
(DESIGN § 6): each disagreement is attributed by its cause to T4b (a view read in) or T4c (an
optional fact left out) and counted under that milestone while it is pending; an
unattributed one fails its walk. ``--without MILESTONE`` takes its rows as removed, so its
disagreements fail their walks (its engine walked as if it had removed them). ``--strict``
draws neither cause, and every disagreement fails. ``--implementation module:attribute``
(or a registered name; ``today`` by default) names the implementation compared with the
model, asked through ``support.space_oracle``'s interface; the summary names it. It exits 1
when a walk failed. The walks are imported by their command alone, so the matrix prints
whatever state the walks are in."""

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
    walks.add_argument(
        "--strict",
        action="store_true",
        help="draw neither cause (no view read in, every fact given, no unset); any "
        "disagreement fails",
    )
    walks.add_argument(
        "--implementation",
        default="today",
        help="the implementation compared with the model: a registered name, or "
        "module:attribute (support.space_oracle.load)",
    )
    arguments = parser.parse_args(argv)
    if arguments.command != "walks":
        print(markdown())
        return 0
    from support.space_oracle import load  # noqa: PLC0415

    from .walks import seeds, table, walk  # noqa: PLC0415

    implementation = load(arguments.implementation)
    without = [item.strip() for item in arguments.without.split(",") if item.strip()]
    tallies = [
        walk(seed, arguments.steps, without, arguments.strict, implementation)
        for seed in seeds(arguments.seeds)
    ]
    print(table(tallies, without))
    failed = [tally for tally in tallies if tally.failure]
    for tally in failed:
        print(f"\n{tally.failure}")
    return 1 if failed else 0


sys.exit(main(sys.argv[1:]))
