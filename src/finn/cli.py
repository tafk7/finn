# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The finn command."""

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="finn")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("doctor", help="report what this environment gives FINN", add_help=False)
    commands.add_parser("resources", help="list, fetch, locate and clean resources", add_help=False)
    args, rest = parser.parse_known_args(argv)
    if args.command == "resources":
        from finn.resources._cli import main as resources  # noqa: PLC0415

        return resources(rest)
    from finn import doctor  # noqa: PLC0415

    return doctor.main(rest)


if __name__ == "__main__":
    sys.exit(main())
