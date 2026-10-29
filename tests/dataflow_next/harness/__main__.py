# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the harness over sources, each in a process of its own:

    python -m dataflow_next.harness SOURCE [SOURCE ...] [--out DIRECTORY] [--tree LABEL]

Before any source runs it prints one header line naming what it runs on: the package's
source directory, the line's ``finn.dataflow`` directory, the sources asked for, and the
git commit and branch of the tree holding the package (``(no git metadata)`` where there is
no ``.git``, as in an archive). ``--tree LABEL`` adds ``tree LABEL`` beside the git state: a
run from an archive passes the commit it was archived from, so the log names it. For each
source it writes ``SOURCE.json`` (the table) and prints its summary; it exits 1 when any
source's table has a failing verdict, or the translation's round trip fails.

Each ``costs`` entry of the table is one priced call: ``target`` (its label), ``verdict``
(its class), ``line_cost`` and ``next_cost`` (a ``steps.Cost`` each, written as its text,
or null where the line has none: ``live_window`` in elements, ``reads`` in source beats,
``narrowest_lanes`` in lanes), ``latency`` (a sentence naming what a larger latency costs,
empty where none is counted) and ``detail`` (a note on the costs or the chain).

The ``fifo`` section compares the FIFO depths the line sizes (``fifo``, its module docstring
the classes and the faults): ``rules`` and ``frames`` (the model's), ``distinct_calls`` and
``calls_made`` (``fifo_sizing.size`` and ``least_depth`` calls, recorded in the source's
process), ``counts`` (each class, zeros included: ``same``, ``smaller``,
``larger_latency_counted``, ``differs``, ``slower_than_period``, ``untranslatable``,
``crashed``, and counted apart, ``not_modelled``, ``raised`` and ``stubbed``: a call made
while a test had stood in for the line's ``least_depth``), ``apart`` (each of the last three
and ``untranslatable`` by its reason), ``coverage`` (the rows compared, by depth and
acceptance) and ``gap`` (what it does not cover, said), ``finding_counts`` and ``findings``
(``network_not_kept``: a network at the line's depths that does not keep the period, with its
``run``, the channels that each restore it raised to unbounded and whether the memory channels
alone do; ``period_other_than_p``; ``slower_than_period``: a call whose bottleneck is above the
period, with the period, the line's depth, the network's ``bottleneck`` and its period
``at_depth``; and ``not_modelled``, counted), ``differences`` (every call answered otherwise
than the line, or slower than the period, with its run in ``detail``), ``faults``, ``joint`` (a
partition's joint minimal set: its ``order``, ``capacities``, ``total``, ``passes`` and whether
it keeps the period, or why it is ``refused``), ``entries`` (one a distinct call: ``key``,
``period``, the ``line``'s answer and the ``package``'s, the class (``verdict``), ``detail``,
its ``depth`` and ``acceptance`` in the coverage and how many times it was ``made``) and
``passed``.

The ``fifo`` section fails a source only on a broken tool (a ``faults`` entry): a part of a call
the package cannot translate; a call with sides only whose run at the line's depth differs from
the line's ``simulate`` and ``within``, word by word, or whose ``least`` differs from the line's
while the bottleneck is not above the period; a crash in the comparison; a result that does not
keep the period by its own criterion. A difference, a finding and a count apart are counted and
listed (each difference with its run), and fail nothing. The positions-based classes of the
other sections fail as they did.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from dataflow_next.harness.sources import PREPARED, forbid_tools
from pathlib import Path


def package_directory(name: str) -> Path:
    """The resolved source directory of the package ``name``, found without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"no package {name}")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def git_state(directory: Path) -> str:
    """The commit, branch and cleanliness of the tree holding ``directory``, read only if a
    ``.git`` (a directory, or a worktree's file) is found at or above it; else
    ``(no git metadata)``."""
    if not any((each / ".git").exists() for each in (directory, *directory.parents)):
        return "(no git metadata)"

    def git(*arguments: str) -> str:
        done = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.stdout.strip() if done.returncode == 0 else ""

    commit = git("rev-parse", "--short=9", "HEAD")
    if not commit:
        return "(no git metadata)"
    branch = git("branch", "--show-current") or "detached"
    changed = len(git("status", "--porcelain").splitlines())
    return f"{commit} on {branch}, {f'dirty, changed paths: {changed}' if changed else 'clean'}"


def header(sources: list[str], tree: str | None = None) -> str:
    """The one line that names what the harness runs on; ``tree``, if given, is the label
    of the tree (an archived commit, say), written beside the git state."""
    package = package_directory("finn.dataflow_next")
    labelled = f"tree {tree}; " if tree else ""
    return (
        f"dataflow_next harness: finn.dataflow_next {package}; "
        f"finn.dataflow {package_directory('finn.dataflow')}; "
        f"sources {', '.join(sources)}; {labelled}git {git_state(package)}"
    )


def one(source: str, out: Path) -> int:
    from dataflow_next.harness import fifo  # noqa: PLC0415
    from dataflow_next.harness.fifo_record import FifoRecorder  # noqa: PLC0415
    from dataflow_next.harness.record import Recorder  # noqa: PLC0415
    from dataflow_next.harness.replay import round_trip, table  # noqa: PLC0415

    run = PREPARED[source]()
    forbid_tools()
    started = time.monotonic()
    with Recorder() as recorder, FifoRecorder() as sizing:
        recorder.source = source
        run()
    recorded = time.monotonic() - started
    calls = list(recorder.recorded())
    trips = round_trip(calls)
    found = table(source, calls)
    sized = fifo.table(sizing.recorded())
    total = time.monotonic() - started
    report = {
        "source": source,
        "seconds_recording": round(recorded, 1),
        "seconds_in_all": round(total, 1),
        "distinct_calls": found.calls,
        "calls_made": sum(call.count for call in calls),
        "round_trip_failures": trips,
        "counts": [list(row) for row in found.counts],
        "totals": found.totals,
        # One entry a priced call. line_cost and next_cost: steps.Cost, live_window in
        # elements, reads in source beats, narrowest_lanes in lanes (written as text).
        "costs": [
            {
                "target": priced.target.label,
                "verdict": priced.verdict,
                "line_cost": priced.line_cost,
                "next_cost": priced.next_cost,
                "latency": priced.latency,
                "detail": priced.note,
            }
            for priced in found.priced
        ],
        "failures": [
            asdict(failure) | {"target": failure.target.label} for failure in found.failures
        ],
        "fifo": sized.report(),
        "passed": found.passed and not trips and sized.passed,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{source}.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    sys.stdout.write(
        f"{source}: {found.calls} distinct calls ({report['calls_made']} made), {found.totals}, "
        f"round trip failures {len(trips)}, {'PASS' if report['passed'] else 'FAIL'}\n"
        f"{source}: {sized.summary()}\n"
    )
    for line in sized.lines():
        sys.stdout.write(f"{source}: {line}\n")
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", choices=sorted(PREPARED))
    parser.add_argument("--out", type=Path, default=Path("/tmp/dataflow-package/harness"))
    parser.add_argument(
        "--tree", metavar="LABEL", help="name the tree in the header, e.g. the archived commit"
    )
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.in_process:
        return one(arguments.sources[0], arguments.out)
    sys.stdout.write(header(arguments.sources, arguments.tree) + "\n")
    sys.stdout.flush()
    failed = 0
    for source in arguments.sources:
        command = [
            sys.executable,
            "-m",
            "dataflow_next.harness",
            source,
            "--out",
            str(arguments.out),
            "--in-process",
        ]
        failed |= subprocess.run(command, check=False).returncode
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
