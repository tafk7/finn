#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One isolated measurement. Run each case in a fresh process; stdout is JSON."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import tracemalloc
from typing import cast

import finn.kernels.space as baseline
from finn.kernels.space.occurrence import state
from finn.kernels.space import _self_prototype as candidate


def self_fixture(kind: str, size: int, work: Counter[str]):
    if kind == "chain":
        members: dict[str, object] = {"seed": candidate.Param(int)}
        previous = "seed"
        for index in range(size):

            def make(name: str):
                def step(self: candidate.Space) -> int:
                    work["body"] += 1
                    return cast(int, getattr(self, name)) + 1

                return candidate.derived(step)

            current = f"step{index}"
            members[current] = make(previous)
            previous = current
        family = type("Chain", (candidate.Space,), members)
        return lambda: family(seed=0), lambda point: getattr(point, previous), size

    class Leaf(candidate.Space):
        @candidate.derived
        def value(self) -> int:
            work["leaf_body"] += 1
            return 1

    names = tuple(f"child{i}" for i in range(size))
    members = {name: candidate.Subspace(Leaf) for name in names}
    if kind == "narrow":
        names = names[:1]

    def total(self: candidate.Space) -> int:
        work["aggregate_starts"] += 1
        value = 0
        for name in names:
            work["loop"] += 1
            value += getattr(self, name).value
        return value

    members["total"] = candidate.derived(total)
    family = type("FanIn", (candidate.Space,), members)
    return family, lambda point: point.total, len(names)


def baseline_fixture(kind: str, size: int, work: Counter[str]):
    if kind == "chain":
        previous = baseline.Param(int)
        members: dict[str, object] = {"seed": previous}

        def step(*, value: int) -> int:
            work["body"] += 1
            return value + 1

        for index in range(size):
            current = baseline.Derived(step, aliases={"value": previous})
            members[f"step{index}"] = current
            previous = current
        family = type("ExplicitChain", (baseline.Space,), members)
        return lambda: family(seed=0), lambda point: getattr(point, f"step{size - 1}"), size

    class Leaf(baseline.Space):
        @baseline.derived
        def value() -> int:
            work["leaf_body"] += 1
            return 1

    placements = {f"child{i}": baseline.Subspace(Leaf) for i in range(size)}
    selected = tuple(placements)[:1] if kind == "narrow" else tuple(placements)
    aliases = {f"v{i}": placements[name].ref(Leaf.value) for i, name in enumerate(selected)}
    parameters = ", ".join(f"{name}: int" for name in aliases)
    values = ", ".join(aliases)
    namespace = {"work": work}
    # Signature generation is solely for the matched existing explicit-dependency API.
    exec(
        f"def total(*, {parameters}) -> int:\n"
        "    work['aggregate_starts'] += 1\n"
        "    result = 0\n"
        f"    for value in ({values},):\n"
        "        work['loop'] += 1\n"
        "        result += value\n"
        "    return result\n",
        namespace,
    )
    members = dict(placements)
    members["total"] = baseline.Derived(namespace["total"], aliases=aliases)
    family = type("ExplicitFanIn", (baseline.Space,), members)
    return family, lambda point: point.total, len(selected)


def memory() -> dict[str, int]:
    status = Path("/proc/self/status").read_text()
    entries = {
        line.split(":")[0]: line.split(":")[1].strip().split()[0]
        for line in status.splitlines()
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:"))
    }
    return {
        "rss_bytes": int(entries["VmRSS"]) * 1024,
        "peak_rss_bytes": int(entries["VmHWM"]) * 1024,
        "threads": int(entries["Threads"]),
        "python_threads": threading.active_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler", choices=("baseline", "replay", "recursive", "greenlet"), required=True
    )
    parser.add_argument("--fixture", choices=("chain", "fanin", "narrow"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    work: Counter[str] = Counter()
    before = memory()
    started = time.perf_counter()
    factory = baseline_fixture if args.scheduler == "baseline" else self_fixture
    bind, query, expected = factory(args.fixture, args.size, work)
    point = bind()
    prepare_seconds = time.perf_counter() - started
    prepared_memory = memory()
    if args.memory:
        tracemalloc.start()
    result: dict[str, object] = {
        "scheduler": args.scheduler,
        "fixture": args.fixture,
        "size": args.size,
        "python": sys.version,
        "recursion_limit": sys.getrecursionlimit(),
        "prepare_seconds": prepare_seconds,
        "memory_before": before,
        "memory_prepared": prepared_memory,
        "tracemalloc_enabled": args.memory,
        "processes": 1
        + len(Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()),
        "child_processes": len(
            Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
        ),
    }
    with candidate.using_scheduler("replay" if args.scheduler == "baseline" else args.scheduler):
        started = time.perf_counter()
        try:
            value = query(point)
            result["cold_seconds"] = time.perf_counter() - started
            assert value == expected, (value, expected)
            result["status"] = "available"
            result["value"] = value
            result["work"] = dict(work)
            if args.scheduler != "baseline":
                result["engine_work"] = asdict(point._snapshot.work)
            else:
                result["baseline_dependency_reads"] = sum(
                    len(entry.dependencies) for entry in state(point).snapshot.cache.values()
                )
                result["baseline_callback_starts"] = sum(
                    work[name] for name in ("body", "leaf_body", "aggregate_starts")
                )
            started = time.perf_counter()
            for _ in range(100):
                assert query(point) == expected
            result["hot_seconds_per_query"] = (time.perf_counter() - started) / 100
            result["hot_added_body_work"] = dict(work) != result["work"]
        except Exception as error:
            result["cold_seconds"] = time.perf_counter() - started
            causes = []
            cause: BaseException | None = error
            while cause is not None:
                causes.append(type(cause).__name__)
                cause = cause.__cause__
            result["status"] = "error"
            result["error_types"] = causes
            result["error_message"] = str(error)[:500]
            result["work"] = dict(work)
            if args.scheduler != "baseline":
                result["engine_work"] = asdict(point._snapshot.work)
    gc.collect()
    result["memory_after_query"] = memory()
    if args.memory:
        current, peak = tracemalloc.get_traced_memory()
        result["traced_retained_bytes"] = current
        result["traced_peak_bytes"] = peak
        tracemalloc.stop()
    del point
    gc.collect()
    result["memory_after_discard"] = memory()
    result["resource_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
