#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Fresh-process comparison and separate frame/native-stack attribution."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import tracemalloc
from weakref import ref
from types import GeneratorType

import greenlet
import finn.kernels.space as explicit


def load_api(previous):
    name = "finn.kernels.space._self_prototype"
    if previous is None:
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name, previous)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    explicit._self_prototype = module
    return module


def mixed_fixture(api, is_explicit, depth, prefix, work):
    if is_explicit:

        def leaf() -> int:
            work["leaf_body"] += 1
            return 1
    else:

        def leaf(self) -> int:
            work["leaf_body"] += 1
            return 1

    leaf_family = type("Leaf", (api.Space,), {"value": api.derived(leaf)})
    members = {"seed": api.Param(int)}
    previous = members["seed"] if is_explicit else "seed"
    for index in range(depth):
        if is_explicit:

            def step(*, value: int) -> int:
                work["body"] += 1
                return value + 1

            declaration = api.Derived(step, aliases={"value": previous})
        else:

            def make(name):
                def step(self) -> int:
                    work["body"] += 1
                    return getattr(self, name) + 1

                return api.derived(step)

            declaration = make(previous)
        name = f"step{index}"
        members[name] = declaration
        previous = declaration if is_explicit else name
    names = tuple(f"child{i}" for i in range(prefix))
    placements = {name: api.Subspace(leaf_family) for name in names}
    members.update(placements)
    if is_explicit:
        aliases = {f"v{i}": placements[name].ref(leaf_family.value) for i, name in enumerate(names)}
        aliases["tail"] = previous
        parameters = ", ".join(f"{key}: int" for key in aliases)
        values = ", ".join(key for key in aliases if key != "tail")
        namespace = {"work": work}
        exec(
            f"def total(*, {parameters}) -> int:\n"
            "    work['aggregate_starts'] += 1\n    result = 0\n"
            f"    for value in ({values},):\n"
            "        work['loop'] += 1\n        result += value\n"
            "    return result + tail\n",
            namespace,
        )
        total = api.Derived(namespace["total"], aliases=aliases)
    else:

        def total(self) -> int:
            work["aggregate_starts"] += 1
            value = 0
            for name in names:
                work["loop"] += 1
                value += getattr(self, name).value
            return value + getattr(self, previous)

        total = api.derived(total)
    members["total"] = total
    family = type("Mixed", (api.Space,), members)
    return lambda: family(seed=0), lambda point: point.total, depth + prefix, names


def attach_profile(api, target):
    references, profile = [], {}
    original, driver = greenlet.greenlet, greenlet.getcurrent()

    def factory(*args, **kwargs):
        continuation = original(*args, **kwargs)
        references.append(ref(continuation))
        return continuation

    def trace(event, args):
        if profile or event != "switch" or args[1] is not driver or len(references) < target:
            return
        live = [r() for r in references if r() is not None and not r().dead]
        if len(live) < target:
            return
        counts, shallow, seen = Counter(), Counter(), set()
        for continuation in live:
            current = continuation.gr_frame
            while current is not None:
                if id(current) not in seen:
                    seen.add(id(current))
                    role = (
                        "framework"
                        if current.f_globals.get("__name__") == api.__name__
                        else "author_or_fixture"
                    )
                    counts[f"{role}:{current.f_code.co_name}"] += 1
                    shallow[role] += sys.getsizeof(current)
                current = current.f_back
        engine_frames = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, GeneratorType)
            and obj.gi_code.co_name == "_native_steps"
            and obj.gi_frame is not None
        ]
        profile.update(
            {
                "active_continuations": len(live),
                "saved_native_stack_bytes": sum(g._stack_saved for g in live),
                "frame_counts": dict(counts),
                "frame_shallow_bytes": dict(shallow),
                "engine_generator_count": len(engine_frames),
                "engine_generator_frame_shallow_bytes": sum(
                    sys.getsizeof(g.gi_frame) for g in engine_frames
                ),
                "traced_at_sample": tracemalloc.get_traced_memory(),
                "top_allocations": [
                    str(item) for item in tracemalloc.take_snapshot().statistics("lineno")[:10]
                ],
            }
        )

    greenlet.greenlet = factory
    prior = greenlet.settrace(trace)

    def restore():
        greenlet.settrace(prior)
        greenlet.greenlet = original

    return profile, references, restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("current", "previous", "explicit"), required=True
    )
    parser.add_argument("--previous-source", type=Path)
    parser.add_argument("--fixture", choices=("chain", "fanin", "narrow", "mixed"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--prefix", type=int, default=4096)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.implementation == "previous" and args.previous_source is None:
        parser.error("previous requires --previous-source")
    api = load_api(args.previous_source if args.implementation == "previous" else None)
    benchmark = runpy.run_path("scripts/benchmark-self-space.py")
    memory = benchmark["memory"]
    work = Counter()
    started = time.perf_counter()
    if args.fixture == "mixed":
        bind, query, expected, prewarm = mixed_fixture(
            explicit if args.implementation == "explicit" else api,
            args.implementation == "explicit",
            args.size,
            args.prefix,
            work,
        )
    else:
        factory = (
            benchmark["baseline_fixture"]
            if args.implementation == "explicit"
            else benchmark["self_fixture"]
        )
        bind, query, expected = factory(args.fixture, args.size, work)
        prewarm = ()
    point = bind()
    preparation_seconds = time.perf_counter() - started
    prepared_memory = memory()
    profile, references, restore = {}, [], lambda: None
    rows = []
    with api.using_scheduler("greenlet"):
        for generation in range(args.rounds):
            if generation:
                point = bind()
            for name in prewarm:
                assert getattr(point, name).value == 1
            before_query_memory = memory()
            work_before = dict(work)
            engine_before = (
                asdict(point._snapshot.work) if args.implementation != "explicit" else {}
            )
            if args.trace or args.profile:
                tracemalloc.start()
            if args.profile:
                profile, references, restore = attach_profile(api, args.size)
            snapshot_ref = ref(point._snapshot) if args.implementation != "explicit" else ref(point)
            started = time.perf_counter()
            assert query(point) == expected
            seconds = time.perf_counter() - started
            restore()
            after_cold = dict(work)
            engine_after = asdict(point._snapshot.work) if args.implementation != "explicit" else {}
            started = time.perf_counter()
            for _ in range(100):
                assert query(point) == expected
            hot_seconds = (time.perf_counter() - started) / 100
            assert dict(work) == after_cold
            gc.collect()
            after_query = memory()
            traced = tracemalloc.get_traced_memory() if args.trace or args.profile else None
            if args.trace or args.profile:
                tracemalloc.stop()
            del point
            gc.collect()
            rows.append(
                {
                    "generation": generation,
                    "cold_seconds": seconds,
                    "hot_seconds": hot_seconds,
                    "work_before": work_before,
                    "work_after_cold": after_cold,
                    "query_work": dict(Counter(after_cold) - Counter(work_before)),
                    "engine_before": engine_before,
                    "engine_after_cold": engine_after,
                    "before_query_memory": before_query_memory,
                    "query_engine_work": {
                        key: engine_after[key] - engine_before[key]
                        for key in engine_after
                        if key != "max_pending"
                    },
                    "after_query_memory": after_query,
                    "after_discard_memory": memory(),
                    "snapshot_reclaimed": snapshot_ref() is None
                    if args.implementation != "explicit"
                    else None,
                    "configuration_reclaimed": snapshot_ref() is None
                    if args.implementation == "explicit"
                    else None,
                    "traced_retained_and_peak": traced,
                    "profile": profile,
                    "remaining_profiled_continuations": sum(r() is not None for r in references),
                }
            )
            assert snapshot_ref() is None
    runtime = (
        Path(explicit.__file__).with_name("_runtime.py")
        if args.implementation == "explicit"
        else Path(api.__file__)
    )
    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "fixture": args.fixture,
                "size": args.size,
                "prefix": args.prefix if args.fixture == "mixed" else None,
                "instrumentation": "frame-inspection-and-tracemalloc"
                if args.profile
                else "tracemalloc"
                if args.trace
                else "none",
                "preparation_seconds": preparation_seconds,
                "prepared_memory": prepared_memory,
                "python": sys.version,
                "recursion_limit": sys.getrecursionlimit(),
                "greenlet": greenlet.__version__,
                "runtime_source": str(runtime),
                "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                "candidate_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "checkout_dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
                ),
                "rows": rows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
