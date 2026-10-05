#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the public Space runtime without importing its implementation records.

Example after installing the checkout or setting PYTHONPATH=src:
    python scripts/benchmark-space.py --output /tmp/space-performance.json

Space classes are called to declare nodes, and their design spaces opened with
``design_space(Root(...))``.
Compilation itself is measured through ``finn.core.space.compiler.compile_model``.
Native self-read and real-kernel workloads run in fresh subprocesses; the kernels
are built and configured through ``finn.kernels``' public construction path.
Time and memory are observations, not CI thresholds. Semantic work assertions
check what was evaluated and that discarded point populations release caches.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
import weakref
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def increment(*, value: int) -> int:
    return value + 1


@dataclass(frozen=True)
class CachedPayload:
    value: int
    data: bytes


def source_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(directory.rglob("*.py")):
        digest.update(source.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def repository_state(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def compile_model(api, space_type):
    """The canonical model of a Space class; imported here because it measures compilation."""
    return importlib.import_module(f"{api.__name__}.compiler").compile_model(space_type)


def flat_space(api, count: int):
    source = api.Param(semantics=api.default_semantics(int))
    members = {"source": source}
    for index in range(count):
        members[f"item{index}"] = api.Derived(increment, aliases={"value": source})
    return api.composite("FlatSpace", members)


def repeated_space(api, count: int):
    class Child(api.Space):
        extent: int = api.Param()
        lanes: int = api.Decision(values=(1, 2, 4))
        width = extent * lanes
        physical = api.View(width)

    # Each child is a node declaration: calling the Space class places it once.
    return api.composite(
        "RepeatedSpace", {f"child{index}": Child(extent=16) for index in range(count)}
    )


def deep_space(api, count: int):
    class Leaf(api.Space):
        value = api.Const(1)

    space_type = Leaf
    for index in range(count):
        enabled = api.Const(True)
        space_type = api.composite(
            f"Level{index}", {"enabled": enabled, "child": space_type(when=enabled)}
        )
    return space_type


def compilation(api, label: str, count: int, factory) -> dict[str, object]:
    started = time.perf_counter()
    space_type = factory(api, count)
    author_seconds = time.perf_counter() - started
    gc.collect()
    started = time.perf_counter()
    timed_model = compile_model(api, space_type)
    compile_seconds = time.perf_counter() - started
    counts = asdict(api.inspection.statistics(timed_model))
    del timed_model
    del space_type
    gc.collect()

    # Measure a separate compile, so tracemalloc overhead does not contaminate
    # the reported ordinary compile time. Keep the model alive through GC.
    retained_space_type = factory(api, count)
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    started = time.perf_counter()
    retained_model = compile_model(api, retained_space_type)
    memory_compile_seconds = time.perf_counter() - started
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert asdict(api.inspection.statistics(retained_model)) == counts
    started = time.perf_counter()
    reused_model = compile_model(api, retained_space_type)
    reuse_seconds = time.perf_counter() - started
    assert reused_model is retained_model
    return {
        "fixture": label,
        "size": count,
        "statistics": counts,
        "authoring_seconds": author_seconds,
        "compile_seconds": compile_seconds,
        "traced_compile_seconds": memory_compile_seconds,
        "retained_compile_bytes": current - before,
        "peak_compile_bytes": peak - before,
        "reuse_seconds": reuse_seconds,
        "canonical_reuse": True,
    }


def constant_expressions(api, terms: int, trials: int) -> dict[str, object]:
    """Measure the deliberate tradeoff of runtime evaluation of constant expressions."""
    base = api.Const(7)
    names = tuple(f"term{index}" for index in range(terms))

    def total(self) -> int:
        return sum(getattr(self, name) for name in names)

    space_type = api.composite(
        "ConstantExpressions",
        {
            "base": base,
            "total": api.derived(total),
            **{name: base + index for index, name in enumerate(names)},
        },
    )
    started = time.perf_counter()
    model = compile_model(api, space_type)
    prepare_seconds = time.perf_counter() - started
    expected = 7 * terms + terms * (terms - 1) // 2
    # A formal-free root reuses the prepared model: each design_space only binds.
    points = [api.design_space(space_type()) for _ in range(trials)]
    started = time.perf_counter()
    assert all(point.total == expected for point in points)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    assert all(point.total == expected for point in points)
    warm_seconds = time.perf_counter() - started
    evidence = api.inspection.explain(points[0], space_type.total)
    assert evidence.result == api.Available(expected)
    return {
        "terms": terms,
        "trials": trials,
        "result": expected,
        "prepare_seconds": prepare_seconds,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "demanded_nodes": len(evidence.nodes),
        "statistics": asdict(api.inspection.statistics(model)),
    }


def batch_updates(api, count: int, *, dependent: bool) -> dict[str, object]:
    work = Counter()

    def prerequisite(*, previous: int) -> int:
        work["prerequisite"] += 1
        return previous

    def membership(*, candidate: int, limit: int) -> bool:
        work["membership"] += 1
        return candidate == limit + 1

    source = api.Const(0)
    members = {"source": source}
    previous = source
    requests = []
    for index in range(count):
        limit = api.Derived(prerequisite, aliases={"previous": previous})
        decision = api.Decision(
            domain=api.domain(accepts=membership, limit=limit),
            semantics=api.default_semantics(int),
        )
        members[f"limit{index}"] = limit
        members[f"choice{index}"] = decision
        requests.append((decision, index + 1 if dependent else 1))
        if dependent:
            previous = decision
    space_type = api.composite("DependentBatch" if dependent else "IndependentBatch", members)
    model = compile_model(api, space_type)
    base = api.design_space(space_type())
    changes = [base.field(reference).change(value) for reference, value in reversed(requests)]
    started = time.perf_counter()
    report = base.try_with_choices(*changes)
    seconds = time.perf_counter() - started
    assert report.accepted
    assert len(report.outcomes) == count
    assert work["membership"] == count
    assert work["prerequisite"] == count
    started = time.perf_counter()
    selection = api.selections.capture(report.instance)
    capture_seconds = time.perf_counter() - started
    assert len(selection.entries) == count
    return {
        "fixture": "dependent" if dependent else "independent",
        "changes": count,
        "submitted_order": "reverse declaration order",
        "statistics": asdict(api.inspection.statistics(model)),
        "update_seconds": seconds,
        "membership_calls": work["membership"],
        "prerequisite_calls": work["prerequisite"],
        "captured_choices": len(selection.entries),
        "capture_seconds": capture_seconds,
    }


def replacement_validation(api, count: int) -> dict[str, object]:
    work = Counter()

    def membership(*, candidate: int) -> bool:
        work["membership"] += 1
        return candidate in {0, 1}

    members = {
        f"choice{index}": api.Decision(
            domain=api.domain(accepts=membership), semantics=api.default_semantics(int)
        )
        for index in range(count)
    }
    space_type = api.composite("ReplacementValidation", members)
    base = api.design_space(space_type())
    configured = base.with_choices(
        *(base.field(reference).change(0) for reference in members.values())
    )
    work.clear()
    first = next(iter(members.values()))
    started = time.perf_counter()
    revised = configured.with_choices(configured.field(first).change(1))
    seconds = time.perf_counter() - started
    assert revised is not configured
    assert work["membership"] == count
    return {
        "choices": count,
        "changed": 1,
        "validated": work["membership"],
        "seconds": seconds,
    }


def narrow_query(api, branches: int) -> dict[str, object]:
    work = Counter()

    class Selected(api.Space):
        value: int = api.Param()

        @api.view
        def physical(self) -> int:
            work["selected"] += 1
            return self.value + 1

    class Inactive(api.Space):
        @api.view
        def physical(self) -> int:
            work["inactive"] += 1
            raise AssertionError("an inactive alternative was demanded")

    source = api.Param(semantics=api.default_semantics(int))
    members = {"source": source}
    for index in range(branches):
        # The structural choice is a Decision over nodes; ``choice.physical`` is
        # the selected candidate's member by name (replaces the accepted export).
        choice = api.Decision({"selected": Selected(value=source), "inactive": Inactive()})
        members[f"branch{index}"] = choice
        members[f"output{index}"] = api.View(choice.physical)
    space_type = api.composite("NarrowQuery", members)
    model = compile_model(api, space_type)
    assert work == Counter()
    base = api.design_space(space_type(source=7))
    selector = api.inspection.choices(model)[0].selector
    point = base.with_choices(base.field(selector).change("selected"))
    output = space_type.output0
    started = time.perf_counter()
    answer = point.query(output)
    miss_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cached = point.query(output)
    hit_seconds = time.perf_counter() - started
    assert answer == cached == api.Available(8)
    evidence = api.inspection.explain(point, output)
    assert work["selected"] == 1 and work["inactive"] == 0
    return {
        "branches": branches,
        "alternatives_per_branch": 2,
        "statistics": asdict(api.inspection.statistics(model)),
        "miss_seconds": miss_seconds,
        "hit_seconds": hit_seconds,
        "selected_callbacks": work["selected"],
        "inactive_callbacks": work["inactive"],
        "demanded_nodes": len(evidence.nodes),
        "demanded_edges": sum(len(node.dependencies) for node in evidence.nodes),
    }


def wide_choice(api, alternatives: int, trials: int = 30) -> dict[str, object]:
    work = Counter()

    class Leaf(api.Space):
        @api.view
        def physical(self) -> int:
            work["leaf"] += 1
            return 1

    choice = api.Decision({f"case{index}": Leaf() for index in range(alternatives)})
    space_type = api.composite(
        "WideChoice", {"implementation": choice, "physical": api.View(choice.physical)}
    )
    model = compile_model(api, space_type)
    # A Decision over nodes always has a selector, even with one candidate.
    selector = api.inspection.choices(model)[0].selector
    output = space_type.physical
    measurements = []
    for _ in range(trials):
        base = api.design_space(space_type())
        point = base.with_choices(base.field(selector).change(f"case{alternatives - 1}"))
        started = time.perf_counter()
        answer = point.query(output)
        measurements.append(time.perf_counter() - started)
        assert answer == api.Available(1)
    assert work["leaf"] == trials
    return {
        "alternatives": alternatives,
        "trials": trials,
        "statistics": asdict(api.inspection.statistics(model)),
        "query_seconds_min": min(measurements),
        "query_seconds_median": statistics.median(measurements),
        "query_seconds_max": max(measurements),
        "leaf_callbacks_per_query": work["leaf"] / trials,
        "timing_scope": (
            "cold selected-candidate view query only; start and selector membership excluded"
        ),
    }


def cache_reclamation(api, population: int) -> dict[str, object]:
    created = []
    payload_semantics = api.ValueSemantics.immutable_nominal(CachedPayload)

    def member(*, candidate: int) -> bool:
        return candidate >= 0

    class Example(api.Space):
        choice: int = api.Decision(domain=api.domain(accepts=member))

        @api.derived(semantics=payload_semantics)
        def output(self) -> CachedPayload:
            payload = CachedPayload(self.choice, bytes((self.choice % 256,)) * 1024)
            # Track the callback-created object, not a detached public copy.
            created.append(weakref.ref(payload))
            return payload

    base = api.design_space(Example())
    configurations = []
    started = time.perf_counter()
    for candidate in range(population):
        point = base.with_choices(choice=candidate)
        point.query(Example.output)
        configurations.append(point)
    creation_seconds = time.perf_counter() - started
    del point
    gc.collect()
    while_held = sum(reference() is not None for reference in created)
    assert len(created) == population and while_held == population
    configurations.clear()
    started = time.perf_counter()
    gc.collect()
    reclamation_seconds = time.perf_counter() - started
    after_release = sum(reference() is not None for reference in created)
    assert after_release == 0
    # Both compilation and the original choice-free root are still alive here.
    assert isinstance(base.field(Example.choice).state, api.Available)
    return {
        "population": population,
        "payload_bytes_per_object": 1024,
        "callback_created_objects": len(created),
        "alive_with_configurations_held": while_held,
        "alive_after_configurations_released": after_release,
        "creation_seconds": creation_seconds,
        "collection_seconds_after_release": reclamation_seconds,
        "snapshot_policy": (
            "explicit immutable nominal semantics preserves each callback-created object"
        ),
    }


def rss_bytes() -> int:
    """Current Linux resident set; includes native continuation allocations."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmRSS is required for the native-memory benchmark")


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def self_workload(api, shape: str, depth: int, width: int) -> dict[str, object]:
    """Every graph edge is an ordinary self read, with no declared aliases."""
    work = Counter()
    members = {"source": api.Const(1)}

    def computation(name: str, dependency: str):
        def evaluate(self) -> int:
            work[name] += 1
            return getattr(self, dependency) + 1

        return api.derived(evaluate)

    if shape in ("chain", "mixed"):
        previous = "source"
        for index in range(depth):
            name = f"step{index}"
            members[name] = computation(name, previous)
            previous = name
    if shape in ("fan_in", "mixed"):
        for index in range(width):
            name = f"leaf{index}"
            members[name] = computation(name, "source")

    def output(self) -> int:
        work["output"] += 1
        # Mixed work reads a previously cached wide prefix before starting the
        # cold deep dependency. The prefix must not be replayed on suspension.
        total = (
            sum(getattr(self, f"leaf{index}") for index in range(width)) if shape != "chain" else 0
        )
        return total + (getattr(self, f"step{depth - 1}") if shape != "fan_in" else 0)

    members["output"] = api.view(output)
    space_type = api.composite("Self" + shape.title(), members)
    started = time.perf_counter()
    compile_model(api, space_type)
    prepare_seconds = time.perf_counter() - started
    expected = (2 * width if shape != "chain" else 0) + (depth + 1 if shape != "fan_in" else 0)
    callback_count = 1 + (width if shape != "chain" else 0) + (depth if shape != "fan_in" else 0)

    def prepare_point():
        work.clear()
        point = api.design_space(space_type())
        if shape == "mixed":
            for index in range(width):
                assert getattr(point, f"leaf{index}") == 2
        return point

    point = prepare_point()
    before_rss = rss_bytes()
    started = time.perf_counter()
    assert point.output == expected
    cold_seconds = time.perf_counter() - started
    after_rss = rss_bytes()
    assert len(work) == callback_count and all(value == 1 for value in work.values()), work
    before_hits = work.copy()
    hits = []
    for _ in range(30):
        started = time.perf_counter()
        assert point.output == expected
        hits.append(time.perf_counter() - started)
    assert work == before_hits
    peak = peak_rss_bytes()
    del point
    gc.collect()
    after_release_rss = rss_bytes()

    # Repeat separately for Python allocation measurements. Native process
    # high-water and ordinary timing above exclude allocation tracing.
    point = prepare_point()
    tracemalloc.start()
    assert point.output == expected
    traced_live, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(work) == callback_count and all(value == 1 for value in work.values()), work
    return {
        "shape": shape,
        "depth": depth if shape != "fan_in" else 0,
        "width": width if shape != "chain" else 0,
        "prepare_seconds": prepare_seconds,
        "cold_view_seconds": cold_seconds,
        "cached_view_seconds_median": statistics.median(hits),
        "callback_count_including_cached_prefix": callback_count,
        "cold_callbacks_per_second": (callback_count - (width if shape == "mixed" else 0))
        / cold_seconds,
        "cached_prefix_callbacks": width if shape == "mixed" else 0,
        "rss_before_query_bytes": before_rss,
        "rss_after_query_bytes": after_rss,
        "rss_after_release_bytes": after_release_rss,
        "ordinary_process_peak_rss_bytes": peak,
        "query_traced_live_bytes": traced_live,
        "query_traced_peak_bytes": traced_peak,
        "all_callbacks_run_once": True,
    }


def kernel_fixtures(api):
    """Each kernel workload by name: its root node, the step that configures a point of
    it, and the member path of the module view measured.

    A leaf (the FIFO) is its own root; a kernel on streams (the packed Dotp core,
    MatMul) sits in a root that declares them beside it. Choices are committed as a
    flow commits them, by their stable decision keys (``finn.kernels.commit``).
    """
    kernels = importlib.import_module("finn.kernels")
    FifoKernel, MatMulKernel, PackedDotpKernel, commit = (
        kernels.FifoKernel,
        kernels.MatMulKernel,
        kernels.PackedDotpKernel,
        kernels.commit,
    )
    Channel = importlib.import_module("finn.kernels.channels").Channel
    tensor = importlib.import_module("finn.dataflow.tensor")
    Tensor, ScalarEncoding = tensor.Tensor, tensor.ScalarEncoding
    dtype = importlib.import_module("finn.dataflow.datatypes").resolve_qonnx_datatype_name
    platform = importlib.import_module("finn.kernels.target").Platform(
        period_ns=5.0,
        dsp=kernels.DspBlock.DSP48E2,
        uram=True,
        uram_init=True,
        clk2x=True,
        control_ports=1,
        memory_ports=0,
        aie=False,
    )
    int3, int8 = dtype("INT3"), dtype("INT8")

    def streams(x, w, y):
        return {
            "x": Channel(tensor=x, port="in0_V", platform=platform),
            "w": Channel(tensor=w, port="in1_V", platform=platform),
            "y": Channel(tensor=y, port="out0_V", platform=platform),
        }

    # A dot-product core between three boundary streams.
    dotp_streams = streams(
        Tensor((1, 4), ScalarEncoding(int3)),
        Tensor((4, 4), ScalarEncoding(int3)),
        Tensor((1, 4), ScalarEncoding(int8)),
    )
    dotp_root = api.composite(
        "PlacedDotp",
        {
            **dotp_streams,
            "compute": PackedDotpKernel(
                x_stream=dotp_streams["x"],
                w_stream=dotp_streams["w"],
                y_stream=dotp_streams["y"],
                result_dtype=int8,
                platform=platform,
            ),
        },
    )

    # A MatMul (M=2, K=4, N=4) on streams that carry the tensors it derives.
    exact_result_dtype = importlib.import_module("finn.kernels.matmul").exact_result_dtype
    matmul_streams = streams(
        Tensor((2, 4), ScalarEncoding(int3)),
        Tensor((4, 4), ScalarEncoding(int3)),
        Tensor((2, 4), ScalarEncoding(exact_result_dtype(4, int3, int3))),
    )
    matmul_root = api.composite(
        "PlacedMatMul",
        {
            **matmul_streams,
            "matmul": MatMulKernel(
                m=2,
                n=4,
                k=4,
                activation_dtype=int3,
                weights_dtype=int3,
                platform=platform,
                x_stream=matmul_streams["x"],
                w_stream=matmul_streams["w"],
                y_stream=matmul_streams["y"],
            ),
        },
    )

    return {
        "fifo": (
            FifoKernel(word_bits=16, depth=32, platform=platform),
            lambda point, index: point.with_choices(ram_style=("block", "distributed")[index % 2]),
            "module",
        ),
        "dotp": (
            dotp_root(),
            lambda point, index: commit(
                point,
                {
                    "compute.pe": 2,
                    "compute.simd": 2,
                    "compute.reducer": "tree",
                    "compute.compute_pumping": bool(index % 2),
                },
            ),
            "compute.module",
        ),
        # The compute core is a Decision over kernels; ``packed`` is its entry.
        "matmul": (
            matmul_root(),
            lambda point, index: commit(
                point,
                {
                    "matmul.compute": "packed",
                    "matmul.compute.packed.pe": (1, 2)[index % 2],
                    "matmul.compute.packed.simd": 2,
                    "matmul.compute.packed.compute_pumping": False,
                    "matmul.compute.packed.reducer": "tree",
                },
            ),
            "matmul.compute.module",
        ),
    }


def kernel_workload(api, name: str, trials: int) -> dict[str, object]:
    """Measure repeated replacement, cold/cached accepted module views and retention."""
    node, configure, view = kernel_fixtures(api)[name]
    base = api.design_space(node)

    def accepted(point):
        # The view is a member path below the root: the module the kernel builds.
        target = point
        for part in view.split("."):
            target = getattr(target, part)
        return target

    # Warm compilation/import allocations before reporting exploration costs.
    warm = configure(base, 0)
    accepted(warm)
    del warm
    gc.collect()
    before_rss = rss_bytes()
    times = {"replacement": [], "cold_view": [], "cached_view": []}
    root_refs = []
    point = base
    for index in range(trials):
        started = time.perf_counter()
        point = configure(point, index)
        times["replacement"].append(time.perf_counter() - started)
        root_refs.append(weakref.ref(point))
        started = time.perf_counter()
        result = accepted(point)
        times["cold_view"].append(time.perf_counter() - started)
        assert result.sources
        started = time.perf_counter()
        cached = accepted(point)
        times["cached_view"].append(time.perf_counter() - started)
        assert cached == result
    del point, result, cached
    gc.collect()
    assert all(reference() is None for reference in root_refs)
    repeated_rss = rss_bytes()
    ordinary_peak = peak_rss_bytes()

    # Hold an independent population, then release it with base/model retained.
    tracemalloc.start()
    population = []
    for index in range(trials):
        point = configure(base, index)
        accepted(point)
        population.append(point)
    del point
    gc.collect()
    held_bytes, peak_bytes = tracemalloc.get_traced_memory()
    held_rss = rss_bytes()
    population_refs = [weakref.ref(point) for point in population]
    population.clear()
    gc.collect()
    released_bytes = tracemalloc.get_traced_memory()[0]
    released_rss = rss_bytes()
    tracemalloc.stop()
    assert all(reference() is None for reference in population_refs)
    return {
        "kernel": name,
        "trials": trials,
        "view": view,
        "seconds": {
            label: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for label, values in times.items()
        },
        "configurations_per_second": trials / sum(times["replacement"] + times["cold_view"]),
        "rss_before_exploration_bytes": before_rss,
        "rss_after_repeated_exploration_bytes": repeated_rss,
        "ordinary_process_peak_rss_bytes": ordinary_peak,
        "held_population_rss_bytes": held_rss,
        "released_population_rss_bytes": released_rss,
        "held_population_traced_bytes": held_bytes,
        "peak_population_traced_bytes": peak_bytes,
        "released_population_traced_bytes": released_bytes,
        "retained_old_configuration_count": sum(reference() is not None for reference in root_refs),
        "retained_population_count": sum(reference() is not None for reference in population_refs),
    }


def isolated_workload(kind: str, name: str, sizes: dict[str, int]) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            f"{kind}:{name}",
            "--worker-config",
            json.dumps(sizes),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"{kind}:{name} failed:\n{result.stdout}{result.stderr}")
    return json.loads(result.stdout)


def markdown(report: dict[str, object]) -> str:
    environment = report["environment"]
    lines = [
        "# Space performance measurements",
        "",
        f"Recorded {report['recorded_at_utc']}; suite `{report['suite']}`. "
        "All semantic work assertions passed.",
        "",
        f"Python: `{environment['python']}`; greenlet `{environment['greenlet']}`.",
        f"Platform: `{environment['platform']}`.",
        f"Source revision: `{environment['repository']['revision']}`; "
        f"dirty: `{environment['repository']['dirty']}`.",
        f"Space source SHA-256: `{environment['space_source_sha256']}`.",
        "",
    ]
    if "compilation" in report:
        lines += [
            "## Preparation",
            "",
            "Ordinary timing excludes allocation tracing. Memory comes from a separate "
            "preparation with declarations already allocated and the prepared definition held. "
            "Traced Python allocation excludes native continuation stacks and RSS.",
            "",
            "| Fixture | Size | Nodes | Known edges | Prepare s | Reuse µs | "
            "Retained MiB | Peak MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report["compilation"]:
            counts = item["statistics"]
            lines.append(
                f"| {item['fixture']} | {item['size']} | {counts['nodes']} | "
                f"{counts['potential_edges']} | {item['compile_seconds']:.6f} | "
                f"{item['reuse_seconds'] * 1e6:.2f} | "
                f"{item['retained_compile_bytes'] / 2**20:.3f} | "
                f"{item['peak_compile_bytes'] / 2**20:.3f} |"
            )
        lines += [
            "",
            "## Admission and query work",
            "",
            "Both explicit-input batches use reverse declaration order.",
            "",
            "| Batch | Changes | Membership calls | Prerequisite calls | Commit s | Capture s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in report["batch_updates"]:
            lines.append(
                f"| {item['fixture']} | {item['changes']} | {item['membership_calls']} | "
                f"{item['prerequisite_calls']} | {item['update_seconds']:.6f} | "
                f"{item['capture_seconds']:.6f} |"
            )
        replacement = report["replacement_validation"]
        lines += [
            "",
            f"Replacing 1 of {replacement['choices']} choices revalidated "
            f"{replacement['validated']} memberships in {replacement['seconds']:.6f} s.",
            "",
            "| Choices | Selected callbacks | Inactive callbacks | "
            "Demanded nodes | Edges | Cold µs | Cached µs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report["narrow_queries"]:
            lines.append(
                f"| {item['branches']} | {item['selected_callbacks']} | "
                f"{item['inactive_callbacks']} | {item['demanded_nodes']} | "
                f"{item['demanded_edges']} | "
                f"{item['miss_seconds'] * 1e6:.2f} | {item['hit_seconds'] * 1e6:.2f} |"
            )
        lines += [
            "",
            "| Alternatives | Median cold µs | Minimum µs | Maximum µs | Calls/query |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in report["wide_choices"]:
            lines.append(
                f"| {item['alternatives']} | {item['query_seconds_median'] * 1e6:.2f} | "
                f"{item['query_seconds_min'] * 1e6:.2f} | {item['query_seconds_max'] * 1e6:.2f} | "
                f"{item['leaf_callbacks_per_query']:g} |"
            )
        cache = report["cache_reclamation"]
        lines += [
            "",
            f"Cache reclamation tracked {cache['callback_created_objects']} "
            f"callback-created immutable payloads of {cache['payload_bytes_per_object']} "
            f"bytes each. {cache['alive_with_configurations_held']} were alive with "
            f"configurations held; {cache['alive_after_configurations_released']} remained "
            "after release and collection, with the base and prepared definition retained.",
            "",
        ]
    if "self_workloads" in report:
        lines += [
            "## Ordinary self reads",
            "",
            "Each scenario runs in a fresh subprocess. Every callback runs once, and cached "
            "view calls run no callbacks. Mixed work warms the wide prefix before a cold "
            "view reads that prefix and enters a deep chain.",
            "",
            "| Work | Depth | Width | Cold view s | Cached µs | Callback count | "
            "Process peak MiB | Traced query peak MiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report["self_workloads"]:
            lines.append(
                f"| {item['shape']} | {item['depth']} | {item['width']} | "
                f"{item['cold_view_seconds']:.6f} | "
                f"{item['cached_view_seconds_median'] * 1e6:.2f} | "
                f"{item['callback_count_including_cached_prefix']} | "
                f"{item['ordinary_process_peak_rss_bytes'] / 2**20:.3f} | "
                f"{item['query_traced_peak_bytes'] / 2**20:.3f} |"
            )
        lines.append("")
    if "kernel_workloads" in report:
        lines += [
            "## Repeated kernel configurations",
            "",
            "Choices alternate on each immutable replacement, committed by their decision "
            "keys. Each measures the module view of the kernel it configures: the FIFO's own, "
            "the packed Dotp core's on its streams, and MatMul's selected compute core's. "
            "Timings exclude source rendering and hardware execution. A separate "
            "allocation run holds an independent population and then releases it.",
            "",
            "| Kernel | Trials | Replacement µs | Cold view µs | Cached view µs | Configs/s | "
            "Held traced MiB | Released traced MiB | Old configs retained |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report["kernel_workloads"]:
            times = item["seconds"]
            lines.append(
                f"| {item['kernel']} | {item['trials']} | "
                f"{times['replacement']['median'] * 1e6:.2f} | "
                f"{times['cold_view']['median'] * 1e6:.2f} | "
                f"{times['cached_view']['median'] * 1e6:.2f} | "
                f"{item['configurations_per_second']:.1f} | "
                f"{item['held_population_traced_bytes'] / 2**20:.3f} | "
                f"{item['released_population_traced_bytes'] / 2**20:.3f} | "
                f"{item['retained_old_configuration_count']} |"
            )
        lines.append("")
    if "constant_expressions" in report:
        item = report["constant_expressions"]
        lines += [
            "## Constant expressions",
            "",
            f"{item['terms']} constant terms across {item['trials']} configurations: "
            f"prepare {item['prepare_seconds']:.6f} s; cold reads {item['cold_seconds']:.6f} s; "
            f"warm reads {item['warm_seconds']:.6f} s. "
            f"The first query demanded {item['demanded_nodes']} nodes.",
            "",
            "Removing folding intentionally adds runtime expression work. Compare this "
            "scenario separately from parameter-based expressions, and preserve operand "
            "validation and inactive arithmetic suppression.",
            "",
        ]
    lines += [
        "## Interpretation and limits",
        "",
        "Timings are observations on this machine, not CI thresholds. No cross-snapshot cache "
        "sharing is assumed. Runtime/kernel subprocess RSS includes the interpreter, imports, "
        "prepared definitions and native allocations. Its ordinary process peak is captured "
        "before the separate tracing run; it is not an attribution of all bytes to greenlets. "
        "Current RSS can stay high after objects are reclaimed because allocators retain pages. "
        "Weak-reference assertions establish configuration/payload reclamation, not complete "
        "release of the process resident set. Deep scope names retain their existing storage cost.",
        "",
        "JSON contains per-scenario current/peak RSS, traced allocation, throughput and "
        "minimum/median/maximum view timings. Native stack peaks are visible in process RSS, "
        "not necessarily in tracemalloc. These workloads supplement semantic failure, cleanup "
        "and cancellation tests; they do not prove a universal memory bound.",
        "",
        "Reproduce with `scripts/benchmark-space.py --suite all --output <report.json> "
        "--markdown <PERFORMANCE.md>` and the sizes in the JSON configuration.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "generic", "runtime", "kernels"), default="all")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--flat", type=int, default=5000)
    parser.add_argument("--children", type=int, default=500)
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--branches", type=int, default=500)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--self-depth", type=int, default=20000)
    parser.add_argument("--fan-in", type=int, default=2000)
    parser.add_argument("--kernel-trials", type=int, default=100)
    arguments = parser.parse_args()
    api = importlib.import_module("finn.core.space")
    if arguments.worker:
        kind, name = arguments.worker.split(":", 1)
        sizes = json.loads(arguments.worker_config)
        result = (
            self_workload(api, name, sizes["self_depth"], sizes["fan_in"])
            if kind == "runtime"
            else kernel_workload(api, name, sizes["kernel_trials"])
        )
        print(json.dumps(result))
        return
    if arguments.output is None:
        parser.error("--output is required")
    sizes = {
        name: getattr(arguments, name)
        for name in (
            "flat",
            "children",
            "depth",
            "batch",
            "branches",
            "population",
            "self_depth",
            "fan_in",
            "kernel_trials",
        )
    }
    if any(value < 1 for value in sizes.values()):
        parser.error("fixture sizes must be positive")
    compile_model(api, api.composite("Warmup", {}))
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 2,
        "suite": arguments.suite,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "module": "finn.core.space",
            "greenlet": importlib.import_module("greenlet").__version__,
            "repository": repository_state(root),
            "space_source_sha256": source_digest(Path(api.__file__).parent),
            "kernel_source_sha256": source_digest(root / "src/finn/kernels"),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "configuration": sizes,
    }
    if arguments.suite in ("all", "generic"):
        report.update(
            {
                "compilation": [
                    compilation(api, "flat", arguments.flat, flat_space),
                    compilation(api, "repeated children", arguments.children, repeated_space),
                    compilation(api, "guarded scope depth", arguments.depth, deep_space),
                ],
                "constant_expressions": constant_expressions(
                    api, arguments.batch, arguments.population
                ),
                "batch_updates": [
                    batch_updates(api, arguments.batch, dependent=False),
                    batch_updates(api, arguments.batch, dependent=True),
                ],
                "replacement_validation": replacement_validation(api, arguments.batch),
                "narrow_queries": [narrow_query(api, 1), narrow_query(api, arguments.branches)],
                "wide_choices": [wide_choice(api, 2), wide_choice(api, max(2, arguments.branches))],
                "cache_reclamation": cache_reclamation(api, arguments.population),
            }
        )
        first, second = report["narrow_queries"]
        assert first["demanded_nodes"] == second["demanded_nodes"]
        assert first["demanded_edges"] == second["demanded_edges"]
    if arguments.suite in ("all", "runtime"):
        report["self_workloads"] = [
            isolated_workload("runtime", name, sizes) for name in ("chain", "fan_in", "mixed")
        ]
    if arguments.suite in ("all", "kernels"):
        report["kernel_workloads"] = [
            isolated_workload("kernels", name, sizes) for name in ("fifo", "dotp", "matmul")
        ]
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(markdown(report))
    print(json.dumps({"output": str(arguments.output), "semantic_assertions": "passed"}))


if __name__ == "__main__":
    main()
