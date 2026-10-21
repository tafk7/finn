#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the public Space runtime without importing its implementation records.

Example after installing the checkout or setting PYTHONPATH=src:
    python scripts/benchmark-space.py --output /tmp/space-performance.json

Space classes are called to declare nodes, and their design spaces opened with
``design_space(Root(...))``; a kernel is its own root, as a network binds it.
Compilation itself is measured through ``finn.core.space.compiler.compile_model``.
Native self-read and real-kernel workloads run in fresh subprocesses; the kernels
are built and configured through ``finn.kernels``' public construction path.
Time and memory are observations, not CI thresholds. Semantic work assertions
check what was evaluated and that discarded point populations release caches. The
memo's calls and hits are read through a compiled model's evaluator: every
configuration of a model shares the memo, so an authored function is called once for
the values it reads, and later configurations find its outcome. Run with
FINN_SPACE_CHECK_PURITY=1, each memo hit runs its function again; the assertions hold
both ways.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
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

# The purity check (FINN_SPACE_CHECK_PURITY=1) runs an authored function again at each
# memo hit, to compare the answers: it then runs once per call and once per hit.
PURITY_CHECKED = os.environ.get("FINN_SPACE_CHECK_PURITY") == "1"


def memo_work(model, *kinds: str) -> dict[str, dict[str, int]]:
    """The memo's calls and hits of each kind of authored function in ``model`` (a
    node's kind, or a domain's membership, listing, enumeration or order; every kind
    counted where none is named), since its counters were last reset. Every
    configuration of a model shares its memo: an authored function is called again only
    where what it reads answers differently."""
    counters = model.evaluator.counters
    kinds = kinds or tuple(sorted({*counters.calls, *counters.hits}))
    return {
        "calls": {kind: counters.calls.get(kind, 0) for kind in kinds},
        "hits": {kind: counters.hits.get(kind, 0) for kind in kinds},
    }


def memo_traces_kept(api) -> int:
    """How many outcomes a model's memo keeps of each authored function, the newest."""
    return importlib.import_module(f"{api.__name__}._memo").TRACES_KEPT


def ran(memo: dict[str, dict[str, int]], **kinds: str) -> dict[str, int]:
    """How often each named authored function ran, by its kind's calls in ``memo_work``
    (and its hits too under the purity check)."""
    return {
        name: memo["calls"][kind] + (memo["hits"][kind] if PURITY_CHECKED else 0)
        for name, kind in kinds.items()
    }


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
    model.evaluator.counters.reset()
    points = [api.design_space(space_type()) for _ in range(trials)]
    started = time.perf_counter()
    assert all(point.total == expected for point in points)
    cold_seconds = time.perf_counter() - started
    memo = memo_work(model, "derived")
    started = time.perf_counter()
    assert all(point.total == expected for point in points)
    warm_seconds = time.perf_counter() - started
    evidence = api.inspection.explain(points[0], space_type.total)
    assert evidence.result == api.Available(expected)
    # The configurations share the model's memo: each term and the total is called once,
    # by the first configuration read; every later one finds them.
    assert memo["calls"]["derived"] == terms + 1, memo
    return {
        "terms": terms,
        "trials": trials,
        "result": expected,
        "prepare_seconds": prepare_seconds,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "demanded_nodes": len(evidence.nodes),
        "statistics": asdict(api.inspection.statistics(model)),
        "memo": memo,
    }


def batch_updates(api, count: int, *, dependent: bool) -> dict[str, object]:
    work = Counter()

    def prerequisite(*, previous: int) -> int:
        work["prerequisite"] += 1
        return previous

    def membership(*, candidate: int) -> bool:
        work["membership"] += 1
        return candidate >= 1

    def follows(*, choice: int, limit: int) -> bool:
        work["rule"] += 1
        return choice == limit + 1

    # A domain's membership reads facts only: how a choice follows the one before it
    # is a rule of the Space holding both.
    source = api.Const(0)
    members = {"source": source}
    previous = source
    requests = []
    for index in range(count):
        limit = api.Derived(prerequisite, aliases={"previous": previous})
        decision = api.Decision(
            domain=api.domain(accepts=membership),
            semantics=api.default_semantics(int),
        )
        members[f"limit{index}"] = limit
        members[f"choice{index}"] = decision
        members[f"follows{index}"] = api.Constraint(
            follows, aliases={"choice": decision, "limit": limit}
        )
        requests.append((decision, index + 1 if dependent else 1))
        if dependent:
            previous = decision
    space_type = api.composite("DependentBatch" if dependent else "IndependentBatch", members)
    model = compile_model(api, space_type)
    base = api.design_space(space_type())
    changes = [base.field(reference).change(value) for reference, value in reversed(requests)]
    model.evaluator.counters.reset()
    work.clear()
    started = time.perf_counter()
    report = base.try_with_choices(*changes)
    seconds = time.perf_counter() - started
    assert report.accepted
    assert len(report.outcomes) == count
    memo = memo_work(model, "domain membership", "derived", "constraint")
    # The memo keeps each function's outcomes apart: each choice's membership, each limit
    # and each rule is called once, on the one value it reads once the batch is in (the
    # old bound, one evaluation each, holds). What else reads them finds them in the memo.
    assert memo["calls"] == {"domain membership": count, "derived": count, "constraint": count}
    assert work == ran(
        memo, membership="domain membership", prerequisite="derived", rule="constraint"
    )
    return {
        "fixture": "dependent" if dependent else "independent",
        "changes": count,
        "submitted_order": "reverse declaration order",
        "statistics": asdict(api.inspection.statistics(model)),
        "update_seconds": seconds,
        "membership_calls": memo["calls"]["domain membership"],
        "prerequisite_calls": memo["calls"]["derived"],
        "rule_calls": memo["calls"]["constraint"],
        "memo": memo,
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
    model = api.inspection.model(base)
    model.evaluator.counters.reset()
    work.clear()
    first = next(iter(members.values()))
    started = time.perf_counter()
    revised = configured.with_choices(configured.field(first).change(1))
    seconds = time.perf_counter() - started
    assert revised is not configured
    memo = memo_work(model, "domain membership")
    # Only the changed choice is admitted again: no retained choice's admission read it,
    # and its new value was never asked before.
    assert memo["calls"]["domain membership"] == 1, memo
    assert work == ran(memo, membership="domain membership")
    return {
        "choices": count,
        "changed": 1,
        "validated": memo["calls"]["domain membership"],
        "memo": memo,
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
    memo = {"calls": {"derived": 0}, "hits": {"derived": 0}}
    for _ in range(trials):
        base = api.design_space(space_type())
        point = base.with_choices(base.field(selector).change(f"case{alternatives - 1}"))
        model.evaluator.counters.reset()
        before = work["leaf"]
        started = time.perf_counter()
        answer = point.query(output)
        measurements.append(time.perf_counter() - started)
        assert answer == api.Available(1)
        query = memo_work(model, "derived")
        # The query evaluates one authored function, the selected candidate's view: called
        # or found in the memo, never an inactive alternative's.
        assert query["calls"]["derived"] + query["hits"]["derived"] == 1
        assert work["leaf"] - before == ran(query, leaf="derived")["leaf"]
        for count in ("calls", "hits"):
            memo[count]["derived"] += query[count]["derived"]
    # Every binding of the model shares its memo: the selected view, reading nothing that
    # changes between bindings, is called once in all, and found by every later query.
    assert memo["calls"]["derived"] == 1
    return {
        "alternatives": alternatives,
        "trials": trials,
        "statistics": asdict(api.inspection.statistics(model)),
        "query_seconds_min": min(measurements),
        "query_seconds_median": statistics.median(measurements),
        "query_seconds_max": max(measurements),
        "leaf_callbacks_per_query": (memo["calls"]["derived"] + memo["hits"]["derived"]) / trials,
        "memo": memo,
        "timing_scope": (
            "selected-candidate view query on a fresh binding; start and selector membership "
            "excluded; the first query calls the view, every later one finds it in the memo"
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
    alive = [index for index, reference in enumerate(created) if reference() is not None]
    # Released configurations hold nothing; the model's memo keeps the newest outcomes of
    # each function (``TRACES_KEPT``), each holding its payload, for later bindings to find.
    kept = min(population, memo_traces_kept(api))
    assert alive == list(range(population - kept, population))
    # Nothing else holds them: once the memo drops its outcomes, every payload is collected.
    model = api.inspection.model(base)
    budget = model.evaluator.memo.budget
    model.evaluator.memo.budget = 0
    gc.collect()
    after_memo = sum(reference() is not None for reference in created)
    model.evaluator.memo.budget = budget
    assert after_memo == 0
    # Both compilation and the original choice-free root are still alive here.
    assert isinstance(base.field(Example.choice).state, api.Available)
    return {
        "population": population,
        "payload_bytes_per_object": 1024,
        "callback_created_objects": len(created),
        "alive_with_configurations_held": while_held,
        "alive_after_configurations_released": len(alive),
        "alive_after_memo_released": after_memo,
        "memo_traces_kept": kept,
        "creation_seconds": creation_seconds,
        "collection_seconds_after_release": reclamation_seconds,
        "snapshot_policy": (
            "explicit immutable nominal semantics preserves each callback-created object"
        ),
    }


def rss_bytes() -> int:
    """Current Linux resident set; includes native allocations."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmRSS is required for the native-memory benchmark")


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def self_workload(api, shape: str, depth: int, width: int) -> dict[str, object]:
    """Every graph edge is an ordinary self read, with no declared aliases.

    A callback counts its completions: past the engine's depth guard a method is
    abandoned and run again on an explicit stack, so it may start more than once."""
    work = Counter()
    members = {"source": api.Const(1)}

    def computation(name: str, dependency: str):
        def evaluate(self) -> int:
            value = getattr(self, dependency) + 1
            work[name] += 1
            return value

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
        # Mixed work reads a previously cached wide prefix before starting the cold deep
        # dependency. Where the depth guard runs this again, the prefix is read from the
        # world's entries: its callbacks never run again.
        total = (
            sum(getattr(self, f"leaf{index}") for index in range(width)) if shape != "chain" else 0
        )
        total += getattr(self, f"step{depth - 1}") if shape != "fan_in" else 0
        work["output"] += 1
        return total

    members["output"] = api.view(output)
    space_type = api.composite("Self" + shape.title(), members)
    started = time.perf_counter()
    model = compile_model(api, space_type)
    prepare_seconds = time.perf_counter() - started
    expected = (2 * width if shape != "chain" else 0) + (depth + 1 if shape != "fan_in" else 0)
    callback_count = 1 + (width if shape != "chain" else 0) + (depth if shape != "fan_in" else 0)

    def completed_once() -> None:
        # Each callback completes once. Under the purity check, one the depth guard
        # abandoned is run again once it completes, and the two answers compared.
        assert len(work) == callback_count, work
        assert all(value == 1 or (PURITY_CHECKED and value > 1) for value in work.values()), work

    def prepare_point():
        # Every binding of a model shares its memo: drop it, so each measured query is
        # cold, every callback called.
        budget = model.evaluator.memo.budget
        model.evaluator.memo.budget = 0
        model.evaluator.memo.budget = budget
        work.clear()
        point = api.design_space(space_type())
        if shape == "mixed":
            for index in range(width):
                assert getattr(point, f"leaf{index}") == 2
        return point

    point = prepare_point()
    model.evaluator.counters.reset()
    before_rss = rss_bytes()
    started = time.perf_counter()
    assert point.output == expected
    cold_seconds = time.perf_counter() - started
    after_rss = rss_bytes()
    memo = memo_work(model, "derived")
    completed_once()
    # Each cold callback is called, never found in the memo; one the depth guard
    # abandoned is called again.
    cold = callback_count - (width if shape == "mixed" else 0)
    assert memo["calls"]["derived"] >= cold and memo["hits"]["derived"] == 0, memo
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
    completed_once()
    return {
        "shape": shape,
        "depth": depth if shape != "fan_in" else 0,
        "width": width if shape != "chain" else 0,
        "prepare_seconds": prepare_seconds,
        "cold_view_seconds": cold_seconds,
        "cached_view_seconds_median": statistics.median(hits),
        "callback_count_including_cached_prefix": callback_count,
        "cold_callbacks_per_second": cold / cold_seconds,
        "memo": memo,
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
    it, and the member path of the netlist view measured (the library module the kernel
    builds, at its empty label).

    Each kernel is its own root, bound as a network binds it (``finn.kernels.network``):
    the FIFO; the packed Dotp core and MatMul on the tensors their channels carry.
    Choices are committed as a flow commits them, by their stable decision keys
    (``finn.kernels.configure.commit``).
    """
    FifoKernel = importlib.import_module("finn.kernels.fifo").FifoKernel
    MatMulKernel = importlib.import_module("finn.kernels.matmul").MatMulKernel
    PackedDotpKernel = importlib.import_module("finn.kernels.dotp").PackedDotpKernel
    commit = importlib.import_module("finn.kernels.configure").commit
    target = importlib.import_module("finn.platform.capabilities")
    Fabric = target.Fabric
    tensor = importlib.import_module("finn.dataflow.tensor")
    Tensor, ScalarEncoding = tensor.Tensor, tensor.ScalarEncoding
    traversal = importlib.import_module("finn.dataflow.traversal")
    BeatSequence, vector_major = traversal.BeatSequence, traversal.vector_major
    dtype = importlib.import_module("finn.dataflow.datatypes").resolve_qonnx_datatype_name
    platform = target.Platform(
        period_ns=5.0,
        dsp=target.DspBlock.DSP48E2,
        fabric=Fabric.ULTRASCALE,
        uram=True,
        uram_init=True,
        resources=None,
    )
    int3, int8 = dtype("INT3"), dtype("INT8")
    # The FIFO's words: 32 of 16 bits, in order.
    words = Tensor((32, 1), ScalarEncoding(dtype("INT16")))

    # A dot-product core on three tensors.
    dotp_tensors = (
        Tensor((1, 4), ScalarEncoding(int3)),
        Tensor((4, 4), ScalarEncoding(int3)),
        Tensor((1, 4), ScalarEncoding(int8)),
    )

    # A MatMul (M=2, K=4, N=4) on the tensors it derives.
    datatype_range = importlib.import_module("finn.kernels.matmul").datatype_range
    range_dtype = importlib.import_module("finn.kernels.values.domains").range_dtype
    matmul_tensors = (
        Tensor((2, 4), ScalarEncoding(int3)),
        Tensor((4, 4), ScalarEncoding(int3)),
        Tensor(
            (2, 4),
            ScalarEncoding(
                range_dtype(*datatype_range(4, ScalarEncoding(int3), ScalarEncoding(int3)))
            ),
        ),
    )

    return {
        "fifo": (
            FifoKernel(
                input_tensor=words,
                arriving=BeatSequence(vector_major((32, 1), 1)),
                depth=32,
                platform=platform,
            ),
            lambda point, index: point.with_choices(ram_style=("block", "distributed")[index % 2]),
            "netlist",
        ),
        "dotp": (
            PackedDotpKernel(
                x_tensor=dotp_tensors[0],
                w_tensor=dotp_tensors[1],
                # What a parent states from the operation's facts: N and K.
                columns=4,
                widest_reduction=4,
                result_range=(-128, 127),  # int8's: the core's accumulator is INT8
                platform=platform,
            ),
            lambda point, index: commit(
                point,
                {
                    "pe": 2,
                    "simd": 2,
                    "reducer": "tree",
                    "compute_pumping": bool(index % 2),
                },
            ),
            "netlist",
        ),
        # The compute core is a Decision over kernels; ``packed`` is its entry.
        "matmul": (
            MatMulKernel(
                platform=platform,
                x_tensor=matmul_tensors[0],
                w_tensor=matmul_tensors[1],
            ),
            lambda point, index: commit(
                point,
                {
                    "compute": "packed",
                    "compute.packed.pe": (1, 2)[index % 2],
                    "compute.packed.simd": 2,
                    "compute.packed.compute_pumping": False,
                    "compute.packed.reducer": "tree",
                },
            ),
            "compute.netlist",
        ),
    }


def kernel_workload(api, name: str, trials: int) -> dict[str, object]:
    """Measure repeated replacement, cold/cached accepted netlist views and retention."""
    node, configure, view = kernel_fixtures(api)[name]
    base = api.design_space(node)

    def accepted(point):
        # The view is a member path below the root: the netlist of the module the kernel
        # builds.
        target = point
        for part in view.split("."):
            target = getattr(target, part)
        return target

    # Warm compilation/import allocations before reporting exploration costs.
    warm = configure(base, 0)
    accepted(warm)
    del warm
    gc.collect()
    model = api.inspection.model(base)
    model.evaluator.counters.reset()
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
        assert result.instances[0][1].sources
        started = time.perf_counter()
        cached = accepted(point)
        times["cached_view"].append(time.perf_counter() - started)
        assert cached == result
    # What the repeated configurations call and find in the model's memo: alternating
    # choices revisit the values an earlier configuration read.
    memo = memo_work(model)
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
        "memo": memo,
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
        f"Python: `{environment['python']}`.",
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
            "Traced Python allocation excludes native allocations and RSS.",
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
            "Both explicit-input batches use reverse declaration order. Calls are the "
            "memo's: an authored function called, not found in the model's memo.",
            "",
            "| Batch | Changes | Membership calls | Prerequisite calls | Rule calls | Commit s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in report["batch_updates"]:
            lines.append(
                f"| {item['fixture']} | {item['changes']} | {item['membership_calls']} | "
                f"{item['prerequisite_calls']} | {item['rule_calls']} | "
                f"{item['update_seconds']:.6f} |"
            )
        replacement = report["replacement_validation"]
        lines += [
            "",
            f"Replacing 1 of {replacement['choices']} independent choices ran "
            f"{replacement['validated']} memberships in {replacement['seconds']:.6f} s; "
            "the retained choices keep their admissions.",
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
            "Each wide-choice query is on a fresh binding: the first calls the selected view, "
            "every later one finds it in the model's memo.",
            "",
            "| Alternatives | Median µs | Minimum µs | Maximum µs | Views/query | "
            "Memo calls | Memo hits |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report["wide_choices"]:
            lines.append(
                f"| {item['alternatives']} | {item['query_seconds_median'] * 1e6:.2f} | "
                f"{item['query_seconds_min'] * 1e6:.2f} | {item['query_seconds_max'] * 1e6:.2f} | "
                f"{item['leaf_callbacks_per_query']:g} | {item['memo']['calls']['derived']} | "
                f"{item['memo']['hits']['derived']} |"
            )
        cache = report["cache_reclamation"]
        lines += [
            "",
            f"Cache reclamation tracked {cache['callback_created_objects']} "
            f"callback-created immutable payloads of {cache['payload_bytes_per_object']} "
            f"bytes each. {cache['alive_with_configurations_held']} were alive with "
            f"configurations held; {cache['alive_after_configurations_released']} remained "
            "after release and collection, with the base and prepared definition retained: "
            f"the newest {cache['memo_traces_kept']}, which the model's memo keeps. "
            f"{cache['alive_after_memo_released']} remained once the memo dropped them.",
            "",
        ]
    if "self_workloads" in report:
        lines += [
            "## Ordinary self reads",
            "",
            "Each scenario runs in a fresh subprocess, its memo dropped before each measured "
            "query. Every callback completes once, and cached view calls run no callbacks. "
            "Mixed work warms the wide prefix before a cold view reads that prefix and enters "
            "a deep chain.",
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
            "keys. Each measures the netlist view of the kernel it configures: the FIFO's own, "
            "the packed Dotp core's on its streams, and MatMul's selected compute core's. "
            "Timings exclude source rendering and hardware execution. A view is cold in its "
            "configuration's world; the model's memo, shared by every configuration, may "
            "already hold what it computes (the JSON's `memo`). A separate allocation run "
            "holds an independent population and then releases it.",
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
            "Removing folding intentionally adds runtime expression work, once per model: "
            "the configurations share its memo, so the first configuration read calls each "
            "term, and every later one finds them. Compare this scenario separately from "
            "parameter-based expressions, and preserve operand validation and inactive "
            "arithmetic suppression.",
            "",
        ]
    lines += [
        "## Interpretation and limits",
        "",
        "Timings are observations on this machine, not CI thresholds. Every configuration of a "
        "model shares its memo of authored functions; the JSON's `memo` fields name its calls "
        "and hits. Runtime/kernel subprocess RSS includes the interpreter, imports, "
        "prepared definitions and native allocations. Its ordinary process peak is captured "
        "before the separate tracing run; it is not an attribution of all bytes to the engine. "
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
        # What a narrow query evaluates stays narrow: no inactive alternative's callback
        # runs (``narrow_query``). What it demands grows with the branches: a view reads
        # its Space's validity, and a view at the root reads every branch's piece of it.
        first, second = report["narrow_queries"]
        assert first["inactive_callbacks"] == second["inactive_callbacks"] == 0
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
