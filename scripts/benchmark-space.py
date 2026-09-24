#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the public Space runtime without importing its implementation records.

Example after installing the checkout or setting PYTHONPATH=src:
    python scripts/benchmark-space.py --output /tmp/space-performance.json

--module allows a candidate public facade to be measured before its final move.
Time and memory are observations, not CI thresholds. Semantic work assertions
check what was evaluated and that discarded point populations release caches.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import weakref


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


def flat_family(api, count: int):
    source = api.Param(int)
    members = {"source": source}
    for index in range(count):
        members[f"item{index}"] = api.Derived(increment, aliases={"value": source})
    return type("FlatFamily", (api.Space,), members)


def repeated_family(api, count: int):
    class Child(api.Space):
        extent = api.Param(int)
        lanes = api.Decision(int, values=(1, 2, 4))
        width = extent * lanes
        physical = api.View(width)

    return type(
        "RepeatedFamily",
        (api.Space,),
        {f"child{index}": api.Subspace(Child, extent=16) for index in range(count)},
    )


def deep_family(api, count: int):
    class Leaf(api.Space):
        value = api.Const(1)

    family = Leaf
    for index in range(count):
        enabled = api.Const(True)
        family = type(
            f"Level{index}",
            (api.Space,),
            {"enabled": enabled, "child": api.Subspace(family, when=enabled)},
        )
    return family


def compilation(api, label: str, count: int, factory) -> dict[str, object]:
    started = time.perf_counter()
    family = factory(api, count)
    author_seconds = time.perf_counter() - started
    gc.collect()
    started = time.perf_counter()
    timed_model = api.compile_space(family)
    compile_seconds = time.perf_counter() - started
    counts = asdict(api.inspection.statistics(timed_model))
    del timed_model
    del family
    gc.collect()

    # Measure a separate compile, so tracemalloc overhead does not contaminate
    # the reported ordinary compile time. Keep the model alive through GC.
    retained_family = factory(api, count)
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    started = time.perf_counter()
    retained_model = api.compile_space(retained_family)
    memory_compile_seconds = time.perf_counter() - started
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert asdict(api.inspection.statistics(retained_model)) == counts
    started = time.perf_counter()
    reused_model = api.compile_space(retained_family)
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


def batch_refinement(api, count: int, *, dependent: bool) -> dict[str, object]:
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
        decision = api.Decision(int, domain=api.domain(accepts=membership, limit=limit))
        members[f"limit{index}"] = limit
        members[f"choice{index}"] = decision
        requests.append((decision, index + 1 if dependent else 1))
        if dependent:
            previous = decision
    family = type("DependentBatch" if dependent else "IndependentBatch", (api.Space,), members)
    model = api.compile_space(family)
    base = model.bind()
    changes = [
        api.refinement.change(base, reference, value) for reference, value in reversed(requests)
    ]
    started = time.perf_counter()
    report = api.refinement.commit(base, *changes)
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
        "refine_seconds": seconds,
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
        f"choice{index}": api.Decision(int, domain=api.domain(accepts=membership))
        for index in range(count)
    }
    family = type("ReplacementValidation", (api.Space,), members)
    base = family()
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
    physical_key = api.ViewKey("physical", int)

    class Selected(api.Space):
        value = api.Param(int)

        @api.view
        def physical(*, value: int) -> int:
            work["selected"] += 1
            return value + 1

        exports = {physical_key: physical}

    class Inactive(api.Space):
        @api.view
        def physical() -> int:
            work["inactive"] += 1
            raise AssertionError("an inactive alternative was demanded")

        exports = {physical_key: physical}

    source = api.Param(int)
    members = {"source": source}
    first = None
    for index in range(branches):
        choice = api.SubspaceChoice(
            {"selected": api.Subspace(Selected, value=source), "inactive": api.Subspace(Inactive)},
            exports=(physical_key,),
        )
        members[f"branch{index}"] = choice
        if first is None:
            first = choice
    family = type("NarrowQuery", (api.Space,), members)
    model = api.compile_space(family)
    assert work == Counter()
    base = model.bind({source: 7})
    selector = api.inspection.choices(model)[0].selector
    assert selector is not None and first is not None
    point = base.with_choices(base.field(selector).change("selected"))
    output = first.accepted(physical_key)
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
    physical_key = api.ViewKey("physical", int)

    class Leaf(api.Space):
        @api.view
        def physical() -> int:
            work["leaf"] += 1
            return 1

        exports = {physical_key: physical}

    choice = api.SubspaceChoice(
        {f"case{index}": api.Subspace(Leaf) for index in range(alternatives)},
        exports=(physical_key,),
    )
    family = type("WideChoice", (api.Space,), {"implementation": choice})
    model = api.compile_space(family)
    selector = api.inspection.choices(model)[0].selector
    output = choice.accepted(physical_key)
    measurements = []
    for _ in range(trials):
        base = model.bind()
        point = (
            base
            if selector is None
            else base.with_choices(base.field(selector).change(f"case{alternatives - 1}"))
        )
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
        "timing_scope": "cold accepted-output query only; start and selector membership excluded",
    }


def cache_reclamation(api, population: int) -> dict[str, object]:
    created = []
    payload_semantics = api.ValueSemantics.immutable_nominal(CachedPayload)

    def member(*, candidate: int) -> bool:
        return candidate >= 0

    class Family(api.Space):
        choice = api.Decision(int, domain=api.domain(accepts=member))

        @api.derived(semantics=payload_semantics)
        def output(*, choice: int) -> CachedPayload:
            payload = CachedPayload(choice, bytes((choice % 256,)) * 1024)
            # Track the callback-created object, not a detached public copy.
            created.append(weakref.ref(payload))
            return payload

    model = api.compile_space(Family)
    base = model.bind()
    configurations = []
    started = time.perf_counter()
    for candidate in range(population):
        point = base.with_choices(choice=candidate)
        point.query(Family.output)
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
    assert isinstance(base.field(Family.choice).state, api.Available)
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


def markdown(report: dict[str, object]) -> str:
    environment = report["environment"]
    lines = [
        "# Space performance measurements",
        "",
        f"Recorded {report['recorded_at_utc']}. All semantic work assertions passed.",
        "",
        f"Python: `{environment['python']}`. Platform: `{environment['platform']}`.",
        f"Source revision: `{environment['repository']['revision']}`; "
        f"dirty: `{environment['repository']['dirty']}`.",
        f"Space source SHA-256: `{environment['space_source_sha256']}`.",
        "",
        "## Compilation",
        "",
        "Time is measured without allocation tracing. Memory comes from a separate compile: "
        "net traced live Python allocation after garbage collection, with the compiled model held. "
        "Source declarations were created before tracing. This excludes prior allocations and RSS; "
        "it can include retained library allocations made during compilation. Peak is the traced "
        "compile high-water mark. Neither number is a recursive size estimate of the model.",
        "",
        "Authored declarations count effective members per instantiated scope, child placements "
        "including choice cases, and structural choices. Generated computation nodes are excluded "
        "from that declaration count. Potential edges are direct, not transitive closures.",
        "",
        "| Family | Size | Declarations | Scopes | Nodes | Edges | "
        "Compile seconds | Cached prepare µs | Retained MiB | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["compilation"]:
        counts = item["statistics"]
        lines.append(
            f"| {item['fixture']} | {item['size']} | {counts['authored_declarations']} | "
            f"{counts['scopes']} | {counts['nodes']} | {counts['potential_edges']} | "
            f"{item['compile_seconds']:.6f} | {item['reuse_seconds'] * 1e6:.2f} | "
            f"{item['retained_compile_bytes'] / 2**20:.3f} | "
            f"{item['peak_compile_bytes'] / 2**20:.3f} |"
        )
    lines += [
        "",
        "## Atomic monotone commitment",
        "",
        "Both batches are submitted in reverse declaration order. Each decision has an explicit "
        "prerequisite callback and membership callback. Dependent decisions consume the preceding "
        "decision; independent decisions each consume the same constant source through separate "
        "prerequisite declarations.",
        "",
        "| Batch | Changes | Membership calls | Prerequisite calls | "
        "Commit seconds | Capture seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["refinement"]:
        lines.append(
            f"| {item['fixture']} | {item['changes']} | {item['membership_calls']} | "
            f"{item['prerequisite_calls']} | {item['refine_seconds']:.6f} | "
            f"{item['capture_seconds']:.6f} |"
        )
    replacement = report["replacement_validation"]
    lines += [
        "",
        "Configuration replacement revalidates every retained commitment before publication. "
        f"Changing 1 of {replacement['choices']} choices validated "
        f"{replacement['validated']} memberships in {replacement['seconds']:.6f} seconds.",
    ]
    lines += [
        "",
        "## Narrow query work",
        "",
        "One selected output is queried twice, then explained through the public evidence service. "
        "Other choices remain unresolved, and the unselected evaluator raises if demanded.",
        "",
        "| Choices | Model nodes | Selected callbacks | Inactive callbacks | "
        "Demanded nodes | Demanded edges | Miss µs | Hit µs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["narrow_queries"]:
        lines.append(
            f"| {item['branches']} | {item['statistics']['nodes']} | "
            f"{item['selected_callbacks']} | "
            f"{item['inactive_callbacks']} | {item['demanded_nodes']} | {item['demanded_edges']} | "
            f"{item['miss_seconds'] * 1e6:.2f} | {item['hit_seconds'] * 1e6:.2f} |"
        )
    lines += [
        "",
        "## Choice dispatch",
        "",
        "The final alternative is selected in each independent configuration. Timings include "
        "only the cold accepted-output query, excluding construction and selector membership. "
        "Each query invokes "
        "exactly one leaf evaluator. Median/minimum/maximum are observations over "
        "30 configurations, not thresholds.",
        "",
        "| Alternatives | Median µs | Minimum µs | Maximum µs | Callbacks per query |",
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
        "## Cache reclamation",
        "",
        f"{cache['callback_created_objects']} callback-created immutable payloads each contain "
        f"{cache['payload_bytes_per_object']} bytes. "
        "Weak references were created inside the callback. "
        f"{cache['alive_with_configurations_held']} remained alive while their configurations "
        "were held; "
        f"{cache['alive_after_configurations_released']} remained after releasing the "
        "configurations "
        "and collecting. The compiled model and original choice-free root remained alive. "
        "This checks actual cached "
        "outputs, rather than weak references to discarded public copies.",
        "",
        "## Interpretation and limits",
        "",
        "Compilation includes all declared alternatives. "
        "The narrow-query callback and evidence counts "
        "measure demanded work separately. No cross-snapshot cache reuse is introduced or assumed. "
        "Arbitrary user-defined membership is not claimed to be constant time. Deep scopes include "
        "their full relative diagnostic names in the measured allocation.",
        "",
        "Deep scope chains retain increasingly long relative names at each level, even though "
        "each scope and node has one compiled record. This name-storage cost remains in the "
        "current representation. Selected outputs use a compiled lookup while retaining the "
        "ordered alternatives for structural inspection.",
        "",
        "These are one-process measurements on the recorded machine. "
        "Timing and memory do not establish CI performance thresholds. "
        "Existing 20,000-node chain and 2,000-scope tests supply separate "
        "iterative correctness evidence; this report does not replace them.",
        "",
        "Reproduce with `scripts/benchmark-space.py --output <report.json> "
        "--markdown <PERFORMANCE.md>` "
        "and the fixture sizes recorded in the JSON configuration. The client uses only supported "
        "Space, inspection, and selection operations.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="finn.kernels.space")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--flat", type=int, default=5000)
    parser.add_argument("--children", type=int, default=500)
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--branches", type=int, default=500)
    parser.add_argument("--population", type=int, default=200)
    arguments = parser.parse_args()
    sizes = {
        name: getattr(arguments, name)
        for name in ("flat", "children", "depth", "batch", "branches", "population")
    }
    if any(value < 1 for value in sizes.values()):
        parser.error("fixture sizes must be positive")
    api = importlib.import_module(arguments.module)
    api.compile_space(type("Warmup", (api.Space,), {}))
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "module": arguments.module,
            "repository": repository_state(root),
            "space_source_sha256": source_digest(Path(api.__file__).parent),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "configuration": sizes,
        "compilation": [
            compilation(api, "flat", arguments.flat, flat_family),
            compilation(api, "repeated children", arguments.children, repeated_family),
            compilation(api, "guarded scope depth", arguments.depth, deep_family),
        ],
        "refinement": [
            batch_refinement(api, arguments.batch, dependent=False),
            batch_refinement(api, arguments.batch, dependent=True),
        ],
        "replacement_validation": replacement_validation(api, arguments.batch),
        "narrow_queries": [narrow_query(api, 1), narrow_query(api, arguments.branches)],
        "wide_choices": [wide_choice(api, 2), wide_choice(api, max(2, arguments.branches))],
        "cache_reclamation": cache_reclamation(api, arguments.population),
    }
    first, second = report["narrow_queries"]
    assert first["demanded_nodes"] == second["demanded_nodes"]
    assert first["demanded_edges"] == second["demanded_edges"]
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(markdown(report))
    print(json.dumps({"output": str(arguments.output), "semantic_assertions": "passed"}))


if __name__ == "__main__":
    main()
