# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's kernel choices through the DSE seam, and saving them.

``ExploreKernelChoices(strategies)`` explores a partition's body, the model of
KernelOps the kernel path's cut made (``CutKernelPartition``; ``ExplorePartition``, the
build's, explores it through the parent graph's node, ``partition_body``, and saves it
in its file), so which KernelOps go together is the
cut's alone: a model holding any other node is refused. It builds the body's shell root
once (``shell_root``: their kernels, the channels between them and the channels on its
boundary; the nodes' and the channels' saved choices replayed, a stale one dropped with
why; members named as the graph, ``MatMul_0``), gives every choice the body leaves open
its baseline (``Seam.baseline``: what ``BaselineChoices`` does, a no-op on a body the
cut created; on a body made by hand, after ``fresh``, or for a choice replay dropped as
stale, the report names what it baselined), runs the strategies in order through one ``Seam``
(``finn.kernels.explore``), each reopening the choices it decides, exploring and
settling, from the point the one before returned, and then:

- refuses a point a member refuses (``Seam.refusals``) or its shell does not admit
  (``admission_refusal``: its AXI-Lite buses, memory ports and doubled clock against
  the shell's row), by name (a ``KernelOpError`` carrying their findings);
- refuses a run where a strategy changed anything but the transports with FIFOs placed
  before it and no ``size_fifos`` after it, naming the FIFOs: the engine keeps a
  stored depth valid when its ends' folding changes, so the run checks it. A list
  where a strategy that folds follows the last ``size_fifos`` is refused before
  anything runs (``refuse_folding_after_sizing``);
- persists every value of the point where it is stated, each owner's whole
  (``persist``): a kernel's on its node, a channel's on its tensor (``finn.channel``);
  a required choice no strategy set stays open, and hardware generation refuses it;
- keeps what it found (``explored``): the point, its cost and the report, a log of this
  run: the choices the body left open and the baseline took (``baselined``); each
  strategy with every value it changed (``changed``: before and after, by owner), how
  many it reopened and settling kept, the ones settling re-baselined, a pin's held
  keys, attempts and time; every value by its owner (a node or a tensor, by graph
  name, and the key there, as ``kernel_choices.json`` names it), with who set it last
  in this run (``saved``: the body, ``baseline``, or a strategy); the keys held; the
  choices left open (a required one named so); each FIFO's memory style beside the
  storage FinnLib's selection gives it and what it uses (``fifo_storage``: its
  ``auto`` is FinnLib's, not Vivado's); whether FIFOs were sized; the dropped choices
  with why, per member cycles, buffering and resources, the bottleneck, and the
  shell's resources against the platform's, ``resources``), so an outer search can
  compare; and every limitation the run found (``findings``: the strategies' and the
  resources'), in the report where each was found. Who set a value is logged here,
  never stored in the model.

Until the build's completion policy goes, ``completion`` (``Baseline()`` by default)
still completes the point on a copy for the report (``completion``, ``completed``):
on a settled point it has nothing left to complete but a required choice.

The resources are the shell root's (``shell_resources``), each member its own
statement (``finn.kernels.base.RESOURCES``): its partition (the partition's kernels
and its channels, boundary channels included), each end and its static region, and
their sum, ``used``; where members state none (a memory pinned ``auto``, which
Vivado places), ``used`` is the sum of those that
do, a lower bound (``lower_bound``), each other named with why (``unstated``, its
findings), and the
ends and the static region are not counted. The ends' and the static region's are out
of context, which the report says with how far that overstates the placed shell; the
``ip`` shell has neither, so its sum is its partition's. The platform's are its part's totals
(``Platform.resources``), nothing subtracted: what the platform has, not a budget,
and ``part`` states the part's facts as the part catalog has them (its device, the
device's SLRs, on a reduced die of several SLRs each SLR's site capacity under the
totals as a cap with which caps are proven or inferred, the devices that share them)
and where they come from (``source``);
``share`` is the fraction of each the shell uses, ``binding`` the one it uses most of,
and ``over`` each it uses more of than the part has. A point over the part is a
limitation (``resources-over-part``, in the result's and the resources' ``findings``)
naming the binding resource, whichever strategy chose it: the report ranks and states
it, and refuses nothing (RC5).

The shell is the model's target's (``shell_root``): its row offers the boundary
channels its ends (none: the ``ip`` shell). Where a channel has an end, the report
has its row (``ends``: its facts and its cycles a frame, a frame a call and 16) and
says what the cycles leave out (``memory_latency``: unmeasured, each call adds it).

``fresh`` clears the nodes' choices and the channel choices the body's tensors state
before the root is built: every choice starts at its baseline (the ``baseline``
strategy does the same within a list, the pins kept).

A strategy is written as a spec, ``{"strategy": name, **parameters}``
(``strategy(spec)``, the names ``KERNEL_STRATEGIES``), as a build configuration
lists them: ``[{"strategy": "target_throughput", "fps": 1000000}, {"strategy":
"size_fifos"}]``, or ``[{"strategy": "max_throughput", "within": {"lut": 0.5}},
{"strategy": "size_fifos"}]``. A list runs as written: nothing is appended, and nothing is
read from anywhere else. Pins go first, held for the run; sizing after the last
strategy that folds. A completion policy is named (``completion(name)``, the names
``KERNEL_COMPLETIONS``): ``baseline`` unless the build names another.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn.core.space import Available, Finding, FindingKind, finding_record, finding_text
from finn.custom_op.kernels.base import KernelOpError, clear_channel_choices, kernel_op, read_target
from finn.custom_op.kernels.shell import (
    ShellResources,
    admission_refusal,
    persist,
    shell_resources,
    shell_root,
)
from finn.custom_op.partition.kernel_partitions import KERNEL_OPS_DOMAIN, partition_body
from finn.kernels.base import RESOURCES
from finn.kernels.ends import MEMORY_LATENCY, EndContract
from finn.kernels.explore import (
    Baseline,
    Bottleneck,
    Completed,
    Completion,
    Cost,
    DebugPlaceholder,
    ExploreError,
    Explorer,
    MaxThroughput,
    Pinned,
    Placeholder,
    Rebaseline,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.fifo import FifoKernel
from finn.kernels.target import Platform
from finn.kernels.utilization import SHELL_CHARACTERISED, binding, over, total
from finn.platform import part_report

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

KERNEL_STRATEGIES: Mapping[str, Callable[..., Explorer]] = {
    "pinned": Pinned,
    "baseline": Rebaseline,
    "target_throughput": TargetThroughput,
    "max_throughput": MaxThroughput,
    "size_fifos": SizeFifos,
    "placeholder": DebugPlaceholder,
}
"""The strategies a spec names, each made from the spec's other keys: ``pinned``
(``path``), ``baseline`` (every choice not held back at its baseline),
``target_throughput`` (``fps``, ``relax``), ``max_throughput`` (``within``, required:
``{resource: fraction}`` of the part's), ``size_fifos`` (``method``, ``margin``,
``ram_style``, ``frames``), and ``placeholder``, for debugging (every open required
choice at its first case)."""

#: No value: a key absent from a mapping of values.
_ABSENT = object()

KERNEL_COMPLETIONS: Mapping[str, Callable[[], Completion]] = {
    "baseline": Baseline,
    "placeholder": Placeholder,
}
"""The completion policies a build names: ``baseline`` (every open choice that is not
required at its kernel's baseline, the default), ``placeholder`` (also the required
ones, for debugging)."""


def completion(name: str) -> Completion:
    """The completion policy ``name`` names (``KERNEL_COMPLETIONS``)."""
    if name not in KERNEL_COMPLETIONS:
        raise ValueError(
            f"{name!r} names no kernel completion policy (one of {sorted(KERNEL_COMPLETIONS)})"
        )
    return KERNEL_COMPLETIONS[name]()


def strategy(spec: Mapping[str, Any]) -> Explorer:
    """The strategy ``spec`` writes: ``{"strategy": name, **parameters}``."""
    parameters = dict(spec)
    name = parameters.pop("strategy", None)
    if name not in KERNEL_STRATEGIES:
        raise ValueError(
            f"{dict(spec)} names no kernel strategy (one of {sorted(KERNEL_STRATEGIES)})"
        )
    try:
        return KERNEL_STRATEGIES[name](**parameters)
    except TypeError as error:
        raise ValueError(f"{dict(spec)}: {error}") from error


@dataclass(frozen=True)
class Explored:
    """An exploration's result: the point of the shell root (every value, settled after
    the last strategy), its completion by the build's policy on a copy (None where the
    policy refused; on a settled point it completes no more than a required choice),
    the cost of the completed point, the report (JSON values), and every limitation
    the run found (``findings``: each strategy's, then the resources'), never a
    refusal (RC5)."""

    point: Any
    completed: Completed[Any] | None
    cost: Cost
    report: Mapping[str, Any]
    findings: tuple[Finding, ...] = ()


RESOURCES_COUNTED = (
    "the shell: its partition (the partition's kernels and channels, boundary channels "
    "included), each end and its static region, each its own statement"
)
"""What the report's resources count, as it states it."""

RESOURCES_LOWER_BOUND = (
    "a lower bound: the members that state their own resources (the partition's kernels "
    "and channels, boundary channels included); the members named under unstated state "
    "none, and the ends and the static region are not counted while they do not"
)
"""What the report's resources count where members state none, as it states it."""


def resources_exact(platform: Platform | None) -> str:
    """Which of the report's resources are exact and which are models, as it states it,
    naming the primitive table the memories are counted in (the platform's fabric's,
    ``finn.kernels.utilization.PRIMITIVES``); on a board the shell was not built and
    timed on, the report follows it with its row's ``caveat``."""
    table = (
        "no fabric's primitive table, the platform not stated"
        if platform is None
        else f"the {platform.fabric.value} primitive table (finn.kernels.utilization.PRIMITIVES)"
    )
    return (
        "the partition's DSP slices and every memory's block RAM, UltraRAM and LUTRAM "
        f"storage are the RTL's, in {table}; a FIFO's shallow hi space, which Vivado "
        "places, is a model; its other LUTs and FFs are models (about 10 %); a memory "
        "in auto, other than a FIFO's, is placed by Vivado, not stated (unstated); the "
        "ends and the static region are models, "
        + SHELL_CHARACTERISED
        + ", which overstate the placed shell (by about 28 % of its LUTs, TFC on Ultra96)"
    )


def _resources_report(
    cost: Cost,
    split: ShellResources | tuple[Finding, ...],
    platform: Platform | None,
    part: Mapping[str, object],
    caveat: str | None,
) -> tuple[dict[str, object], tuple[Finding, ...]]:
    """The shell's resources by member and their sum against the platform's part totals;
    where members state none, the sum of those that do, a lower bound (``lower_bound``),
    and each that does not, with why, its findings (``unstated``); the part's facts with
    their source (``part_report``); ``exact`` names the shell row's ``caveat`` for its
    board, if any; and the limitation where the shell uses more than the part has
    (``findings``, also returned)."""
    unstated = dict(cost.unstated)
    if not isinstance(split, ShellResources) and not unstated:
        unstated["shell"] = split
    stated = None if not isinstance(split, ShellResources) or unstated else split
    lower_bound = stated is None and bool(cost.unstated)
    used = stated.total if stated is not None else None
    if lower_bound:
        used = total(cost.resources.values())
    totals = None if platform is None else platform.resources
    exact = resources_exact(platform)
    share = None
    most = None
    exceeded: dict[str, dict[str, int]] = {}
    if used is not None and totals is not None:
        available = asdict(totals)
        share = {
            key: round(count / available[key], 4)
            for key, count in asdict(used).items()
            if available[key]
        }
        most = binding(used, available)
        exceeded = {
            key: {"used": count, "platform": limit}
            for key, (count, limit) in over(used, available).items()
        }
    findings = _over_part(most, exceeded)
    report: dict[str, object] = {
        "used": None if used is None else asdict(used),
        "shell": None
        if stated is None
        else {
            "partition": asdict(stated.partition),
            "ends": {name: asdict(each) for name, each in stated.ends},
            "static_region": {name: asdict(each) for name, each in stated.static_region},
        },
        "lower_bound": lower_bound,
        "unstated": {
            name: [finding_record(finding) for finding in why] for name, why in unstated.items()
        },
        "platform": None if totals is None else asdict(totals),
        "part": dict(part),
        "share": share,
        "binding": most,
        "over": exceeded,
        "findings": [finding_record(finding) for finding in findings],
        "counted": RESOURCES_LOWER_BOUND if lower_bound else RESOURCES_COUNTED,
        "exact": exact if caveat is None else f"{exact}; {caveat}",
    }
    return report, findings


def _over_part(most: str | None, exceeded: Mapping[str, Mapping[str, int]]) -> tuple[Finding, ...]:
    """What the report states where the shell uses more than the part has (RC5): a
    ``resources-over-part`` limitation naming the binding resource, and each resource
    over with its count and the part's."""
    if not exceeded:
        return ()
    counts = ", ".join(
        f"{key} {each['used']} of {each['platform']}" for key, each in exceeded.items()
    )
    return (
        Finding(
            FindingKind.LIMITATION,
            "resources-over-part",
            ExploreKernelChoices.__name__,
            f"the point uses more than the platform's part has, most of {most}: {counts}",
            (("binding", most), ("over", tuple(exceeded))),
        ),
    )


def _cost_report(seam: Seam, cost: Cost, resources: dict[str, object]) -> dict[str, object]:
    bottleneck = cost.bottleneck
    return {
        "members": {
            name: {
                "cycles": cost.cycles.get(name),
                "buffering": cost.buffering.get(name),
                "resources": asdict(cost.resources[name]) if name in cost.resources else None,
            }
            for name in seam.members
        },
        "bottleneck": None
        if bottleneck is None
        else {"members": list(bottleneck.members), "cycles": bottleneck.cycles},
        "buffering": sum(cost.buffering.values()),
        "resources": resources,
    }


def _fifo_storage(seam: Seam, point: Any) -> dict[str, dict[str, object]]:
    """Each FIFO's memory style on the completed ``point``, by the graph name (a node or a
    tensor) and key that would persist it, beside the storage the RTL selects for it
    (FinnLib's ``auto``: shift, distributed, block or ultra) and what it uses."""
    found: dict[str, dict[str, object]] = {}
    declared = seam.declared(point)
    for key, value in seam.chosen(point).items():
        path, _, name = key.rpartition(".")
        if declared.get(key) is not FifoKernel or name != "ram_style":
            continue
        # A selector's case names no attribute: the selector's value is the case's Space.
        fifo = reduce(lambda found, name: getattr(found, name, found), path.split("."), point)
        # A FIFO whose depth no strategy sized (required, left open) has neither yet.
        storage = fifo.query(FifoKernel.storage)
        used = fifo.query(FifoKernel.exports[RESOURCES])
        node, attribute = seam.owner(key) or ("", key)
        found.setdefault(node, {})[attribute] = {
            "ram_style": value,
            "storage": storage.value.effective_style if isinstance(storage, Available) else None,
            "resources": asdict(used.value) if isinstance(used, Available) else None,
        }
    return found


def _owned(seam: Seam, key: str) -> str:
    """``key`` as its owner names it: ``<graph name>.<key>``."""
    node, attribute = seam.owner(key) or ("", key)
    return f"{node}.{attribute}"


def _by_owner(seam: Seam, values: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """``values`` by the graph name (a node or a tensor) and key that persist each."""
    found: dict[str, dict[str, object]] = {}
    for key, value in values.items():
        node, attribute = seam.owner(key) or ("", key)
        found.setdefault(node, {})[attribute] = value
    return found


def _completed_by_owner(seam: Seam, completed: Completed[Any]) -> dict[str, dict[str, object]]:
    """Each completed value by the graph name and key that would persist it, with who
    completed it."""
    return _by_owner(
        seam,
        {
            key: {"value": value, "by": completed.made_by[key]}
            for key, value in completed.values.items()
        },
    )


def _fifos(
    strategies: Sequence[Explorer], explorers: Sequence[Mapping[str, Any]], placed: Sequence[str]
) -> str:
    """Whether FIFOs were sized, said so that a design without sizing does not read as
    sized: by ``size_fifos`` in the list (the channels it sized, the FIFOs placed), or
    not, and why."""
    sized = [
        report for explorer, report in zip(strategies, explorers) if isinstance(explorer, SizeFifos)
    ]
    if not sized:
        held = f"; FIFOs placed: {', '.join(placed)}" if placed else ""
        return "not sized (no size_fifos in the list): each transport as it was" + held
    channels = len(sized[-1]["channels"])
    if not channels:
        return "not sized: size_fifos found every transport held"
    return f"sized by size_fifos: {channels} channels, {len(placed)} FIFOs placed"


def refuse_folding_after_sizing(strategies: Sequence[Explorer]) -> None:
    """Refuse a list where a strategy that changes folding (``folds``) follows the last
    ``size_fifos``, naming both: the engine keeps a stored FIFO depth valid when its ends'
    folding changes, so the depths it sized would go stale."""
    last = max(
        (index for index, explorer in enumerate(strategies) if isinstance(explorer, SizeFifos)),
        default=None,
    )
    if last is None:
        return
    after = [
        f"{explorer.strategy} (strategy {index + 1})"  # type: ignore[attr-defined]
        for index, explorer in enumerate(strategies)
        if index > last and getattr(explorer, "folds", False)
    ]
    if after:
        raise ExploreError(
            f"{', '.join(after)} follows the last size_fifos (strategy {last + 1}): FIFO "
            "depths are sized at the folding before them and would go stale; size FIFOs "
            "after the last strategy that folds"
        )


def _placed(
    seam: Seam, point: Any, chosen: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Every FIFO placed on ``point`` (or in its root's values ``chosen``): each transport
    at its FIFO case, with its depth (None where it is open)."""
    chosen = seam.chosen(point) if chosen is None else chosen
    found: dict[str, object] = {}
    for key, fifo in seam.transports(point).items():
        if chosen.get(key) == fifo:
            depth = next(
                (
                    value
                    for name, value in chosen.items()
                    if name.startswith(f"{key}.{fifo}.") and name.endswith(".depth")
                ),
                None,
            )
            found[key] = depth
    return found


def _end_rows(point: Any, ends: Sequence[str]) -> dict[str, dict[str, object]]:
    """Each end's facts and cycles a frame, by its boundary channel."""
    rows: dict[str, dict[str, object]] = {}
    for name in ends:
        contract: EndContract = getattr(point, name).end_contract
        rows[name] = {
            "kind": contract.kind,
            "direction": contract.direction,
            "port": contract.port,
            "tdata": contract.tdata,
            "beats": contract.beats,
            "lanes": contract.lanes,
            "element": contract.element.dtype.name,
            "memory_width": contract.memory_width,
            "words": contract.words,
            "converter": contract.converter,
            "converter_kind": contract.converter_kind,
            "call_cycles": contract.call_cycles,
            "frames_per_call": contract.frames_per_call,
            "control_buses": contract.control_buses,
            "memory_ports": contract.memory_ports,
            "cycles": contract.cycles,
            "cycles_16_frames_a_call": contract.cycles_at(16),
        }
    return rows


def explore_kernel_choices(
    model: ModelWrapper,
    strategies: Sequence[Explorer],
    *,
    fresh: bool = False,
    completion: Completion | None = None,
) -> Explored:
    """The KernelOps of a partition's body, ``model``, explored by ``strategies`` and every
    value persisted, the point completed by ``completion`` (``Baseline()`` by default)
    for its report, on the shell root of the model's target; see the module docstring."""
    if not model.graph.node:
        raise KernelOpError("no KernelOp to explore")
    others = [node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN]
    if others:
        raise KernelOpError(
            f"{', '.join(others)}: not KernelOps; exploration reads a partition's body, the "
            "KernelOps the cut put together (CutKernelPartition, partition_body)"
        )
    refuse_folding_after_sizing(strategies)
    nodes = list(model.graph.node)
    if fresh:
        for node in nodes:
            op = kernel_op(model, node)
            op.save(dict.fromkeys(op.choices()))
        clear_channel_choices(model)
    started = time.perf_counter()
    root = shell_root(model, nodes)
    seam = Seam(root.members, root.owners, read_target(model).platform, completion)
    saved = seam.chosen(root.point)
    # A choice the body leaves open takes its baseline before the strategies run.
    point = seam.baseline(root.point)
    values = seam.chosen(point)
    baselined = [key for key in values if key not in saved]
    # Who set each value in this run: the body (``saved``), the baseline, or a strategy.
    set_by = {key: "saved" if key in saved else "baseline" for key in values}
    explorers: list[dict[str, object]] = []
    stale: tuple[str, dict[str, object]] | None = None
    for explorer in strategies:
        attempts, began, runs = seam.attempts, time.perf_counter(), len(seam.runs)
        before = seam.chosen(point)
        point = explorer.explore(seam, point)
        report = explorer.report()
        after = seam.chosen(point)
        label = str(getattr(explorer, "label", report["strategy"]))
        changed = [
            key
            for key in dict.fromkeys([*before, *after])
            if before.get(key, _ABSENT) != after.get(key, _ABSENT)
        ]
        for key in changed:
            if key in after:
                set_by[key] = label
            else:
                set_by.pop(key, None)
        done = seam.runs[runs:]
        entry: dict[str, object] = {
            **report,
            "committed": sum(key in after for key in changed),
            "changed": {_owned(seam, key): [before.get(key), after.get(key)] for key in changed},
            "reopened": len({key for run in done for key in run.reopened}),
            "kept": len({key for run in done for key in run.kept}),
            "re-baselined": [_owned(seam, key) for run in done for key in run.baselined],
            "attempts": seam.attempts - attempts,
            "seconds": round(time.perf_counter() - began, 3),
        }
        if isinstance(explorer, Pinned):
            entry["held"] = [_owned(seam, key) for key in explorer.held]
        explorers.append(entry)
        # A FIFO placed before a strategy that changes anything but the transports, and
        # not sized again after it, is stale: its depth was sized at the folding before.
        if isinstance(explorer, SizeFifos):
            stale = None
        elif any(seam.transport_of(point, key) is None for key in changed):
            kept = {
                key: depth
                for key, depth in _placed(seam, point, before).items()
                if key not in seam.held and _placed(seam, point, after).get(key, _ABSENT) == depth
            }
            if kept:
                stale = (str(report["strategy"]), kept)
    refused = seam.refusals(point)
    if refused:
        raise KernelOpError(
            "the explored point is refused: "
            + "; ".join(
                f"{name}: " + "; ".join(map(finding_text, why)) for name, why in refused.items()
            ),
            findings=tuple(finding for why in refused.values() for finding in why),
        )
    if stale is not None:
        name, fifos = stale
        raise KernelOpError(
            f"{name} changed the design with FIFOs placed and no size_fifos after it: "
            + ", ".join(f"{_owned(seam, key)} (depth {depth})" for key, depth in fifos.items())
            + " would keep depths sized at another folding; run size_fifos after it"
        )
    completed: Completed[Any] | None = None
    completion_report: dict[str, object] = {"policy": seam.completion.name}
    try:
        completed = seam.completion.complete(seam, point, sizing=True)
    except ExploreError as error:
        completion_report["refused"] = str(error)
    costed = point if completed is None else completed.point
    unadmitted = admission_refusal(costed)
    if unadmitted:
        raise KernelOpError(
            f"the explored point is refused by the {root.row.shell!r} shell: "
            + "; ".join(map(finding_text, unadmitted)),
            findings=unadmitted,
        )
    persist(model, root, point)
    left = [
        _owned(seam, choice.key) + (" (required)" if choice.required else "")
        for choice in seam.choices(point)
    ]
    if completed is not None:
        completion_report["open"] = [
            _owned(seam, choice.key) + (" (required)" if choice.required else "")
            for choice in completed.open
        ]
        if completed.sizing is not None:
            completion_report["sizing"] = completed.sizing
    cost = seam.cost(costed)
    resources, over_part = _resources_report(
        cost,
        shell_resources(costed),
        seam.platform,
        part_report(read_target(model).part),
        root.row.caveat,
    )
    ends: dict[str, object] = {}
    if root.ends and not cost.waiting and not cost.refused:
        ends = {"ends": _end_rows(costed, root.ends), "memory_latency": MEMORY_LATENCY}
    final = seam.chosen(point)
    report = {
        "strategies": explorers,
        "baselined": [_owned(seam, key) for key in baselined],
        "choices": _by_owner(
            seam, {key: {"value": final[key], "by": set_by[key]} for key in final}
        ),
        "held": _by_owner(seam, seam.held),
        "open": left,
        "completion": completion_report,
        "completed": {} if completed is None else _completed_by_owner(seam, completed),
        "fifo_storage": _fifo_storage(seam, costed),
        "fifos": _fifos(strategies, explorers, [_owned(seam, key) for key in _placed(seam, point)]),
        "fresh": fresh,
        "dropped": dict(root.dropped),
        **_cost_report(seam, cost, resources),
        **ends,
        "seconds": round(time.perf_counter() - started, 3),
    }
    found = tuple(
        finding for explorer in strategies for finding in getattr(explorer, "findings", ())
    )
    return Explored(point, completed, cost, report, (*found, *over_part))


def shell_bottleneck(
    model: ModelWrapper, completion: Completion | None = None
) -> Bottleneck | None:
    """The slowest members of a partition model of KernelOps' shell root, ends included,
    and their cycles a frame, its saved choices replayed and completed as hardware
    generation completes them, by ``completion`` (``Baseline()`` by default; None where
    a member's cycles wait on a choice the policy leaves open)."""
    root = shell_root(model, model.graph.node)
    seam = Seam(root.members, root.owners, read_target(model).platform, completion)
    return seam.cost(seam.completion.complete(seam, root.point, sizing=True).point).bottleneck


class ExploreKernelChoices(Transformation):
    """A partition body's KernelOps explored by ``strategies``, in order, and every value
    saved, the point completed by ``completion`` for the report, on the shell root of the
    model's target; ``explored`` holds the result (``explore_kernel_choices``)."""

    def __init__(
        self,
        strategies: Sequence[Explorer],
        *,
        fresh: bool = False,
        completion: Completion | None = None,
    ) -> None:
        super().__init__()
        self.strategies = tuple(strategies)
        self.fresh = fresh
        self.completion = completion
        self.explored: Explored | None = None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.explored = explore_kernel_choices(
            model,
            self.strategies,
            fresh=self.fresh,
            completion=self.completion,
        )
        return model, False


class ExplorePartition(Transformation):
    """The kernel path's parent graph's partition explored through its node: its body
    opened through the partition node (``partition_body``), its KernelOps explored by
    ``strategies``, in order, every value saved (``explore_kernel_choices``), and the
    body saved again in its file, as the cut saved it; the parent graph itself
    unchanged. ``explored`` holds the result and ``body`` the body as saved."""

    def __init__(
        self,
        strategies: Sequence[Explorer],
        *,
        fresh: bool = False,
        completion: Completion | None = None,
    ) -> None:
        super().__init__()
        self.strategies = tuple(strategies)
        self.fresh = fresh
        self.completion = completion
        self.explored: Explored | None = None
        self.body: ModelWrapper | None = None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        _, body, body_file = partition_body(model)
        self.explored = explore_kernel_choices(
            body,
            self.strategies,
            fresh=self.fresh,
            completion=self.completion,
        )
        body.save(body_file)
        self.body = body
        return model, False


#: Where the build writes the exploration's report (``Explored.report``), under its
#: output directory; the bitfile's build reads the throughput asked from it.
EXPLORATION_REPORT = "report/kernel_exploration.json"


__all__ = [
    "EXPLORATION_REPORT",
    "KERNEL_COMPLETIONS",
    "KERNEL_STRATEGIES",
    "ExploreKernelChoices",
    "ExplorePartition",
    "Explored",
    "completion",
    "explore_kernel_choices",
    "refuse_folding_after_sizing",
    "shell_bottleneck",
    "strategy",
]
