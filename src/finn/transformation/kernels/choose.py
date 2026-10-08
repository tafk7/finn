# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's open kernel choices through the DSE seam, and saving them.

``ExploreKernelChoices(strategies, completion=...)`` explores a partition's body, the
model of KernelOps the kernel path's cut made (``CutKernelPartition``; the build opens
it through the parent graph's node, ``partition_body``), so which KernelOps go together
is the cut's alone: a model holding any other node is refused. It builds the body's
shell root once (``shell_root``: their Partition, its kernels and the channels between
them, and the channels on its boundary; the nodes' saved choices replayed, a stale one
dropped with why; members named by path, ``partition.MatMul_0``),
runs the strategies in order through one ``Seam`` (``finn.kernels.explore``), each
from the point the one before returned, and then:

- refuses a point a member refuses (``Seam.refusals``); what no strategy chose
  stays open, for the completion policy to complete wherever the point is costed
  or generated;
- completes the point as hardware generation will (``Completion.complete`` with
  sizing, on a copy that is not stored) and refuses a completed point its shell
  does not admit (``admission_refusal``: its AXI-Lite buses, memory ports and
  doubled clock against the shell's row), by name;
- persists the point's commitments on the nodes that own them, each node's whole
  (``persist``): the choices made on purpose, never a completed one; a saved choice
  is never changed (an explorer fills open choices only), and a stale one is
  cleared;
- keeps what it found (``explored``): the committed point, the cost of the
  completed one and the report (the strategies, each with the choices it
  committed, attempts and time, and the completed values it read, if any; every
  committed choice by its owner, with the strategy that made it, ``saved`` for
  one the model held before; the completion policy and every value it completed,
  by owner, with who completed it; the required choices it leaves open, which
  hardware generation refuses; whether FIFOs were sized; the dropped choices with
  why, per member cycles, buffering and resources, the bottleneck, and the
  shell's resources against the platform's, ``resources``), so an outer search can
  compare.

The resources are the shell root's (``shell_resources``), each member its own
statement (``finn.kernels.base.RESOURCES``): its partition (the partition's kernels
and its channels, boundary channels included), each end and its static region, and
their sum, ``used``; where members state none, ``used`` is the sum of those that
do, a lower bound (``lower_bound``), each other named with why (``unstated``), and the
ends and the static region are not counted. The ends' and the static region's are out
of context, which the report says with how far that overstates the placed shell; the
``ip`` shell has neither, so its sum is its partition's. The platform's are its part's totals
(``Platform.resources``), nothing subtracted: what the platform has, not a budget,
and ``part`` states the part's facts as the part catalog has them (its device, the
device's SLRs, the devices that share them) and where they come from (``source``);
``share`` is the fraction of each the shell uses, ``binding`` the one it uses most of,
and ``over`` each it uses more of than the part has. A point over the part is a warning
(``ResourceBudgetWarning``, and the report's ``warning``) naming the binding resource,
whichever strategy chose it: the report ranks and warns, and refuses nothing (RC5).

The shell is the model's target's (``shell_root``): its row offers the boundary
channels its ends (none: the ``ip`` shell). Where a channel has an end, the report
has its row (``ends``: its facts and its cycles a frame, a frame a call and 16) and
says what the cycles leave out (``memory_latency``: unmeasured, each call adds it).

``fresh`` clears the nodes' choices before the root is built, so the strategies
explore from scratch; otherwise a saved choice is pinned and an exploration
resumes from what was saved.

A strategy is written as a spec, ``{"strategy": name, **parameters}``
(``strategy(spec)``, the names ``KERNEL_STRATEGIES``), as a build configuration
lists them: ``[{"strategy": "target_throughput", "fps": 1000000}, {"strategy":
"size_fifos"}]``, or ``[{"strategy": "max_throughput", "within": {"lut": 0.5}},
{"strategy": "size_fifos"}]``. A list runs as written: nothing is appended, and nothing is
read from anywhere else. A completion policy is named (``completion(name)``, the
names ``KERNEL_COMPLETIONS``): ``baseline`` unless the build names another.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.base import KernelOpError, kernel_op, read_target
from finn.custom_op.kernels.shell import (
    ShellResources,
    admission_refusal,
    persist,
    shell_resources,
    shell_root,
)
from finn.kernels.ends import MEMORY_LATENCY, EndContract
from finn.kernels.explore import (
    Baseline,
    Bottleneck,
    Completed,
    Completion,
    Cost,
    ExploreError,
    Explorer,
    MaxThroughput,
    Pinned,
    Placeholder,
    ResourceBudgetWarning,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.target import Platform
from finn.kernels.utilization import SHELL_CHARACTERISED, binding, over, total
from finn.platform import TargetRefused
from finn.platform import part as catalog_part
from finn.transformation.fpgadataflow.kernel_partitions import KERNEL_OPS_DOMAIN

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

KERNEL_STRATEGIES: Mapping[str, Callable[..., Explorer]] = {
    "pinned": Pinned,
    "target_throughput": TargetThroughput,
    "max_throughput": MaxThroughput,
    "size_fifos": SizeFifos,
}
"""The strategies a spec names, each made from the spec's other keys: ``pinned``
(``path``), ``target_throughput`` (``fps``, ``relax``), ``max_throughput``
(``within``, required: ``{resource: fraction}`` of the part's), ``size_fifos``
(``method``, ``margin``, ``ram_style``, ``frames``)."""

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
    """An exploration's result: the configured point of the shell root (what the
    strategies committed), its completion as hardware generation completes it (None
    where the policy refused), the cost of the completed point, and the report (JSON
    values)."""

    point: Any
    completed: Completed[Any] | None
    cost: Cost
    report: Mapping[str, Any]


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

RESOURCES_EXACT = (
    "the partition's DSP slices, and block RAM and UltraRAM where a memory's style is "
    "explicit, are the RTL's; its LUTs, FFs and auto memories are models (about 10 % on "
    "LUT and FF); the ends and the static region are models, " + SHELL_CHARACTERISED + ", "
    "which overstate the placed shell (by about 28 % of its LUTs, TFC on Ultra96)"
)
"""Which of the report's resources are exact and which are models, as it states it; on
a board the shell was not built and timed on, followed by its row's ``caveat``."""


def part_report(name: str) -> dict[str, object]:
    """The target's part as the part catalog states it (``finn.platform.catalog``):
    its device, the device's resources per SLR (``None`` where Vivado states no split),
    the devices that share them, and
    where its facts come from (``source``); a part the catalog does not have says
    why, with no source."""
    try:
        found = catalog_part(name)
    except TargetRefused as refused:
        return {"name": name, "source": None, "refused": str(refused)}
    device = found.device
    return {
        "name": found.name,
        "device": device.name,
        "architecture": device.architecture,
        "family": device.family,
        "slrs": None if device.slrs is None else [asdict(slr) for slr in device.slrs],
        "shared_with": list(device.shared_with),
        "source": found.source,
    }


def _resources_report(
    cost: Cost,
    split: ShellResources | str,
    platform: Platform | None,
    part: Mapping[str, object],
    caveat: str | None,
) -> dict[str, object]:
    """The shell's resources by member and their sum against the platform's part totals;
    where members state none, the sum of those that do, a lower bound (``lower_bound``),
    and each that does not, with why (``unstated``); the part's facts with their source
    (``part_report``); ``exact`` names the shell row's ``caveat`` for its board, if any."""
    unstated = dict(cost.unstated)
    if isinstance(split, str) and not unstated:
        unstated["shell"] = split
    stated = None if isinstance(split, str) or unstated else split
    lower_bound = stated is None and bool(cost.unstated)
    used = stated.total if stated is not None else None
    if lower_bound:
        used = total(cost.resources.values())
    totals = None if platform is None else platform.resources
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
    return {
        "used": None if used is None else asdict(used),
        "shell": None
        if stated is None
        else {
            "partition": asdict(stated.partition),
            "ends": {name: asdict(each) for name, each in stated.ends},
            "static_region": {name: asdict(each) for name, each in stated.static_region},
        },
        "lower_bound": lower_bound,
        "unstated": unstated,
        "platform": None if totals is None else asdict(totals),
        "part": dict(part),
        "share": share,
        "binding": most,
        "over": exceeded,
        "warning": _over_warning(most, exceeded),
        "counted": RESOURCES_LOWER_BOUND if lower_bound else RESOURCES_COUNTED,
        "exact": RESOURCES_EXACT if caveat is None else f"{RESOURCES_EXACT}; {caveat}",
    }


def _over_warning(most: str | None, exceeded: Mapping[str, Mapping[str, int]]) -> str | None:
    """What the report warns of where the shell uses more than the part has (RC5): the
    binding resource, and each resource over with its count and the part's."""
    if not exceeded:
        return None
    counts = ", ".join(
        f"{key} {each['used']} of {each['platform']}" for key, each in exceeded.items()
    )
    return f"the point uses more than the platform's part has, most of {most}: {counts}"


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


def _choices_by_owner(seam: Seam, made_by: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Each committed choice, by the node and attribute that persist it (as
    ``kernel_choices.json`` names it), with who made it."""
    found: dict[str, dict[str, str]] = {}
    for key, strategy_name in made_by.items():
        node, attribute = seam.owner(key) or ("", key)
        found.setdefault(node, {})[attribute] = strategy_name
    return found


def _owned(seam: Seam, key: str) -> str:
    """``key`` as its owner names it: ``node.attribute``."""
    node, attribute = seam.owner(key) or ("", key)
    return f"{node}.{attribute}"


def _completed_by_owner(seam: Seam, completed: Completed[Any]) -> dict[str, dict[str, object]]:
    """Each completed value by the node and attribute that would persist it, with who
    completed it."""
    found: dict[str, dict[str, object]] = {}
    for key, value in completed.values.items():
        node, attribute = seam.owner(key) or ("", key)
        found.setdefault(node, {})[attribute] = {"value": value, "by": completed.made_by[key]}
    return found


def _read_flag(explorer: Explorer, read: Sequence[str]) -> str:
    """What the report says of a strategy that read completed values (DSE12: what it
    committed from them is stored)."""
    named = "read completed choices: " + ", ".join(read)
    return f"sized at completed folding ({named})" if isinstance(explorer, SizeFifos) else named


def _fifos(
    strategies: Sequence[Explorer],
    explorers: Sequence[Mapping[str, Any]],
    completed: Completed[Any] | None,
    policy: Completion,
) -> str:
    """Whether FIFOs were sized, said so that a design without sizing does not read as
    sized: by ``size_fifos`` in the chain (the channels it sized), by the completion
    on its completed copy, or not, and why."""
    sized = [
        report for explorer, report in zip(strategies, explorers) if isinstance(explorer, SizeFifos)
    ]
    completing = (
        0 if completed is None or completed.sizing is None else len(completed.sizing["channels"])
    )
    if not sized:
        if completing:
            return f"sized at completion by {policy.name}: {completing} channels"
        return "not sized (no size_fifos in the chain)"
    channels = sum(len(report["channels"]) for report in sized)
    if not channels:
        return "not sized: size_fifos found no open transport (each was saved before)"
    return f"sized by size_fifos: {channels} channels"


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
    """The KernelOps of a partition's body, ``model``, explored by ``strategies`` and their
    choices persisted, the point completed by ``completion`` (``Baseline()`` by default)
    for its report, on the shell root of the model's target; see the module docstring."""
    if not model.graph.node:
        raise KernelOpError("no KernelOp to explore")
    others = [node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN]
    if others:
        raise KernelOpError(
            f"{', '.join(others)}: not KernelOps; exploration reads a partition's body, the "
            "KernelOps the cut put together (CutKernelPartition, partition_body)"
        )
    nodes = list(model.graph.node)
    if fresh:
        for node in nodes:
            op = kernel_op(model, node)
            op.save(dict.fromkeys(op.choices()))
    started = time.perf_counter()
    root = shell_root(model, nodes)
    seam = Seam(root.members, root.owners, read_target(model).platform, completion)
    point = root.point
    # Who made each choice: the model before the strategies, or the strategy that
    # committed it (the point it returned commits the choice, the one before did not).
    made_by = dict.fromkeys(seam.chosen(point), "saved")
    explorers: list[dict[str, object]] = []
    for explorer in strategies:
        attempts, began, reads = seam.attempts, time.perf_counter(), len(seam.reads)
        point = explorer.explore(seam, point)
        report = explorer.report()
        committed = [key for key in seam.chosen(point) if key not in made_by]
        made_by.update(dict.fromkeys(committed, str(report["strategy"])))
        entry: dict[str, object] = {
            **report,
            "committed": len(committed),
            "attempts": seam.attempts - attempts,
            "seconds": round(time.perf_counter() - began, 3),
        }
        read = {key: None for values in seam.reads[reads:] for key in values}
        if read:
            entry["read_completed"] = _read_flag(explorer, [_owned(seam, key) for key in read])
        explorers.append(entry)
    refused = seam.refusals(point)
    if refused:
        raise KernelOpError(
            "the explored point is refused: "
            + "; ".join(f"{name}: {why}" for name, why in refused.items())
        )
    completed: Completed[Any] | None = None
    completion_report: dict[str, object] = {"policy": seam.completion.name}
    try:
        completed = seam.completion.complete(seam, point, sizing=True)
    except ExploreError as error:
        completion_report["refused"] = str(error)
    if completed is not None:
        unadmitted = admission_refusal(completed.point)
        if unadmitted is not None:
            raise KernelOpError(
                f"the explored point is refused by the {root.row.shell!r} shell: {unadmitted}"
            )
    persist(model, root, point)
    if completed is not None:
        completion_report["open"] = [
            _owned(seam, choice.key) + (" (required)" if choice.required else "")
            for choice in completed.open
        ]
        if completed.sizing is not None:
            completion_report["sizing"] = completed.sizing
    costed = point if completed is None else completed.point
    cost = seam.cost(costed)
    resources = _resources_report(
        cost,
        shell_resources(costed),
        seam.platform,
        part_report(read_target(model).part),
        root.row.caveat,
    )
    ends: dict[str, object] = {}
    if root.ends and not cost.waiting and not cost.refused:
        ends = {"ends": _end_rows(costed, root.ends), "memory_latency": MEMORY_LATENCY}
    report = {
        "strategies": explorers,
        "choices": _choices_by_owner(seam, made_by),
        "completion": completion_report,
        "completed": {} if completed is None else _completed_by_owner(seam, completed),
        "fifos": _fifos(strategies, explorers, completed, seam.completion),
        "fresh": fresh,
        "dropped": dict(root.dropped),
        **_cost_report(seam, cost, resources),
        **ends,
        "seconds": round(time.perf_counter() - started, 3),
    }
    warning = resources["warning"]
    if isinstance(warning, str):
        warnings.warn(warning, ResourceBudgetWarning, stacklevel=2)
    return Explored(point, completed, cost, report)


def partition_bottleneck(
    model: ModelWrapper, completion: Completion | None = None
) -> Bottleneck | None:
    """The slowest members of a partition model of KernelOps and their cycles a frame, on
    its shell root (members by path), its saved choices replayed and completed as
    hardware generation completes them, by ``completion`` (``Baseline()`` by default;
    None where a member's cycles wait on a choice the policy leaves open)."""
    root = shell_root(model, model.graph.node)
    seam = Seam(root.members, root.owners, read_target(model).platform, completion)
    return seam.cost(seam.completion.complete(seam, root.point, sizing=True).point).bottleneck


class ExploreKernelChoices(Transformation):
    """Every open choice of a partition body's KernelOps explored by ``strategies``, in
    order, and saved, the point completed by ``completion`` for the report, on the shell root
    of the model's target; ``explored`` holds the result (``explore_kernel_choices``)."""

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


__all__ = [
    "KERNEL_COMPLETIONS",
    "KERNEL_STRATEGIES",
    "ExploreKernelChoices",
    "Explored",
    "completion",
    "explore_kernel_choices",
    "partition_bottleneck",
    "strategy",
]
