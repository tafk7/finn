# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a partition body's kernel choices through the DSE seam, one strategy at a
time, and saving them.

``explore_kernel_choices(model, strategy)`` explores a partition's body, the model of
KernelOps the build's cut made (``CutKernelPartition``; ``explore_partition``, which
the build's strategy steps run, opens it through the parent graph's node,
``partition_body``, and saves it in its file), so which KernelOps go together is the
cut's alone: a model holding any other node is refused. It builds the body's shell root
(``shell_root``: their kernels, the channels between them and the channels on its
boundary; the nodes' and the channels' saved
choices replayed, a stale one dropped with why; members named as the graph,
``MatMul_0``), presents it to its ``Seam`` (``finn.kernels.explore``) as a network
(``ShellRoot.network``: each choice keyed by its owner, a node or a tensor, and the key
there), holds the pins the body
records (``HELD``), gives every choice the body leaves open its baseline
(``Seam.baseline``: what ``BaselineChoices`` does, a no-op on a body the cut created; on
a body made by hand, or for a choice replay dropped as stale, the report names what it
baselined), runs the strategy, which reopens the choices it decides, explores and
settles, and then:

- refuses a point a member refuses (``Seam.refusals``) or its shell does not admit
  (``admission_refusal``: its AXI-Lite buses, memory ports and doubled clock against
  the shell's row), by name (a ``KernelOpError`` carrying their findings);
- persists every value of the point where it is stated, each owner's whole
  (``persist``): a kernel's on its node, a channel's on its tensor (``finn.channel``);
  a required choice no strategy set stays open, and emission refuses it by name
  (``configured_root``);
- records the pins held, those the body recorded and those the strategy pinned, in the
  body (``HELD``, ``{graph name: {key: value}}``), so that every later exploration of
  the body holds them: a pin holds for the build, and re-baselining keeps it; and a
  target throughput's frames a second (``OBJECTIVE_FPS``), which another strategy that
  may change the folding removes (``FOLDING``); and whether its FIFOs are stale
  (``FIFOS_STALE``): set by any strategy but FIFO sizing and the debug placeholder where
  the body holds a FIFO, removed by FIFO sizing;
- keeps what it found (``explored``): the point, its cost and the report: the choices
  the body left open and the baseline took (``baselined``); the strategy with every
  value it changed (``changed``: before and after, by owner), how many it reopened and
  settling kept, the ones settling re-baselined, a pin's held keys, attempts and time;
  every value by its owner (a node or a tensor, by graph name, and the key there, as
  ``kernel_choices.json`` names it), with who set it (``saved``: the body, ``baseline``,
  or the strategy); the keys held; the choices left open (a required one named so); each
  FIFO's memory style beside the storage FinnLib's selection gives it and what it uses
  (``fifo_storage``: its ``auto`` is FinnLib's, not Vivado's); whether FIFOs were
  sized; the dropped choices with why, per member cycles, buffering and resources, the
  bottleneck, and the shell's resources against the platform's, ``resources``), so an
  outer search can compare; and every limitation it found (``findings``: the
  strategy's and the resources'), in the report where each was found. Who set a value
  is logged in the report, never stored in the model.

``exploration_summary`` gives an exploration's lines for a build's log.

Pins go first, and FIFO sizing after the last strategy that changes the folding: the
engine keeps a stored FIFO depth valid when its ends' folding changes, so a FIFO sized
before it is marked stale (``FIFOS_STALE``), and packaging refuses the body.

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
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn.core.graph.annotations.choices import FIFOS_STALE, HELD, OBJECTIVE_FPS, held_choices
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, partition_body
from finn.core.space import Available, Finding, FindingKind, finding_record, finding_text
from finn.custom_op.kernels.rooted import Rooted
from finn.custom_op.kernels.shell import (
    ShellResources,
    admission_refusal,
    persist,
    shell_resources,
    shell_root,
)
from finn.kernels.base import RESOURCES
from finn.kernels.ends import MEMORY_LATENCY, EndContract
from finn.kernels.explore import (
    Bottleneck,
    Cost,
    DebugPlaceholder,
    Explorer,
    MaxThroughput,
    Pinned,
    Rebaseline,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.fifo import FifoKernel
from finn.kernels.network import Key, dotted
from finn.kernels.utilization import SHELL_CHARACTERISED, binding, over, total
from finn.platform.capabilities import Platform
from finn.platform.catalog import part_report

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

#: The strategies other than a target throughput that may change the kernels' folding:
#: each removes the body's ``OBJECTIVE_FPS``, which no longer names the throughput its
#: folding was explored for.
FOLDING = (Pinned, MaxThroughput, Rebaseline)

#: The strategies that keep the kernels' folding; any other, a user's own included, may
#: change it.
_KEEPS_FOLDING = (SizeFifos, DebugPlaceholder)

#: No value: a key absent from a mapping of values.
_ABSENT = object()


@dataclass(frozen=True)
class Explored:
    """An exploration's result: the point of the shell root (every value, settled after
    the strategy), its cost, the report (JSON values), and every limitation it found
    (``findings``: the strategy's, then the resources'), never a refusal (RC5)."""

    point: Any
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


def _cost_report(network: Rooted, cost: Cost, resources: dict[str, object]) -> dict[str, object]:
    bottleneck = cost.bottleneck
    return {
        "members": {
            name: {
                "cycles": cost.cycles.get(name),
                "buffering": cost.buffering.get(name),
                "resources": asdict(cost.resources[name]) if name in cost.resources else None,
            }
            for name in network.members
        },
        "bottleneck": None
        if bottleneck is None
        else {"members": list(bottleneck.members), "cycles": bottleneck.cycles},
        "buffering": sum(cost.buffering.values()),
        "resources": resources,
    }


def _fifo_storage(seam: Seam, network: Rooted) -> dict[str, dict[str, object]]:
    """Each FIFO's memory style, by the graph name (a node or a tensor) and key that
    persist it, beside the storage the RTL selects for it (FinnLib's ``auto``: shift,
    distributed, block or ultra) and what it uses."""
    found: dict[str, dict[str, object]] = {}
    declared = seam.declared(network)
    for (owner, key), value in seam.chosen(network).items():
        path, _, name = key.rpartition(".")
        if declared.get((owner, key)) is not FifoKernel or name != "ram_style":
            continue
        # A selector's case names no attribute: the selector's value is the case's Space.
        fifo = reduce(
            lambda found, name: getattr(found, name, found),
            path.split("."),
            network.member(owner),
        )
        # A FIFO whose depth no strategy sized (required, left open) has neither yet.
        storage = fifo.query(FifoKernel.storage)
        used = fifo.query(FifoKernel.exports[RESOURCES])
        found.setdefault(owner, {})[key] = {
            "ram_style": value,
            "storage": storage.value.effective_style if isinstance(storage, Available) else None,
            "resources": asdict(used.value) if isinstance(used, Available) else None,
        }
    return found


def _by_owner(values: Mapping[Key, object]) -> dict[str, dict[str, object]]:
    """``values`` by the graph name (a node or a tensor) and key that persist each."""
    found: dict[str, dict[str, object]] = {}
    for (owner, key), value in values.items():
        found.setdefault(owner, {})[key] = value
    return found


def _fifos(
    strategy: Explorer | None, entry: Mapping[str, Any] | None, placed: Sequence[str]
) -> str:
    """Whether this exploration sized the FIFOs, said so that a design without sizing does
    not read as sized: by ``size_fifos`` (the channels it sized, the FIFOs placed), or not,
    with the FIFOs placed as they were."""
    if not isinstance(strategy, SizeFifos) or entry is None:
        held = f"; FIFOs placed: {', '.join(placed)}" if placed else ""
        return "not sized here (no size_fifos): each transport as it was" + held
    channels = len(entry["channels"])
    if not channels:
        return "not sized: size_fifos found every transport held"
    return f"sized by size_fifos: {channels} channels, {len(placed)} FIFOs placed"


def _placed(seam: Seam, network: Rooted) -> dict[Key, object]:
    """Every FIFO placed on ``network``: each transport at its FIFO case, with its depth
    (None where it is open)."""
    chosen = seam.chosen(network)
    found: dict[Key, object] = {}
    for key, fifo in seam.transports(network).items():
        if chosen.get(key) == fifo:
            depth = next(
                (
                    value
                    for (owner, name), value in chosen.items()
                    if owner == key[0]
                    and name.startswith(f"{key[1]}.{fifo}.")
                    and name.endswith(".depth")
                ),
                None,
            )
            found[key] = depth
    return found


def _end_rows(network: Rooted, ends: Sequence[str]) -> dict[str, dict[str, object]]:
    """Each end's facts and cycles a frame, by its boundary channel."""
    rows: dict[str, dict[str, object]] = {}
    for name in ends:
        contract: EndContract = network.member(name).end_contract
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


def explore_kernel_choices(model: ModelWrapper, strategy: Explorer | None = None) -> Explored:
    """The KernelOps of a partition's body, ``model``, explored by ``strategy`` (None: only
    the body's open choices given their baselines) and every value persisted, on the shell
    root of the model's target, the body's pins held and recorded; see the module
    docstring."""
    if not model.graph.node:
        raise KernelOpError("no KernelOp to explore")
    others = [node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN]
    if others:
        raise KernelOpError(
            f"{', '.join(others)}: not KernelOps; exploration reads a partition's body, the "
            "KernelOps the cut put together (CutKernelPartition, partition_body)"
        )
    nodes = list(model.graph.node)
    started = time.perf_counter()
    root = shell_root(model, nodes)
    seam = Seam(read_target(model).platform)
    network = root.network()
    seam.held.update(seam.keyed(network, held_choices(model), HELD.entry))
    saved = seam.chosen(network)
    # A choice the body leaves open takes its baseline before the strategy runs; a pin
    # the body records must hold there.
    network = seam.baseline(network)
    values = seam.chosen(network)
    baselined = [key for key in values if key not in saved]
    # Who set each value: the body (``saved``), the baseline, or the strategy.
    set_by = {key: "saved" if key in saved else "baseline" for key in values}
    entry: dict[str, object] | None = None
    if strategy is not None:
        attempts, began = seam.attempts, time.perf_counter()
        before = seam.chosen(network)
        network = strategy.explore(seam, network)
        report = strategy.report()
        after = seam.chosen(network)
        label = str(getattr(strategy, "label", report["strategy"]))
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
        entry = {
            **report,
            "committed": sum(key in after for key in changed),
            "changed": {dotted(key): [before.get(key), after.get(key)] for key in changed},
            "reopened": len({key for run in seam.runs for key in run.reopened}),
            "kept": len({key for run in seam.runs for key in run.kept}),
            "re-baselined": [dotted(key) for run in seam.runs for key in run.baselined],
            "attempts": seam.attempts - attempts,
            "seconds": round(time.perf_counter() - began, 3),
        }
        if isinstance(strategy, Pinned):
            entry["held"] = [dotted(key) for key in strategy.held]
    refused = seam.refusals(network)
    if refused:
        raise KernelOpError(
            "the explored point is refused: "
            + "; ".join(
                f"{name}: " + "; ".join(map(finding_text, why)) for name, why in refused.items()
            ),
            findings=tuple(finding for why in refused.values() for finding in why),
        )
    point = network.point
    unadmitted = admission_refusal(point)
    if unadmitted:
        raise KernelOpError(
            f"the explored point is refused by the {root.row.shell!r} shell: "
            + "; ".join(map(finding_text, unadmitted)),
            findings=unadmitted,
        )
    persist(model, root, point)
    if seam.held:
        model.set(HELD, _by_owner(seam.held))
    if isinstance(strategy, TargetThroughput):
        model.set(OBJECTIVE_FPS, strategy.fps)
    elif isinstance(strategy, FOLDING):
        model.delete(OBJECTIVE_FPS)
    if isinstance(strategy, SizeFifos):
        model.delete(FIFOS_STALE)
    elif strategy is not None and not isinstance(strategy, _KEEPS_FOLDING):
        if _placed(seam, network):
            model.set(FIFOS_STALE, True)
    left = [
        dotted(choice.key) + (" (required)" if choice.required else "")
        for choice in seam.choices(network)
    ]
    cost = seam.cost(network)
    resources, over_part = _resources_report(
        cost,
        shell_resources(point),
        seam.platform,
        part_report(read_target(model).part),
        root.row.caveat,
    )
    ends: dict[str, object] = {}
    if root.ends and not cost.waiting and not cost.refused:
        boundary = [root.owners.get(path, path) for path in root.ends]
        ends = {"ends": _end_rows(network, boundary), "memory_latency": MEMORY_LATENCY}
    final = seam.chosen(network)
    placed = [dotted(key) for key in _placed(seam, network)]
    explored = {
        "strategy": entry,
        "baselined": [dotted(key) for key in baselined],
        "choices": _by_owner({key: {"value": final[key], "by": set_by[key]} for key in final}),
        "held": _by_owner(seam.held),
        "open": left,
        "fifo_storage": _fifo_storage(seam, network),
        "fifos": _fifos(strategy, entry, placed),
        "dropped": dict(root.dropped),
        **_cost_report(network, cost, resources),
        **ends,
        "seconds": round(time.perf_counter() - started, 3),
    }
    found = tuple(getattr(strategy, "findings", ()))
    return Explored(point, cost, explored, (*found, *over_part))


def explore_partition(parent: ModelWrapper, strategy: Explorer) -> Explored:
    """The partition of the build's parent graph ``parent`` explored by ``strategy``:
    its body opened through the partition node (``partition_body``), explored
    (``explore_kernel_choices``) and saved in its file. The parent graph is unchanged;
    a later exploration of the body holds the pins this one recorded."""
    _, body, body_file = partition_body(parent)
    explored = explore_kernel_choices(body, strategy)
    body.save(body_file)
    return explored


def exploration_summary(explored: Explored, label: str) -> list[tuple[int, str]]:
    """An exploration's lines for a build's log, each with its level: how many values
    the strategy ``label`` changed; each limitation found, a warning; each value a
    debugging strategy set; the choices left open; and the FIFOs."""
    report = explored.report
    changed = len(report["strategy"]["changed"])
    lines = [(logging.INFO, f"Kernel choices: {label} changed {changed}")]
    lines += [(logging.WARNING, f"Kernel choices: {each.message}") for each in explored.findings]
    lines += [
        (logging.INFO, f"{entry['by']}: {node}.{attribute} = {entry['value']!r}")
        for node, held in report["choices"].items()
        for attribute, entry in held.items()
        if str(entry["by"]).startswith("DEBUG")
    ]
    if report["open"]:
        left = ", ".join(report["open"])
        lines.append((logging.INFO, f"Kernel choices: open, refused at packaging: {left}"))
    lines.append((logging.INFO, f"FIFOs: {report['fifos']}"))
    return lines


def shell_bottleneck(model: ModelWrapper) -> Bottleneck | None:
    """The slowest members of a partition model of KernelOps' shell root, ends included,
    and their cycles a frame, its saved choices replayed (None where a member's cycles
    wait on a choice left open)."""
    root = shell_root(model, model.graph.node)
    return Seam(read_target(model).platform).cost(root.network()).bottleneck


class ExploreKernelChoices(Transformation):
    """A partition body's KernelOps explored by ``strategy`` and every value saved, on the
    shell root of the model's target, the body's pins held and recorded; ``explored``
    holds the result (``explore_kernel_choices``)."""

    def __init__(self, strategy: Explorer | None = None) -> None:
        super().__init__()
        self.strategy = strategy
        self.explored: Explored | None = None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.explored = explore_kernel_choices(model, self.strategy)
        return model, False


__all__ = [
    "ExploreKernelChoices",
    "Explored",
    "explore_kernel_choices",
    "explore_partition",
    "exploration_summary",
    "shell_bottleneck",
]
