# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a partition body's kernel choices through the DSE seam, a list of strategies
on one seam, and saving them.

``explore_kernel_choices(model, strategies)`` explores a partition's body, the model of
KernelOps the build's cut made (``CutKernelPartition``; the build's ``step_explore`` opens
it through the parent graph's node, ``partition_body``, and saves it in its file), so which
KernelOps go together is the cut's alone: a model holding any other node is refused. It
binds the body's network in its shell (``finn.custom_op.kernels.shell.in_shell``: their
kernels, the channels between them and the channels on its boundary, members named as the
graph, ``MatMul_0``; the nodes' and the channels' saved choices set as it binds, a stale one
dropped with why), which its ``Seam`` (``finn.kernels.explore``) explores, each choice keyed
by its owner, a node or a tensor, and the key there; runs each strategy of the list in
order on that one seam (the keys a ``Hold`` fixes stay held for the strategies after it),
each reopening the choices it decides, searching, and giving back the prior values of those
it left open (or, where one is refused, setting it by its policy, reported); and then (a
choice the body leaves open stays open: conversion and the cut set the rest, and nothing
fills a value in but a strategy):

- checks nothing of the FIFOs of itself: a strategy may (``Seam.check_fifos``, or
  ``Seam.checked``, which drops each FIFO, not held, that no longer keeps its producer
  within its idle time, named with why and left open, a required choice FIFO sizing sizes
  again), and the build checks once, before emission (``verify_partition``,
  ``finn.custom_op.kernels.shell``);
- refuses, once, for the whole list, and writes nothing: a ``Hold`` whose key the bind
  dropped as stale, where the seam refuses it (``pin-refused``, the bind's findings first
  among its causes); a held key not at its held value at the end (``held-not-kept``,
  ``Seam.unkept``); a network a member's validity refuses (``Seam.refusals``), its shell's
  budgets among them (its shell member's rules: its AXI-Lite buses, memory ports and doubled
  clock against the shell's row), by name (a ``KernelOpError`` carrying their findings);
- persists every value of the network where it is stated, each owner's whole
  (``persist``): a kernel's on its node, a channel's on its tensor (``finn.channel``);
  a required choice no strategy set stays open, and emission refuses it by name
  (``configured_network``);
- records, right after, what the search left to later readers (``OUTPUT_SEARCH``): the
  design it persisted, the keys held (those the list's holds fixed, ``{graph name: [key,
  ...]}``), so that the build's verification holds them, and a target throughput's frames a
  second, which a later entry that changes any kernel's choice removes;
- keeps what it found (``explored``): the network, its cost and the report. ``strategies``
  has an entry for each strategy, in order, taken from what the seam did beyond what it had
  done before the entry: the strategy's own report, every value it changed (``changed``:
  before and after, by owner), how many it reopened and gave back their prior values
  (``kept``), those it set by their policy where the prior value is refused (``by_policy``),
  those it left open (``unset``), what the fold rule did for its changes
  (``finn.kernels.explore.Seam.set``): the stored channel choices the seam cleared, each
  standing in the way of a change, with why (``cleared``), and the channel choices it set by
  the channel's policy, cleared or brought into play by a change, with their values
  (``settled``), a ``Hold``'s held keys, attempts and time; whether FIFOs were sized
  (``fifos``); how they were judged (``fifos_judged``: ``model``, where the strategy sized
  them or checked them; ``not checked``; or what the strategy's report states); and what the
  entry's last check of the transports found (``fifo_check``, only where it checked). The
  rows of the whole point are written once: every value by its owner (a node or a tensor,
  by graph name, and the key there, as the log ``report/kernel_choices.json`` names it),
  with who set it (``choices``: the body (``saved``), the entry that changed it (the fold
  rule's values among its own), or that entry by the choice's policy, ``<strategy>, its
  policy``); the keys held; the choices left open (a required one named so); each FIFO's
  memory style beside the storage FinnLib's selection gives it and what it uses
  (``fifo_storage``: its ``auto`` is FinnLib's, not Vivado's); the dropped choices with why
  (``dropped``: replay's, and those a strategy dropped through ``Seam.checked``), per member
  cycles, buffering and resources, the bottleneck, and the shell's resources against the
  platform's, ``resources``), so an outer search can compare; and every limitation it found
  (``findings``: each strategy's, its last check's and the resources'), in the report where
  each was found. Who set a value is logged in the report, never stored in the model.

``exploration_summary`` gives an exploration's lines for a build's log.

A ``Hold`` goes first, and FIFO sizing after the last strategy that changes the folding: a FIFO
sized before a fold that leaves it too small for its stream is refused when the build
verifies the partition (``fifo-stale``), unless a strategy dropped it to be sized again.

The resources are the shell's (``shell_resources``), each member its own
statement (``finn.kernels.base.KernelSpace.resources``): its partition (the partition's kernels
and its channels, boundary channels included), each mover and its static region, and
their sum, ``used``; where members state none (a memory in ``auto``, which
Vivado places), ``used`` is the sum of those that
do, a lower bound (``lower_bound``), each other named with why (``unstated``, its
findings), and the
movers and the static region are not counted. The movers' and the static region's are out
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
it, and refuses nothing.

The shell is the model's target's (``finn.custom_op.kernels.shell``): its row offers the
boundary channels its movers (none: the ``ip`` shell). Where a channel has a mover, the
report has its row (``movers``: its facts and its cycles a frame, a frame a call and 16) and
says what the cycles leave out (``memory_latency``: unmeasured, each call adds it).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

from finn.core.graph.annotations.outputs import OUTPUT_SEARCH, OUTPUTS
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, design_digest
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import ExploreError, Hold, Key, dotted
from finn.core.space import Available
from finn.custom_op.kernels.base import why
from finn.custom_op.kernels.network import persist, stale_text
from finn.custom_op.kernels.shell import (
    InShell,
    ShellResources,
    in_shell,
    shell_resources,
)
from finn.kernels.explore import (
    FIFOS_BY_MODEL,
    FIFOS_NOT_CHECKED,
    Bottleneck,
    Cost,
    FifoCheck,
    KernelExplorer,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.kernels.fifo import FifoKernel
from finn.kernels.movers import MEMORY_LATENCY, MoverContract
from finn.kernels.partition import Partition
from finn.kernels.policies import Required
from finn.kernels.utilization import SHELL_CHARACTERISED, binding, over, total
from finn.platform.capabilities import Platform
from finn.platform.catalog import part_report

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

#: No value: a key absent from a mapping of values.
_ABSENT = object()


@dataclass(frozen=True)
class Explored:
    """An exploration's result: the body's network (every value, as the strategy left
    it), its cost, the report (JSON values), and every limitation it found
    (``findings``: the strategy's, its last check's of the transports where it checked,
    then the resources'), never a refusal."""

    network: InShell
    cost: Cost
    report: Mapping[str, Any]
    findings: tuple[Finding, ...] = ()


RESOURCES_COUNTED = (
    "the shell: its partition (the partition's kernels and channels, boundary channels "
    "included), each mover and its static region, each its own statement"
)
"""What the report's resources count, as it states it."""

RESOURCES_LOWER_BOUND = (
    "a lower bound: the members that state their own resources (the partition's kernels "
    "and channels, boundary channels included); the members named under unstated state "
    "none, and the movers and the static region are not counted while they do not"
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
        "movers and the static region are models, "
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
            "movers": {name: asdict(each) for name, each in stated.movers},
            "static_region": {name: asdict(each) for name, each in stated.static_region},
        },
        "lower_bound": lower_bound,
        "unstated": {name: [finding.record() for finding in why] for name, why in unstated.items()},
        "platform": None if totals is None else asdict(totals),
        "part": dict(part),
        "share": share,
        "binding": most,
        "over": exceeded,
        "findings": [finding.record() for finding in findings],
        "counted": RESOURCES_LOWER_BOUND if lower_bound else RESOURCES_COUNTED,
        "exact": exact if caveat is None else f"{exact}; {caveat}",
    }
    return report, findings


def _over_part(most: str | None, exceeded: Mapping[str, Mapping[str, int]]) -> tuple[Finding, ...]:
    """What the report states where the shell uses more than the part has: a
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
            "explore_kernel_choices",
            f"the point uses more than the platform's part has, most of {most}: {counts}",
            (("binding", most), ("over", tuple(exceeded))),
        ),
    )


def _cost_report(network: Partition, cost: Cost, resources: dict[str, object]) -> dict[str, object]:
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


def _fifo_storage(seam: Seam, network: Partition) -> dict[str, dict[str, object]]:
    """Each FIFO's memory style, by the graph name (a node or a tensor) and key that
    persist it, beside the storage the RTL selects for it (FinnLib's ``auto``: shift,
    distributed, block or ultra) and what it uses: each ``ChannelFifo``'s, found by its
    class (``Seam.fifos``), its memory its buffer's ``ram_style``."""
    found: dict[str, dict[str, object]] = {}
    memories = set(seam.fifos(network).values())
    for (owner, key), value in seam.chosen(network).items():
        if (owner, key) not in memories:
            continue
        path = key.rpartition(".")[0]
        # A selector's case names no attribute: the selector's value is the case's Space.
        fifo = reduce(
            lambda found, name: getattr(found, name, found),
            path.split("."),
            network.member(owner),
        )
        # A FIFO whose depth no strategy sized (required, left open) has neither yet.
        storage = fifo.query(FifoKernel.storage)
        used = fifo.query(FifoKernel.resources)
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
    strategy: KernelExplorer | None, entry: Mapping[str, Any] | None, placed: Sequence[str]
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


def _judged(entry: Mapping[str, Any] | None, check: FifoCheck | None) -> str:
    """How this exploration judged the FIFOs (``fifos_judged``): as its strategy's report
    states it (``SizeFifos``: by the model), else by the model where the strategy checked
    them (``Seam.check_fifos``), else not at all."""
    if entry is not None and "fifos_judged" in entry:
        return str(entry["fifos_judged"])
    return FIFOS_NOT_CHECKED if check is None else FIFOS_BY_MODEL


def _placed(seam: Seam, network: Partition) -> dict[Key, object]:
    """Every FIFO placed on ``network``: each transport at its FIFO case, with its depth
    (None where it is open): its ``ChannelFifo``'s own, found by its class."""
    chosen = seam.chosen(network)
    fifos = seam.fifos(network)
    found: dict[Key, object] = {}
    for key, fifo in seam.transports(network).items():
        if chosen.get(key) == fifo:
            depth = next(
                (
                    chosen[held]
                    for held in fifos
                    if held[0] == key[0]
                    and held[1].startswith(f"{key[1]}.{fifo}.")
                    and held in chosen
                ),
                None,
            )
            found[key] = depth
    return found


def _mover_rows(network: Partition, movers: Sequence[str]) -> dict[str, dict[str, object]]:
    """Each mover's facts and cycles a frame, by its boundary channel."""
    rows: dict[str, dict[str, object]] = {}
    for name in movers:
        contract: MoverContract = network.member(name).mover_contract
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


def _hold_refused(
    error: ExploreError, strategy: KernelExplorer, bound: Partition
) -> ExploreError | None:
    """``error``, a hold's ``pin-refused``, as the entry that states a held key the bind
    dropped as stale refuses it: the bind's findings for those keys (``domain-membership``,
    the stored value outside its domain) first among its causes, then the seam's own; None
    where the error is not that (a key held to another value, or one the bind kept)."""
    if not isinstance(strategy, Hold):
        return None
    kept = [dotted(key) for key in strategy.held if key in bound.dropped]
    if error.findings[0].code != "pin-refused" or not kept:
        return None
    finding = error.findings[0]
    dropped = tuple(each for key in kept for each in bound.stale.get(key, ()))
    return ExploreError(
        finding.code,
        finding.owner,
        f"{finding.message}; the body stored another value for {', '.join(kept)}, which the "
        "bind dropped as stale",
        causes=(*dropped, *finding.causes),
    )


@dataclass(frozen=True)
class _Mark:
    """What the seam held before an entry: its runs, attempts, fallbacks and checks."""

    runs: int
    attempts: int
    fallbacks: frozenset[Key]
    checks: int


def _entry_report(
    seam: Seam,
    strategy: KernelExplorer,
    before: Mapping[Key, object],
    after: Mapping[Key, object],
    changed: Sequence[Key],
    since: _Mark,
    began: float,
) -> dict[str, Any]:
    """One entry's report: its strategy's own, then what it changed (``changed``: before
    and after, by owner; ``committed``), and what its runs and the changes it made through
    the seam did, taken from what lies beyond ``since`` (the seam's runs, attempts,
    fallbacks and checks before the entry): the choices it ``reopened`` and gave back their
    prior values (``kept``), set by their policy (``by_policy``), left open (``unset``),
    cleared (``cleared``) or settled (``settled``), its ``attempts``, ``seconds``, and the
    keys a ``Hold`` held; the choices it fell back on a listed order for where a ranking
    was not read; and how it left the FIFOs: ``fifos`` (the FIFOs placed on its result),
    ``fifos_judged`` and, where it checked them, ``fifo_check``."""
    runs = seam.runs[since.runs :]
    report = strategy.report()
    entry: dict[str, Any] = {
        **report,
        "committed": sum(key in after for key in changed),
        "changed": {dotted(key): [before.get(key), after.get(key)] for key in changed},
        "reopened": len({key for run in runs for key in run.reopened}),
        "kept": len({key for run in runs for key in run.kept}),
        "by_policy": [dotted(key) for run in runs for key in run.by_policy],
        "unset": {dotted(key): why(found) for run in runs for key, found in run.unset.items()},
        "cleared": {dotted(key): why(found) for run in runs for key, found in run.cleared.items()},
        "settled": {dotted(key): value for run in runs for key, value in run.settled.items()},
        "attempts": seam.attempts - since.attempts,
        "seconds": round(time.perf_counter() - began, 3),
    }
    if isinstance(strategy, Hold):
        entry["held"] = [dotted(key) for key in strategy.held]
    fell = {key: found for key, found in seam.fallbacks.items() if key not in since.fallbacks}
    if fell:
        # A ranking a policy could not read: its choice offered in listed order.
        entry["fallbacks"] = {dotted(key): why((finding,)) for key, finding in fell.items()}
    return entry


def explore_kernel_choices(
    model: ModelWrapper, strategies: Sequence[KernelExplorer] = ()
) -> Explored:
    """The KernelOps of a partition's body, ``model``, explored by ``strategies``, in
    order, on one seam (the holds of one lasting through the rest), and every value
    persisted, on its network in the shell of the model's target, the search's record kept;
    no strategy: only checked and reported. See the module docstring. The body is changed
    only at the end, by the persisted values and the record; a refusal or an exception
    before leaves it as it was."""
    if not model.graph.node:
        raise KernelOpError.refused(
            "kernel-ops-absent", "explore_kernel_choices", "no KernelOp to explore"
        )
    others = [node.name for node in model.graph.node if node.domain != KERNEL_OPS_DOMAIN]
    if others:
        raise KernelOpError.refused(
            "body-not-kernel-ops",
            "explore_kernel_choices",
            f"{', '.join(others)}: not KernelOps; exploration reads a partition's body, the "
            "KernelOps the cut put together (CutKernelPartition, partition_body)",
        )
    nodes = list(model.graph.node)
    started = time.perf_counter()
    bound = in_shell(model, nodes)
    seam = Seam(read_target(model).platform)
    network = bound
    # Who set each value: the body (what it stated), or the entry that changed it.
    set_by = dict.fromkeys(network.chosen(), "saved")
    entries: list[dict[str, Any]] = []
    found: list[Finding] = []
    objective: float | None = None
    for strategy in strategies:
        since = _Mark(len(seam.runs), seam.attempts, frozenset(seam.fallbacks), len(seam.checks))
        began = time.perf_counter()
        before = seam.chosen(network)
        try:
            network = strategy.explore(seam, network)
        except ExploreError as error:
            enriched = _hold_refused(error, strategy, bound)
            if enriched is None:
                raise
            raise enriched from error
        after = seam.chosen(network)
        report = strategy.report()
        label = str(getattr(strategy, "label", report["strategy"]))
        changed = [
            key
            for key in dict.fromkeys([*before, *after])
            if before.get(key, _ABSENT) != after.get(key, _ABSENT)
        ]
        by_policy = {key for run in seam.runs[since.runs :] for key in run.by_policy}
        for key in changed:
            if key not in after:
                set_by.pop(key, None)
            elif key in by_policy:
                set_by[key] = f"{label}, its policy"
            else:
                set_by[key] = label
        entry = _entry_report(seam, strategy, before, after, changed, since, began)
        # The entry's last check of the FIFOs, where it made one (Seam.check_fifos).
        checks = seam.checks[since.checks :]
        check = checks[-1] if checks else None
        entry["fifos"] = _fifos(strategy, entry, [dotted(key) for key in _placed(seam, network)])
        entry["fifos_judged"] = _judged(entry, check)
        if check is not None:
            entry["fifo_check"] = check.report()
        entries.append(entry)
        found += [*getattr(strategy, "findings", ()), *(() if check is None else check.findings)]
        if isinstance(strategy, TargetThroughput):
            objective = strategy.fps
        elif any(owner in network.kernels for owner, _ in changed):
            # The folding is no longer the one the target throughput explored.
            objective = None
    unkept = seam.unkept(network)
    if unkept:
        raise KernelOpError(
            "a held choice was not kept: "
            + "; ".join(f"{each.owner}: {each.message}" for each in unkept),
            tuple(each.owner for each in unkept),
            findings=unkept,
        )
    refused = seam.refusals(network)
    if refused:
        raise KernelOpError(
            "the explored point is refused: "
            + "; ".join(f"{name}: " + "; ".join(map(str, why)) for name, why in refused.items()),
            findings=tuple(
                network.located(name, finding) for name, why in refused.items() for finding in why
            ),
        )
    left = [
        dotted(key) + (" (required)" if isinstance(seam.policy(network, key), Required) else "")
        for key in network.undecided()
    ]
    cost = seam.cost(network)
    resources, over_part = _resources_report(
        cost,
        shell_resources(network),
        seam.platform,
        part_report(read_target(model).part),
        network.row.caveat,
    )
    movers: dict[str, object] = {}
    if network.movers and not cost.waiting and not cost.refused:
        movers = {"movers": _mover_rows(network, network.movers), "memory_latency": MEMORY_LATENCY}
    final = seam.chosen(network)
    explored = {
        "strategies": entries,
        "choices": _by_owner({key: {"value": final[key], "by": set_by[key]} for key in final}),
        "held": _by_owner(seam.held),
        "open": left,
        "fifo_storage": _fifo_storage(seam, network),
        "dropped": {
            **{key: stale_text(key, each) for key, each in bound.stale.items()},
            **{dotted(key): why(each) for key, each in seam.dropped.items()},
        },
        **_cost_report(network, cost, resources),
        **movers,
        "seconds": round(time.perf_counter() - started, 3),
    }
    held = _by_owner(seam.held)
    record = {
        # The design the record describes is the persisted body's, known only once it is
        # persisted; the body as it stands stands in for it here, as a digest of the same
        # form, so that the record is checked whole before anything is written.
        "design": design_digest(model),
        "held": {owner: sorted(held[owner]) for owner in sorted(held)},
        "objective_fps": objective,
    }
    OUTPUT_SEARCH.encode(record)
    # Setting the record reads the body's stored outputs back: a namespace stored malformed
    # is refused here, before anything is written.
    model.namespace(OUTPUTS)
    # The last two writes: the values, then the record that says whose they are.
    persist(model, network)
    model.set(OUTPUT_SEARCH, {**record, "design": design_digest(model)})
    return Explored(network, cost, explored, (*found, *over_part))


def exploration_summary(explored: Explored) -> list[str]:
    """An exploration's lines for a build's log: a line for each entry, how many values it
    changed and how it left the FIFOs (a ``max_throughput`` entry, how far it reached);
    each value a debugging strategy set; and the choices left open. Its limitations are its
    ``findings``."""
    report = explored.report
    lines: list[str] = []
    for entry in report["strategies"]:
        lines.append(
            f"Kernel choices: {entry['strategy']} changed {len(entry['changed'])}; "
            f"FIFOs: {entry['fifos']}"
        )
        if entry["strategy"] == "max_throughput":
            lines.append(
                f"Kernel choices: max_throughput reached {entry['bottleneck']['cycles']} cycles "
                f"a frame within {entry['budget']} ({'fits' if entry['fits'] else 'does not fit'}; "
                f"binding: {entry['binding']}; {len(entry['tried'])} budgets tried)"
            )
    lines += [
        f"{entry['by']}: {node}.{attribute} = {entry['value']!r}"
        for node, held in report["choices"].items()
        for attribute, entry in held.items()
        if str(entry["by"]).startswith("DEBUG")
    ]
    if report["open"]:
        lines.append(f"Kernel choices: open, refused at packaging: {', '.join(report['open'])}")
    return lines


def shell_bottleneck(model: ModelWrapper) -> Bottleneck | None:
    """The slowest members of a partition model of KernelOps' network, movers included,
    and their cycles a frame, its saved choices replayed (None where a member's cycles
    wait on a choice left open)."""
    bound = in_shell(model, model.graph.node)
    return Seam(read_target(model).platform).cost(bound).bottleneck


__all__ = [
    "Explored",
    "explore_kernel_choices",
    "exploration_summary",
    "shell_bottleneck",
]
