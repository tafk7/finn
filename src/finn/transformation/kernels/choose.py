# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a partition body's kernel choices through the DSE seam, one strategy at a
time, and saving them.

``explore_kernel_choices(model, strategy)`` explores a partition's body, the model of
KernelOps the build's cut made (``CutKernelPartition``; ``explore_partition``, which
the build's strategy steps run, opens it through the parent graph's node,
``partition_body``, and saves it in its file), so which KernelOps go together is the
cut's alone: a model holding any other node is refused. It binds the body's network in
its shell (``finn.custom_op.kernels.shell.in_shell``: their kernels, the channels between
them and the channels on its boundary, members named as the graph, ``MatMul_0``; the nodes'
and the channels' saved choices set as it binds, a stale one dropped with why), which its
``Seam`` (``finn.kernels.explore``) explores, each choice keyed by its owner, a node or a
tensor, and the key there; holds the keys the body records (``HELD``; a held key the
bound network does not state is refused, ``held-not-taken``, ``held-owner-unknown``); runs
the strategy, which reopens the choices it decides, searches, and gives back the prior
values of those it left open (or, where one is refused, sets it by its policy, reported);
and then (a choice the body leaves open stays open: conversion and the cut set the rest,
and nothing fills a value in but a strategy):

- checks nothing of the FIFOs of itself: a strategy may (``Seam.check_fifos``, or
  ``Seam.checked``, which drops each FIFO, not held, that no longer keeps its producer
  within its idle time, named with why and left open, a required choice FIFO sizing sizes
  again), and the build checks once, before emission (``verify_partition``,
  ``finn.custom_op.kernels.shell``);
- refuses a network a member's validity refuses (``Seam.refusals``) or over its shell's
  budgets (``budget_refusal``: its AXI-Lite buses, memory ports and doubled clock against
  the shell's row), by name (a ``KernelOpError`` carrying their findings);
- persists every value of the network where it is stated, each owner's whole
  (``persist``): a kernel's on its node, a channel's on its tensor (``finn.channel``);
  a required choice no strategy set stays open, and emission refuses it by name
  (``configured_network``);
- records the keys held, those the body recorded and those a ``Hold`` fixed, in the
  body as keys (``HELD``, ``{graph name: [key, ...]}``; the values are the ones the nodes
  and tensors state), so that every later exploration of the body holds them: a hold lasts
  for the build, and going back to the defaults keeps it; and a
  target throughput's frames a second (``OBJECTIVE_FPS``), which a later exploration
  whose strategy changes any kernel's choice removes;
- keeps what it found (``explored``): the network, its cost and the report: the strategy
  with every value it changed (``changed``: before and after, by owner), how many it
  reopened and gave back their prior values (``kept``), those it set by their policy
  where the prior value is refused (``by_policy``), those it left open (``unset``), the
  stored channel choices the interim channel rule derived again on its way
  (``rederived``) and those it set as the new facts brought them into play
  (``interim``), a ``Hold``'s held keys, attempts and time; every value by its owner (a
  node or a tensor, by graph name, and the key there, as the log
  ``report/kernel_choices.json`` names it), with who set it (``saved``: the body, the
  strategy, the strategy by the choice's policy, ``<strategy>, its policy``, or the
  interim channel rule, a channel's value its rebuild gave it on the strategy's way,
  ``interim channel rule``); the keys
  held; the choices left open (a required one named so); each FIFO's memory style beside
  the storage FinnLib's selection gives it and what it uses
  (``fifo_storage``: its ``auto`` is FinnLib's, not Vivado's); whether FIFOs were
  sized; how they were judged (``fifos_judged``: ``model``, where the strategy sized them
  or checked them; ``not checked``; or what the strategy's report states); the dropped
  choices with why (``dropped``: replay's, and those the strategy dropped through
  ``Seam.checked``), what the strategy's last check of the transports found
  (``fifo_check``, only where it checked), per member cycles, buffering and resources, the
  bottleneck, and the shell's resources against the platform's, ``resources``), so an
  outer search can compare; and every limitation it found (``findings``: the strategy's,
  its last check's and the resources'), in the report where each was found. Who set a
  value is logged in the report, never stored in the model.

``exploration_summary`` gives an exploration's lines for a build's log.

A ``Hold`` goes first, and FIFO sizing after the last strategy that changes the folding: a FIFO
sized before a fold that leaves it too small for its stream is refused when the build
verifies the partition (``fifo-stale``), unless a strategy dropped it to be sized again.

The resources are the shell's (``shell_resources``), each member its own
statement (``finn.kernels.base.RESOURCES``): its partition (the partition's kernels
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

from qonnx.transformation.base import Transformation

from finn.core.graph.annotations.choices import HELD, OBJECTIVE_FPS, held_keys
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, partition_body
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Available
from finn.custom_op.kernels.base import why
from finn.custom_op.kernels.network import Partition, persist, stale_text
from finn.custom_op.kernels.shell import (
    InShell,
    ShellResources,
    budget_refusal,
    in_shell,
    shell_resources,
)
from finn.kernels.base import RESOURCES
from finn.kernels.composer import Key
from finn.kernels.explore import (
    FIFOS_BY_MODEL,
    FIFOS_NOT_CHECKED,
    Bottleneck,
    Cost,
    Explorer,
    FifoCheck,
    Hold,
    Seam,
    SizeFifos,
    TargetThroughput,
    dotted,
)
from finn.kernels.fifo import FifoKernel
from finn.kernels.movers import MEMORY_LATENCY, MoverContract
from finn.kernels.policies import Required
from finn.kernels.utilization import SHELL_CHARACTERISED, binding, over, total
from finn.platform.capabilities import Platform
from finn.platform.catalog import part_report

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

#: No value: a key absent from a mapping of values.
_ABSENT = object()

#: Who set a channel's value the network's rebuild gave it, as a report names it.
INTERIM = "interim channel rule"


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
            ExploreKernelChoices.__name__,
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


def _held_values(
    network: Partition, saved: Mapping[Key, object], held: Mapping[str, Sequence[str]]
) -> dict[Key, object]:
    """The values of the keys the body's mark holds (``held``, by owner), the ones
    ``network`` states (``saved``). A held key is the user's exact value, so one the network
    does not take is refused, by name, never dropped and set another way: an owner that is
    no member of the network (``held-owner-unknown``), or a key the bound network states no
    value for, because the bind dropped the stored one or it is gone (``held-not-taken``;
    the bind's findings are its causes). A key stored out of play (dormant) or waiting is
    stated, by its stored value."""
    members = set(network.members)
    kernels = set(network.kernels)
    values: dict[Key, object] = {}
    lost: list[Finding] = []
    refused_keys: list[str] = []
    for owner, keys in held.items():
        for key in keys:
            if owner not in members:
                code, reason = "held-owner-unknown", "no member of the network owns it"
                subject: tuple[tuple[str, str], ...] = ()
            elif (owner, key) not in saved:
                code, reason = "held-not-taken", "the bound network states no value for it"
                subject = (("node" if owner in kernels else "tensor", owner),)
            else:
                values[(owner, key)] = saved[(owner, key)]
                continue
            refused_keys.append(dotted((owner, key)))
            lost.append(
                Finding(
                    FindingKind.REJECTION,
                    code,
                    "explore_kernel_choices",
                    f"{dotted((owner, key))}: held by the body, but {reason}; the held value "
                    "was not taken, and nothing takes its place",
                    (("keys", (key,)), *subject),
                    causes=network.dropped.get((owner, key), ()),
                )
            )
    if lost:
        raise KernelOpError(
            "a held choice is refused: " + "; ".join(each.message for each in lost),
            tuple(refused_keys),
            findings=lost,
        )
    return values


def explore_kernel_choices(model: ModelWrapper, strategy: Explorer | None = None) -> Explored:
    """The KernelOps of a partition's body, ``model``, explored by ``strategy`` (None:
    nothing changed, only checked and reported) and every value persisted, on its network
    in the shell of the model's target, the body's holds kept and recorded; see the module
    docstring."""
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
    # What the body states: every stored choice, in play or not.
    saved = network.chosen()
    # The body's mark names the keys held; their values are the ones the network states.
    seam.held.update(_held_values(network, saved, held_keys(model)))
    # Who set each value: the body (``saved``), or the strategy.
    set_by = dict.fromkeys(saved, "saved")
    entry: dict[str, object] | None = None
    changed: list[Key] = []
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
        by_policy = {key for run in seam.runs for key in run.by_policy}
        by_rule = {key for run in seam.runs for key in (*run.rederived, *run.interim)}
        for key in changed:
            if key not in after:
                set_by.pop(key, None)
            elif key in by_policy:
                set_by[key] = f"{label}, its policy"
            else:
                set_by[key] = INTERIM if key in by_rule else label
        entry = {
            **report,
            "committed": sum(key in after for key in changed),
            "changed": {dotted(key): [before.get(key), after.get(key)] for key in changed},
            "reopened": len({key for run in seam.runs for key in run.reopened}),
            "kept": len({key for run in seam.runs for key in run.kept}),
            "by_policy": [dotted(key) for run in seam.runs for key in run.by_policy],
            "unset": {
                dotted(key): why(found) for run in seam.runs for key, found in run.unset.items()
            },
            "rederived": {
                dotted(key): why(found) for run in seam.runs for key, found in run.rederived.items()
            },
            "interim": [dotted(key) for run in seam.runs for key in run.interim],
            "attempts": seam.attempts - attempts,
            "seconds": round(time.perf_counter() - began, 3),
        }
        if isinstance(strategy, Hold):
            entry["held"] = [dotted(key) for key in strategy.held]
        if seam.fallbacks:
            # A ranking a policy could not read: its choice offered in listed order.
            entry["fallbacks"] = {
                dotted(key): why((finding,)) for key, finding in seam.fallbacks.items()
            }
    refused = seam.refusals(network)
    if refused:
        raise KernelOpError(
            "the explored point is refused: "
            + "; ".join(f"{name}: " + "; ".join(map(str, why)) for name, why in refused.items()),
            findings=tuple(
                network.located(name, finding) for name, why in refused.items() for finding in why
            ),
        )
    over_budget = budget_refusal(network)
    if over_budget:
        raise KernelOpError(
            f"the explored point is refused by the {network.row.shell!r} shell: "
            + "; ".join(map(str, over_budget)),
            findings=over_budget,
        )
    persist(model, network)
    if seam.held:
        held = _by_owner(seam.held)
        model.set(HELD, {owner: sorted(held[owner]) for owner in sorted(held)})
    if isinstance(strategy, TargetThroughput):
        model.set(OBJECTIVE_FPS, strategy.fps)
    elif any(owner in network.kernels for owner, _ in changed):
        # The folding is no longer the one the target throughput explored.
        model.delete(OBJECTIVE_FPS)
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
    placed = [dotted(key) for key in _placed(seam, network)]
    # The strategy's last check of the FIFOs, where it made one (Seam.check_fifos).
    check = seam.checks[-1] if seam.checks else None
    explored = {
        "strategy": entry,
        "choices": _by_owner({key: {"value": final[key], "by": set_by[key]} for key in final}),
        "held": _by_owner(seam.held),
        "open": left,
        "fifo_storage": _fifo_storage(seam, network),
        "fifos": _fifos(strategy, entry, placed),
        "fifos_judged": _judged(entry, check),
        "dropped": {
            **{key: stale_text(key, found) for key, found in bound.stale.items()},
            **{dotted(key): why(found) for key, found in seam.dropped.items()},
        },
        **({} if check is None else {"fifo_check": check.report()}),
        **_cost_report(network, cost, resources),
        **movers,
        "seconds": round(time.perf_counter() - started, 3),
    }
    found = tuple(getattr(strategy, "findings", ()))
    checked = () if check is None else check.findings
    return Explored(network, cost, explored, (*found, *checked, *over_part))


def explore_partition(parent: ModelWrapper, strategy: Explorer) -> Explored:
    """The partition of the build's parent graph ``parent`` explored by ``strategy``:
    its body opened through the partition node (``partition_body``), explored
    (``explore_kernel_choices``) and saved in its file. The parent graph is unchanged;
    a later exploration of the body holds the keys this one recorded."""
    _, body, body_file = partition_body(parent)
    explored = explore_kernel_choices(body, strategy)
    body.save(body_file)
    return explored


def exploration_summary(explored: Explored, label: str) -> list[str]:
    """An exploration's lines for a build's log: how many values the strategy ``label``
    changed; each value a debugging strategy set; the choices left open; and the FIFOs.
    Its limitations are its ``findings``."""
    report = explored.report
    changed = len(report["strategy"]["changed"])
    lines = [f"Kernel choices: {label} changed {changed}"]
    lines += [
        f"{entry['by']}: {node}.{attribute} = {entry['value']!r}"
        for node, held in report["choices"].items()
        for attribute, entry in held.items()
        if str(entry["by"]).startswith("DEBUG")
    ]
    if report["open"]:
        lines.append(f"Kernel choices: open, refused at packaging: {', '.join(report['open'])}")
    lines.append(f"FIFOs: {report['fifos']}")
    return lines


def shell_bottleneck(model: ModelWrapper) -> Bottleneck | None:
    """The slowest members of a partition model of KernelOps' network, movers included,
    and their cycles a frame, its saved choices replayed (None where a member's cycles
    wait on a choice left open)."""
    bound = in_shell(model, model.graph.node)
    return Seam(read_target(model).platform).cost(bound).bottleneck


class ExploreKernelChoices(Transformation):
    """A partition body's KernelOps explored by ``strategy`` and every value saved, on its
    network in the shell of the model's target, the body's holds kept and recorded;
    ``explored`` holds the result (``explore_kernel_choices``)."""

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
