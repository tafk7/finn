# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A DataflowCore's core in its shell (``shell_point``): the core of its body's KernelOps
(``finn.custom_op.kernels.network``) in the shell of the model's target
(``finn.platform.shells.shell_row``, ``body_row``), one design space
(``finn.kernels.shell.ShellKernel``): the core, the mover the row offers on each of its edges
(``row_offer``) and the row's limits (``row_facts``). The node's point
(``finn.custom_op.partition.DataflowCore``), which a step holding only the body reads by the
node's name. Every row composes one, the ``ip`` row's with no movers and no limit.

- **rules**, the shell's typed limits (its shell member, ``finn.kernels.shell_rules``): the
  units it places within the row's ``units`` (``units-exceeded``); the control windows of
  its movers and its core, placed in order each aligned to itself, within its control
  ``aperture`` (``control-window-exceeded``); its movers' memory ports within its memory
  target's ``masters`` (``memory-masters-exceeded``); no memory port of the core's own
  (``core-memory-port``); and a member that takes the doubled clock on a row that gives
  one (``clocks.doubled``; ``clock-unavailable``). The ``ip`` row bounds none but the
  core's memory ports. Each reads the members' derived values, so a value that breaks one
  is refused in the shell, where the seam, the cut and emission read it. Emission checks
  the core's netlist (its AXI-Lite buses' windows, its AXI4 initiators and its doubled
  clock) against the rules' (``finn.kernels.shell_rules.counts``): a difference is an
  error naming both (``shell-counts-differ``);
- **resources** (``shell_resources``, by member; the seam reads their sum, the shell's
  ``Kernel.resources``): its core's members' own statements, each mover's (its contract's)
  and its row's static region's (``ShellRow.fixed``, ``ShellRules.static_region``), at
  the memory ports and AXI-Lite buses the core and its movers connect. The movers' and the
  static region's are out of context (``SHELL_CHARACTERISED``), which overstates the placed
  shell. The ``ip`` shell has neither, so its resources are its core's.

``configured_network`` is a core as emission builds it (``Configured``: its composition in
its shell, its module and its cycles a frame), bound only: a stored choice it dropped as
stale, a rule its stored choices break (a member's, a mover's, or a budget of its shell's),
a choice left open, refuse it, named. ``verify_partition`` is the one
check of its FIFOs against their streams before emission (a strategy may call the seam's
check too, ``Seam.check_fifos``), and records the design it verified on the body; emission
(packaging, the shell's integration, the testbench) compares it (``refuse_unverified``),
and refuses a design never verified or changed since. Simulation and elaboration only
bind. The cut makes a body's channel choices (``make_cut_choices``) on its core in its
shell, where a boundary channel's side is its mover's, by the channel's policies
(``Channel.policies``), and writes them with ``persist``. Each is given the core's name,
its node's (``finn.custom_op.partition.DataflowCore``), which names its refusals; its module's
name leaves it out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from finn.core.graph.annotations.outputs import OUTPUT_VERIFIED
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest, search_record
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import Key, dotted
from finn.core.space import (
    Available,
    Inapplicable,
    QueryResult,
    Rejected,
    Unresolved,
    inspection,
)
from finn.custom_op.kernels.network import (
    bound_from,
    composed,
    declaration,
    persist,
    stale_text,
)
from finn.kernels.artifacts.module import Composed
from finn.kernels.base import Kernel
from finn.kernels.composer import DataflowCoreDeclaration
from finn.kernels.configure import refusal
from finn.kernels.dataflow_core import Bound, DataflowCoreKernel
from finn.kernels.explore import CoreComposition, FifoCheck, Seam
from finn.kernels.grouper import GrouperKernel
from finn.kernels.iodma import IodmaKernel
from finn.kernels.movers import MoverContract, MoverOffer
from finn.kernels.policies import Required, policy
from finn.kernels.shell import ShellKernel, grouper_name, shell
from finn.kernels.shell_rules import SHELL, ShellRules, counts, netlist_counts
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources
from finn.platform.shells import ShellRow, mover_offer, placed_offer, shell_row

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class ShellResources:
    """A shell's resources by member, each its own statement: ``partition``, its core (its
    kernels and channels, the boundary channels' stages included); ``movers``, each mover
    by its name in the shell (``in0``); ``groupers``, the grouper of each core presenting a
    control bus, by its name in the shell (``DataflowCore_0_grouper``; a core presenting
    none has none in hardware); ``static_region``, each part of its row's static region by
    its role (``FixedShell.resources``). ``total`` is their sum, the shell's
    ``Kernel.resources``."""

    partition: Resources
    movers: tuple[tuple[str, Resources], ...]
    groupers: tuple[tuple[str, Resources], ...]
    static_region: tuple[tuple[str, Resources], ...]

    @property
    def total(self) -> Resources:
        named = (*self.movers, *self.groupers, *self.static_region)
        return total((self.partition, *(used for _, used in named)))


def row_facts(row: ShellRow) -> dict[str, object]:
    """The facts of ``row`` a partition in its shell holds
    (``finn.kernels.shell_rules.ShellRules``), as plain values: the shell's name, the most
    units it places, its control's least window and aperture, its memory targets (each
    name and the most mover ports it takes), whether it gives an aligned doubled clock, and
    its static region, as the row states them."""
    control = row.control
    return {
        "shell": row.shell,
        "units": row.units,
        "least_window": None if control is None else control.least_window,
        "aperture": None if control is None else control.aperture,
        "targets": tuple((target.name, target.masters) for target in row.memory),
        "doubled": row.clocks.doubled,
        "fixed": row.fixed,
    }


def row_offer(row: ShellRow) -> MoverOffer | None:
    """The mover ``row`` offers each edge of a core: the offer its one placed mover kind
    makes at its width cap (``placed_offer``, ``mover_offer``), or None. A row a core cannot
    place as it is declared is refused (``offer-unplaced``)."""
    offer = placed_offer(row)
    if offer is None:
        return None
    (mover,) = offer.movers
    assert offer.width_cap is not None
    return mover_offer(mover, offer.width_cap)


def body_row(model: ModelWrapper) -> ShellRow:
    """The row of the shell of ``model``'s target (``shell_row``)."""
    target = read_target(model)
    return shell_row(target.shell, target.board)


def _in_shell(
    row: ShellRow, offer: MoverOffer | None
) -> Callable[[DataflowCoreDeclaration], ShellKernel]:
    """The shell of ``row`` around a declared core, ``offer`` on each of its edges."""
    facts = row_facts(row)

    def made(declared: DataflowCoreDeclaration) -> ShellKernel:
        return shell((declared,), row=facts, offer=offer)

    return made


def shell_point(model: ModelWrapper, name: str) -> Bound[ShellKernel]:
    """The core of ``model``'s KernelOps, a DataflowCore's body (``network``), its module
    named ``name`` (the node's), in the shell of the model's target: its row (``body_row``)
    offers the mover on each edge (``row_offer``) and states its limits (``row_facts``);
    its stored choices bound. The node's point
    (``finn.custom_op.partition.DataflowCore.point``)."""
    row = body_row(model)
    offer = row_offer(row)
    return bound_from(model, model.graph.node, name=name, made=_in_shell(row, offer), offer=offer)


def shell_space(model: ModelWrapper, name: str) -> ShellKernel:
    """The core of ``model``'s KernelOps in the shell of its target, as ``shell_point``
    makes it, its design space with nothing chosen: the node's
    (``finn.custom_op.partition.DataflowCore.bound``)."""
    row = body_row(model)
    offer = row_offer(row)
    declared = declaration(model, model.graph.node, name=name, offer=offer)
    return composed(declared, _in_shell(row, offer))


def _together(results: Iterable[QueryResult[Any]]) -> Rejected | Unresolved:
    """Results that are not available, as one: refused where one is, else waiting."""
    held = list(results)
    findings = tuple(
        dict.fromkeys(finding for each in held for finding in getattr(each, "findings", ()))
    )
    if any(isinstance(each, Rejected) for each in held):
        return Rejected(findings)
    return Unresolved(findings)


# -- the shell, as functions over its composition ------------------------------------------


def _netlist(
    network: CoreComposition,
) -> QueryResult[tuple[tuple[int | None, ...], int, bool]]:
    """``netlist_counts`` of its core's netlist and its edge's buses
    (``CoreComposition.netlist``); or what its netlist waits on, or why it is refused."""
    found = network.netlist()
    if not isinstance(found, Available):
        return _together([found])
    return Available(netlist_counts(*found.value))


def mover_contracts(network: CoreComposition) -> QueryResult[tuple[tuple[str, MoverContract], ...]]:
    """Each mover of its shell, by name, with its contract (``IodmaKernel.contract``): none
    on the ``ip`` shell."""
    found: list[tuple[str, MoverContract]] = []
    waiting: list[QueryResult[object]] = []
    for name in network.movers:
        placed = network.query(network.placed(name), IodmaKernel.contract)
        if isinstance(placed, Available):
            found.append((name, placed.value))
        else:
            waiting.append(placed)
    if waiting:
        return _together(waiting)
    return Available(tuple(found))


def _counts_agree(network: CoreComposition, label: str) -> None:
    """Refuse ``network``, decided, whose core's netlist's count (``netlist_counts``) is
    not its shell member's (``finn.kernels.shell_rules.counts``): (control windows, memory
    ports, doubled clock), the two named (``shell-counts-differ``), or either not known.
    Emission's check that its shell's rules placed what it emits."""
    found = _netlist(network)
    ruled = counts(network.member(SHELL))
    netlist = found.value if isinstance(found, Available) else found
    stated = ruled.value if isinstance(ruled, Available) else ruled
    if netlist != stated:
        raise KernelOpError.refused(
            "shell-counts-differ",
            "configured_network",
            f"{label}: the netlist's count (control windows, memory ports, doubled clock) "
            f"{netlist} is not the shell's rules' {stated}",
            node=label,
        )


def _resources(network: CoreComposition) -> QueryResult[ShellResources]:
    """``shell_resources``, as a query's result: its core's, each mover's, each grouper's
    whose core presents a control bus, and its static region's, each its own statement."""
    partition = network.core_answer(Kernel.resources)
    movers = [
        (name, network.query(network.placed(name), Kernel.resources)) for name in network.movers
    ]
    placed = [
        network.placed(grouper_name(core))
        for core in network.names.cores
        if network.placed(grouper_name(core)) in network.names.placed
    ]
    windows = [network.query(name, GrouperKernel.window) for name in placed]
    groupers = [
        (network.names.placed[name], network.query(name, Kernel.resources))
        for name, window in zip(placed, windows)
        if not isinstance(window, Available) or window.value
    ]
    region = network.query(SHELL, ShellRules.static_region)
    found = [partition, *windows, *(used for _, used in (*movers, *groupers)), region]
    if not all(isinstance(each, Available) for each in found):
        return _together(each for each in found if not isinstance(each, Available))
    assert isinstance(partition, Available) and isinstance(region, Available)
    return Available(
        ShellResources(
            partition.value,
            tuple((name, used.value) for name, used in movers if isinstance(used, Available)),
            tuple((name, used.value) for name, used in groupers if isinstance(used, Available)),
            region.value,
        )
    )


@dataclass(frozen=True)
class Configured:
    """A logically configured partition and its physical module, without a timing query.

    Every choice in play is stored and validity/presence and the shell's rules are complete.
    Emission does not prove a rate or timed FIFO sufficiency. The existing ``cycles``
    readout remains an explicit, separate request until timing consumers migrate.
    """

    network: CoreComposition
    module: Composed
    row: ShellRow

    @property
    def cycles(self) -> int:
        """Request the existing cycle readout; never infer it from logical configuration.

        Its shell's class answers it (``Kernel.frame_cycles``: the most any member takes,
        its movers among them). A
        missing or refused member estimate fails this request, not configuration or
        emission. This preserves the held timing consumers' contract without making them a
        prerequisite of the untimed build bridge.
        """
        cycles = self.network.answer(Kernel.frame_cycles)
        if not isinstance(cycles, Available):
            found = getattr(cycles, "findings", ())
            raise KernelOpError(
                "cycle characterization unavailable: "
                + "; ".join(map(Finding.subject_line, found)),
                findings=found,
            )
        return cycles.value

    def channel(self, tensor: str) -> Any:
        """The channel carrying ``tensor``: its point."""
        return self.network.member(tensor)

    def node(self, name: str) -> str:
        """Member ``name``'s node in the module: the prefix of its instances."""
        return self.network.node(name)


def core_grouping(network: CoreComposition) -> tuple[tuple[int, ...], int] | None:
    """Where the grouper of ``network``'s core places its control buses in its unit
    (``GrouperKernel.map``) and the unit's window; None where its shell places no grouper
    (the ``ip`` shell) or its core presents no bus (its unit is the core). A
    ``KernelOpError`` with the findings of a grouper answering neither."""
    core = network.names.core
    name = None if core is None else network.placed(grouper_name(core))
    if name is None or name not in network.names.placed:
        return None
    mapped = network.query(name, GrouperKernel.map)
    window = network.query(name, GrouperKernel.window)
    if not isinstance(mapped, Available) or not isinstance(window, Available):
        found = _together(each for each in (mapped, window) if not isinstance(each, Available))
        raise KernelOpError(
            f"{core}: its grouper places no control bus: "
            + "; ".join(map(Finding.subject_line, found.findings)),
            findings=found.findings,
        )
    return (mapped.value, window.value[0]) if window.value else None


def shell_resources(network: CoreComposition) -> ShellResources | tuple[Finding, ...]:
    """The resources of ``network``, a core in its shell (``shell_point``), by member and
    their total (``ShellResources``); or why it states none, its findings: a member that
    states none, or the open choices they wait on (each ``decision-unassigned``)."""
    found = _resources(network)
    return found.value if isinstance(found, Available) else found.findings


def configured_network(
    model: ModelWrapper, name: str
) -> tuple[Configured, tuple[tuple[str, str], ...]]:
    """A DataflowCore's body's core, named ``name`` (the node's), in the shell of
    its target (``shell_point``), bound with
    its nodes' and its tensors' channel choices, as emission builds it (``Configured``),
    and its boundary (tensor, port). Every choice is stored where it is created (a
    KernelOp's at conversion, a channel's at the cut) and replaced by the strategies, so
    nothing is completed here. A stale choice; an open choice in play, each by name: one no
    value of which is possible (each refused value's rules; a ``choice-no-valid-value``
    cause each, owned by the key, the rules' findings its causes, as the cut reports it), a
    required one no strategy set ("required: add size_fifos or
    debug_placeholder to step_explore's strategies"), any other one never set ("not set: the cut,
    MakeCutChoices, or a strategy sets it"; one finding, ``choices-open``); a partition
    over the shell's limits (its shell member's validity: "refused by the <shell> shell",
    each broken limit a finding), or whose module is refused (by a member, or by a rule of
    the partition's own), or a member's validity refuses (a rule its stored choices break,
    which the bind binds, or a mover's or a connection's rule: "refused", each finding
    naming its member, where it is one's); or
    graph inputs and outputs
    out of port order refuse, named. Its netlist's count (``netlist_counts``) must be its
    shell's rules' (``shell-counts-differ``, naming both, otherwise). A dormant
    choice (stored under a case not selected) is not open, and refuses nothing. It only
    binds: it does not check the FIFOs against their streams (``verify_partition`` does), so
    a simulation can run a FIFO the model calls too small."""
    label = name
    row = body_row(model)
    held = shell_point(model, name)
    if held.stale:
        keys = tuple(held.stale)
        message = f"{label}: stale choices, refused by the partition: " + "; ".join(
            stale_text(key, found) for key, found in held.stale.items()
        )
        raise KernelOpError(
            message,
            keys,
            findings=(
                Finding(
                    FindingKind.REJECTION,
                    "choices-stale",
                    "configured_network",
                    message,
                    (("keys", keys), ("node", label)),
                    tuple(finding for found in held.stale.values() for finding in found),
                ),
            ),
        )
    bound = CoreComposition(held.point)
    open_keys = bound.undecided()
    if open_keys:
        required = {
            key
            for key, item in bound.decisions().items()
            if isinstance(policy(bound.owner(key[0]), item), Required)
        }
        listed = bound.viable(key for key in open_keys if key not in required)
        impossible = {
            key: _impossible(bound, key, viable.values)
            for key, viable in listed
            if viable.values is not None and not viable.cases
        }
        named = [
            f"{dotted(key)}: {impossible[key].message}"
            if key in impossible
            else f"{dotted(key)} (required: add size_fifos or debug_placeholder to "
            "step_explore's strategies)"
            if key in required
            else f"{dotted(key)} (not set: the cut, MakeCutChoices, or a strategy sets it)"
            for key in open_keys
        ]
        text = f"{label}: open choices, refused at emission: " + "; ".join(named)
        keys = tuple(dotted(key) for key in open_keys)
        raise KernelOpError(
            text,
            keys,
            findings=(
                Finding(
                    FindingKind.REJECTION,
                    "choices-open",
                    "configured_network",
                    text,
                    (("keys", keys), ("node", label)),
                    tuple(impossible.values()),
                ),
            ),
        )
    over_limits = refusal(bound.validity(SHELL))
    if over_limits:
        raise KernelOpError(
            f"{label}: refused by the {row.shell!r} shell: " + "; ".join(map(str, over_limits)),
            findings=over_limits,
        )
    module = bound.core_answer(DataflowCoreKernel.module)
    found = [] if isinstance(module, Available) else list(module.findings)
    if not found and isinstance(module, Available):
        _counts_agree(bound, label)
        # A rule a member's stored choices break that its module does not read.
        found = [
            bound.located(name, finding)
            for name in (*bound.members, *bound.names.placed)
            for finding in refusal(bound.validity(name))
        ]
    if found or not isinstance(module, Available):
        refused = tuple(dict.fromkeys(found))
        raise KernelOpError(
            f"{label}: refused: " + "; ".join(map(Finding.subject_line, refused)),
            findings=refused,
        )
    # Computable readouts do not prove that arbitrary rules or presence resolved.
    # Refused rules retain their detailed diagnostics above; now require completeness.
    validity = inspection.validity(bound.point)
    waiting = [
        key
        for key, state in inspection.states(bound.point)
        if isinstance(state, (inspection.Open, inspection.Waiting))
    ]
    unsettled = [
        (name, result)
        for name, result in inspection.presence(bound.point)
        if not isinstance(result, (Available, Inapplicable))
    ]
    if validity.result != Available(True) or waiting or unsettled:
        names = (
            tuple(piece.owner for piece in validity.pieces)
            + tuple(waiting)
            + tuple(name for name, _ in unsettled)
        )
        text = f"{label}: validity or presence not complete at emission: " + ", ".join(names)
        validity_findings = (
            () if isinstance(validity.result, Available) else validity.result.findings
        )
        raise KernelOpError(
            text,
            findings=(
                Finding(
                    FindingKind.REJECTION,
                    "design-incomplete",
                    "configured_network",
                    text,
                    (("node", label),),
                    tuple(
                        dict.fromkeys(
                            (
                                *validity_findings,
                                *(
                                    finding
                                    for _, result in unsettled
                                    for finding in result.findings
                                ),
                            )
                        )
                    ),
                ),
            ),
        )
    ports = dict(bound.boundary)
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [item.name for item in model.graph.input if item.name not in initializers]
    expected = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    expected |= {item.name: f"m_axis_{index}" for index, item in enumerate(model.graph.output)}
    if ports != expected:
        raise KernelOpError.refused(
            "ports-out-of-order",
            "configured_network",
            f"{label}: the partition's ports {ports} are not its graph's inputs and"
            f" outputs in order {expected}",
            node=label,
        )
    return Configured(bound, module.value, row), bound.boundary


def _impossible(
    bound: CoreComposition, key: Key, values: tuple[inspection.Possible | inspection.Refused, ...]
) -> Finding:
    """Why open choice ``key`` of ``bound`` can take no value (``choice-no-valid-value``,
    owned by its dotted key): each listed value with the rules it breaks (each by its key in
    the choice's member, another member's by its dotted key), those rules' findings, each
    once, its causes."""

    def named(rule: str) -> str:
        owner, below = bound.pair(key[0], rule)
        return below if owner == key[0] else dotted((owner, below))

    refused = [item for item in values if isinstance(item, inspection.Refused)]
    why = "; ".join(
        f"{item.value!r} breaks " + ", ".join(named(rule.rule) for rule in item.broken)
        for item in refused
    )
    causes = tuple(
        dict.fromkeys(
            finding for item in refused for rule in item.broken for finding in rule.result.findings
        )
    )
    return Finding(
        FindingKind.REJECTION,
        "choice-no-valid-value",
        dotted(key),
        f"no value is possible: {why or 'its domain lists none'}",
        causes=causes,
    )


def _held_values(network: CoreComposition, held: Mapping[str, Sequence[str]]) -> dict[Key, object]:
    """The values of the keys the search's record holds (``held``, by owner), the ones
    ``network`` states. A held key is the user's exact value, so one the network does not
    take is refused, by name, never dropped: an owner that is no member of the network
    (``held-owner-unknown``), or a key the bound network states no value for, because it is
    gone (``held-not-taken``; a stored one the bind dropped refuses the network before,
    ``configured_network``). A key stored out of play (dormant) or waiting is stated, by its
    stored value."""
    members = set(network.members)
    kernels = set(network.kernels)
    saved = network.chosen()
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
                    "verify_partition",
                    f"{dotted((owner, key))}: held by the search's record, but {reason}; the "
                    "held value was not taken, and nothing takes its place",
                    (("keys", (key,)), *subject),
                )
            )
    if lost:
        raise KernelOpError(
            "a held choice is refused: " + "; ".join(each.message for each in lost),
            tuple(refused_keys),
            findings=lost,
        )
    return values


def verify_partition(model: ModelWrapper, name: str) -> tuple[Configured, FifoCheck]:
    """A DataflowCore's body's configured network, named ``name``
    (``configured_network``, whose refusals it
    keeps) with its FIFOs checked against their streams at its bottleneck period
    (``Seam.check_fifos``: sizing's own model, whatever changed since a FIFO was sized, an
    edit by hand included), and what the check found; the design it verified recorded on
    the model (``OUTPUT_VERIFIED``: its ``design_digest``), which emission compares
    (``refuse_unverified``). The keys the search held are the ones of its record, read only
    while the record is the body's own (``search_record``: current; a stale one or none holds
    nothing); one the network does not take is refused, named (``held-not-taken``,
    ``held-owner-unknown``). A FIFO not held that no longer keeps its producer within its
    idle time is refused, named (``fifo-stale``: add size_fifos to step_explore's strategies), and
    nothing is recorded; a held one (``fifo-held-stale``) and a ``direct`` that throttles
    its producer (``direct-throttles``) are limitations, the check's ``findings``. This
    held timed path explicitly requests cycles first; unavailable/refused estimates
    leave the verification marker unchanged. The model is changed only by the record;
    its caller saves it."""
    label = name
    point, _ = configured_network(model, name)
    # This held consumer performs a timed FIFO check. Retain its former eager
    # cycle-refusal precondition explicitly: FifoCheck(None) is not verification.
    _ = point.cycles
    seam = Seam(read_target(model).platform)
    _, record = search_record(model)
    seam.held.update(_held_values(point.network, {} if record is None else record["held"]))
    found = seam.check_fifos(point.network)
    stale = tuple(finding for each in found.stale.values() for finding in each)
    if stale:
        raise KernelOpError(
            f"{label}: FIFOs too small for their streams, refused at verification (add "
            "size_fifos to step_explore's strategies): "
            + "; ".join(map(Finding.subject_line, stale)),
            findings=stale,
        )
    model.set(OUTPUT_VERIFIED, design_digest(model))
    return point, found


def refuse_unverified(model: ModelWrapper, label: str) -> None:
    """Refuse to emit a partition model whose design was not verified as it is now: it
    records none (``design-unverified``: run step_verify_partition), or the one it records
    (``OUTPUT_VERIFIED``, by ``verify_partition``) is not its ``design_digest`` now
    (``design-changed``: it changed since; verify it again). Read from the model alone, in
    milliseconds: nothing is bound or checked."""
    verified = model.get(OUTPUT_VERIFIED)
    if verified is None:
        raise KernelOpError.refused(
            "design-unverified",
            "refuse_unverified",
            f"{label}: its design was never verified (finn.outputs verified): run "
            "step_verify_partition",
            node=label,
        )
    now = design_digest(model)
    if verified != now:
        raise KernelOpError.refused(
            "design-changed",
            "refuse_unverified",
            f"{label}: the design changed since step_verify_partition verified it (verified "
            f"{verified}, now {now}): run step_verify_partition again",
            node=label,
        )


# -- a body's choices, on its network in its shell ---------------------------------------


@dataclass(frozen=True)
class CutChoices:
    """What the cut did to a body's choices (``make_cut_choices``), by dotted key: the
    stored choices the bind dropped as stale, each a value outside its domain, with the
    kernel's findings (``stale``: left open, for the channel's policies (a channel's) or a
    strategy (a kernel's) to set, and emission refuses one left open until one does); the
    kernel choices it set by their KernelOps' policies, left open by a body that never went
    through conversion (``late``: one ``choice-converted-late`` finding each, naming the
    value); the choices it could set no value for, left open (``unset``: why,
    ``choice-no-valid-value``); and the choices a first valid policy set in listed order,
    its ranking not read (``fallbacks``: one ``choice-order-unknown`` finding each, naming
    the view and why)."""

    stale: Mapping[str, tuple[Finding, ...]]
    late: Mapping[str, tuple[Finding, ...]]
    unset: Mapping[str, tuple[Finding, ...]]
    fallbacks: Mapping[str, tuple[Finding, ...]] = field(default_factory=dict)


def _late(key: str, value: object) -> Finding:
    return Finding(
        FindingKind.LIMITATION,
        "choice-converted-late",
        key,
        f"left open by a body that never went through conversion: set at the cut to "
        f"{value!r} by conversion's policy",
        (("value", value),) if type(value) in (bool, int, float, str) else (),
    )


def make_cut_choices(model: ModelWrapper, name: str) -> CutChoices:
    """The cut's choices of a DataflowCore's body, ``model``, persisted, each owner's whole
    (``persist``), on the core of its KernelOps, named ``name`` (the node's), in
    its shell (``shell_point``), their stored choices bound (a value outside its domain
    dropped as stale), the policies of its members applied once over its design space
    (``CoreComposition.defaulted``): one pass
    in the root's flow order, which interleaves the members, each choice's policy read from
    its member's owner (a kernel's its KernelOp, a channel's the cut), each value first
    valid where setting it is accepted (no rule of the partition newly false):

    1. each kernel choice the body leaves open, in play and not required, that the bind
       did not drop, set by its KernelOp's policies: a body never converted for it (one
       made by hand), reported (``CutChoices.late``);
    2. each channel's open choices in play that are not required, those the bind dropped
       among them, set by the channel's policies (a channel created here has both its sides
       only in the partition). A required choice (a FIFO's depth) stays open for a
       strategy.

    A kernel choice the bind dropped as stale stays open, whatever an earlier member set. A
    member refused on its own is skipped, none of its choices set (emission names it): it
    is read when the pass reaches the member's first choice. A choice no value can be set
    for is left open and the pass goes on (``CutChoices.unset``). Returns what it dropped,
    set late, could not set, and set in listed order where a ranking could not be read
    (``CutChoices``)."""
    held = shell_point(model, name)
    stale = dict(held.stale)
    bound = CoreComposition(held.point)
    done = bound.defaulted(kept=[key for key in held.dropped if key[0] in bound.kernels])
    names = bound.names

    def keyed(finding: Finding) -> tuple[str, Finding]:
        key = names.pair(finding.owner)
        return dotted(key), bound.local(key[0], finding)

    unset = {each: (finding,) for each, finding in map(keyed, done.unset)}
    fallbacks = {each: (finding,) for each, finding in map(keyed, done.fallbacks)}
    after = bound.made(done.point)
    before = bound.chosen()
    late = {
        key: value
        for key, value in after.chosen().items()
        if key not in before and key[0] in bound.kernels
    }
    persist(model, after.point)
    return CutChoices(
        stale,
        {dotted(key): (_late(dotted(key), value),) for key, value in late.items()},
        unset,
        fallbacks,
    )


__all__ = [
    "Configured",
    "CutChoices",
    "ShellResources",
    "body_row",
    "configured_network",
    "core_grouping",
    "make_cut_choices",
    "mover_contracts",
    "refuse_unverified",
    "row_facts",
    "row_offer",
    "shell_point",
    "shell_resources",
    "shell_space",
    "verify_partition",
]
