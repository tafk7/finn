# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition in its shell (``InShell``, bound by ``in_shell``): the partition of its
KernelOps (``finn.custom_op.kernels.network``) with the movers the row of the model's
target's shell offers (``finn.platform.shells.shell_row``) on its boundary channels and
the row's budgets as rules of its own, and the shell as functions over it, over its netlist
(``Partition.netlist``: the module's buses counted without building the module,
``finn.platform.shells.bus_counts``) and its channels' mover contracts.

- **budgets**, rules of the partition (its shell member, ``finn.kernels.budgets``, given
  the row's facts, ``row_facts``): the AXI-Lite buses its kernels present and its movers
  present within the row's ``control_budget``, and the memory ports its kernels initiate
  within its ``memory_ports``, each refused as ``interface-budget-exceeded``; a member
  that takes the doubled clock needs a row that supplies one (``clk2x``), refused as
  ``clock-unavailable``. The ``ip`` row bounds neither count. Each reads the members'
  derived values, so a value that breaks one is refused in the partition, where the seam,
  the cut and emission read it. Emission checks the netlist's count (``bus_counts``, and
  the movers' contracts, ``MOVER``) against the rules' (``finn.kernels.budgets.counts``):
  a difference is an error naming both (``shell-counts-differ``);
- **resources** (``shell_resources``; ``InShell.own_resources``, which the seam reads):
  the sum of its members' own statements (``Partition.own_resources``), each mover's (its
  contract's) and its row's static region's, at the memory ports and AXI-Lite buses it
  connects (the module's and its movers'). The movers' and the static region's are out of context
  (``SHELL_CHARACTERISED``), which overstates the placed shell. The ``ip`` shell has
  neither, so its resources are its members'.

``configured_network`` is a partition as emission builds it (``Configured``: the
partition, its module and its cycles a frame), bound only: a stored choice it dropped as
stale, a rule its stored choices break (a member's, or a budget of its shell's), a choice
left open, refuse it, named.
``verify_partition`` is the one
check of its FIFOs against their streams before emission (a strategy may call the seam's
check too, ``Seam.check_fifos``), and records the design it verified on the body; emission
(packaging, the shell's integration, the testbench) compares it (``refuse_unverified``),
and refuses a design never verified or changed since. Simulation and elaboration only
bind. The cut makes a body's channel choices (``make_cut_choices``) on its partition in
its shell, which alone holds both sides of a boundary channel, by the channel's policies
(``Channel.policies``), and writes them with ``persist``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from onnx import NodeProto

from finn.core.graph.annotations.choices import held_keys
from finn.core.graph.annotations.outputs import OUTPUT_VERIFIED
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Available, Inapplicable, QueryResult, Rejected, inspection
from finn.custom_op.kernels.network import PARTITION, PRODUCER, _network, persist, stale_text
from finn.kernels.artifacts.module import Composed
from finn.kernels.base import CYCLES
from finn.kernels.budgets import SHELL, counts
from finn.kernels.composer import Key
from finn.kernels.configure import refusal
from finn.kernels.explore import FifoCheck, Seam, dotted
from finn.kernels.movers import MOVER, MoverContract
from finn.kernels.partition import Partition, _together
from finn.kernels.policies import Required, policy
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources
from finn.platform.shells import ShellRow, bus_counts, shell_row

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class ShellResources:
    """A shell's resources by member, each its own statement: ``partition``, the module
    (its kernels and channels, the boundary channels' stages included); ``movers``, each
    mover by its boundary channel's node in the module (``member``); ``static_region``,
    each of its row's static IPs by its Vivado IP. ``total`` is their sum."""

    partition: Resources
    movers: tuple[tuple[str, Resources], ...]
    static_region: tuple[tuple[str, Resources], ...]

    @property
    def total(self) -> Resources:
        return total((self.partition, *(used for _, used in self.movers + self.static_region)))


class InShell(Partition):
    """A partition in its shell (``in_shell``): its shell's ``row``, whose movers its
    boundary channels were offered and whose budgets are its shell member's rules (its
    facts, ``row_facts``). What it uses (``own_resources``, which the seam reads)
    is the shell's: its members', its movers' and its static region's sum
    (``shell_resources``). Immutable, as a partition: a change returns another, which keeps
    its row."""

    __slots__ = ("row",)

    row: ShellRow

    def __init__(self, *args: Any, row: ShellRow, **given: Any) -> None:
        super().__init__(*args, **given)
        self.row = row

    def own_resources(self) -> QueryResult[Resources]:
        """What the shell uses: its members', its movers' and its static region's sum
        (``shell_resources``)."""
        found = _resources(self)
        return Available(found.value.total) if isinstance(found, Available) else _together([found])


def row_facts(row: ShellRow) -> dict[str, object]:
    """The facts of ``row`` a partition in its shell holds (``finn.kernels.budgets.ShellBudgets``):
    the shell's name and its budgets, as the row states them."""
    return {
        "shell": row.shell,
        "control_budget": row.control_budget,
        "memory_ports": row.memory_ports,
        "clk2x": row.clk2x,
    }


def in_shell(model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = PARTITION) -> InShell:
    """The partition of ``nodes``, KernelOp nodes of ``model`` (``network``), in the shell of
    the model's target: its row (``shell_row``) offers its movers on the boundary channels,
    and its budgets are the partition's rules (``row_facts``)."""
    target = read_target(model)
    row = shell_row(target.shell, target.board)
    return _network(model, list(nodes), name, row.movers, partial(InShell, row=row), row_facts(row))


# -- the shell, as functions over the partition --------------------------------------------


def _buses(network: Partition) -> QueryResult[tuple[int, int, bool]]:
    """``bus_counts`` of its netlist and boundary buses (``Partition.netlist``), its top's
    pins counted without building it; or what its netlist waits on, or why it is
    refused."""
    found = network.netlist()
    if not isinstance(found, Available):
        return _together([found])
    return Available(bus_counts(*found.value))


def _mover_contracts(network: InShell) -> QueryResult[tuple[tuple[str, MoverContract], ...]]:
    """Each mover its boundary channels place, by its channel: none on the ``ip`` shell."""
    found: list[tuple[str, MoverContract]] = []
    waiting: list[QueryResult[object]] = []
    for name in network.movers:
        placed = network.query(name, MOVER)
        if isinstance(placed, Available):
            found.extend((name, contract) for contract in placed.value)
        elif not isinstance(placed, Inapplicable):
            waiting.append(placed)
    if waiting:
        return _together(waiting)
    return Available(tuple(found))


def _counts_agree(network: InShell, label: str) -> None:
    """Refuse ``network``, decided, whose netlist's count (``bus_counts``, its movers'
    AXI-Lite buses added from their contracts) is not its shell member's
    (``finn.kernels.budgets.counts``): (AXI-Lite buses, memory ports, doubled clock), the
    two named (``shell-counts-differ``), or either not known. Emission's check of the
    rules' count."""
    buses = _buses(network)
    contracts = _mover_contracts(network)
    ruled = counts(network.member(SHELL))
    netlist: object = buses
    if isinstance(buses, Available) and isinstance(contracts, Available):
        control, memory, doubled = buses.value
        movers = sum(contract.control_buses for _, contract in contracts.value)
        netlist = (control + movers, memory, doubled)
    stated = ruled.value if isinstance(ruled, Available) else ruled
    if netlist != stated:
        raise KernelOpError.refused(
            "shell-counts-differ",
            "configured_network",
            f"{label}: the netlist's count (AXI-Lite buses, memory ports, doubled clock) "
            f"{netlist} is not the shell's rules' {stated}",
            node=label,
        )


def _resources(network: InShell) -> QueryResult[ShellResources]:
    """``shell_resources``, as a query's result, read once of a network."""

    def read() -> QueryResult[ShellResources]:
        members = Partition.own_resources(network)
        if not isinstance(members, Available):
            return _together([members])
        partition = members.value
        contracts = _mover_contracts(network)
        if not isinstance(contracts, Available):
            return _together([contracts])
        movers = tuple(
            (network.node(name), contract.resources) for name, contract in contracts.value
        )
        row = network.row
        region = row.static_region
        if region is None:
            return Available(ShellResources(partition, movers, ()))
        buses = _buses(network)
        if not isinstance(buses, Available):
            return _together([buses])
        control, memory, _ = buses.value
        masters = memory + sum(contract.memory_ports for _, contract in contracts.value)
        slaves = control + sum(contract.control_buses for _, contract in contracts.value)
        if masters < 1 or slaves < 1:
            message = (
                f"the {row.shell!r} shell's static region connects at least one memory port "
                f"and one AXI-Lite bus; the partition and its movers connect {masters} and "
                f"{slaves}"
            )
            return Rejected(
                (Finding(FindingKind.REJECTION, "shell-resources", "resources", message),)
            )
        static = region.resources(masters=masters, slaves=slaves)
        return Available(ShellResources(partition, movers, static))

    found: QueryResult[ShellResources] = network._read("resources", read)
    return found


@dataclass(frozen=True)
class Configured:
    """A partition's network as emission builds it (``configured_network``): every choice
    in play stored, none open, within its shell's budgets (its shell member's rules); its
    ``module`` and its ``cycles`` a frame, its slowest member's."""

    network: InShell
    module: Composed
    cycles: int

    @property
    def row(self) -> ShellRow:
        """Its shell's row."""
        return self.network.row

    def channel(self, tensor: str) -> Any:
        """The channel carrying ``tensor``: its point."""
        return self.network.member(tensor)

    def node(self, name: str) -> str:
        """Member ``name``'s node in the module: the prefix of its instances."""
        return self.network.node(name)


def shell_resources(subject: InShell | Configured) -> ShellResources | tuple[Finding, ...]:
    """The resources of a partition's network (or of its configured one) by member and
    their total (``ShellResources``); or why it states none, its findings: a member that
    states none, or the open choices they wait on (each ``decision-unassigned``)."""
    found = _resources(subject.network if isinstance(subject, Configured) else subject)
    return found.value if isinstance(found, Available) else found.findings


def configured_network(
    model: ModelWrapper, label: str
) -> tuple[Configured, tuple[tuple[str, str], ...]]:
    """A partition model's partition in the shell of its target (``in_shell``), bound with
    its nodes' and its tensors' channel choices, as emission builds it (``Configured``),
    and its boundary (tensor, port). Every choice is stored where it is created (a
    KernelOp's at conversion, a channel's at the cut) and replaced by the strategies, so
    nothing is completed here. A stale choice; an open choice in play, each by name: one no
    value of which is possible (each refused value's rules; a ``choice-no-valid-value``
    cause each, owned by the key, the rules' findings its causes, as the cut reports it), a
    required one no strategy set ("required: run step_size_fifos or
    step_placeholder_choices"), any other one never set ("not set: the cut,
    MakeCutChoices, or a strategy sets it"; one finding, ``choices-open``); a partition
    over the shell's budgets (its shell member's validity: "refused by the <shell> shell",
    each broken budget a finding), or whose module or cycles its members refuse, or a
    member's validity refuses (a rule its stored choices break, which the bind binds:
    "refused by its members", each finding naming its member); or graph inputs and outputs
    out of port order refuse, named. Its netlist's count (``bus_counts`` and its movers')
    must be its shell's rules' (``shell-counts-differ``, naming both, otherwise). A dormant
    choice (stored under a case not selected) is not open, and refuses nothing. It only
    binds: it does not check the FIFOs against their streams (``verify_partition`` does), so
    a simulation can run a FIFO the model calls too small."""
    bound = in_shell(model, model.graph.node)
    if bound.stale:
        keys = tuple(bound.stale)
        message = f"{label}: stale choices, refused by the partition: " + "; ".join(
            stale_text(key, found) for key, found in bound.stale.items()
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
                    tuple(finding for found in bound.stale.values() for finding in found),
                ),
            ),
        )
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
            else f"{dotted(key)} (required: run step_size_fifos or step_placeholder_choices)"
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
    over_budget = refusal(bound.validity(SHELL))
    if over_budget:
        raise KernelOpError(
            f"{label}: refused by the {bound.row.shell!r} shell: "
            + "; ".join(map(str, over_budget)),
            findings=over_budget,
        )
    module = bound.module_of(PRODUCER)
    cycles = {name: bound.query(name, CYCLES) for name in bound.members}
    found = [] if isinstance(module, Available) else list(module.findings)
    for name, each in cycles.items():
        if not isinstance(each, Available):
            found.extend(bound.located(name, finding) for finding in each.findings)
    if not found and isinstance(module, Available):
        _counts_agree(bound, label)
        # A rule a member's stored choices break that its module and cycles do not read.
        found = [
            bound.located(name, finding)
            for name in bound.members
            for finding in refusal(bound.validity(name))
        ]
    if found or not isinstance(module, Available):
        refused = tuple(dict.fromkeys(found))
        raise KernelOpError(
            f"{label}: refused by its members: " + "; ".join(map(Finding.subject_line, refused)),
            findings=refused,
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
    slowest = max(each.value for each in cycles.values() if isinstance(each, Available))
    return Configured(bound, module.value, slowest), bound.boundary


def _impossible(
    bound: Partition, key: Key, values: tuple[inspection.Possible | inspection.Refused, ...]
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


def verify_partition(model: ModelWrapper, label: str) -> tuple[Configured, FifoCheck]:
    """A partition model's configured network (``configured_network``, whose refusals it
    keeps) with its FIFOs checked against their streams at its bottleneck period
    (``Seam.check_fifos``: sizing's own model, whatever changed since a FIFO was sized, an
    edit by hand included), and what the check found; the design it verified recorded on
    the model (``OUTPUT_VERIFIED``: its ``design_digest``), which emission compares
    (``refuse_unverified``). A FIFO not held that no longer keeps its producer within its
    idle time is refused, named (``fifo-stale``: run step_size_fifos), and nothing is
    recorded; a held one (``fifo-held-stale``) and a ``direct`` that throttles its
    producer (``direct-throttles``) are limitations, the check's ``findings``. The model
    is changed only by the record; its caller saves it."""
    point, _ = configured_network(model, label)
    seam = Seam(read_target(model).platform)
    # Only which keys are held matters here, not their values (exploration checks those).
    stated = seam.chosen(point.network)
    seam.held.update(
        {
            (owner, key): stated[(owner, key)]
            for owner, keys in held_keys(model).items()
            for key in keys
            if (owner, key) in stated
        }
    )
    found = seam.check_fifos(point.network)
    stale = tuple(finding for each in found.stale.values() for finding in each)
    if stale:
        raise KernelOpError(
            f"{label}: FIFOs too small for their streams, refused at verification (run "
            "step_size_fifos): " + "; ".join(map(Finding.subject_line, stale)),
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


def make_cut_choices(model: ModelWrapper) -> CutChoices:
    """The cut's choices of a partition's body, ``model``, persisted, each owner's whole
    (``persist``), on the partition of its KernelOps bound in its shell (``in_shell``),
    their stored choices bound (a value outside its domain dropped as stale), the policies
    of its members applied once over its design space (``Partition.defaulted``): one pass
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
    bound = in_shell(model, model.graph.node)
    stale = dict(bound.stale)
    done = bound.defaulted(kept=[key for key in bound.dropped if key[0] in bound.kernels])
    names = bound.composed.names

    def keyed(finding: Finding) -> tuple[str, Finding]:
        key = names.pair(finding.owner)
        return dotted(key), bound.local(key[0], finding)

    unset = {each: (finding,) for each, finding in map(keyed, done.unset)}
    fallbacks = {each: (finding,) for each, finding in map(keyed, done.fallbacks)}
    after = bound.at(done.point)
    before = bound.chosen()
    late = {
        key: value
        for key, value in after.chosen().items()
        if key not in before and key[0] in bound.kernels
    }
    persist(model, after)
    return CutChoices(
        stale,
        {dotted(key): (_late(dotted(key), value),) for key, value in late.items()},
        unset,
        fallbacks,
    )


__all__ = [
    "Configured",
    "CutChoices",
    "InShell",
    "ShellResources",
    "make_cut_choices",
    "configured_network",
    "in_shell",
    "refuse_unverified",
    "row_facts",
    "shell_resources",
    "verify_partition",
]
