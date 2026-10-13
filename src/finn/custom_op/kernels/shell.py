# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's network in its shell, as functions over the network
(``finn.custom_op.kernels.network``): over its module and its channels' end contracts.

- **admission** (``admission_refusal``, by the row's ``finn.platform.shells.budgets``):
  the AXI-Lite buses the module presents and its ends present (each end's contract,
  ``END``) within the row's ``control_budget``, and the AXI memory ports the module
  initiates within its ``memory_ports``, each refused as ``interface-budget-exceeded``;
  a module that takes an aligned doubled clock needs a row that supplies one
  (``clk2x``), refused as ``clock-unavailable``. The ``ip`` row bounds neither count.
  Of the configured module, so a network is admitted once it is decided;
- **resources** (``shell_resources``; ``Partition.own_resources``, which the seam reads):
  the sum of its members' own statements, each end's (its contract's) and its row's
  static region's, at the memory ports and AXI-Lite buses it connects (the module's
  and its ends'). The ends' and the static region's are out of context
  (``SHELL_CHARACTERISED``), which overstates the placed shell. The ``ip`` shell has
  neither, so its resources are its members'.

``configured_root`` is a partition's network as emission builds it (``Configured``: the
network, its module and its cycles a frame).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from finn.core.graph.errors import KernelOpError
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Available, Inapplicable, QueryResult, Rejected, Unresolved
from finn.custom_op.kernels.network import Partition, member, network
from finn.kernels.artifacts.module import Composed
from finn.kernels.base import CYCLES, RESOURCES
from finn.kernels.ends import END, EndContract
from finn.kernels.network import dotted
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources
from finn.platform.shells import ShellRow, budgets, bus_counts

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass(frozen=True)
class ShellResources:
    """A shell's resources by member, each its own statement: ``partition``, the module
    (its kernels and channels, the boundary channels' stages included); ``ends``, each
    end by its boundary channel's node in the module (``member``); ``static_region``,
    each of its row's static IPs by its Vivado IP. ``total`` is their sum."""

    partition: Resources
    ends: tuple[tuple[str, Resources], ...]
    static_region: tuple[tuple[str, Resources], ...]

    @property
    def total(self) -> Resources:
        return total((self.partition, *(used for _, used in self.ends + self.static_region)))


# -- the shell, as functions over the network ----------------------------------------------


def _together(results: Iterable[QueryResult[Any]]) -> Rejected | Unresolved:
    """Results that are not available, as one: refused where one is, else waiting."""
    held = list(results)
    findings = tuple(
        dict.fromkeys(finding for each in held for finding in getattr(each, "findings", ()))
    )
    if any(isinstance(each, Rejected) for each in held):
        return Rejected(findings)
    return Unresolved(findings)


def _buses(network: Partition) -> QueryResult[tuple[int, int, bool]]:
    """``bus_counts`` of its module; or what its module waits on, or why it is
    refused."""
    module = network.module_of()
    if not isinstance(module, Available):
        return _together([module])
    return Available(bus_counts(module.value.abi))


def _end_contracts(network: Partition) -> QueryResult[tuple[tuple[str, EndContract], ...]]:
    """Each end its boundary channels place, by its channel: none on the ``ip`` shell."""
    found: list[tuple[str, EndContract]] = []
    waiting: list[QueryResult[object]] = []
    for name in network.ends:
        placed = network.query(name, END)
        if isinstance(placed, Available):
            found.extend((name, contract) for contract in placed.value)
        elif not isinstance(placed, Inapplicable):
            waiting.append(placed)
    if waiting:
        return _together(waiting)
    return Available(tuple(found))


def admission_refusal(network: Partition) -> tuple[Finding, ...]:
    """Why the shell refuses ``network`` (its ``budgets``, of its module's ``bus_counts``
    and its ends' AXI-Lite buses): its findings, none where it is admitted or not decided
    far enough to say."""
    buses = _buses(network)
    contracts = _end_contracts(network)
    if not isinstance(buses, Available) or not isinstance(contracts, Available):
        return ()
    ends = sum(contract.control_buses for _, contract in contracts.value)
    return budgets(network.row, buses.value, ends)


def _resources(network: Partition) -> QueryResult[ShellResources]:
    """``shell_resources``, as a query's result, read once of a network."""

    def read() -> QueryResult[ShellResources]:
        used = [network.query(name, RESOURCES) for name in network.members]
        missing = [each for each in used if not isinstance(each, Available)]
        if missing:
            return _together(missing)
        partition = total(each.value for each in used if isinstance(each, Available))
        contracts = _end_contracts(network)
        if not isinstance(contracts, Available):
            return _together([contracts])
        ends = tuple((member(name), contract.resources) for name, contract in contracts.value)
        row = network.row
        region = row.static_region
        if region is None:
            return Available(ShellResources(partition, ends, ()))
        buses = _buses(network)
        if not isinstance(buses, Available):
            return _together([buses])
        control, memory, _ = buses.value
        masters = memory + sum(contract.memory_ports for _, contract in contracts.value)
        slaves = control + sum(contract.control_buses for _, contract in contracts.value)
        if masters < 1 or slaves < 1:
            message = (
                f"the {row.shell!r} shell's static region connects at least one memory port "
                f"and one AXI-Lite bus; the partition and its ends connect {masters} and "
                f"{slaves}"
            )
            return Rejected(
                (Finding(FindingKind.REJECTION, "shell-resources", "resources", message),)
            )
        static = region.resources(masters=masters, slaves=slaves)
        return Available(ShellResources(partition, ends, static))

    found: QueryResult[ShellResources] = network._read("resources", read)
    return found


@dataclass(frozen=True)
class Configured:
    """A partition's network as emission builds it (``configured_root``): every choice
    stored, none open, admitted by its shell; its ``module`` and its ``cycles`` a frame,
    its slowest member's."""

    network: Partition
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


def shell_resources(subject: Partition | Configured) -> ShellResources | tuple[Finding, ...]:
    """The resources of a partition's network (or of its configured one) by member and
    their total (``ShellResources``); or why it states none, its findings: a member that
    states none, or the open choices they wait on (each ``decision-unassigned``)."""
    found = _resources(subject.network if isinstance(subject, Configured) else subject)
    return found.value if isinstance(found, Available) else found.findings


def _about(finding: Finding) -> str:
    """A finding as text after the node or tensor it is about, where it names one."""
    details = dict(finding.details)
    about = details.get("node", details.get("tensor"))
    return str(finding) if about is None else f"{about}: {finding}"


def configured_root(
    model: ModelWrapper, label: str
) -> tuple[Configured, tuple[tuple[str, str], ...]]:
    """A partition model's network (``network``, in the shell of its target), replayed
    from its nodes' and its tensors' channel choices (a Decision with one viable case is
    forced, nothing to commit), as emission builds it (``Configured``), and its boundary
    (tensor, port). Every choice is stored where it is created (a KernelOp's at
    conversion, a channel's at the cut) and replaced by the strategies, so nothing is
    completed here. A stale choice; an open choice, each by name: one no case of which is
    viable (each case's reason), a required one no strategy set ("required: run
    step_size_fifos or step_placeholder_choices"), any other one never baselined ("not
    created: BaselineChoices"); a network the shell does not admit
    (``admission_refusal``), or whose module or cycles its members refuse; or graph
    inputs and outputs out of port order refuse, named."""
    bound = network(model, model.graph.node)
    if bound.stale:
        raise KernelOpError(
            f"{label}: stale choices, refused by the partition: "
            + "; ".join(f"{key}: {reason}" for key, reason in bound.stale.items()),
            tuple(bound.stale),
        )
    open_keys = bound.undecided()
    if open_keys:
        required = {key for key, item in bound.decisions().items() if item.required}
        unviable = {
            key: "; ".join(f"{case}: {reason}" for case, reason in item.refused.items())
            for key, item in bound.viable()
            if not item.cases
        }
        named = [
            f"{dotted(key)}: no case is viable: {unviable[key]}"
            if key in unviable
            else f"{dotted(key)} (required: run step_size_fifos or step_placeholder_choices)"
            if key in required
            else f"{dotted(key)} (not created: BaselineChoices)"
            for key in open_keys
        ]
        raise KernelOpError(
            f"{label}: open choices, refused at emission: " + "; ".join(named),
            tuple(dotted(key) for key in open_keys),
        )
    unadmitted = admission_refusal(bound)
    if unadmitted:
        raise KernelOpError(
            f"{label}: refused by the {bound.row.shell!r} shell: "
            + "; ".join(map(str, unadmitted)),
            findings=unadmitted,
        )
    module = bound.module_of()
    cycles = {name: bound.query(name, CYCLES) for name in bound.members}
    found = [] if isinstance(module, Available) else list(module.findings)
    for name, each in cycles.items():
        if not isinstance(each, Available):
            found.extend(bound.located(name, finding) for finding in each.findings)
    if found or not isinstance(module, Available):
        refused = tuple(dict.fromkeys(found))
        raise KernelOpError(
            f"{label}: refused by its members: " + "; ".join(map(_about, refused)),
            findings=refused,
        )
    ports = dict(bound.boundary)
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [item.name for item in model.graph.input if item.name not in initializers]
    expected = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    expected |= {item.name: f"m_axis_{index}" for index, item in enumerate(model.graph.output)}
    if ports != expected:
        raise KernelOpError(
            f"{label}: the partition's ports {ports} are not its graph's inputs and"
            f" outputs in order {expected}"
        )
    slowest = max(each.value for each in cycles.values() if isinstance(each, Available))
    return Configured(bound, module.value, slowest), bound.boundary


__all__ = [
    "Configured",
    "ShellResources",
    "admission_refusal",
    "configured_root",
    "shell_resources",
]
