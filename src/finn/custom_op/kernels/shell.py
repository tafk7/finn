# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell root: a set of KernelOp nodes' Partition and the channels on its boundary.

Every exploration and packaging of KernelOps reads one root, built by
``shell_root(model, nodes)`` for the model's target, which changes no graph. Its
shell is the target's (``read_target``): the root reads that shell's row for the
target's board (``finn.platform.shell_row``), which states its ends and budgets.

- **members**: each boundary channel, at its tensor's member (``Reshape_0_out0``),
  declared here as the Partition declares it (``finn.custom_op.kernels.partition``:
  its tensor, its port ``s_axis_<i>`` or ``m_axis_<j>``, pinned ``direct`` when a
  KernelOp outside consumes it), and the Partition at ``partition``, its reference
  inputs supplied with them: input channels, the Partition, output channels. So
  what crosses the boundary is the shell's, and the Partition's own channels and
  kernels are below it (``partition.MatMul_0``, ``partition.MatMul_0_out0``);
- **ends**: the ends the row offers (``ShellRow.ends``, ``finn.kernels.ends.EndOffer``),
  supplied to each boundary channel whose free side meets the host (not one a
  KernelOp outside the nodes produces or consumes); each such channel places its
  ``end`` from them (``Channel.end``: one offered is forced, so nothing is
  persisted), and its cycles are the channel's. The **``ip`` shell** offers none:
  no end on any boundary channel; its IP is the module the shells read
  (``PackagePartition``), and a testbench drives the same pins. Either way the
  module is the same: an end binds no RTL;
- **admission** (``Shell.interfaces``): what the module presents, within what the
  row takes. The AXI-Lite buses the module presents and its ends present (each
  end's ``END_CONTROL``) are within the row's ``control_budget``, and the AXI
  memory ports the module initiates (a bus beside its streams that it initiates)
  within its ``memory_ports``, each refused as ``interface-budget-exceeded``; a
  module that takes an aligned doubled clock needs a row that supplies one
  (``clk2x``), refused as ``clock-unavailable``. The ``ip`` row bounds neither
  count. The counts are of the configured module, so a point is admitted once it
  is decided (``admission_refusal``);
- **resources** (``RESOURCES``, ``shell_resources``): the sum of its members' own
  statements: its partition (the module: the Partition's kernels and channels and
  the boundary channels' stages), each end (its contract's, ``END``) and its row's
  static region, at the memory ports and AXI-Lite buses it connects (the module's
  and its ends'). The ends' and the static region's are out of context
  (``SHELL_CHARACTERISED``), which overstates the placed shell. The ``ip`` shell has
  neither, so its resources are its partition's;
- **paths**: a key of the root is a member path and the key below it
  (``partition.MatMul_0.compute.packed.pe``, ``Reshape_0_out0.transport``): the
  owners, the replayed choices, the dropped ones and the members whose cost a
  design space exploration reads (``members``: channels in node order, then
  kernels, as the Partition declares them) are all named by path, and a key's
  member is the longest member path that prefixes it
  (``finn.kernels.configure.member_of``);
- **replay**: every node's choices, kernel and edge alike, together; a choice
  the current facts refuse or make inapplicable (a key nested under a selector
  a fact change un-forced: ``compute.packed.pe`` once ``compute`` is open) is
  stale: dropped and reported with why (``dropped``), with those the Partition
  found stale before replay, and what it chose is open again, or forced (an
  edge's adapter selectors are forced, never persisted). Nothing is written:
  ``persist`` writes a configured point's choices back, each node's whole, which
  clears the dropped ones;
- **its module** is its Partition's IP: the Partition's netlist in place (its
  instances and buses named as the graph's nodes and tensors, as the Partition
  names them) and its boundary channels' stages, with the boundary's pins. It is
  named and identified as the Partition (its stem ``finn_<name>``; the Partition's
  producer), since an end is integrated beside the IP, not in it.

The class is kept with its Partition's and its row's offers (``SHELLS``, by
``ShellKey``), so a call on the same facts compiles nothing again; the row is
supplied, and the choices replayed, on a fresh design space every call.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from onnx import NodeProto

from finn.core.space import (
    Available,
    ConstraintGroup,
    Members,
    Param,
    Rejected,
    Space,
    Unresolved,
    View,
    composite,
    constraint,
    default_semantics,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.custom_op.kernels.base import (
    KernelOpError,
    committed,
    kernel_op,
    read_target,
    typed_choices,
)
from finn.custom_op.kernels.cache import LeastRecentlyUsed
from finn.custom_op.kernels.partition import (
    KERNEL_OPS,
    Partition,
    Partitioned,
    member,
    partition,
)
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.module import BuildError, BusExport, Fragment, ProducerIdentity, merge
from finn.kernels.base import Kernel
from finn.kernels.configure import chosen, describe, member_of
from finn.kernels.ends import END, END_CONTROL, EndContract, EndOffer
from finn.kernels.utilization import RESOURCES_SEMANTICS, Resources, total
from finn.platform import ShellRow, shell_row

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

S = TypeVar("S", bound=Space)

PARTITION = "partition"
"""The shell root's member that is its Partition."""


@dataclass(frozen=True)
class ShellResources:
    """A shell root's resources by member, each its own statement: ``partition``, the
    module (the Partition's kernels and channels and the boundary channels' stages);
    ``ends``, each end by its boundary channel's member path; ``static_region``, each
    of its row's static IPs by its Vivado IP. ``total`` is their sum."""

    partition: Resources
    ends: tuple[tuple[str, Resources], ...]
    static_region: tuple[tuple[str, Resources], ...]

    @property
    def total(self) -> Resources:
        return total((self.partition, *(used for _, used in self.ends + self.static_region)))


class Shell(Kernel):
    """A shell root: its boundary channels and its Partition (``PARTITION``), admitted
    by its shell's ``row``; its resources the sum of its partition's, its ends' and
    its row's static region's."""

    id = "finn.custom_op.kernels.shell"
    version = 1

    row: ShellRow = Param()
    end_control = Members(END_CONTROL)
    placed_ends = Members(END)

    @derived
    def module_buses(self) -> tuple[int, int]:
        """The AXI-Lite buses the module presents and the AXI memory ports it initiates
        (a bus beside its streams that it initiates)."""
        buses = [pin for pin in self.composed_abi.pins if isinstance(pin, Bus)]
        control = sum(bus.protocol is StandardProtocol.AXILITE for bus in buses)
        memory = sum(
            bus.protocol is not StandardProtocol.AXIS and bus.endpoint is Endpoint.INITIATOR
            for bus in buses
        )
        return control, memory

    @constraint
    def interfaces(self) -> bool | Rejected:
        """The module's AXI-Lite buses and its ends', and the memory ports it initiates,
        within the row's budgets; its doubled clock, if it takes one, the row's."""
        row, abi = self.row, self.composed_abi
        control, memory = self.module_buses
        ends = sum(item.value for item in self.end_control)
        if row.control_budget is not None and control + ends > row.control_budget:
            return reject(
                "interface-budget-exceeded",
                f"the {row.shell!r} shell takes {row.control_budget} AXI-Lite buses; the "
                f"partition presents {control} and its ends {ends}",
            )
        if row.memory_ports is not None and memory > row.memory_ports:
            return reject(
                "interface-budget-exceeded",
                f"the {row.shell!r} shell gives a partition {row.memory_ports} memory "
                f"ports; it initiates {memory}",
            )
        if abi.clock_alignments and not row.clk2x:
            return reject(
                "clock-unavailable",
                f"the partition takes an aligned doubled clock (ap_clk2x), which the "
                f"{row.shell!r} shell does not supply",
            )
        return True

    admission = ConstraintGroup(interfaces)

    @derived(semantics=default_semantics(ShellResources))
    def resources_by_member(self) -> ShellResources | Rejected:
        """Its partition's resources (its members' sum), each end's and its row's static
        region's, at the memory ports and AXI-Lite buses the module and its ends
        connect."""
        partition = total(item.value for item in self.member_resources)
        contracts: list[tuple[str, EndContract]] = [
            (str(item.node), contract) for item in self.placed_ends for contract in item.value
        ]
        ends = tuple((node, contract.resources) for node, contract in contracts)
        region = self.row.static_region
        if region is None:
            return ShellResources(partition, ends, ())
        control, memory = self.module_buses
        masters = memory + sum(contract.memory_ports for _, contract in contracts)
        slaves = control + sum(contract.control_buses for _, contract in contracts)
        if masters < 1 or slaves < 1:
            return reject(
                "shell-resources",
                f"the {self.row.shell!r} shell's static region connects at least one "
                f"memory port and one AXI-Lite bus; the partition and its ends connect "
                f"{masters} and {slaves}",
            )
        return ShellResources(partition, ends, region.resources(masters=masters, slaves=slaves))

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        """The sum of its members' statements (``resources_by_member``)."""
        split: ShellResources = self.resources_by_member
        return split.total

    by_member = View(resources_by_member)

    def producer_identity(self) -> ProducerIdentity:
        """The Partition's: the shell's module is its Partition's IP."""
        return ProducerIdentity(Partition.id, str(Partition.version))

    @derived
    def fragment(self) -> Fragment | Rejected:
        """Each member's netlist under its node, the Partition's in place (``inlined``):
        the IP's instances and buses are named as the Partition names them."""
        exports = tuple(
            BusExport(item.node, item.child, item.port)
            for located in self.presented
            for item in located.value
        )
        try:
            merged = merge(
                *(item.value.under(str(item.node)) for item in self.netlists),
                Fragment(exports=exports),
            )
            return merged.inlined(PARTITION)
        except BuildError as error:
            return reject("kernel-netlist", str(error))


@dataclass(frozen=True)
class ShellKey:
    """What a shell root's class is built from, by value: its Partition's class, the
    ends its row offers (none: the ``ip`` shell) and the boundary tensors offered them."""

    partition: type[Partition]
    offers: tuple[EndOffer, ...]
    ended: tuple[str, ...]


SHELLS: LeastRecentlyUsed[type[Shell]] = LeastRecentlyUsed(16)
"""The process's shell root classes by ``ShellKey``: the 16 most recently used."""


@dataclass(frozen=True)
class ShellRoot:
    """The configured root; each member's owning node and attribute prefix, by member
    path; the choices replay dropped as stale, each with why; each boundary tensor's
    port; its members by path (channels, then kernels), whose cost a design space
    exploration reads; the boundary channels offered ends, by member path; and the
    shell's row, which supplies the ends and admits the point."""

    point: Any
    owners: Mapping[str, tuple[str, str]]
    dropped: Mapping[str, str]
    boundary: tuple[tuple[str, str], ...]
    members: tuple[str, ...]
    ends: tuple[str, ...]
    row: ShellRow

    def owner(self, key: str) -> tuple[str, str] | None:
        """The node that persists ``key`` and the key there (its attribute): its longest
        owned member path's owner."""
        path = member_of(self.owners, key)
        if path is None:
            return None
        node, prefix = self.owners[path]
        return node, prefix + key[len(path) + 1 :]


def _class(built: Partitioned, key: ShellKey) -> type[Shell]:
    """The shell root's class: input channels, the Partition, output channels; the
    channels ``key.ended`` names offered ``key.offers``."""
    declared = dict(built.channels)
    channels = {
        member(tensor): declared[tensor].channel(
            built.platform, end_offer=key.offers if tensor in key.ended else ()
        )
        for tensor, _ in built.boundary
    }
    if PARTITION in channels:
        raise KernelOpError(f"a boundary tensor is named {PARTITION}, the shell's Partition")
    ports = {member(tensor): port for tensor, port in built.boundary}
    inputs = {name: each for name, each in channels.items() if ports[name].startswith("s_axis_")}
    outputs = {name: each for name, each in channels.items() if name not in inputs}
    # Its reference inputs, one per boundary channel, are the composite's own.
    lifted: Any = built.space
    members = {**inputs, PARTITION: lifted(**channels), **outputs}
    return composite(built.space.__name__, members, base=Shell)


def _replay(point: S, choices: Mapping[str, object]) -> tuple[S, dict[str, str]]:
    """``choices`` replayed on ``point``: together when the facts accept them all;
    otherwise every refused or inapplicable one is dropped as stale, with why, and the
    rest replayed again, until they are accepted (what a dropped one chose is open
    again, or forced). A refusal of the batch that names none of its keys is resolved
    one choice at a time."""
    remaining = typed_choices([point], choices)
    dropped: dict[str, str] = {}
    while remaining:
        together = committed(point, remaining)
        if not isinstance(together, dict):
            return together, dropped
        named = {key: why for key, why in together.items() if key in remaining}
        if not named:
            for key, value in remaining.items():
                alone = committed(point, {key: value})
                if isinstance(alone, dict):
                    dropped[key] = "; ".join(alone.values())
                else:
                    point = alone
            return point, dropped
        dropped |= named
        remaining = {key: value for key, value in remaining.items() if key not in named}
    return point, dropped


def _ended(model: ModelWrapper, built: Partitioned) -> tuple[str, ...]:
    """The boundary tensors whose free side meets the host: neither an output handed on
    to a KernelOp outside nor an input a KernelOp produces."""
    declared = dict(built.channels)
    found = []
    for tensor, port in built.boundary:
        if port.startswith("s_axis_"):
            producer = model.find_producer(tensor)
            if producer is not None and producer.domain == KERNEL_OPS:
                continue
        elif declared[tensor].direct:
            continue
        found.append(tensor)
    return tuple(found)


def shell_root(
    model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = "partition"
) -> ShellRoot:
    """The shell root of ``nodes``, KernelOp nodes of ``model``, their Partition named
    ``name``, in the shell of the model's target: its row's ends offered on the
    boundary channels that meet the host, its budgets admitting the point; see the
    module docstring."""
    built = partition(model, nodes, name=name)
    target = read_target(model)
    row = shell_row(target.shell, target.board)
    boundary = {member(tensor) for tensor, _ in built.boundary}
    key = ShellKey(built.space, row.ends, _ended(model, built) if row.ends else ())

    def path(key: str) -> str:
        """A key of the Partition as the shell root names it."""
        head = key.partition(".")[0]
        return key if head in boundary else f"{PARTITION}.{key}"

    root: Any = SHELLS.get(key, lambda: _class(built, key))
    point, dropped = _replay(
        design_space(root(row=row)), {path(key): value for key, value in built.choices.items()}
    )
    stale = {path(key): why for key, why in built.stale.items()}
    members = (*(path(member(tensor)) for tensor, _ in built.channels), *map(path, built.kernels))
    owners = {path(name): owner for name, owner in built.owners.items()}
    ends = tuple(member(tensor) for tensor in key.ended)
    return ShellRoot(point, owners, stale | dropped, built.boundary, members, ends, row)


def shell_resources(point: Any) -> ShellResources | str:
    """``point``'s resources by member and their total (``Shell.resources_by_member``), a
    point of a shell root; or why it states none: a member that states none, or the
    open choices they wait on."""
    answer = point.query(type(point).by_member)
    if isinstance(answer, Available):
        found: ShellResources = answer.value
        return found
    if isinstance(answer, Unresolved):
        awaited = dict.fromkeys(
            item.owner for item in answer.findings if item.code == "decision-unassigned"
        )
        return "waits on " + ", ".join(awaited)
    return describe([answer])


def admission_refusal(point: Any) -> str | None:
    """Why the shell refuses ``point``, a point of a shell root (``Shell.interfaces``),
    or ``None``: admitted, or not decided far enough to say."""
    admitted = inspection.admission(point)
    return describe([admitted]) if isinstance(admitted, Rejected) else None


def persist(model: ModelWrapper, root: ShellRoot, point: Any) -> dict[str, dict[str, object]]:
    """Write ``point``'s choices, a configured point of ``root``'s class, back on the nodes
    that own them, each node's whole: every Decision ``point`` commits (a forced one is
    never committed) on its owner (``ShellRoot.owner``), and a choice a node holds that
    ``point`` does not commit (one replay dropped as stale) cleared. Returns what each
    node now holds."""
    per_node: dict[str, dict[str, object]] = {node: {} for node, _ in root.owners.values()}
    for key, value in chosen(point).items():
        owner = root.owner(key)
        if owner is None:
            raise KernelOpError(f"{key}: no node of the partition owns it")
        node, attribute = owner
        per_node[node][attribute] = value
    by_name = {node.name: node for node in model.graph.node}
    for node, values in per_node.items():
        op = kernel_op(model, by_name[node])
        held = op.choices()
        if values or held:
            op.save(dict.fromkeys(held) | values)
    return per_node


__all__ = [
    "PARTITION",
    "SHELLS",
    "Shell",
    "ShellKey",
    "ShellResources",
    "ShellRoot",
    "admission_refusal",
    "persist",
    "shell_resources",
    "shell_root",
]
