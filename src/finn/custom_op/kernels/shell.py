# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shell root: a set of KernelOp nodes as one Kernel, in its shell.

Every exploration and packaging of KernelOps reads one root, built by
``shell_root(model, nodes)`` straight from the nodes and the model's target, which
changes no graph. Its shell is the target's (``read_target``): the root reads that
shell's row for the target's board (``finn.platform.shell_row``), which states its ends
and budgets.

- **channels**, one per ONNX tensor, in node order, each declared from the graph
  before any kernel is placed: a node's graph inputs, the parameter channels it owns
  (named after the initializer), its outputs. A channel's tensor is the graph's
  value_info and annotation; a parameter channel's is the initializer's over its
  values' range, and its contents the node's value of it (``Facts.values``), which its
  source stores; every channel's platform is the model's target's. Only the nodes'
  ONNX inputs and outputs are boundaries, named by the shell's convention
  ``s_axis_<i>`` and ``m_axis_<j>``. A channel has one consumer, so a tensor read
  more than once is refused by name before any owner is chosen (``tensor-fan-out``):
  read by two of the nodes (an edge, or a parameter both own), or read by one and an
  output boundary too (a graph output, or read outside the nodes). Fan-out waits for
  a channel that forks;
- **kernels**, one per node: its op's placement (``KernelOp.place``, the one its node
  root is generated from) with literal formals, on these channels;
- **members**, named as the graph: channels by tensor and kernels by node (``\\W`` as
  ``_``), in the order input boundary channels, the other channels, the kernels,
  output boundary channels (``Reshape_0_out0``, ``MatMul_0``, ``MatMul_0_out0``); two
  members of one name (a node and a tensor, or two nodes) are refused, not renamed;
- **ends**: the ends the row offers (``ShellRow.ends``, ``finn.kernels.ends.EndOffer``),
  supplied to each boundary channel: its free side meets the host, since the nodes are
  every KernelOp of the model (a second partition of KernelOps is refused by name;
  several partitions are to be designed as one shell root, when CNV or Alveo needs
  them); each channel places its ``end`` from them (``Channel.end``: one offered is
  forced, so nothing is persisted), and its cycles are the channel's. The **``ip``
  shell** offers none: no end on any boundary channel; its IP is the module the shells
  read (``PackagePartition``), and a testbench drives the same pins. Either way the
  module is the same: an end binds no RTL;
- **owners**: each member's node and attribute prefix, how a choice made in the root
  goes back to the node that persists it: a kernel's on its node, an edge's on its
  consumer, a parameter channel's on its value owner. An output boundary (a graph
  output) is its producer's, under its output port;
- **paths**: a key of the root is a member and the key below it
  (``MatMul_0.compute.packed.pe``, ``Reshape_0_out0.transport``): the owners, the
  replayed choices, the dropped ones and the members whose cost a design space
  exploration reads (``members``: channels in node order, then kernels) are all named
  so, and a key's member is the longest member path that prefixes it
  (``finn.kernels.configure.member_of``);
- **replay**: every node's choices, kernel and edge alike, together; a choice the
  current facts refuse or make inapplicable (a key nested under a selector a fact
  change un-forced: ``compute.packed.pe`` once ``compute`` is open) is stale: dropped
  and reported with why (``dropped``), with those stale before replay (held under an
  output port whose channel a KernelOp now consumes), and what it chose is open again,
  or forced (an edge's adapter selectors are forced, never persisted). Nothing is
  written: ``persist`` writes a configured point's choices back, each node's whole,
  which clears the dropped ones;
- **admission** (``Shell.interfaces``): what the module presents, within what the
  row takes. The AXI-Lite buses the module presents and its ends present (each
  end's contract, ``END``) are within the row's ``control_budget``, and the AXI
  memory ports the module initiates (a bus beside its streams that it initiates)
  within its ``memory_ports``, each refused as ``interface-budget-exceeded``; a
  module that takes an aligned doubled clock needs a row that supplies one
  (``clk2x``), refused as ``clock-unavailable``. The ``ip`` row bounds neither
  count. The counts are of the configured module, so a point is admitted once it
  is decided (``admission_refusal``);
- **resources** (``RESOURCES``, ``shell_resources``): the sum of its members' own
  statements: its partition (the module: its kernels and channels, the boundary
  channels' stages included), each end (its contract's, ``END``) and its row's static
  region, at the memory ports and AXI-Lite buses it connects (the module's and its
  ends'). The ends' and the static region's are out of context
  (``SHELL_CHARACTERISED``), which overstates the placed shell. The ``ip`` shell has
  neither, so its resources are its partition's;
- **its module** is the partition's IP, the cut's (``PARTITION``): its kernels'
  and channels' netlists, its instances and buses named as the graph's nodes and
  tensors, with the boundary's pins. It is named after the partition (its stem
  ``finn_<name>``) and identified as the partition's IP (``Shell.id``), since an end
  is integrated beside the IP, not in it.

The class is kept by what it is built from, by value (``ShellKey``): the name; the
target's platform; each channel as declared (the tensor it carries, rows and
annotation, its port); each node's name, op class, node-root class, facts key (its
formals and owned values by value, as the bind cache keys them), owned parameter ports
and tensors; and the ends the row offers. A call on the same facts compiles nothing
again, and reads an owned value only to build it; ``SHELLS`` keeps the 16 most
recently used. The row is supplied, and the choices replayed, on a fresh design space
every call.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from onnx import NodeProto

from finn.core.space import (
    Available,
    ConstraintGroup,
    Finding,
    FindingKind,
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
    KernelOp,
    KernelOpError,
    committed,
    edge_tensor,
    kernel_op,
    read_target,
    typed_choices,
)
from finn.custom_op.kernels.cache import Facts, LeastRecentlyUsed
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen, describe, member_of
from finn.kernels.ends import END, EndContract, EndOffer
from finn.kernels.target import Platform
from finn.kernels.utilization import RESOURCES_SEMANTICS, Resources, total
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform import ShellRow, shell_row

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

S = TypeVar("S", bound=Space)

KERNEL_OPS = "finn.custom_op.kernels"

PARTITION = "partition"
"""The partition's name: the cut's node and body file, and the IP made from it, which
is the shell root's module (``finn_partition``)."""


@dataclass(frozen=True)
class ShellResources:
    """A shell root's resources by member, each its own statement: ``partition``, the
    module (its kernels and channels, the boundary channels' stages included); ``ends``,
    each end by its boundary channel's member; ``static_region``, each of its row's
    static IPs by its Vivado IP. ``total`` is their sum."""

    partition: Resources
    ends: tuple[tuple[str, Resources], ...]
    static_region: tuple[tuple[str, Resources], ...]

    @property
    def total(self) -> Resources:
        return total((self.partition, *(used for _, used in self.ends + self.static_region)))


class Shell(Kernel):
    """A shell root: a set of nodes' channels and kernels, admitted by its shell's
    ``row``; its resources the sum of its partition's, its ends' and its row's static
    region's."""

    # Its module is the partition's IP, which an end sits beside: identified as the
    # partition's, so the IP's name and digest are the partition's.
    id = "finn.custom_op.kernels.partition"
    version = 1

    row: ShellRow = Param()
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
        ends = sum(contract.control_buses for item in self.placed_ends for contract in item.value)
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


@dataclass(frozen=True)
class Declared:
    """A channel as the shell root declares it: the tensor it carries, rows and
    annotation (a parameter channel's over its values' range), and its boundary port."""

    tensor: Tensor
    port: str | None = None

    def channel(
        self,
        platform: Platform,
        contents: IntegerTensorValue | None = None,
        end_offer: tuple[EndOffer, ...] = (),
    ) -> Channel:
        """The channel on the target's ``platform``, carrying ``contents`` (an owned
        parameter's value) when given, its free side offered the ends ``end_offer``
        when any."""
        settings: dict[str, Any] = {"tensor": self.tensor}
        if contents is not None:
            settings["contents"] = contents
        if end_offer:
            settings["end_offer"] = end_offer
        if self.port is not None:
            settings["port"] = self.port
        return Channel(platform=platform, **settings)


@dataclass(frozen=True)
class Placement:
    """What placing a node's kernel reads: the node's name (its member), its op class,
    node-root class and facts key (its formals and owned values by value), the parameter
    ports it owns, and its tensors."""

    node: str
    op: type[KernelOp]
    root: type[Kernel]
    facts: tuple[Hashable, ...]
    owned: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ShellKey:
    """What a shell root's class is built from, by value: its name, the target's
    platform, its channels as declared, in order, its nodes' placements, in order, and
    the ends its row offers its boundary channels (none: the ``ip`` shell)."""

    name: str
    platform: Platform
    channels: tuple[tuple[str, Declared], ...]
    kernels: tuple[Placement, ...]
    offers: tuple[EndOffer, ...]


SHELLS: LeastRecentlyUsed[type[Shell]] = LeastRecentlyUsed(16)
"""The process's shell root classes by ``ShellKey``: the 16 most recently used."""


def member(name: str) -> str:
    """A graph name as a member name."""
    return re.sub(r"\W", "_", name)


def _boundary(model: ModelWrapper, nodes: list[NodeProto], owned: set[str]) -> dict[str, str]:
    """The port of each boundary tensor, inputs then outputs, in node order: an input
    no node produces (and not a parameter: owned, or an initializer the op reads as a
    fact) is ``s_axis_<i>``; an output read outside ``nodes`` is ``m_axis_<j>``."""
    produced = {tensor for node in nodes for tensor in node.output}
    inside = {id(node) for node in nodes}
    used_outside = {
        tensor for node in model.graph.node if id(node) not in inside for tensor in node.input
    } | {output.name for output in model.graph.output}
    inputs = [
        tensor
        for tensor in dict.fromkeys(t for node in nodes for t in node.input)
        if tensor not in produced and tensor not in owned and model.get_initializer(tensor) is None
    ]
    outputs = [tensor for node in nodes for tensor in node.output if tensor in used_outside]
    ports = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    return ports | {tensor: f"m_axis_{index}" for index, tensor in enumerate(outputs)}


def _second_partition(model: ModelWrapper, nodes: list[NodeProto]) -> list[str]:
    """The KernelOps of ``model`` outside ``nodes``: a second partition of KernelOps."""
    inside = {id(node) for node in nodes}
    return [
        node.name
        for node in model.graph.node
        if node.domain == KERNEL_OPS and id(node) not in inside
    ]


def _fan_out(
    model: ModelWrapper,
    nodes: list[NodeProto],
    owned: list[dict[str, str]],
    ports: Mapping[str, str],
) -> list[Finding]:
    """Each tensor whose channel would have more than one consumer, in node order: read
    by two of ``nodes`` (an input that is not an initializer, or a parameter a node owns),
    or by one and at its output port too (``m_axis_<j>``)."""
    readers: dict[str, list[str]] = {}
    for node, tensors in zip(nodes, owned):
        for tensor in node.input:
            if tensor in tensors.values() or model.get_initializer(tensor) is None:
                readers.setdefault(tensor, []).append(node.name)
    found = []
    for tensor, names in readers.items():
        port = ports.get(tensor, "")
        if port.startswith("m_axis_"):
            message = f"read by {', '.join(names)} and leaves the partition at {port}"
        elif len(names) > 1:
            message = f"read by {', '.join(names)}"
        else:
            continue
        found.append(Finding(FindingKind.LIMITATION, "tensor-fan-out", tensor, message))
    return found


def _channels(
    model: ModelWrapper,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    owned: list[dict[str, str]],
    ports: Mapping[str, str],
) -> dict[str, Declared]:
    """The root's channels by tensor, in node order: a node's inputs on an edge or the
    boundary, the parameter channels it owns (the initializer's tensor; its value, the
    contents, is bound when the class is built), its outputs."""
    parameters = {tensor for tensors in owned for tensor in tensors.values()}
    channels: dict[str, Declared] = {}

    def declare(tensor: str, label: str) -> None:
        if tensor not in channels:
            carried = edge_tensor(model, tensor, label)
            channels[tensor] = Declared(carried, ports.get(tensor))

    for node, op, tensors in zip(nodes, ops, owned):
        for tensor in node.input:
            if tensor not in parameters and model.get_initializer(tensor) is None:
                declare(tensor, op.label)
        for tensor in tensors.values():
            channels[tensor] = Declared(edge_tensor(model, tensor, op.label))
        for tensor in node.output:
            declare(tensor, op.label)
    return channels


@dataclass
class _Owners:
    """Each member's owner (node, attribute prefix); the nodes' choices split by what
    declares them, and those stale before replay (an output's, now an edge another
    KernelOp owns)."""

    owners: dict[str, tuple[str, str]]
    kernel_choices: dict[str, object]
    edge_choices: dict[str, object]
    stale: list[str]


def _owners(
    nodes: list[NodeProto], ops: list[KernelOp], channels: Iterable[str], produced: set[str]
) -> _Owners:
    """Each node's kernel member and channels' owners, and its choices as root keys: a
    kernel's under the kernel's member, an input or owned channel's under the channel's,
    and an output's under the channel's where its node is the producer that owns it
    (``produced``: the output boundaries)."""
    found = _Owners({}, {}, {}, [])
    channel_members = {member(tensor) for tensor in channels}
    kernels: set[str] = set()
    for node, op in zip(nodes, ops):
        kernel = member(node.name)
        if kernel in channel_members or kernel in kernels:
            other = "a tensor" if kernel in channel_members else "another node"
            raise KernelOpError(f"{node.name}: a node and {other} are both named {kernel}")
        kernels.add(kernel)
        by_port = op.inputs() | {
            port: tensor for port, tensor in zip(op.outputs, node.output) if tensor in produced
        }
        found.owners[kernel] = (node.name, "")
        for port, tensor in by_port.items():
            found.owners[member(tensor)] = (node.name, f"{port}.")
        for attribute, value in op.choices().items():
            head, _, rest = attribute.partition(".")
            if head in by_port:
                found.edge_choices[f"{member(by_port[head])}.{rest}"] = value
            elif head in op.outputs:
                tensor = node.output[op.outputs.index(head)]
                found.stale.append(f"{member(tensor)}.{rest}")
            else:
                found.kernel_choices[f"{kernel}.{attribute}"] = value
    return found


def _class(
    key: ShellKey,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    facts: list[Facts],
    owned: list[dict[str, str]],
) -> type[Shell]:
    """The shell root's class: its channels as ``key`` declares them (an owned
    parameter's with its node's value as contents, a boundary channel's offered
    ``key.offers``) and each node's kernel placed on them from its ``facts``; members
    in the order input boundary channels, the other channels, the kernels, output
    boundary channels."""
    contents: dict[str, IntegerTensorValue] = {}
    for each, tensors in zip(facts, owned):
        if tensors:
            values = each.values()
            contents |= {tensor: values[port] for port, tensor in tensors.items()}
    channels: dict[str, Channel] = {
        tensor: each.channel(key.platform, contents.get(tensor), key.offers if each.port else ())
        for tensor, each in key.channels
    }
    ports = {tensor: each.port or "" for tensor, each in key.channels}
    kernels = {
        member(node.name): op.place(each, channels) for node, op, each in zip(nodes, ops, facts)
    }

    def on(side: str) -> dict[str, Channel]:
        """The channels at the ports ``<side><i>`` (``""``: no port, none on the boundary)."""
        return {
            member(tensor): channel
            for tensor, channel in channels.items()
            if ports[tensor].rstrip("0123456789") == side
        }

    members = {**on("s_axis_"), **on(""), **kernels, **on("m_axis_")}
    return composite(key.name, members, base=Shell)


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


def shell_root(
    model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = PARTITION
) -> ShellRoot:
    """The shell root of ``nodes``, every KernelOp node of ``model``, its module the
    partition ``name``'s IP, in the shell of the model's target: its row's ends offered
    on the boundary channels, its budgets admitting the point; see the module
    docstring. A tensor read more than once is refused (``tensor-fan-out``)."""
    nodes = list(nodes)
    outside = _second_partition(model, nodes)
    if outside:
        raise KernelOpError(
            f"{', '.join(outside)}: KernelOps outside the partition {name!r}, a second "
            "partition of KernelOps, refused until several partitions are designed as one "
            "shell root"
        )
    ops = [kernel_op(model, node) for node in nodes]
    facts = [op.facts() for op in ops]
    owned = [op.owned(each) for op, each in zip(ops, facts)]
    ports = _boundary(model, nodes, {tensor for tensors in owned for tensor in tensors.values()})
    fanned = _fan_out(model, nodes, owned, ports)
    if fanned:
        raise KernelOpError(
            f"{name}: a channel has one consumer, and fan-out is not designed yet (a channel "
            "that forks): "
            + "; ".join(f"{each.owner}: {each.code}: {each.message}" for each in fanned)
        )
    declared = _channels(model, nodes, ops, owned, ports)
    outputs = {tensor for node in nodes for tensor in node.output if tensor in ports}
    found = _owners(nodes, ops, declared, outputs)

    target = read_target(model)
    row = shell_row(target.shell, target.board)
    key = ShellKey(
        name,
        target.platform,
        tuple(declared.items()),
        tuple(
            Placement(
                node.name,
                type(op),
                each.root,
                each.key,
                each.owned,
                tuple(node.input),
                tuple(node.output),
            )
            for node, op, each in zip(nodes, ops, facts)
        ),
        row.ends,
    )
    root: Any = SHELLS.get(key, lambda: _class(key, nodes, ops, facts, owned))
    point, dropped = _replay(design_space(root(row=row)), found.kernel_choices | found.edge_choices)
    stale = dict.fromkeys(found.stale, "held under an output port a KernelOp now consumes")
    members = (*(member(tensor) for tensor in declared), *(member(node.name) for node in nodes))
    ends = tuple(member(tensor) for tensor in ports) if row.ends else ()
    return ShellRoot(point, found.owners, stale | dropped, tuple(ports.items()), members, ends, row)


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
    "Declared",
    "Placement",
    "Shell",
    "ShellKey",
    "ShellResources",
    "ShellRoot",
    "admission_refusal",
    "persist",
    "shell_resources",
    "member",
    "shell_root",
]
