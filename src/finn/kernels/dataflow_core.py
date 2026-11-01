# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The DataflowCore's design space: ``DataflowCoreKernel``.

The core of a dataflow partition (the DataflowCore: the cut's node,
``finn.custom_op.partition``, and the IP made of its body) is a Kernel. Its members are its
kernels, one per KernelOp node of its body, and its channels, one per tensor, each named by
its graph name; it has no choices of its own, every choice being a member's. The composer
makes each structure's class on it, ``DataflowCoreKernel_<digest>``
(``finn.kernels.composer``); a test may declare one by hand.

**Sealed** (``Kernel.sealed``). Its realization is one module of its own, its IP, which a
parent places as one instance; inside it its members are one flat netlist of leaves. Its
ports are its edge ports, each a node alias of a member channel's outer port
(``s_axis_0 = x.outer_in``, ``m_axis_0 = y.outer_out``,
``finn.kernels.channels.Channel``), named as the side's label, which is also its bus: the
alias keeps the channel as the port's logical owner (its stream, its keys), and the
module's pins are the physical endpoint, its own (``finn.kernels.port.exposed_pins``). An
edge port aliasing anything but a member channel's outer port is refused when the class is
made (``core-edge-unchanneled``, ``finn.kernels.base``): where a stream leaves a module, the
boundary rule applies, which the channel's outer port presents.

**Its facts**: its name (``name``: the node's, which its module's name leaves out) and, for
a slice, the port beside each edge channel at a frozen neighbour, read at the
neighbour's stored choices (``<channel>-frozen``, ``finn.kernels.channels.Neighbours``: its
stream pins and its pace), from which its class answers that channel's neighbours and hops
as it does a member's. **Its class facts**: the order its module places its members in
(``Kernel.instance_order``), its keyed names (``names``: each member's graph name and each
choice's ``(member, key)``, ``Names``) and each kernel's policies' owner (``policy_owners``,
``finn.kernels.policies.Owner``; ``Unstated`` where none is given), all the declaration's.

**Its answers**: its cycles, the most any member takes, and its resources and bits held,
its members' sums, are the base's (``Kernel.frame_cycles``, ``Kernel.resources``,
``Kernel.buffering``), as are its member channels' neighbours and hops and its netlist
(``Kernel.netlist``, in its instance order). Its own is its module's edge (``edge_buses``): each
edge port's AXIS bus, in its instance order; refused for a slice (some of a graph's kernels,
declared with an edge at a frozen neighbour, ``Side.neighbour``), whose edge there is a
kernel's port outside its module and no AXIS port of it: a slice is not emitted
(``slice-not-emitted``); and refused where a member channel places an outer port (its side
the boundary) that no edge port names, which its class cannot see, its sides being facts
(``port-unconnected``, the port named). Its module (``module``), its IP, is its netlist
with its edge's buses (``core_module``), every marker pin of its edge ruled
(``top_markers``), named by its kind and its content (``STEM``: ``finn_dataflowcore_<digest>``,
``finn.kernels.artifacts.module.module_name``), whatever its node is called, and derived by
its class (``PRODUCER``: its ``id``).

``edge_buses`` reads each member channel through a helper of its own, ``-edge-<i>`` in its
instance order, each reading its channel's sides and its edge ports by the names its
closure fixes (``finn.core.space``'s read inference), then the next: named as the base names
what it installs, so that no member meets it (``finn.kernels.base``). A frozen neighbour's
port is read by the base's reading of a side with no member port, which the class states
for each side of an edge channel at one (``-<channel>-producer``, ``-<channel>-consumer``).

**Bind is assignment** (``bound``): a core's stored choices, by key, are bound with the
engine's ``bind`` on its design space, never refused there. A stored value outside its
domain over the facts (or a key that is no choice) is dropped as stale, with why
(``Bound.dropped``, and ``Bound.stale`` by dotted key). A stored set that breaks a rule
binds, and the core reads refused: the member's validity names the rule. Nothing is filled
in: a choice a change brings into play is open until a policy or a strategy sets it (the
channel's policies at the cut, the seam's fold rule after it). A node binds its body's
stored choices so (``finn.custom_op.partition.DataflowCore.point``), and so does a slice or
a test core, which has no node.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.search import Key, dotted
from finn.core.space import (
    BrokenRule,
    Param,
    Rejected,
    Unresolved,
    bind,
    inspection,
    reject,
    view,
)
from finn.core.space._aliases import alias_table
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.artifacts.module import Composed as Module
from finn.kernels.artifacts.module import Fragment, ProducerIdentity
from finn.kernels.artifacts.top import compose as top
from finn.kernels.base import _END, Kernel, _connects, _installed, _members, _read_none
from finn.kernels.channels import Neighbours
from finn.kernels.policies import Owner
from finn.kernels.sides import SideKind

PRODUCER = ProducerIdentity("finn.dataflow_core", "1")
"""What derives a core's module, its IP (``DataflowCoreKernel.module``): its class, by its
``id``."""
#: A core module's stem, its kind: the module is ``<STEM>_<digest>``, its own name left out
#: of the digest (``finn.kernels.artifacts.module.module_name``).
STEM = "finn_dataflowcore"

_EDGE = "edge"

SEPARATOR = "-"
"""Between a root member's member name and its fact: a member's name never holds it."""

FROZEN = "frozen"
"""The fact of an edge channel at a frozen neighbour that holds the port beside it
(``<channel>-frozen``, the module docstring)."""


def root_name(member: str, fact: str) -> str:
    """The name of the root member holding member ``member``'s ``fact``: a Param, or a
    channel's side (``mm0-x_tensor``, ``h0-producer``)."""
    return f"{member}{SEPARATOR}{fact}"


# Names are a record its class holds and FINN reads, never a value a Space holds: a plain
# class.


class Names:
    """A core's names, a fact of its class (``DataflowCoreKernel.names``): each graph name's
    member name (``members``, kernels then channels, ranked), its channels' graph names
    (``channels``), each root member's owner and fact (``roots``: a Param's, a side's), by
    its name, and, in a shell, its shell member's (``shell``: its own owner; None outside
    one).

    Inside the class a key is a path (``mm0.compute.packed.pe``); at FINN's surface it is the
    pair (owner, key), the owner a graph name: ``path`` and ``pair`` map one to the other.
    A member's answers are read as its own space gives them (``local``, ``relative``): its
    keys and the owners of its findings relative to it, a finding of another member's
    naming that member (``tensor`` for a channel, ``node`` for a kernel)."""

    members: Mapping[str, str]
    roots: Mapping[str, Key]
    shell: str | None
    channels: frozenset[str]

    def __init__(
        self,
        members: Mapping[str, str],
        roots: Mapping[str, Key],
        shell: str | None = None,
        *,
        channels: Iterable[str] = (),
    ) -> None:
        self.members = MappingProxyType(dict(members))
        self.roots = MappingProxyType(dict(roots))
        self.shell = shell
        self.channels = frozenset(channels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Names):
            return NotImplemented
        return (dict(self.members), dict(self.roots), self.shell, self.channels) == (
            dict(other.members),
            dict(other.roots),
            other.shell,
            other.channels,
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Names({dict(self.members)!r})"

    @cached_property
    def owners(self) -> dict[str, str]:
        """Each member name's graph name, the shell member's its own name."""
        found = {member: owner for owner, member in self.members.items()}
        if self.shell is not None:
            found[self.shell] = self.shell
        return found

    def _member(self, owner: str) -> str:
        return owner if owner == self.shell else self.members[owner]

    def path(self, key: Key) -> str:
        """The path of ``key`` (owner, key) inside the class (``mm0.compute.packed.pe``); a
        member's own (``(owner, "")``) its name."""
        owner, below = key
        member = self._member(owner)
        return f"{member}.{below}" if below else member

    def pair(self, path: str) -> Key:
        """The pair (owner, key) of ``path``, a path below a member."""
        member, _, below = path.partition(".")
        return self.owners[member], below

    def root(self, owner: str, fact: str) -> str:
        """The root member holding ``owner``'s ``fact``."""
        return root_name(self._member(owner), fact)

    def below(self, path: str) -> Key | None:
        """The member a path is below, by graph name, and its key there; None for a root
        Param's or a side's."""
        member, _, below = path.partition(".")
        owner = self.owners.get(member)
        return None if owner is None else (owner, below)

    def relative(self, name: str, path: str) -> str:
        """``path`` as an answer about member ``name`` names it: relative to ``name`` where
        it is below it, else as it is."""
        found = self.below(path)
        return found[1] if found is not None and found[0] == name and found[1] else path

    def local(self, name: str, finding: Finding) -> Finding:
        """``finding``, of an answer about member ``name``, as the member's own space gave
        it: owned by its key in the member it is below, naming that member where it is
        another (``tensor`` for a channel, ``node`` for a kernel); its causes the same."""
        causes = tuple(self.local(name, cause) for cause in finding.causes)
        found = self.below(finding.owner)
        if found is None or not found[1]:
            return finding if causes == finding.causes else replace(finding, causes=causes)
        owner, below = found
        made = replace(finding, owner=below, causes=causes)
        if owner == name:
            return made
        return made.about(**{"tensor" if owner in self.channels else "node": owner})

    def rule(self, rule: BrokenRule) -> BrokenRule:
        """A rule broken, its findings relative to the member it is below (its path and the
        stored choices standing kept, ``pair`` their keys)."""
        found = self.below(rule.rule)
        name = rule.rule if found is None else found[0]
        return replace(
            rule, result=Rejected(tuple(self.local(name, each) for each in rule.result.findings))
        )


def _edge_from(
    channel: str, entering: str | None, leaving: str | None, following: str
) -> Callable[[Kernel], tuple[Any, ...]]:
    """The reading of member channel ``channel`` (the module docstring): its frozen
    neighbours' ports, by label, the pins of its edge ports, ``entering`` (its outer
    input's) and ``leaving`` (its outer output's), each where it places it, and the outer
    ports it places that no edge port names; then the next helper's."""

    def edge_from(self: Kernel) -> tuple[Any, ...]:
        held = getattr(self, channel)
        frozen = tuple(
            side.label for side in (held.producer, held.consumer) if side.kind is SideKind.NEIGHBOUR
        )
        pins: list[Any] = []
        unaliased: list[str] = []
        for port, alias, placed in (
            ("outer_in", entering, held.entering),
            ("outer_out", leaving, held.leaving),
        ):
            if not placed:
                continue
            if alias is None:
                unaliased.append(port)
            else:
                pins.extend(getattr(self, alias).physical_pins)
        return ((channel, frozen, tuple(pins), tuple(unaliased)), *getattr(self, following)())

    return edge_from


def _install_edge(cls: type[Kernel]) -> None:
    """The helpers ``-edge-<i>``, one a member channel in its instance order
    (``Kernel.instance_order``, then the others as declared), each with its edge ports by
    the alias naming its outer port (the module docstring)."""
    channels = [name for name, item in _members(cls).items() if _connects(item)]
    rank = {name: index for index, name in enumerate(cls.instance_order)}
    channels.sort(key=lambda name: rank.get(name, len(rank)))
    edges: dict[tuple[str, str], str] = {}
    for label, entry in alias_table(cls).items():
        for end in entry.ends:
            channel, _, port = end.path.partition(".")
            edges.setdefault((channel, port), label)
    for index, channel in enumerate(channels):
        following = _installed(_EDGE, str(index + 1)) if index + 1 < len(channels) else _END
        setattr(
            cls,
            _installed(_EDGE, str(index)),
            _edge_from(
                channel,
                edges.get((channel, "outer_in")),
                edges.get((channel, "outer_out")),
                following,
            ),
        )
    if not channels:
        setattr(cls, _installed(_EDGE, "0"), _read_none)


def _frozen_side(fact: str, role: str) -> Callable[[Kernel], tuple[Any, Any]]:
    """The reading of an edge channel's side ``role`` at a frozen neighbour (the module
    docstring): the port beside it and its pace, as the fact ``fact`` gives them; neither
    where that side is no frozen neighbour's."""

    def beside(self: Kernel) -> tuple[Any, Any]:
        given: Neighbours = getattr(self, fact)
        return getattr(given, role), getattr(given, f"{role}_pace")

    return beside


def _install_frozen(cls: type[Kernel]) -> None:
    """For each member channel given the port beside it at a frozen neighbour
    (``<channel>-frozen``), the readings of its sides the base takes for a side with no
    member port (``-<channel>-producer``, ``-<channel>-consumer``; the base reads a side with
    a member port itself)."""
    declared = vars(cls)
    for name, item in _members(cls).items():
        fact = root_name(name, FROZEN)
        if not _connects(item) or fact not in declared:
            continue
        for role in ("producer", "consumer"):
            setattr(cls, _installed(name, role), _frozen_side(fact, role))


class DataflowCoreKernel(Kernel):
    """The design space of a DataflowCore (the module docstring): a sealed Kernel whose
    members are its kernels and channels and whose ports are its channels' outer ports, by
    alias."""

    id = "finn.dataflow_core"
    sealed = True

    # Its keyed names and each kernel's policies' owner, by graph name: facts of its class
    # (the module docstring), the composer's.
    names: ClassVar[Names] = Names({}, {})
    policy_owners: ClassVar[Mapping[str, Owner]] = MappingProxyType({})

    name: str = Param()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _install_edge(cls)
        _install_frozen(cls)
        super().__init_subclass__(**kwargs)

    @view
    def edge_buses(self) -> tuple[Bus, ...] | Rejected:
        """Its module's edge: each edge port's AXIS bus, its module's own pins, in its
        instance order; refused for a slice, each member channel at a frozen neighbour's
        port named (``slice-not-emitted``, the module docstring), and where a member channel
        places an outer port no edge port names, the port named (``port-unconnected``: the
        class cannot see which outer ports are placed, its sides being facts)."""
        read = getattr(self, "-edge-0")()
        refused: list[Finding] = []
        for channel, labels, _, unaliased in read:
            refused.extend(
                Finding(
                    FindingKind.REJECTION,
                    "slice-not-emitted",
                    channel,
                    f"a slice is not emitted: its edge meets {label}, a frozen neighbour's port "
                    "outside its module, and no AXIS port of it",
                )
                for label in labels
            )
            refused.extend(
                Finding(
                    FindingKind.AUTHORING,
                    "port-unconnected",
                    channel,
                    f"{channel}.{port} is placed, its side being the boundary, and exposed by "
                    "no alias: a sealed kernel presents each outer port it places as its edge "
                    "port, by an alias",
                    (("member", channel), ("port", port)),
                )
                for port in unaliased
            )
        if refused:
            return Rejected(tuple(refused))
        buses = tuple(pin for _, _, pins, _ in read for pin in pins)
        assert all(isinstance(pin, Bus) for pin in buses)  # an AXIS port's pins: its bus
        return buses

    @view
    def module(self) -> Module | Rejected:
        """Its module, its IP (``core_module``): its netlist with its edge's buses, its stem
        ``STEM`` and derived by its class (``PRODUCER``);
        refused where its edge or its netlist is (a slice's edge first,
        ``slice-not-emitted``), or where the module is not well formed."""
        edge: tuple[Bus, ...] = self.edge_buses
        fragment: Fragment = self.netlist
        return core_module(PRODUCER, STEM, fragment, edge)


setattr(DataflowCoreKernel, _installed(_EDGE, "0"), _read_none)


def top_markers(fragment: Fragment, boundary: Iterable[Bus]) -> tuple[Finding, ...]:
    """What the module's boundary leaves of its marker pins (an AXIS bus's ``tlast``)
    unruled, as ``channel-top-marker`` findings: an input marker no link takes as its
    source (it would be left unconsumed), and an output marker no link drives (it lacks a
    rule). A boundary of a bus without marker pins leaves none."""
    taken = {
        pin for link in fragment.links if link.source.instance is None for pin, *_ in link.markers
    }
    driven = {
        pin
        for link in fragment.links
        if link.sink.instance is None
        for _, _, pin, _ in link.markers
    }
    unconsumed: list[str] = []
    unruled: list[str] = []
    for bus in boundary:
        for member in bus.signals:
            if member.logical != "tlast":
                continue
            if bus.endpoint is Endpoint.TARGET and member.physical not in taken:
                unconsumed.append(member.physical)
            elif bus.endpoint is not Endpoint.TARGET and member.physical not in driven:
                unruled.append(member.physical)
    found: list[Finding] = []
    if unconsumed:
        found.append(
            Finding(
                FindingKind.REJECTION,
                "channel-top-marker",
                "",
                f"top input markers {unconsumed} would be left unconsumed",
            )
        )
    if unruled:
        found.append(
            Finding(
                FindingKind.REJECTION,
                "channel-top-marker",
                "",
                f"top output markers {unruled} lack rules",
            )
        )
    return tuple(found)


def core_module(
    producer: ProducerIdentity, stem: str, fragment: Fragment, edge: tuple[Bus, ...]
) -> Module | Rejected:
    """A core's module (``finn.kernels.artifacts.top.compose``), its stem ``stem`` and derived by
    ``producer``, of its netlist (``Kernel.netlist``) and its edge
    (``DataflowCoreKernel.edge_buses``); refused where its edge leaves a marker pin unconsumed or
    unruled (``channel-top-marker``, ``top_markers``), or where the top is not well formed
    (``kernel-netlist``)."""
    unruled = top_markers(fragment, edge)
    if unruled:
        return Rejected(unruled)
    try:
        return top(producer, stem, fragment, edge)
    except ArtifactError as error:
        return reject("kernel-netlist", str(error))


# -- binding stored choices ------------------------------------------------------------------


def _primitive(value: object) -> bool:
    """Whether a finding can hold ``value`` as a detail."""
    return type(value) in (bool, int, float, str)


def _not_a_decision(key: Key) -> tuple[Finding, ...]:
    return (
        Finding(
            FindingKind.REJECTION,
            "not-a-decision",
            dotted(key),
            "no choice of the partition" if key[1] else "no member owns it",
        ),
    )


# A bind is a record FINN reads, never a value a Space holds: a plain class.


class Bound:
    """A core's stored choices bound on its design space (``bound``; the module docstring):
    the design space (``point``), and what the bind dropped as stale, by key (``dropped``:
    each with its findings, relative to its member) and by dotted key (``stale``: each a
    value outside its domain, keeping its code, its owner and its details, which add the
    value stored, ``value``, and the values the choice could take, ``candidates``; with what
    its caller found stray, a graph's choices on a tensor that is no channel of it)."""

    __slots__ = ("dropped", "point", "stale")

    point: DataflowCoreKernel
    dropped: Mapping[Key, tuple[Finding, ...]]
    stale: Mapping[str, tuple[Finding, ...]]

    def __init__(
        self,
        point: DataflowCoreKernel,
        dropped: Mapping[Key, tuple[Finding, ...]],
        stale: Mapping[str, tuple[Finding, ...]],
    ) -> None:
        self.point = point
        self.dropped = MappingProxyType(dict(dropped))
        self.stale = MappingProxyType(dict(stale))

    def __repr__(self) -> str:
        return f"Bound({type(self.point).__qualname__}, stale={list(self.stale)})"


def bound(
    core: DataflowCoreKernel,
    stored: Mapping[Key, object] | None = None,
    *,
    stray: Mapping[str, tuple[Finding, ...]] | None = None,
) -> Bound:
    """``stored`` choices, by key, bound on ``core``, a design space of a core's class
    (``finn.kernels.composer.compose``'s, nothing chosen): each key mapped to its path
    (``Names.path``) and bound with the engine's ``bind`` (the module docstring: a value
    outside its domain, or a key that is no choice or no member's, dropped as stale; a rule
    they break binds, read refused); ``stray``, what its caller found stray, added to
    ``Bound.stale``."""
    names = type(core).names
    decisions = {item.key: item.reference for item in inspection.decisions(inspection.model(core))}
    unknown: dict[Key, tuple[Finding, ...]] = {}
    values: dict[object, object] = {}
    for key, value in (stored or {}).items():
        if key[0] not in names.members:
            unknown[key] = _not_a_decision(key)
            continue
        path = names.path(key)
        if path in decisions:
            values[decisions[path]] = value
        else:
            # Owned by its path, as the engine's findings are: relative to the member below.
            unknown[key] = (
                Finding(
                    FindingKind.REJECTION, "not-a-decision", path, "not a Decision of this member"
                ),
            )
    report = bind(core, values)
    dropped = dict(unknown)
    for outcome in report.outcomes:
        if outcome.status == "dropped":
            result = outcome.result
            findings = result.findings if isinstance(result, (Rejected, Unresolved)) else ()
            dropped[names.pair(outcome.owner)] = findings
    point = report.instance
    local = {
        key: tuple(names.local(key[0], each) for each in found) for key, found in dropped.items()
    }
    return Bound(point, local, _described(point, local, stored or {}) | dict(stray or {}))


def _described(
    point: DataflowCoreKernel,
    dropped: Mapping[Key, tuple[Finding, ...]],
    stored: Mapping[Key, object],
) -> dict[str, tuple[Finding, ...]]:
    """What the bind dropped of the stored choices (each a value outside its domain), by
    dotted key, each with its findings kept: the value stored, and the values the choice
    could take now (its possible ones, ``inspection.viable``, read where the dropped choice
    is open), which its message names."""
    if not dropped:
        return {}
    names = type(point).names
    decisions = {item.key: item.reference for item in inspection.decisions(inspection.model(point))}
    listed = [key for key in dropped if key[0] in names.members and names.path(key) in decisions]
    found = (
        inspection.viable(point, [decisions[names.path(key)] for key in listed]) if listed else ()
    )
    cases = {key: item.cases for key, item in zip(listed, found, strict=True)}

    def described(key: Key, finding: Finding) -> Finding:
        details = dict(finding.details)
        message = finding.message
        if key in stored and _primitive(stored[key]):
            details["value"] = stored[key]
        viable = cases.get(key, ())
        if finding.code == "domain-membership" and viable and all(map(_primitive, viable)):
            details["candidates"] = viable
            message = "not one of " + ", ".join(map(str, viable))
        return replace(finding, message=message, details=tuple(details.items()))

    return {
        dotted(key): tuple(described(key, finding) for finding in found)
        for key, found in dropped.items()
    }


__all__ = [
    "FROZEN",
    "PRODUCER",
    "SEPARATOR",
    "STEM",
    "Bound",
    "DataflowCoreKernel",
    "Names",
    "bound",
    "core_module",
    "root_name",
    "top_markers",
]
