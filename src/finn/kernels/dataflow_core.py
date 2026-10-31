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

**Its facts**: its module's name (``name``: the node's, ``finn_<name>`` its module's stem)
and the order its module places its members in (``Kernel.instance_order``, a class fact:
the declaration's).

**Its answers**: its cycles, the most any member takes, and its resources and bits held,
its members' sums, are the base's (``Kernel.frame_cycles``, ``Kernel.resources``,
``Kernel.buffering``), as are its member channels' neighbours and hops and its netlist
(``Kernel.netlist``, in its instance order). Its own is its module's edge (``edge_buses``): each
edge port's AXIS bus, in its instance order; refused for a slice (some of a graph's kernels,
declared with an edge at a frozen neighbour, ``Side.neighbour``), whose edge there is a
kernel's port outside its module and no AXIS port of it: a slice is not emitted
(``slice-not-emitted``). Its module is its netlist with its edge's buses
(``core_module``), every marker pin of its edge ruled (``top_markers``).

``edge_buses`` reads each member channel through a helper of its own, ``__core_edge_<i>`` in its
instance order, each reading its channel's sides and its edge ports by the names its
closure fixes (``finn.core.space``'s read inference), then the next: a name no member takes
(a graph name never starts ``__``, ``finn.kernels.composer``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Param, Rejected, reject, view
from finn.core.space._aliases import alias_table
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.artifacts.module import Composed as Module
from finn.kernels.artifacts.module import Fragment, ProducerIdentity
from finn.kernels.artifacts.top import compose as top
from finn.kernels.base import Kernel, _connects, _members
from finn.kernels.sides import SideKind

_EDGE = "__core_edge"


def _edge_from(
    channel: str, entering: str | None, leaving: str | None, following: str
) -> Callable[[Kernel], tuple[Any, ...]]:
    """The reading of member channel ``channel`` (the module docstring): its frozen
    neighbours' ports, by label, and the pins of its edge ports, ``entering`` (its outer
    input's) and ``leaving`` (its outer output's), each where it has one and places it; then
    the next helper's."""

    def edge_from(self: Kernel) -> tuple[Any, ...]:
        held = getattr(self, channel)
        frozen = tuple(
            side.label for side in (held.producer, held.consumer) if side.kind is SideKind.NEIGHBOUR
        )
        pins: list[Any] = []
        if entering is not None and held.entering:
            pins.extend(getattr(self, entering).physical_pins)
        if leaving is not None and held.leaving:
            pins.extend(getattr(self, leaving).physical_pins)
        return ((channel, frozen, tuple(pins)), *getattr(self, following)())

    return edge_from


def _install_edge(cls: type[Kernel]) -> None:
    """The helpers ``__core_edge_<i>``, one a member channel in its instance order
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
        following = f"{_EDGE}_{index + 1}" if index + 1 < len(channels) else "_module_end"
        setattr(
            cls,
            f"{_EDGE}_{index}",
            _edge_from(
                channel,
                edges.get((channel, "outer_in")),
                edges.get((channel, "outer_out")),
                following,
            ),
        )
    if not channels:
        setattr(cls, f"{_EDGE}_0", Kernel._module_end)


class DataflowCoreKernel(Kernel):
    """The design space of a DataflowCore (the module docstring): a sealed Kernel whose
    members are its kernels and channels and whose ports are its channels' outer ports, by
    alias."""

    id = "finn.dataflow_core"
    sealed = True

    name: str = Param()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _install_edge(cls)
        super().__init_subclass__(**kwargs)

    @view
    def edge_buses(self) -> tuple[Bus, ...] | Rejected:
        """Its module's edge: each edge port's AXIS bus, its module's own pins, in its
        instance order; refused for a slice, each member channel at a frozen neighbour's
        port named (``slice-not-emitted``, the module docstring)."""
        read = getattr(self, "__core_edge_0")()
        frozen = tuple(
            Finding(
                FindingKind.REJECTION,
                "slice-not-emitted",
                channel,
                f"a slice is not emitted: its edge meets {label}, a frozen neighbour's port "
                "outside its module, and no AXIS port of it",
            )
            for channel, labels, _ in read
            for label in labels
        )
        if frozen:
            return Rejected(frozen)
        buses = tuple(pin for _, _, pins in read for pin in pins)
        assert all(isinstance(pin, Bus) for pin in buses)  # an AXIS port's pins: its bus
        return buses


setattr(DataflowCoreKernel, f"{_EDGE}_0", Kernel._module_end)


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
    """A core's module (``finn.kernels.artifacts.top.compose``), named ``stem`` and derived by
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


__all__ = ["DataflowCoreKernel", "core_module", "top_markers"]
