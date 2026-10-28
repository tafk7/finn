# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware design Spaces with actual model children and native outward source.

UntimedKernel states netlist/buffering/resources; KernelSpace also inherits
Timed.cycles. Real PortModel children own intrinsic rules, included ControlModel
Groups share their real parent, and plain ModuleSpec values describe sources.
Each module kernel authors its serial native netlist and frame count. Parents
compose resolved child Members in ordinary source. No lowering/setter/read-list
or generated flat callback table is installed on a kernel.

Hardware configuration facts/Choices/rules remain inward model computations.
Outward Views keep selected owning non-refusal and actual unresolved inputs.
Kernel is the existing four-answer readout Protocol; no timing/native/container
facility or computational verification is added. Physical source provision,
module ownership, required declarations and resource statements are checked.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from finn.core.space import (
    DefinitionError,
    Inapplicable,
    Members,
    Rejected,
    Space,
    derived,
    reject,
    view,
)
from finn.core.space.declarations import unmet_required
from finn.dataflow.schedule import Access, Affine, Index, Refused, Schedule, bind_extents
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Clocking
from finn.kernels.artifacts.contributions import Contribution
from finn.kernels.artifacts.module import BusExport, Fragment, Link, merge
from finn.kernels.control import Control, control_models, presented
from finn.kernels.library import ModuleSpec
from finn.kernels.transport import StreamContract
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources


class Kernel(Protocol):
    """What a reader asks of a kernel's design point: its four hardware answers, read out
    (its netlist, the cycles a frame takes through it, the bits it holds and what it uses
    of the device). A design point of every ``KernelSpace`` satisfies it, each answer an
    accepted view's value; a reader typed ``Kernel`` reads them and nothing else. An
    ``UntimedKernel`` alone (a channel's FIFO) states no cycles and is no ``Kernel``: it is
    read through its channel.

    These read-out answers are the whole Protocol. A kernel's streams are members of its
    design space, one per port and named by it (``x_stream``), so each class types its own
    and the Protocol holds none of them; a generic reader finds them through the class's
    declarations (``finn.kernels.declaration.declarations``). What a kernel admits, its
    choices and a trial of them are the engine's, asked of any Space.

    Reading an answer here gives its accepted value, and raises where the kernel refuses
    it. A reader wanting the result rather than the accepted value asks
    ``point.query(type(point).netlist)``: ``Available`` with the value, or the refusal."""

    @property
    def netlist(self) -> Fragment:
        """Its netlist, labelled relative to itself."""
        ...

    @property
    def cycles(self) -> int:
        """The clock cycles a frame takes through it."""
        ...

    @property
    def buffering(self) -> int:
        """The bits it holds between a channel's sides."""
        ...

    @property
    def resources(self) -> Resources:
        """What it uses of the device."""
        ...


@dataclass(frozen=True)
class PortContract:
    """The contract a port presents to a channel: its ``contract`` and its ``label``, the
    port's path relative to the member that states it (``x`` for a module's port,
    ``compute.packed.x`` for MatMul's, ``source.memstream.set`` for a channel's source's
    set port, below the label its channel states for it; empty for a contract the member
    presents itself, which is no port's). ``under`` labels it from the network (or kernel
    with children) that holds the member."""

    label: str
    contract: StreamContract

    def under(self, member: str) -> PortContract:
        """The contract as seen from where ``member`` is held."""
        return PortContract(f"{member}.{self.label}" if self.label else member, self.contract)


NATIVE_CLOCKING = Clocking(clock="clk", reset="rst", active_low=False)
"""FinnLib's native ``clk`` and synchronous active-high ``rst``."""


def merged(
    netlists: Iterable[tuple[str, Fragment]],
    presented: Iterable[BusExport] = (),
    links: tuple[Link, ...] = (),
) -> Fragment | Rejected:
    """Each netlist under its node, in order, the control buses presented (each a
    ``BusExport`` with the writes its configuration takes, ``finn.kernels.control.presented``)
    and the ``links`` between them (a channel's hops): a kernel's netlist, and a
    network's."""
    try:
        return merge(
            *(netlist.under(node) for node, netlist in netlists),
            Fragment(links=links, exports=tuple(presented)),
        )
    except ArtifactError as error:
        return reject("kernel-netlist", str(error))


def _declared(cls: type[UntimedKernel], name: str, below: type[UntimedKernel]) -> bool:
    """Whether a class from ``cls`` down to, not including, ``below`` declares ``name``."""
    return any(name in vars(base) for base in cls.__mro__[: cls.__mro__.index(below)])


def _members(cls: type[UntimedKernel], below: type[UntimedKernel]) -> dict[str, object]:
    """The class attributes ``cls`` and its bases down to ``below`` declare, by name."""
    found: dict[str, object] = {}
    for base in reversed(cls.__mro__[: cls.__mro__.index(below)]):
        found.update(vars(base))
    return found


class Timed(Space):
    """A Space stating the clock cycles a frame takes through it: its view ``cycles``.

    The base of what a channel's ``cycles`` counts: ``KernelSpace`` and the movers derive
    from it, and ``member_cycles`` collects ``Timed.cycles`` of the members of a kernel.
    A member that takes none a channel counts (``ChannelFifo``) does not derive it (it is
    an ``UntimedKernel``), so it is no candidate of ``member_cycles``.

    Its own ``cycles`` is the one a class stating none would answer: a refusal, so that a
    ``Timed`` class that is placed without stating its cycles is never taken to take none.
    Every class placed states its own.
    """

    @view
    def cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through it: what a class stating none refuses."""
        return reject("timed-cycles", f"{type(self).__qualname__} states no cycles")


class UntimedKernel(Space):
    """A kernel: the design space of its hardware, binding a library module or placing
    kernels, never both (the module docstring). ``id`` names it, a nonempty string.

    The base of every kernel, which states everything a kernel states but its ``cycles``:
    it exists for a kernel that counts no cycles of its own, a channel's FIFO
    (``finn.kernels.channels.ChannelFifo``), whose beats are counted where they are made.
    That kernel derives it alone, so it is no ``Timed`` and its channel's ``member_cycles``
    does not range over it; every other kernel derives ``KernelSpace``, which adds
    ``Timed``. A check or a type admitting any kernel names this class.

    Its answers are Views. Buffering/resources call ordinary source helpers; legacy
    netlist/cycles still use class-specific lowering while their callers migrate. A
    class may state one itself, calling these. Its members' answers it reaches through
    ``Members`` over the member each states, which only these views read."""

    id: ClassVar[str] = ""

    module_spec: ClassVar[ModuleSpec | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__module__ == __name__ and cls.__qualname__ == "KernelSpace":
            return
        timed = issubclass(cls, KernelSpace)
        below: type[UntimedKernel] = KernelSpace if timed else UntimedKernel
        buses = list(control_models(cls))
        if len(buses) > 1:
            raise DefinitionError.broken(
                "control-bus-repeated",
                "KernelSpace",
                f"{cls.__qualname__} declares the control buses {buses}: "
                "a kernel presents one at most",
                declaration=cls.__qualname__,
            )
        if buses and (
            not _declared(cls, "bus_control", below) or not _declared(cls, "declares_bus", below)
        ):
            raise DefinitionError.broken(
                "control-bus-unaddressed",
                "KernelSpace",
                f"{cls.__qualname__}: an included control states parent bus_control "
                "and declares_bus answers",
                declaration=cls.__qualname__,
            )
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError.broken(
                "kernel-id-invalid",
                "KernelSpace",
                f"{cls.__qualname__} must declare a nonempty string id",
                declaration=cls.__qualname__,
            )
        members = _members(cls, below)
        specs = [(name, item) for name, item in members.items() if isinstance(item, ModuleSpec)]
        for name, spec in specs:
            spec.validate(cls.__qualname__, f"{cls.__qualname__}.{name}")
        if any(spec.constant_sources is None for _, spec in specs) and not _declared(
            cls, "module_sources", below
        ):
            raise DefinitionError.broken(
                "library-module-sources-unstated",
                "KernelSpace",
                f"{cls.__qualname__}: a dynamic ModuleSpec states its module_sources View",
                declaration=cls.__qualname__,
            )
        if len(specs) > 1:
            raise DefinitionError.broken(
                "library-module-repeated",
                "KernelSpace",
                f"{cls.__qualname__} binds the library modules {[name for name, _ in specs]}: "
                "a kernel binds one at most",
                declaration=cls.__qualname__,
            )
        if unmet_required(cls) or not specs:
            return
        from finn.kernels.port_model import port_surfaces  # noqa: PLC0415

        ports = [
            name
            for name, surface in port_surfaces(cls).items()
            if surface.physical_pins is not None
        ]
        if not _declared(cls, "netlist", below):
            raise DefinitionError.broken(
                "module-netlist-unstated",
                "KernelSpace",
                f"{cls.__qualname__}: a ModuleSpec kernel states its native netlist assembly",
                declaration=cls.__qualname__,
            )
        if not ports:
            raise DefinitionError.broken(
                "module-ports-unplaced",
                "KernelSpace",
                f"{cls.__qualname__} declares no port: a kernel with a module states its ports",
                declaration=cls.__qualname__,
            )
        children = sorted(
            name for name, value in members.items() if isinstance(value, UntimedKernel)
        )
        if children:
            raise DefinitionError.broken(
                "module-places-kernels",
                "KernelSpace",
                f"{cls.__qualname__} places the kernels {children}: "
                "a kernel with a module places none",
                declaration=cls.__qualname__,
            )
        if not _declared(cls, "resource_use", below):
            raise DefinitionError.broken(
                "module-resources-unstated",
                "KernelSpace",
                f"{cls.__qualname__} states no resource_use: "
                "a kernel with a module states what it uses of the device",
                declaration=cls.__qualname__,
            )

    # -- its clock, extents --------------------------------------------------------------

    # The clock and reset (and its polarity) its ports and its module run on: a class
    # constant, since no kernel computes their names. A kernel on FinnLib's native pins
    # states ``NATIVE_CLOCKING``.
    clock_domain: ClassVar[Clocking] = Clocking()

    @derived
    def clocking(self) -> Clocking:
        """Its module's clocking: its ``clock_domain``; a kernel with a doubled clock builds
        on it (``replace(type(self).clock_domain, doubled=..., doubling=...)``)."""
        return type(self).clock_domain

    @derived
    def takes_doubled_clock(self) -> bool:
        """Whether it takes the doubled clock: its own ``clocking``'s doubling. A kernel
        with children whose child may take it states its own, reading the child (a
        MatMul its selected core, a channel its source): a parent's ``clocking`` is its
        own, never its children's. What a shell's clock rule reads
        (``finn.kernels.budgets``)."""
        return self.clocking.doubling

    @derived
    def memory_initiators(self) -> int:
        """The memory ports it initiates itself (a memory-mapped bus it is the initiator
        of): none here, and no kernel declares one today. A kernel that does states its
        own; a shell's memory rule counts them (``finn.kernels.budgets``)."""
        return 0

    def _accesses(self) -> list[Access]:
        """Actual input access computations; an indexed module authors its own source."""
        return []

    def _bound(self, extents: Mapping[Index, int] | None = None) -> dict[Index, int] | Rejected:
        try:
            return bind_extents(self._accesses(), extents)
        except Refused as error:
            return reject("kernel-extents", str(error))

    @derived
    def extents(self) -> dict[Index, int] | Rejected:
        """Each index's extent, bound from the tensors its input ports read."""
        return self._bound()

    def bound_schedule(
        self,
        order: tuple[Index, ...],
        factors: Mapping[Index, int] | None = None,
        extents: Mapping[Index, int] | None = None,
    ) -> Schedule | Rejected:
        """The schedule in beat ``order`` (outer to inner), each index's extent bound from the
        ports' tensors (and any ``extents`` the kernel gives), with folding ``factors``."""
        bound = self.extents if extents is None else self._bound(extents)
        if isinstance(bound, Rejected):
            return bound
        missing = [index for index in order if index not in bound]
        if missing:
            return reject("kernel-extents", f"{missing} are bound by no placed port")
        try:
            return Schedule({index: bound[index] for index in order}, factors, order)
        except ValueError as error:
            return reject("kernel-schedule", str(error))

    def spanned(self, index: tuple[Index | Affine, ...]) -> tuple[int, ...] | Rejected:
        """The shape ``index`` spans over the bound extents: an axis read at ``i`` spans
        ``i``'s extent, one at ``oh * OW + ow`` the reach of its terms, plus one. A kernel
        states its output tensor's shape by it, from the extents its inputs bind."""
        extents = self.extents
        shape: list[int] = []
        for axis in index:
            terms = Affine.of(axis).terms
            missing = [i for i, _ in terms if i not in extents]
            if missing:
                return reject("kernel-extents", f"{missing} are bound by no placed port")
            shape.append(1 + sum(c * (extents[i] - 1) for i, c in terms))
        return tuple(shape)

    # -- its own parts of its answers ----------------------------------------------------

    @view
    def held_bits(self) -> int:
        """The bits it holds itself between a channel's sides: none, but for a stage that
        buffers (a FIFO, an ``input_gen``), which states its own."""
        return 0

    @view
    def resource_use(self) -> Resources:
        """What it uses of the device itself: nothing; a kernel with a module states its
        own (a refusal where it has no model of it, never nothing)."""
        return Resources()

    @view
    def module_sources(self) -> tuple[Contribution, ...] | Rejected:
        """Explicit dynamic source product for a ModuleSpec with constant_sources=None.

        Constant-source modules do not read this View. A dynamic module declares
        its own implementation; the class definition check requires it.
        """
        return reject("library-module-sources-unstated", "a dynamic module states its sources")

    # -- its answers, written once ---------------------------------------------------------

    def _netlist(
        self, links: tuple[Link, ...] | None = None, labels: Mapping[str, str] | None = None
    ) -> Fragment | Rejected:
        """``netlist``: its library module at the empty label, its members' netlists under
        their nodes, and the control buses its members present with the writes their
        configuration takes (``merged``). A kernel with neither is refused; one that states
        the ``links`` between them (a channel's hops) is not. A kernel that states the
        ``labels`` some members keep (by node: a channel's placements) places those there,
        in place of their nodes."""
        space_type = type(self)
        placed = labels or {}
        parts: list[tuple[str, Fragment]] = []
        parts.extend(
            (placed.get(str(item.node), str(item.node)), item.value) for item in self.netlists
        )
        if not parts and links is None:
            return reject(
                "kernel-module",
                f"{space_type.__qualname__} declares no module and places no kernel",
            )
        buses = presented(
            (placed.get(str(item.node), str(item.node)), item.value) for item in self.controls
        )
        if isinstance(buses, Rejected):
            return buses
        return merged(parts, buses, links or ())

    def _buffering(self) -> int:
        """``buffering``: its own ``held_bits`` and its members'."""
        return self.held_bits + sum(item.value for item in self.member_buffering)

    def _resources(self) -> Resources:
        """``resources``: its own ``resource_use`` and its members'."""
        return total((self.resource_use, *(item.value for item in self.member_resources)))

    @view
    def netlist(self) -> Fragment | Rejected:
        """Its netlist, labelled relative to itself: its module at the empty label (``under``
        names it by its node where it is held), its members' under theirs."""
        parts = [(str(item.node), item.value) for item in self.netlists]
        if not parts:
            return reject(
                "kernel-module",
                f"{type(self).__qualname__} declares no module and places no kernel",
            )
        buses = presented((str(item.node), item.value) for item in self.controls)
        if isinstance(buses, Rejected):
            return buses
        return merged(parts, buses)

    @derived
    def declares_bus(self) -> bool:
        """Whether it declares a control bus (presented or not): False here, True in a
        class declaring one through its own model answer."""
        return False

    @view
    def bus_control(self) -> Control | Rejected | Inapplicable:
        """The control bus it declares: a ``Control``, presenting nothing while the bus is
        not presented. A kernel declaring none answers inapplicable; one declaring a bus
        states its own parent View here."""
        if not self.declares_bus:
            return Inapplicable()
        return reject("kernel-control", f"{type(self).__qualname__} declares no control bus")

    @view
    def buffering(self) -> int:
        """The bits it and its members hold between a channel's sides."""
        return self.held_bits + sum(item.value for item in self.member_buffering)

    @view
    def resources(self) -> Resources:
        """What it and its members use of the device."""
        return total((self.resource_use, *(item.value for item in self.member_resources)))

    # -- what it reaches of its members: their answers, read by its views only ------------
    # Each ``Members`` names the member it reads, so it follows the views it names.

    netlists = Members(netlist)
    controls = Members(bus_control)
    member_buffering = Members(buffering)
    member_resources = Members(resources)


class KernelSpace(UntimedKernel, Timed):
    """A kernel that takes clock cycles: every kernel but a channel's FIFO, which derives
    ``UntimedKernel`` alone. The base a kernel derives. To the untimed base's answers it adds
    ``cycles``, the most beats any of its declarations carries a frame, or any of its
    members takes (``_cycles``), reaching its members' through ``member_cycles``, over
    ``Timed.cycles``."""

    def _cycles(self, default: int | None = None) -> int | Rejected:
        """``cycles``: the most beats any declaration carries a frame, or any member takes,
        one beat a cycle at best; ``default`` where there is none (a channel of wires
        takes none of its own), else refused."""
        space_type = type(self)
        found = [item.value for item in self.member_cycles]
        if found:
            return max(found)
        if default is not None:
            return default
        if space_type.module_spec is not None:
            return reject(
                "kernel-cycles", f"no port of {space_type.__qualname__} carries a stream's beats"
            )
        return reject("kernel-cycles", f"no member of {space_type.__qualname__} states cycles")

    @view
    def cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through it: the most beats any of its ports carries
        a frame, or any of its members takes."""
        found = [item.value for item in self.member_cycles]
        if found:
            return max(found)
        if type(self).module_spec is not None:
            return reject(
                "kernel-cycles", f"no port of {type(self).__qualname__} carries a stream's beats"
            )
        return reject("kernel-cycles", f"no member of {type(self).__qualname__} states cycles")

    member_cycles = Members(Timed.cycles)


def extent_of(index: Index) -> int:
    """A derived member of a kernel: ``index``'s extent, bound from its placed ports.

    Name it in the class body (``channels = extent_of(c)``) and read that name,
    in a folding factor's ``divisors_of`` domain for instance; used inline inside another
    declaration it is not a member of the class, and linking refuses it.
    """

    def extent(self: UntimedKernel) -> int | Rejected:
        extents = self.extents
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return extents[index]

    return derived(extent)


__all__ = [
    "Kernel",
    "KernelSpace",
    "NATIVE_CLOCKING",
    "PortContract",
    "UntimedKernel",
    "extent_of",
    "merged",
]
