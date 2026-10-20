# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel protocol: two bases, a leaf and a kernel with children, and what they share.

A kernel is a Space whose choices configure hardware. Each base declares what a kernel of
its kind states, with the engine's ``required()``: a class that leaves one unmet is a base
of kernels (dotp's cores share one), not placed. The rest is checked when a class is made.

A **leaf** (``LeafKernel``) binds one FinnLib module. It states its ``rtl_module`` and
``sources`` (an HLS kernel's one source is its build request), its ``parameters``, its
``resource_use`` (a refusal where it has no model of it, never nothing), the constraints
by which it refuses what it cannot build, and its ports (``finn.kernels.port``), each
stream port taking the tensor it reads as a fact (absent: an idle port). It may state its
``clocking``, and ``other_pins``, ``held`` and ``controlled`` for pins that are no port's.
The base derives its module (``codegen``, a ``Leaf``), checks its pins (the constraints
``pins_accounted``, ``clocked``), and takes its cycles as the most beats any of its ports
carries a frame.

A **kernel with children** (``ComposedKernel``) places kernels and channels, and names
for each interface it presents (``interfaces``) the child that serves it. It merges its
members' netlists and control buses, and is no module of its own: a network
(``finn.kernels.network``) composes one. A channel is one (``finn.kernels.channels``).

Every constraint of a kernel, of its ports and of its present children counts toward its
validity (``inspection.validity``), and each of its views is a product: it refuses while
the kernel is refused, and answers while the kernel is undecided. A kernel states one
contract per stream interface: a view ``<name>_end`` of a ``PortEnd`` (on a leaf
``interface("x")``; on a kernel with children the serving child's, labelled below it:
MatMul's ``compute.packed.x``). Every kernel exports its netlist (``NETLIST``), a leaf
its module (``MODULE``), and what an exploration ranks by: ``CYCLES``, ``BUFFERING`` and
``RESOURCES``, views like the rest, so a refused kernel's cost refuses too. A port
exports the values its kernel reads to bind its extents (``ACCESS``) and check its clock
(``CLOCKED``): values, so a port's refusal stands in its kernel's validity and never in
the kernel's extents or domains.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

from finn.core.space import (
    DefinitionError,
    Members,
    Rejected,
    Space,
    ValueSemantics,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    reject,
    required,
    view,
)
from finn.dataflow.schedule import Access, Affine, Index, Refused, Schedule, bind_extents
from finn.kernels.artifacts.abi import Bus, Clocking, Direction, Pin, Signal
from finn.kernels.artifacts.contributions import (
    Contribution,
    CopiedSource,
    GeneratedData,
    HlsSource,
    request_stem,
)
from finn.kernels.artifacts.module import (
    Abi,
    BuildError,
    Composed,
    Fragment,
    Held,
    Leaf,
    Module,
)
from finn.kernels.control import CONTROL, presented
from finn.kernels.defaults import Fixed, stated_defaults
from finn.kernels.transport import StreamContract
from finn.kernels.utilization import RESOURCES_SEMANTICS, total
from finn.platform.capabilities import Resources


def _frozen(name: str, *kinds: type) -> ValueSemantics[object]:
    """A frozen build value: compared by value, shared without a copy."""
    return ValueSemantics(
        kinds[0],
        name,
        lambda value: isinstance(value, kinds),
        lambda left, right: left == right,
        lambda value: value,
    )


MODULE_SEMANTICS = cast("ValueSemantics[Module]", _frozen("module", Leaf, Composed))
MODULE = ViewKey("module", MODULE_SEMANTICS)
"""A leaf's module, its ``Leaf``."""

NETLIST_SEMANTICS = cast("ValueSemantics[Fragment]", _frozen("netlist", Fragment))
NETLIST = ViewKey("netlist", NETLIST_SEMANTICS)
"""A kernel's or a channel's netlist (``finn.kernels.artifacts.module.Fragment``), labelled
relative to itself; whatever holds it (a network, a kernel with children) merges it
under its node."""

BOUNDARY = ViewKey("boundary", default_semantics(tuple))
"""A channel's AXIS bus on its network's boundary: one, or none when both of its ends are
kernels."""

PINS = ViewKey("pins", default_semantics(tuple))
"""A port's pins (signals, or one bus), collected by its kernel into the module's pins."""

CLOCKED = ViewKey("clocked", default_semantics(str))
"""The clock pin a port runs on, collected by its kernel (``clocked``): a value export,
read whatever the port's validity."""

ACCESS = ViewKey("access", default_semantics(Access))
"""A placed scheduled input's read of its tensor, collected by its kernel to bind the
extents of its indices (``finn.dataflow.schedule.bind_extents``): a value export, read
whatever the port's validity, so a port's refusal never refuses its kernel's extents."""

HELD_SEMANTICS = default_semantics(Held)
HELD = ViewKey("held", HELD_SEMANTICS)
"""What an idle port holds, collected by its kernel's leaf."""

BEATS = ViewKey("beats", default_semantics(int))
"""The beats a placed stream port carries a frame, collected by its leaf: the leaf's cycles
are the most of them (``LeafKernel.frame_cycles``)."""

CYCLES = ViewKey("cycles", default_semantics(int))
"""The clock cycles a frame takes through a kernel or a channel: a leaf's the most beats
any of its ports carries; a kernel with children's its slowest member's."""

BUFFERING = ViewKey("buffering", default_semantics(int))
"""The bits a channel's stages hold between its ends: a stage's its own (an
``input_gen``'s buffer, a FIFO's depth); a kernel with children's, a channel's among
them, the sum over its members."""

RESOURCES = ViewKey("resources", RESOURCES_SEMANTICS)
"""What a kernel or a channel uses of the device (``finn.platform.capabilities.Resources``):
a leaf's own statement; a kernel with children's, a channel's among them, the sum over
its members."""


@dataclass(frozen=True)
class PortEnd:
    """The end a port presents on a channel: its ``contract`` and its ``label``, the port's
    path relative to the member that states the end (``x`` for a leaf's port,
    ``compute.packed.x`` for MatMul's, ``source.memstream.set`` for a channel's source's
    set port; empty for an end the member presents itself, which is no port's).
    ``under`` labels it from the network (or kernel with children) that holds the
    member."""

    label: str
    contract: StreamContract

    def under(self, member: str) -> PortEnd:
        """The end as seen from where ``member`` is held."""
        return PortEnd(f"{member}.{self.label}" if self.label else member, self.contract)


PORT_END = default_semantics(PortEnd)
"""A ``PortEnd``'s value semantics: a frozen value, compared by value."""


NATIVE_CLOCKING = Clocking(clock="clk", reset="rst", active_low=False)
"""FinnLib's native ``clk`` and synchronous active-high ``rst``."""


def _module_name(stem: str, contributions: tuple[Contribution, ...]) -> str:
    """``stem``, or the top of the HLS request among ``contributions``, named from it."""
    requests = [item for item in contributions if isinstance(item, HlsSource)]
    if not requests:
        return stem
    function = requests[0].function
    if request_stem(function) != stem:
        raise BuildError(f"{function} is not the top of the HLS module {stem}")
    return function


def _inputs(pin: Pin) -> tuple[str, ...]:
    if isinstance(pin, Signal):
        return (pin.name,) if pin.direction is Direction.IN else ()
    return tuple(name for name, direction in pin.member_directions() if direction is Direction.IN)


class Kernel(Space):
    """A named Space class configuring hardware: a ``LeafKernel`` or a ``ComposedKernel``
    (the module docstring). ``defaults`` fixes the default of a choice it declares, by its
    key (``finn.kernels.defaults``): every other choice is first valid, but a required one,
    which has none."""

    id: ClassVar[str] = ""
    version: ClassVar[int] = 1
    defaults: ClassVar[Mapping[str, Fixed]] = {}

    def __init_subclass__(cls, *, protocol: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "defaults" in vars(cls):
            stated_defaults(cls, cls.defaults)
        if protocol:
            return  # one of the two bases
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError.broken(
                "kernel-id-invalid",
                "Kernel",
                f"{cls.__qualname__} must declare a nonempty string id",
                declaration=cls.__qualname__,
            )
        if type(cls.version) is not int or cls.version < 1:
            raise DefinitionError.broken(
                "kernel-version-invalid",
                "Kernel",
                f"{cls.__qualname__} must declare a positive int version",
                declaration=cls.__qualname__,
            )
        if not issubclass(cls, (LeafKernel, ComposedKernel)):
            raise DefinitionError.broken(
                "kernel-kind-invalid",
                "Kernel",
                f"{cls.__qualname__}: a kernel is a LeafKernel (one module) or a "
                "ComposedKernel (a kernel with children)",
                declaration=cls.__qualname__,
            )


def _declared(cls: type[Kernel], name: str, below: type[Kernel]) -> bool:
    """Whether a class from ``cls`` down to, not including, ``below`` declares ``name``."""
    return any(name in vars(base) for base in cls.__mro__[: cls.__mro__.index(below)])


def _members(cls: type[Kernel], below: type[Kernel]) -> dict[str, object]:
    """The class attributes ``cls`` and its bases down to ``below`` declare, by name."""
    found: dict[str, object] = {}
    for base in reversed(cls.__mro__[: cls.__mro__.index(below)]):
        found.update(vars(base))
    return found


_LEAF_REQUIRED = ("rtl_module", "parameters", "sources", "resource_use")
"""What a leaf states by ``required()``; a class that states them all is checked for the
rest of the protocol when it is made."""


class LeafKernel(Kernel, protocol=True):
    """A kernel binding one module: see the module docstring for what it states."""

    # What every leaf states (``required()``: a leaf that leaves one unmet is not placed).
    rtl_module: ClassVar[str] = required(str)
    parameters = cast("dict[str, int | str]", required(dict))
    sources = cast("tuple[Contribution, ...]", required(tuple))
    resource_use = required(Resources)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        from finn.kernels.port import Port  # noqa: PLC0415 (port.py builds on this module)

        if not all(_declared(cls, name, LeafKernel) for name in _LEAF_REQUIRED):
            return  # a base of leaves (dotp's cores'), which cannot be placed
        members = _members(cls, LeafKernel)
        if not any(isinstance(value, Port) for value in members.values()):
            raise DefinitionError.broken(
                "leaf-ports-unplaced",
                "LeafKernel",
                f"{cls.__qualname__} places no port: a leaf states its ports",
                declaration=cls.__qualname__,
            )
        children = sorted(name for name, value in members.items() if isinstance(value, Kernel))
        if children:
            raise DefinitionError.broken(
                "leaf-places-kernels",
                "LeafKernel",
                f"{cls.__qualname__} places the kernels {children}: a leaf binds one module, "
                "a ComposedKernel places children",
                declaration=cls.__qualname__,
            )

    # -- what a leaf may state ------------------------------------------------------------

    @derived
    def clocking(self) -> Clocking:
        return Clocking()

    @derived
    def other_pins(self) -> tuple[Pin, ...]:
        """Pins that are no port's, such as an AXI-Lite configuration bus."""
        return ()

    @derived
    def held(self) -> Held:
        """What the module itself holds idle, beyond its idle ports and doubled clock."""
        return Held()

    @derived
    def controlled(self) -> tuple[Bus, ...]:
        """The buses among its other pins it presents (its ``Control``, at the port its
        placement names)."""
        return ()

    # -- its ports ------------------------------------------------------------------------

    port_pins = Members(PINS)
    port_holds = Members(HELD)
    port_clocks = Members(CLOCKED)
    port_accesses = Members(ACCESS)
    port_beats = Members(BEATS)

    # -- extents and the schedule --------------------------------------------------------

    def _bound(self, extents: Mapping[Index, int] | None = None) -> dict[Index, int] | Rejected:
        accesses = [replace(item.value, name=item.node or "") for item in self.port_accesses]
        try:
            return bind_extents(accesses, extents)
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
        ``i``'s extent, one at ``oh * OW + ow`` the reach of its terms, plus one. A leaf
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

    # -- its module, and its pins accounted for -------------------------------------------

    @derived
    def holds(self) -> Held:
        """What the module holds: the doubled clock while unused, idle ports, its own."""
        parts = (Held(self.clocking.held()), *(item.value for item in self.port_holds), self.held)
        return Held(
            tuple(pin for part in parts for pin in part.inputs),
            tuple(pin for part in parts for pin in part.unused),
        )

    @derived
    def codegen(self) -> Leaf:
        """The module: clocking, its other pins, then every port's pins; its parameters,
        sources and data; and what it holds."""
        space_type = type(self)
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        pins = tuple(pin for item in self.port_pins for pin in item.value)
        contributions = self.sources
        return Leaf(
            space_type.id,
            str(space_type.version),
            _module_name(space_type.rtl_module, contributions),
            parameters,
            Abi(
                (*clocking.signals(), *self.other_pins, *pins),
                tuple((name, str(value)) for name, value in parameters),
                clocking.alignments(),
            ),
            tuple(item for item in contributions if isinstance(item, (CopiedSource, HlsSource))),
            tuple(item for item in contributions if isinstance(item, GeneratedData)),
            self.holds,
        )

    @constraint
    def pins_accounted(self) -> bool | Rejected:
        """Every input among its other pins is held, or presented as a control bus."""
        accounted = {pin for pin, _ in self.held.inputs} | {
            member.physical for bus in self.controlled for member in bus.signals
        }
        loose = [pin for item in self.other_pins for pin in _inputs(item) if pin not in accounted]
        if loose:
            return reject(
                "kernel-pins",
                f"{type(self).__qualname__}: the inputs {loose} are neither held nor presented",
            )
        return True

    @constraint
    def clocked(self) -> bool | Rejected:
        """Every port runs on the module's clock."""
        clock = self.clocking.clock
        other = [f"{item.node} on {item.value}" for item in self.port_clocks if item.value != clock]
        if other:
            return reject("kernel-clock", f"ports run off the clock {clock}: {', '.join(other)}")
        return True

    @derived(semantics=MODULE_SEMANTICS)
    def built(self) -> Leaf | Composed:
        return self.codegen

    @derived
    def fragment(self) -> Fragment:
        """Its accepted module, the empty label (``under(node)`` names it ``node`` where
        it is held)."""
        leaf = self.module
        assert isinstance(leaf, Leaf)
        return Fragment((("", leaf),))

    module = View(built)
    netlist = View(fragment)

    # -- what a design space exploration ranks by ------------------------------------------

    @derived
    def frame_cycles(self) -> int | Rejected:
        """The clock cycles a frame takes: the most beats any of its ports carries a frame,
        one beat a cycle at best."""
        found = [item.value for item in self.port_beats]
        if not found:
            return reject(
                "kernel-cycles", f"no port of {type(self).__qualname__} carries a stream's beats"
            )
        return max(found)

    @derived
    def held_bits(self) -> int:
        """The bits it holds between a channel's ends: none, but for a stage that buffers
        (a FIFO, an ``input_gen``), which states its own."""
        return 0

    @derived(semantics=RESOURCES_SEMANTICS)
    def stated_resources(self) -> Resources:
        """What it uses of the device: its own statement, ``resource_use``."""
        return self.resource_use

    # Products like the module: a refused leaf's cost refuses with it.
    cycles = View(frame_cycles)
    buffering = View(held_bits)
    resources = View(stated_resources)

    # A leaf adding exports of its own extends these: ``{**LeafKernel.exports, KEY: ...}``.
    exports = {
        NETLIST: netlist,
        MODULE: module,
        CYCLES: cycles,
        BUFFERING: buffering,
        RESOURCES: resources,
    }


class ComposedKernel(Kernel, protocol=True):
    """A kernel with children: see the module docstring for what it states."""

    # The interfaces it presents, by name, each a view ``<name>_end``; a stage presents none.
    interfaces: ClassVar[tuple[str, ...]] = required(tuple)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        names = cls.interfaces
        if not isinstance(names, tuple):
            return  # unmet: the class is a base of kernels with children, not placed
        missing = [name for name in names if not hasattr(cls, f"{name}_end")]
        if missing:
            raise DefinitionError.broken(
                "composed-end-absent",
                "ComposedKernel",
                f"{cls.__qualname__} presents the interfaces {missing} without their ends: "
                "each is a view <name>_end naming the child that serves it",
                declaration=cls.__qualname__,
            )

    # -- its children: their netlists, merged ---------------------------------------------

    netlists = Members(NETLIST)
    controls = Members(CONTROL)

    @derived
    def fragment(self) -> Fragment | Rejected:
        """Each member's netlist under its node, and the control buses its members
        present: what a network, or a parent, merges into its own."""
        return self.composed_fragment()

    def composed_fragment(self) -> Fragment | Rejected:
        """A kernel with children's ``fragment``: each member's netlist under its node, and
        the control buses its members present with the writes their configuration takes
        (``finn.kernels.network.merged``). A kernel that places none is refused."""
        from finn.kernels.network import merged  # noqa: PLC0415 - the network imports the base

        if not self.netlists:
            return reject(
                "kernel-module",
                f"{type(self).__qualname__} declares no module and places no kernel",
            )
        buses = presented((str(item.node), item.value) for item in self.controls)
        if isinstance(buses, Rejected):
            return buses
        return merged(((str(item.node), item.value) for item in self.netlists), buses)

    netlist = View(fragment)

    # -- what a design space exploration ranks by ------------------------------------------

    member_cycles = Members(CYCLES)
    member_buffering = Members(BUFFERING)
    member_resources = Members(RESOURCES)

    @derived
    def frame_cycles(self) -> int | Rejected:
        """The clock cycles a frame takes: its slowest member's."""
        found = [item.value for item in self.member_cycles]
        if not found:
            return reject("kernel-cycles", f"no member of {type(self).__qualname__} states cycles")
        return max(found)

    @derived
    def held_bits(self) -> int:
        """The bits its channels' stages hold, summed over its members."""
        return sum(item.value for item in self.member_buffering)

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        """What it uses of the device: the sum over its members."""
        return total(item.value for item in self.member_resources)

    # Products like the netlist: a refused kernel's cost refuses with it.
    cycles = View(frame_cycles)
    buffering = View(held_bits)
    resources = View(resource_use)

    exports = {
        NETLIST: netlist,
        CYCLES: cycles,
        BUFFERING: buffering,
        RESOURCES: resources,
    }


def interface(port: str) -> PortEnd:
    """A view of a leaf: the end its port ``port`` presents (``Port.contract``), labelled
    ``port``. Name it in the class body after the interface (``x_end = interface("x")``)."""

    def end(self: LeafKernel) -> PortEnd:
        contract: StreamContract = getattr(self, port).contract
        return PortEnd(port, contract)

    return cast(PortEnd, view(semantics=PORT_END)(end))


def extent_of(index: Index) -> int:
    """A derived member of a leaf: ``index``'s extent, bound from its placed ports.

    Name it in the class body (``channels = extent_of(c)``) and read that name,
    in a folding factor's ``divisors_of`` domain for instance; used inline inside another
    declaration it is not a member of the class, and linking refuses it.
    """

    def extent(self: LeafKernel) -> int | Rejected:
        extents = self.extents
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return extents[index]

    return derived(extent)


__all__ = [
    "ACCESS",
    "BEATS",
    "BOUNDARY",
    "BUFFERING",
    "CLOCKED",
    "CYCLES",
    "ComposedKernel",
    "HELD",
    "Kernel",
    "LeafKernel",
    "NATIVE_CLOCKING",
    "NETLIST",
    "PINS",
    "PORT_END",
    "PortEnd",
    "RESOURCES",
    "extent_of",
    "interface",
]
