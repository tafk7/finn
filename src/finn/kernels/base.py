# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel protocol: two bases, a leaf and a kernel with children, and what they share.

A kernel is a Space whose choices configure one module. It is one of two kinds, each
a base that declares what a kernel of its kind must state with the engine's
``required()``: a class that leaves one unmet cannot be placed, nor named as a
Decision's candidate. What ``required()`` cannot name (a group the base's own views
require, the port nodes) is checked when a class that states the rest is made; a
class that leaves some unmet is a base of kernels (dotp's cores share one), and its
subclasses are checked.

A **leaf** (``LeafKernel``) binds one FinnLib module. It states:

- ``rtl_module``: the RTL module it instantiates, and ``sources``, the source files
  that provide it and any data they read. An HLS kernel's one source is its build
  request (``HlsSource``), and its ``rtl_module`` the stem of the request's top: the
  module it instantiates is that top, ``<rtl_module>_<key16>``;
- ``parameters``: the module's parameters, from its choices;
- ``resource_use``: what it uses of the device, from its RTL's parameters and its
  memory styles (``finn.kernels.utilization``); a leaf with no model of it states its
  refusal (``kernel-resources``), never nothing;
- its ``admission``: the constraint group by which it refuses a configuration it
  cannot build (``ConstraintGroup()`` when it refuses none), checked when the class
  is made;
- its ports: one ``Port`` node per interface of its module (``finn.kernels.port``),
  each exporting its pins, its clock and what it holds while idle; at least one,
  checked when the class is made. A leaf places no kernel.

Each stream port takes the tensor it reads, a fact (``x_tensor: Tensor =
Param(required=False)``, given to its port's ``tensor``; absent, the port is an
interface not used, idle), or writes: the leaf's own statement of its output, a view
of its facts (``y_tensor``), whose element is what the module produces. Optionally a
leaf states its ``clocking`` (its clock and reset pins, when not the plain ``ap_clk``
and ``ap_rst_n``), and ``other_pins``, ``held`` and ``controlled``: pins that are no
port's (an AXI-Lite bus), what it holds of them, and the buses among them it presents
through a ``ControlBus`` (``finn.kernels.control``).

The leaf base derives the rest. ``codegen`` is its ``Leaf``: its pins (clocking, other
pins, then every port's), parameters, sources and data, and what it holds (the
doubled clock while unused, idle ports, and its own ``held``). Two constraints account
for its pins: ``pins_accounted`` (every input among its other pins is held or
presented) and ``clocked`` (every port runs on its clock). Its cycles are the most
beats any of its ports carries a frame (``BEATS``: one beat a cycle at best), so a leaf
states no cycles of its own; it holds no channel's buffering.

A **kernel with children** (``ComposedKernel``) places kernels and the channels
between them. It states:

- ``interfaces``: the names of the interfaces it presents, each the end of a view
  ``<name>_end`` that names the child serving it (below); a root, the top of what is
  emitted, presents none, its boundary channels being its pins. Each named view is
  checked when the class is made;
- its children: the kernels and channels it places, its members. It merges their
  netlists (``NETLIST``: each child's and each channel's ``Fragment``, under its node)
  and the control buses its ``ControlBus`` nodes present; one that places no kernel is
  refused (``kernel-module``). It gives its interfaces' tensors to the children that
  read them, and the channels between its children are its own. Placed alone, as the
  root of what is emitted, it is one ``Composed`` module whose pins are ``ap_clk``,
  ``ap_clk2x`` when an instance takes a doubled clock, ``ap_rst_n``, the AXIS bus of
  each boundary channel it declares (inputs, then outputs) and each presented bus. The
  merge is one plain function, which a network of kernels and channels
  (``finn.kernels.network``) composes its module with too: ``composed_fragment``,
  ``composed_abi`` and ``built`` call its steps (``merged``, ``module_abi``,
  ``assembled``). Its cycles are its slowest member's, kernels and channels alike, so a
  root's cycles are its bottleneck; its buffering and resources the sums over its
  members.

**One contract per interface.** A kernel states, for each stream interface by name
(``x``, ``w``, ``y``), its end: a view ``<name>_end`` of a ``PortEnd``, the contract its
port presents there and the port's label relative to the kernel. A leaf states it with
``interface``: ``x_end = interface("x")`` is its port ``x``'s contract, labelled ``x``.
A kernel with children names the child that serves the interface and the port's label
below it (MatMul's ``x_end`` is the selected core's, ``compute.packed.x``). Facts flow
one way: a kernel's input tensors in, its ends and its output tensors out. A channel
(``finn.kernels.channels``) takes the two ends on it as facts, ``producer`` and
``consumer``, each labelled from the Space that declares the channel (``end_of``:
``matmul.compute.packed.x``), so that its hops name the kernel whose port it is
(``^matmul.compute.packed``).

Every kernel exports its netlist (``NETLIST``) and its ``module`` (``MODULE``): the
``Leaf``, or the ``Composed`` module, accepted under its admission (and a leaf's pins
accounted for). It also exports what a design space exploration ranks by: cost, not
validity, so no cost export waits on an admission, and each reads only the choices it
depends on: ``cycles`` (``CYCLES``), the clock cycles a frame takes; ``buffering``
(``BUFFERING``), the bits its channels' stages hold; ``resources`` (``RESOURCES``), what
it uses of the device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

from finn.core.space import (
    ConstraintGroup,
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
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Data,
    Derived,
    Direction,
    Free,
    Pin,
    Reset,
    Signal,
)
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
    ProducerIdentity,
)
from finn.kernels.control import EXPORTED
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
"""A kernel's module: its ``Leaf``, or its children's netlist as one ``Composed`` module."""

NETLIST_SEMANTICS = cast("ValueSemantics[Fragment]", _frozen("netlist", Fragment))
NETLIST = ViewKey("netlist", NETLIST_SEMANTICS)
"""A kernel's or a channel's netlist (``finn.kernels.artifacts.module.Fragment``), labelled
relative to itself; its parent merges its members' under their nodes."""

BOUNDARY = ViewKey("boundary", default_semantics(tuple))
"""A channel's AXIS bus on the root's boundary: one, or none when both of its ends are
kernels."""

PINS = ViewKey("pins", default_semantics(tuple))
"""A port's pins (signals, or one bus), collected by its kernel into the module's pins."""

CLOCKED = ViewKey("clocked", default_semantics(str))
"""The clock pin a port runs on, collected by its kernel (``clocked``)."""

ACCESS = ViewKey("access", default_semantics(Access))
"""A placed scheduled port's read of its tensor, collected by its kernel to bind the
extents of its indices (``finn.dataflow.schedule.bind_extents``)."""

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
"""The bits a channel's stages hold between its ends (an adapter's ``input_gen``
buffers, a FIFO's depth); a kernel with children's is the sum over its members."""

RESOURCES = ViewKey("resources", RESOURCES_SEMANTICS)
"""What a kernel or a channel uses of the device (``finn.platform.capabilities.Resources``):
a leaf's own statement; a kernel with children's and a channel's the sum over its
members and stages."""

# The composed module's clocking pins: its interface convention, not a routing rule.
CLOCK, CLOCK2X, RESET = "ap_clk", "ap_clk2x", "ap_rst_n"


@dataclass(frozen=True)
class PortEnd:
    """The end a port presents on a channel: its ``contract`` and its ``label``, the port's
    path relative to the member that states the end (``x`` for a leaf's port,
    ``compute.packed.x`` for MatMul's, ``source.memstream.set`` for a channel's source's
    set port; empty for an end the member presents itself, which is no port's).
    ``under`` labels it from the Space that places that member."""

    label: str
    contract: StreamContract

    def under(self, member: str) -> PortEnd:
        """The end as seen from the Space where ``member`` is placed."""
        return PortEnd(f"{member}.{self.label}" if self.label else member, self.contract)


PORT_END = default_semantics(PortEnd)
"""A ``PortEnd``'s value semantics: a frozen value, compared by value."""


@dataclass(frozen=True)
class Clocking:
    """A module's clock and reset pins.

    ``clock`` is free-running and ``reset`` synchronous, active low unless
    ``active_low`` is False (FinnLib's native ``clk`` and ``rst``). A
    ``doubled`` pin, when the module has one, is a clock at twice ``clock``
    while ``doubling``, and otherwise an unused input held low.
    """

    clock: str = CLOCK
    reset: str = RESET
    doubled: str | None = None
    doubling: bool = False
    active_low: bool = True

    def signals(self) -> tuple[Signal, ...]:
        doubled = () if self.doubled is None else (self.doubled,)
        clocks = (self.clock, *doubled) if self.doubling else (self.clock,)
        return (
            Signal(self.clock, Direction.IN, 1, Clock(Free())),
            *(
                Signal(
                    name,
                    Direction.IN,
                    1,
                    Clock(Derived(self.clock, 2)) if self.doubling else Data(),
                )
                for name in doubled
            ),
            Signal(
                self.reset,
                Direction.IN,
                1,
                Reset(active_low=self.active_low, synchronous=True, synchronous_to=clocks),
            ),
        )

    def alignments(self) -> tuple[ClockAlignment, ...]:
        if self.doubled is None or not self.doubling:
            return ()
        return (ClockAlignment(self.clock, self.doubled),)

    def held(self) -> tuple[tuple[str, int], ...]:
        """The doubled clock input, held low while unused."""
        if self.doubled is None or self.doubling:
            return ()
        return ((self.doubled, 0),)


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
    """A named Space class configuring one module: a ``LeafKernel`` or a
    ``ComposedKernel`` (the module docstring)."""

    id: ClassVar[str] = ""
    version: ClassVar[int] = 1

    def __init_subclass__(cls, *, protocol: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if protocol:
            return  # one of the two bases
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError(f"{cls.__qualname__} must declare a nonempty string id")
        if type(cls.version) is not int or cls.version < 1:
            raise DefinitionError(f"{cls.__qualname__} must declare a positive int version")
        if not issubclass(cls, (LeafKernel, ComposedKernel)):
            raise DefinitionError(
                f"{cls.__qualname__}: a kernel is a LeafKernel (one module) or a "
                "ComposedKernel (a kernel with children)"
            )

    admission = ConstraintGroup()


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
        if not _declared(cls, "admission", LeafKernel):
            raise DefinitionError(
                f"{cls.__qualname__} states no admission: a leaf states the constraints by "
                "which it refuses what it cannot build (ConstraintGroup() for none)"
            )
        members = _members(cls, LeafKernel)
        if not any(isinstance(value, Port) for value in members.values()):
            raise DefinitionError(f"{cls.__qualname__} places no port: a leaf states its ports")
        children = sorted(name for name, value in members.items() if isinstance(value, Kernel))
        if children:
            raise DefinitionError(
                f"{cls.__qualname__} places the kernels {children}: a leaf binds one module, "
                "a ComposedKernel places children"
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
        """The buses among its other pins it presents through a ``ControlBus``."""
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
        """Its accepted module, the empty label (its parent's ``under(node)`` names it
        ``node``)."""
        leaf = self.module
        assert isinstance(leaf, Leaf)
        return Fragment((("", leaf),))

    module = View(built, requires=(Kernel.admission, pins_accounted, clocked))
    netlist = View(fragment, requires=(Kernel.admission, pins_accounted, clocked))

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
        """The bits its channels' stages hold: a leaf holds none of them."""
        return 0

    @derived(semantics=RESOURCES_SEMANTICS)
    def stated_resources(self) -> Resources:
        """What it uses of the device: its own statement, ``resource_use``."""
        return self.resource_use

    # Cost, not validity: a configuration's admission is the engine's, read apart.
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

    # The interfaces it presents, by name, each a view ``<name>_end``; a root presents none.
    interfaces: ClassVar[tuple[str, ...]] = required(tuple)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        names = cls.interfaces
        if not isinstance(names, tuple):
            return  # unmet: the class is a base of kernels with children, not placed
        missing = [name for name in names if not hasattr(cls, f"{name}_end")]
        if missing:
            raise DefinitionError(
                f"{cls.__qualname__} presents the interfaces {missing} without their ends: "
                "each is a view <name>_end naming the child that serves it"
            )

    def stem(self) -> str:
        """Its composed module's name stem."""
        return "finn_" + type(self).__name__.lower()

    def producer_identity(self) -> ProducerIdentity:
        """What derives its composed module."""
        space_type = type(self)
        return ProducerIdentity(space_type.id, str(space_type.version))

    # -- its children: their netlists, merged ---------------------------------------------

    netlists = Members(NETLIST)
    presented = Members(EXPORTED)
    boundary_buses = Members(BOUNDARY)

    @derived
    def fragment(self) -> Fragment | Rejected:
        """Each member's netlist under its node, and the buses its ``ControlBus`` nodes
        present; its own module is complete only as the root, whose channels it declares,
        so a parent reads its netlist, not its module."""
        return self.composed_fragment()

    def composed_fragment(self) -> Fragment | Rejected:
        """A kernel with children's ``fragment``: each member's netlist under its node, and
        the buses its ``ControlBus`` nodes present with the writes their configuration
        takes (``finn.kernels.network.merged``). A kernel that places none is refused."""
        from finn.kernels.network import merged  # noqa: PLC0415 - the network imports the base

        if not self.netlists:
            return reject(
                "kernel-module",
                f"{type(self).__qualname__} declares no module and places no kernel",
            )
        return merged(
            ((str(item.node), item.value) for item in self.netlists),
            (item for located in self.presented for item in located.value),
        )

    @derived
    def composed_abi(self) -> Abi:
        """Its clocks and reset, each boundary channel's AXIS bus (inputs, then outputs), then
        each presented bus (``finn.kernels.network.module_abi``)."""
        from finn.kernels.network import module_abi  # noqa: PLC0415 - the network imports the base

        return module_abi(
            self.fragment, (bus for item in self.boundary_buses for bus in item.value)
        )

    @derived(semantics=MODULE_SEMANTICS)
    def built(self) -> Leaf | Composed | Rejected:
        from finn.kernels.network import assembled  # noqa: PLC0415 - the network imports the base

        return assembled(self.producer_identity(), self.stem(), self.composed_abi, self.fragment)

    module = View(built, requires=(Kernel.admission, netlists, presented, boundary_buses))
    netlist = View(fragment, requires=(Kernel.admission, netlists, presented, boundary_buses))

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

    # Cost, not validity: a configuration's admission is the engine's, read apart.
    cycles = View(frame_cycles)
    buffering = View(held_bits)
    resources = View(resource_use)

    exports = {
        NETLIST: netlist,
        MODULE: module,
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
    "CLOCK",
    "CLOCK2X",
    "CLOCKED",
    "CYCLES",
    "Clocking",
    "ComposedKernel",
    "HELD",
    "Kernel",
    "LeafKernel",
    "NATIVE_CLOCKING",
    "NETLIST",
    "PINS",
    "PORT_END",
    "PortEnd",
    "RESET",
    "RESOURCES",
    "extent_of",
    "interface",
]
