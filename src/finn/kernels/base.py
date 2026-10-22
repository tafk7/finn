# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel base, ``KernelSpace``, and what every kernel shares.

A kernel is a Space whose choices configure hardware. **``KernelSpace``** is the one base
of kernels: a kernel binds a module of the library or places kernels and channels (its
members), never both (``module-places-kernels``). A kernel states its ``id`` (a nonempty
string), its facts and choices, the constraints by which it refuses what it cannot build,
and its declarations (below). It may state its ``clock_domain``, on which its ports and its
module run (``clocking``). It states no policy for its choices: those are the compiler's
(``finn.kernels.policies``); a ranking it knows is a view a policy may name.

**Its four answers**, each a view written once in the base (``_netlist``, ``_cycles``,
``_buffering``, ``_resources``) and made for each class when it is made, reading its
declarations by the names their lowering fixes:

- ``netlist``: its library module at the empty label (``finn.kernels.library``), its
  members' netlists under their nodes, and the control buses its members present
  (``merged``). A kernel with neither a module nor a member is refused (``kernel-module``).
  A channel states its own, adding its hops (``finn.kernels.channels``).
- ``cycles``: the most beats any of its declarations carries a frame, or any of its
  members takes, one beat a cycle at best; refused where there is none (``kernel-cycles``).
  A channel of wires takes none of its own (0).
- ``buffering``: the bits it holds itself (``held_bits``, a view: none, but for a stage
  that buffers) and its members'.
- ``resources``: what it uses of the device itself (``resource_use``, a view: nothing, but
  for a kernel with a module, which states its own) and its members'.

Nothing in its design space reads them, and each refuses while the kernel is refused.
A class may state one itself, calling the base's body. Its library module is read back
from its netlist (``finn.kernels.library.library_module``). **``Kernel``** is what a
reader of any kernel's design point asks: a ``typing.Protocol`` of the four answers, read
out, and nothing else (its docstring).

**What it reaches of its members**: their answers, through ``Members`` over their exports
(``netlists``, ``controls``, ``member_cycles``, ``member_buffering``,
``member_resources``), read by its views alone. A parent reaches only its present members,
one refused member refusing the answer.

**Its declarations** (``finn.kernels.declaration``): its stream ports
(``finn.kernels.port``), the ports it presents for its members
(``finn.kernels.port.served``), the library module it binds
(``finn.kernels.library.LibraryModule``) and its control bus
(``finn.kernels.control.ControlBus``), each a plain frozen value in its class body. When
the class is made, the base lowers each into ordinary members of the kernel
(``KernelDeclaration.lower``), and makes its answers read what each states of its module,
pins, holds, reads and beats by name.

A kernel binding a module states its parameters, its sources (an HLS kernel's one source
is its build request) and its data in its ``LibraryModule``, its ``resource_use`` (a
refusal where it has no model of it, never nothing) and its ports, each stream port taking
the tensor it reads as a fact (absent: an idle port), and may declare a control bus, held
idle or presented. Each input of its module is a port's, a presented control bus's, or
held (``kernel-pins``). The base binds its indices' extents from its ports' reads of their
tensors (``extents``), from which it states schedules (``bound_schedule``) and shapes
(``spanned``).

**The definition checks**, when a class is made: its ``id`` (``kernel-id-invalid``); a
declaration lowering to a member the class declares itself (``declaration-member-taken``);
at most one library module (``library-module-repeated``) and one control bus
(``control-bus-repeated``); constant sources provide the module
(``library-module-unprovided``). A class leaving a ``required()`` member unmet is a base of
kernels (the dot-product cores'), not placed, and the checks below pass it by. A kernel
with a module states a stream port (``module-ports-unplaced``) and its ``resource_use``
(``module-resources-unstated``), and places no kernel (``module-places-kernels``). A kernel
stating a stream port binds a module (``port-unbound``): a port's pins are its module's,
and a kernel with children presents its children's ports (``served``).

Every constraint of a kernel and of its present members counts toward its validity
(``inspection.validity``). A kernel states one contract per stream port: a view
``<name>_contract`` of a ``PortContract`` (a module's port's, labelled ``x``; a member's,
labelled below it: MatMul's ``compute.packed.x``). Every kernel exports its netlist
(``NETLIST``), its control bus's ``Control`` (``CONTROL``) where it declares one, and what
an exploration ranks by: ``CYCLES``, ``BUFFERING`` and ``RESOURCES``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, cast

from finn.core.space import (
    DefinitionError,
    Members,
    Rejected,
    Space,
    ValueSemantics,
    ViewKey,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.core.space.declarations import unmet_required
from finn.dataflow.schedule import Access, Affine, Index, Refused, Schedule, bind_extents
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Clocking
from finn.kernels.artifacts.module import BusExport, Fragment, Link, merge
from finn.kernels.control import CONTROL, ControlBus, presented
from finn.kernels.declaration import KernelDeclaration, Product, declarations, declared_reads
from finn.kernels.library import LibraryModule
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


NETLIST_SEMANTICS = cast("ValueSemantics[Fragment]", _frozen("netlist", Fragment))
NETLIST = ViewKey("netlist", NETLIST_SEMANTICS)
"""A kernel's or a channel's netlist (``finn.kernels.artifacts.module.Fragment``), labelled
relative to itself; whatever holds it (a network, a kernel with children) merges it
under its node."""

BOUNDARY = ViewKey("boundary", default_semantics(tuple))
"""A channel's AXIS bus on its network's boundary: one, or none when both of its sides are
kernels' ports."""

CYCLES = ViewKey("cycles", default_semantics(int))
"""The clock cycles a frame takes through a kernel or a channel: a kernel binding a module
the most beats any of its ports carries; a kernel with children's its slowest member's."""

BUFFERING = ViewKey("buffering", default_semantics(int))
"""The bits a channel's stages hold between its sides: a stage's its own (an
``input_gen``'s buffer, a FIFO's depth); a kernel with children's, a channel's among
them, the sum over its members."""

RESOURCES = ViewKey("resources", RESOURCES_SEMANTICS)
"""What a kernel or a channel uses of the device (``finn.platform.capabilities.Resources``):
a kernel binding a module its own statement; a kernel with children's, a channel's among
them, the sum over its members."""


class Kernel(Protocol):
    """What a reader asks of a kernel's design point: its four hardware answers, read out
    (its netlist, the cycles a frame takes through it, the bits it holds and what it uses
    of the device). A design point of every ``KernelSpace`` satisfies it, each answer an
    accepted view's value; a reader typed ``Kernel`` reads them and nothing else.

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
    set port; empty for a contract the member presents itself, which is no port's).
    ``under`` labels it from the network (or kernel with children) that holds the
    member."""

    label: str
    contract: StreamContract

    def under(self, member: str) -> PortContract:
        """The contract as seen from where ``member`` is held."""
        return PortContract(f"{member}.{self.label}" if self.label else member, self.contract)


PORT_CONTRACT = default_semantics(PortContract)
"""A ``PortContract``'s value semantics: a frozen value, compared by value."""


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


def _lower(cls: type[KernelSpace]) -> None:
    """Each declaration of ``cls``'s own body, lowered into its members."""
    for name, value in list(vars(cls).items()):
        if not isinstance(value, KernelDeclaration):
            continue
        for member, declaration in value.lower(cls, name).items():
            if member in vars(cls):
                raise DefinitionError.broken(
                    "declaration-member-taken",
                    "KernelSpace",
                    f"{cls.__qualname__}.{name} lowers to {member}, which the class "
                    "declares itself",
                    declaration=f"{cls.__qualname__}.{name}",
                )
            setattr(cls, member, declaration)
            set_name = getattr(declaration, "__set_name__", None)
            if set_name is not None:
                set_name(cls, member)


def _declared(cls: type[KernelSpace], name: str, below: type[KernelSpace]) -> bool:
    """Whether a class from ``cls`` down to, not including, ``below`` declares ``name``."""
    return any(name in vars(base) for base in cls.__mro__[: cls.__mro__.index(below)])


def _members(cls: type[KernelSpace], below: type[KernelSpace]) -> dict[str, object]:
    """The class attributes ``cls`` and its bases down to ``below`` declare, by name."""
    found: dict[str, object] = {}
    for base in reversed(cls.__mro__[: cls.__mro__.index(below)]):
        found.update(vars(base))
    return found


@dataclass(frozen=True)
class _Span:
    """A product of a kernel that spans its declarations: the member ``name``, made by
    ``member`` (a ``view``) from a fresh method calling ``body``, which reads
    the kernel's own members ``own`` and, of each declaration, what ``products`` read of it
    (``KernelDeclaration.reads``)."""

    name: str
    body: Callable[[Any], object]
    own: tuple[str, ...]
    products: tuple[Product, ...]
    member: Callable[[Callable[..., Any]], object]


def _spanning(cls: type[KernelSpace], below: type[KernelSpace], spans: Iterable[_Span]) -> None:
    """The products of ``cls`` that span its declarations, made for it when it is made (a
    ``KernelSpace``'s answers; ``_Span``), and its reads
    of its tensors (``_accesses``, which its extents bind from). Each reads each
    declaration by the names its lowering fixes, its reads declared from ``cls``'s
    declarations below ``below``. A product a class states itself is kept."""
    found = declarations(cls, below)
    cast(Any, cls)._declarations = MappingProxyType(found)

    def paths(products: tuple[Product, ...], own: tuple[str, ...]) -> tuple[str, ...]:
        each = (
            path
            for product in products
            for name, item in found.items()
            for path in item.reads(name, product)
        )
        return tuple(dict.fromkeys((*own, *each)))

    def made(span: _Span) -> Callable[..., object]:
        body = span.body

        def function(self: Any) -> Any:
            return body(self)

        function.__name__ = function.__qualname__ = span.name
        function.__annotations__ = {"return": body.__annotations__["return"]}
        return declared_reads(*paths(span.products, span.own))(function)

    for span in spans:
        if any(
            span.name in vars(base) and not _made(vars(base)[span.name])
            for base in cls.__mro__[: cls.__mro__.index(below)]
        ):
            continue  # stated by the class, or a base of it below ``below``
        function = made(span)
        member = span.member(function)
        cast(Any, function)._spanning = True
        setattr(cls, span.name, member)
        cast(Any, member).__set_name__(cls, span.name)

    @declared_reads(*paths(("access",), ()))
    def _accesses(self: KernelSpace) -> list[Access]:
        return [
            access
            for name, item in found.items()
            if (access := item.access(self, name)) is not None
        ]

    cast(Any, cls)._accesses = _accesses


def _made(member: object) -> bool:
    """Whether ``member`` is a product ``_spanning`` made."""
    return bool(getattr(getattr(member, "function", None), "_spanning", False))


class KernelSpace(Space):
    """A kernel: the design space of its hardware, binding a library module or placing
    kernels, never both (the module docstring). ``id`` names it, a nonempty string.

    Its four answers are views, each written once here (``_netlist``, ``_cycles``,
    ``_buffering``, ``_resources``) and made for each class when it is made, reading the
    class's declarations by the names their lowering fixes (``_spanning``). A class may
    state one itself, calling these. Its members' answers it reaches through ``Members``
    over their exports, which only these views read."""

    id: ClassVar[str] = ""

    # -- what it reaches of its members: their answers, read by its views only ------------

    netlists = Members(NETLIST)
    controls = Members(CONTROL)
    member_cycles = Members(CYCLES)
    member_buffering = Members(BUFFERING)
    member_resources = Members(RESOURCES)

    # Its declarations (``declarations``), by name, in the order they are declared, and its
    # library module among them, by name (None: it binds none).
    _declarations: ClassVar[Mapping[str, KernelDeclaration]] = {}
    _module: ClassVar[tuple[str, LibraryModule] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _lower(cls)
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError.broken(
                "kernel-id-invalid",
                "KernelSpace",
                f"{cls.__qualname__} must declare a nonempty string id",
                declaration=cls.__qualname__,
            )
        found = declarations(cls, KernelSpace)
        modules = [
            (name, declaration)
            for name, declaration in found.items()
            if isinstance(declaration, LibraryModule)
        ]
        if len(modules) > 1:
            raise DefinitionError.broken(
                "library-module-repeated",
                "KernelSpace",
                f"{cls.__qualname__} binds the library modules {[name for name, _ in modules]}: "
                "a kernel binds one at most",
                declaration=cls.__qualname__,
            )
        buses = [name for name, declaration in found.items() if isinstance(declaration, ControlBus)]
        if len(buses) > 1:
            raise DefinitionError.broken(
                "control-bus-repeated",
                "KernelSpace",
                f"{cls.__qualname__} declares the control buses {buses}: a kernel presents "
                "one at most",
                declaration=cls.__qualname__,
            )
        if buses:
            cls.exports = {**cls.exports, CONTROL: getattr(cls, f"{buses[0]}_control")}
        cls._module = modules[0] if modules else None
        _spanning(cls, KernelSpace, _KERNEL_SPANS)
        if unmet_required(cls):
            return  # a base of kernels (the dot-product cores'), which cannot be placed
        ports = [name for name, declaration in found.items() if declaration.module_port]
        if not modules:
            if ports:
                raise DefinitionError.broken(
                    "port-unbound",
                    "KernelSpace",
                    f"{cls.__qualname__} declares the stream ports {ports} and binds no "
                    "library module: a stream port's pins are a module's; a kernel with "
                    "children presents its children's ports (served)",
                    declaration=cls.__qualname__,
                )
            return
        if not ports:
            raise DefinitionError.broken(
                "module-ports-unplaced",
                "KernelSpace",
                f"{cls.__qualname__} declares no port: a kernel with a module states its ports",
                declaration=cls.__qualname__,
            )
        members = _members(cls, KernelSpace)
        children = sorted(name for name, value in members.items() if isinstance(value, KernelSpace))
        if children:
            raise DefinitionError.broken(
                "module-places-kernels",
                "KernelSpace",
                f"{cls.__qualname__} places the kernels {children}: a kernel with a module "
                "places none",
                declaration=cls.__qualname__,
            )
        if not _declared(cls, "resource_use", KernelSpace):
            raise DefinitionError.broken(
                "module-resources-unstated",
                "KernelSpace",
                f"{cls.__qualname__} states no resource_use: a kernel with a module states "
                "what it uses of the device",
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

    @declared_reads()
    def _accesses(self) -> list[Access]:
        """Each declaration's read of a tensor, in declaration order (each class's own:
        ``_spanning``)."""
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

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        """What it uses of the device itself: nothing; a kernel with a module states its
        own (a refusal where it has no model of it, never nothing)."""
        return Resources()

    # -- its answers, written once ---------------------------------------------------------

    def _netlist(self, links: tuple[Link, ...] | None = None) -> Fragment | Rejected:
        """``netlist``: its library module at the empty label, its members' netlists under
        their nodes, and the control buses its members present with the writes their
        configuration takes (``merged``). A kernel with neither is refused; one that states
        the ``links`` between them (a channel's hops) is not."""
        space_type = type(self)
        parts: list[tuple[str, Fragment]] = []
        if space_type._module is not None:
            name, module = space_type._module
            leaf = module.build(self, name)
            if isinstance(leaf, Rejected):
                return leaf
            parts.append(("", Fragment((("", leaf),))))
        parts.extend((str(item.node), item.value) for item in self.netlists)
        if not parts and links is None:
            return reject(
                "kernel-module",
                f"{space_type.__qualname__} declares no module and places no kernel",
            )
        buses = presented((str(item.node), item.value) for item in self.controls)
        if isinstance(buses, Rejected):
            return buses
        return merged(parts, buses, links or ())

    def _cycles(self, default: int | None = None) -> int | Rejected:
        """``cycles``: the most beats any declaration carries a frame, or any member takes,
        one beat a cycle at best; ``default`` where there is none (a channel of wires
        takes none of its own), else refused."""
        space_type = type(self)
        found = [
            beats
            for name, declaration in space_type._declarations.items()
            if (beats := declaration.beats(self, name)) is not None
        ]
        found.extend(item.value for item in self.member_cycles)
        if found:
            return max(found)
        if default is not None:
            return default
        if space_type._module is not None:
            return reject(
                "kernel-cycles", f"no port of {space_type.__qualname__} carries a stream's beats"
            )
        return reject("kernel-cycles", f"no member of {space_type.__qualname__} states cycles")

    def _buffering(self) -> int:
        """``buffering``: its own ``held_bits`` and its members'."""
        return self.held_bits + sum(item.value for item in self.member_buffering)

    def _resources(self) -> Resources:
        """``resources``: its own ``resource_use`` and its members'."""
        return total((self.resource_use, *(item.value for item in self.member_resources)))

    @view(semantics=NETLIST_SEMANTICS)
    @declared_reads("netlists", "controls")
    def netlist(self) -> Fragment | Rejected:
        """Its netlist, labelled relative to itself: its module at the empty label (``under``
        names it by its node where it is held), its members' under theirs."""
        return KernelSpace._netlist(self)

    @view
    @declared_reads("member_cycles")
    def cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through it: the most beats any of its ports carries
        a frame, or any of its members takes."""
        return KernelSpace._cycles(self)

    @view
    @declared_reads("held_bits", "member_buffering")
    def buffering(self) -> int:
        """The bits it and its members hold between a channel's sides."""
        return KernelSpace._buffering(self)

    @view(semantics=RESOURCES_SEMANTICS)
    @declared_reads("resource_use", "member_resources")
    def resources(self) -> Resources:
        """What it and its members use of the device."""
        return KernelSpace._resources(self)

    # A kernel adding exports of its own extends these: ``{**KernelSpace.exports, KEY: ...}``.
    exports = {
        NETLIST: netlist,
        CYCLES: cycles,
        BUFFERING: buffering,
        RESOURCES: resources,
    }


_KERNEL_SPANS = (
    _Span(
        "netlist",
        KernelSpace._netlist,
        ("netlists", "controls"),
        ("module", "pins", "held"),
        view(semantics=NETLIST_SEMANTICS),
    ),
    _Span("cycles", KernelSpace._cycles, ("member_cycles",), ("beats",), view),
    _Span("buffering", KernelSpace._buffering, ("held_bits", "member_buffering"), (), view),
    _Span(
        "resources",
        KernelSpace._resources,
        ("resource_use", "member_resources"),
        (),
        view(semantics=RESOURCES_SEMANTICS),
    ),
)
"""A kernel's answers: each spans its declarations."""


def extent_of(index: Index) -> int:
    """A derived member of a kernel: ``index``'s extent, bound from its placed ports.

    Name it in the class body (``channels = extent_of(c)``) and read that name,
    in a folding factor's ``divisors_of`` domain for instance; used inline inside another
    declaration it is not a member of the class, and linking refuses it.
    """

    def extent(self: KernelSpace) -> int | Rejected:
        extents = self.extents
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return extents[index]

    return derived(extent)


__all__ = [
    "BOUNDARY",
    "BUFFERING",
    "CYCLES",
    "Kernel",
    "KernelSpace",
    "NATIVE_CLOCKING",
    "NETLIST",
    "PORT_CONTRACT",
    "PortContract",
    "RESOURCES",
    "extent_of",
    "merged",
]
