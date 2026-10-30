# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware design Spaces with actual model children and native outward source.

Kernel, the one base, states netlist/cycles/buffering/resources; a channel's FIFO
answers its cycles inapplicable. Real StreamPort children own intrinsic rules,
included ControlModel Groups share their real parent, and plain ModuleSpec values
describe sources.
A kernel binding a module states its binding (ModuleSpec, parameters, sources,
data) and its ports; the base assembles its netlist from them, serially, in the
order the module's pins and holds take (``Kernel.netlist``). Each states its own
frame count. Parents compose resolved child Members in ordinary source. No setter,
read-list or flat callback table is installed on a kernel. Two things are installed
when its class is made, each a function reading by names its closure fixes, so that
its source states its reads: on a kernel with a module, one helper a part of its
module; on a kernel with children, each port it serves (``served``) lowered into the
members a channel and its sizing read of a port, ``<port>_side``,
``<port>_stream_pins``, ``<port>_pace`` and ``<port>_advance``, and, for a selector,
``<child>_case``.

Hardware configuration facts/Choices/rules remain inward model computations.
Outward Views keep selected owning non-refusal and actual unresolved inputs.
A reader types against Kernel and reads its answers; no timing/native/container
facility or computational verification is added. Physical source provision,
module ownership, required declarations and resource statements are checked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

from finn.core.space import (
    DefinitionError,
    Group,
    Inapplicable,
    Members,
    Rejected,
    Space,
    derived,
    reject,
    view,
)
from finn.core.space._nodes import assign, node_record
from finn.core.space.declarations import Declaration, unmet_required
from finn.dataflow.schedule import (
    Access,
    Affine,
    Index,
    Pace,
    Refused,
    Schedule,
    bind_extents,
    schedule_over,
    span,
)
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import Clocking, Pin
from finn.kernels.artifacts.contributions import Contribution, GeneratedData
from finn.kernels.artifacts.module import BusExport, Fragment, Held, Link, merge
from finn.kernels.control import Control, ControlModel, control_models, presented
from finn.kernels.laws import AXIS_CLOCKING, INTAKE_UNSTATED, Advance, Intake
from finn.kernels.library import ModuleSpec, _each, assemble_module, module_unprovided
from finn.kernels.transport import StreamPins
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources


@dataclass(frozen=True)
class PortPins:
    """The stream pins a port gives a channel to wire to (``pins``,
    ``finn.kernels.transport.StreamPins``: its stream on its transport's pins) under its
    ``label``, the port's path relative to the member that states it (``x`` for a module's
    port, ``compute.packed.x`` for MatMul's, ``source.memstream.set`` for a channel's source's
    set port, below the label its channel states for it; empty for pins the member presents
    itself, which are no port's). ``under`` labels it from the network (or kernel with
    children) that holds the member."""

    label: str
    pins: StreamPins

    def under(self, member: str) -> PortPins:
        """The pins as seen from where ``member`` is held."""
        return PortPins(f"{member}.{self.label}" if self.label else member, self.pins)


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


def _declared(cls: type[Kernel], name: str) -> bool:
    """Whether a class from ``cls`` down to, not including, ``Kernel`` declares ``name``."""
    return any(name in vars(base) for base in cls.__mro__[: cls.__mro__.index(Kernel)])


def _members(cls: type[Kernel]) -> dict[str, object]:
    """The class attributes ``cls`` and its bases down to ``Kernel`` declare, by name."""
    found: dict[str, object] = {}
    for base in reversed(cls.__mro__[: cls.__mro__.index(Kernel)]):
        found.update(vars(base))
    return found


# -- the module's parts, read serially ------------------------------------------------------
#
# A module's pins are its other pins' (``OtherPins``), its control bus's, then its ports',
# each in declaration order; its holds its ports', its other pins', then its control bus's.
# The base reads each part's ``pin_values``, ``held_values`` and ``pins_presented`` in that
# order, one after another: a part's refusal is collected with the later parts' (``_each``),
# and a part not yet known stops the reading there. A Space method's reads are found from its
# source (``finn.core.space._reads``), so each read names its part: ``__init_subclass__``
# installs, for a kernel with a module, one helper a part, each reading its part by the name
# its closure holds and then the next helper by name, the last reading none.

_PINS, _HELD, _PRESENTED = "_module_pins", "_module_held", "_module_presented"
_BEATS = "_module_beats"


def _pins_from(part: str, following: str) -> Callable[[Kernel], tuple[Any, ...]]:
    def pins_from(self: Kernel) -> tuple[Any, ...]:
        return (getattr(self, part).pin_values(), *getattr(self, following)())

    return pins_from


def _held_from(part: str, following: str) -> Callable[[Kernel], tuple[Any, ...]]:
    def held_from(self: Kernel) -> tuple[Any, ...]:
        return (getattr(self, part).held_values(), *getattr(self, following)())

    return held_from


def _presented_from(part: str, following: str) -> Callable[[Kernel], tuple[Any, ...]]:
    def presented_from(self: Kernel) -> tuple[Any, ...]:
        return (getattr(self, part).pins_presented(), *getattr(self, following)())

    return presented_from


def _beats_from(part: str, following: str) -> Callable[[Kernel], tuple[Any, ...]]:
    def beats_from(self: Kernel) -> tuple[Any, ...]:
        return (getattr(self, part).frame_beats(), *getattr(self, following)())

    return beats_from


def _module_parts(cls: type[Kernel]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The names of its module's parts in pin order and in hold order (above)."""
    from finn.kernels.port_model import OtherPins, StreamPort, stream_ports  # noqa: PLC0415

    nested = [
        name for names in (control_models(cls), stream_ports(cls)) for name in names if "." in name
    ]
    if nested:
        raise DefinitionError.broken(
            "module-part-nested",
            "Kernel",
            f"{cls.__qualname__} declares {nested} in a Group: "
            "a kernel with a module declares its ports and control bus in its own body",
            declaration=cls.__qualname__,
        )
    members = _members(cls)
    others = tuple(name for name, item in members.items() if isinstance(item, OtherPins))
    controls = tuple(name for name, item in members.items() if isinstance(item, ControlModel))
    ports = tuple(name for name, item in members.items() if isinstance(item, StreamPort))
    return (*others, *controls, *ports), (*ports, *others, *controls)


def _is_stream_port(item: object) -> bool:
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415

    return isinstance(item, StreamPort)


def _supply_ports(cls: type[Kernel]) -> None:
    """Supply each stream port declared in ``cls``'s own body with what is the kernel's,
    where the port does not state it: its ``label``, its name in the kernel, and its
    ``clocking``, the kernel's clock domain (``Kernel.port_clocking``; L8: a port's clock is
    its kernel's). A port declared in a Group states both."""
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415

    for name, value in vars(cls).items():
        record = node_record(value)
        if record is None or not issubclass(record.space_type, StreamPort):
            continue
        if "label" not in record.settings:
            assign(value, "label", name)
        if "clocking" not in record.settings:
            assign(value, "clocking", Kernel.port_clocking)


def _check_advance(cls: type[Kernel]) -> None:
    """Refuse a stream port of ``cls``'s own body stating more than one of what an advance
    rule follows from, its kernel's schedule, the port it governs or stored values
    (``port-advance-ambiguous``), or governing a port its kernel does not have
    (``port-governs-unknown``). CONTRACT.md § 3: a port states what it is, one thing.

    And refuse one presenting a schedule other than its kernel's one ``schedule``
    (``port-schedule-foreign``): its kernel's scheduled ports are projections of one
    schedule, so they present one index point at each iteration (their association) and
    their paces count the same iterations (§ 4). A port presents its kernel's schedule by
    binding it by name (``schedule=schedule``), which a subclass overriding ``schedule``
    then gives it."""
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415

    ports = {name for name, item in _members(cls).items() if isinstance(item, StreamPort)}
    for name, value in vars(cls).items():
        record = node_record(value)
        if record is None or not issubclass(record.space_type, StreamPort):
            continue
        where = f"{cls.__qualname__}.{name}"
        bound = record.bindings
        stated = [key for key in ("schedule", "governs") if bound.get(key) is not None]
        stated += ["stored"] if bound.get("stored") is True else []
        if len(stated) > 1:
            raise DefinitionError.broken(
                "port-advance-ambiguous",
                "Kernel",
                f"{where} states {' and '.join(stated)}: a port's advance rule follows from "
                "one of its kernel's schedule, the port it governs or stored values",
                declaration=where,
            )
        governs = bound.get("governs")
        if isinstance(governs, str) and (governs == name or governs not in ports):
            raise DefinitionError.broken(
                "port-governs-unknown",
                "Kernel",
                f"{where} governs {governs!r}, which is no other stream port of {cls.__qualname__}",
                declaration=where,
            )
        schedule = bound.get("schedule")
        if schedule is not None and not _kernel_schedule(cls, schedule):
            named = schedule.name if isinstance(schedule, Declaration) else None
            raise DefinitionError.broken(
                "port-schedule-foreign",
                "Kernel",
                f"{where} presents "
                + (f"{named!r}" if named else f"a {type(schedule).__name__} of its own")
                + f", not {cls.__qualname__}'s schedule: a scheduled port is a projection of "
                "its kernel's one schedule (schedule=schedule)",
                declaration=where,
            )


def _kernel_schedule(cls: type[Kernel], bound: object) -> bool:
    """Whether ``bound``, what a port of ``cls``'s body binds as its schedule, is its
    kernel's ``schedule``: the declaration of that name of ``cls`` or a kernel class it
    derives (the engine reads a binding by its name, so a subclass's override is read)."""
    return (
        isinstance(bound, Declaration)
        and bound.name == "schedule"
        and isinstance(bound.owner, type)
        and issubclass(cls, bound.owner)
        and issubclass(bound.owner, Kernel)
    )


def _nested_served(group: type[Group], prefix: str) -> list[str]:
    """The served ports declared in ``group`` and its Groups, by dotted path."""
    from finn.kernels.port_model import ServedPort  # noqa: PLC0415

    effective: dict[str, object] = {}
    for base in reversed(group.__mro__):
        effective.update(vars(base))
    found: list[str] = []
    for name, item in effective.items():
        if isinstance(item, ServedPort):
            found.append(prefix + name)
        elif isinstance(item, Group):
            found += _nested_served(type(item), f"{prefix}{name}.")
    return found


# -- the ports a kernel with children serves ------------------------------------------------
#
# A served port (``finn.kernels.port_model.served``) is lowered, when its kernel's class is
# made, into the members a channel and its sizing read of a port: its side and its stream
# pins, each the child's port's labelled below it, and its pace and its advance rule, the
# child's port's. They are derived members of the kernel, not of a child: the engine reads
# a member through a selector one level deep (``compute.x``), never two
# (``compute.x.side``), and a Group reads nothing of its kernel. Each reads its child, its
# port and its case by the names its closure fixes, so that its source states its reads.


def _served_side(child: str, port: str, case: str | None, label: str) -> Callable[[Any], Any]:
    if case is None:

        def side(self: Any) -> Any:
            return getattr(getattr(self, child), port).side.under(label)

        return side

    def side_of_case(self: Any) -> Any:
        return getattr(getattr(self, child), port).side.under(f"{child}.{getattr(self, case)}")

    return side_of_case


def _served_pins(child: str, port: str, case: str | None, label: str) -> Callable[[Any], Any]:
    if case is None:

        def stream_pins(self: Any) -> Any:
            return getattr(getattr(self, child), port).stream_pins.under(label)

        return stream_pins

    def stream_pins_of_case(self: Any) -> Any:
        found = getattr(getattr(self, child), port).stream_pins
        return found.under(f"{child}.{getattr(self, case)}")

    return stream_pins_of_case


def _served_pace(child: str, port: str) -> Callable[[Any], Any]:
    def pace(self: Any) -> Any:
        return getattr(getattr(self, child), port).pace

    return pace


def _served_advance(child: str, port: str) -> Callable[[Any], Any]:
    def advance(self: Any) -> Any:
        return getattr(getattr(self, child), port).advance

    return advance


def _serve_ports(cls: type[Kernel]) -> None:
    """Lower each port ``cls``'s own body serves (above): ``<name>_side``,
    ``<name>_stream_pins``, ``<name>_pace`` and ``<name>_advance``, and, for a selector,
    ``<child>_case``, its case, where the body does not declare it; and record it, by its
    child and the child's port, in ``Kernel.served_names`` (what its intake names it,
    ``Kernel.intake``).
    Refused where the child is not the kernel's (``served-child-unknown``), has no stream
    port so named (``served-port-unknown``), the body declares a member the port is lowered
    into (``served-port-repeated``), or a port is served in a Group, which reads nothing of
    its kernel (``served-port-nested``)."""
    from finn.core.space._nodes import NodeChoice, selected  # noqa: PLC0415
    from finn.kernels.port import served_ports  # noqa: PLC0415
    from finn.kernels.port_model import ServedPort  # noqa: PLC0415
    from finn.kernels.sides import Side  # noqa: PLC0415

    own = dict(vars(cls))
    nested = [
        path
        for name, value in own.items()
        if isinstance(value, Group)
        for path in _nested_served(type(value), f"{name}.")
    ]
    if nested:
        raise DefinitionError.broken(
            "served-port-nested",
            "Kernel",
            f"{cls.__qualname__} serves {nested} in a Group: a kernel serves a child's port "
            "in its own body",
            declaration=cls.__qualname__,
        )
    served = dict(cls.served_names)
    for name, value in own.items():
        if not isinstance(value, ServedPort):
            continue
        where = f"{cls.__qualname__}.{name}"
        child = next((key for key, item in own.items() if item is value.child), None)
        if child is None:
            raise DefinitionError.broken(
                "served-child-unknown",
                "Kernel",
                f"{where} serves a port of no child declared in {cls.__qualname__}'s body",
                declaration=where,
            )
        declared = served_ports(value)
        if not declared or None in declared:
            raise DefinitionError.broken(
                "served-port-unknown",
                "Kernel",
                f"{where} serves {child}.{value.port}, which is no stream port of "
                f"{'each case of ' if isinstance(value.child, NodeChoice) else ''}{child}",
                declaration=where,
            )
        case: str | None = None
        members: dict[str, object] = {}
        if isinstance(value.child, NodeChoice):
            case = f"{child}_case"
            # Lowered once for every port served of the selector; one the body declares is
            # refused below (served-port-repeated).
            if case in own or case not in vars(cls):
                members[case] = selected(value.child)
        label = value.label or child
        side = _served_side(child, value.port, case, label)
        side.__annotations__ = {"return": Side}
        side.__doc__ = f"The side its port {name} gives a channel: {child}'s {value.port}'s."
        pins = _served_pins(child, value.port, case, label)
        pins.__annotations__ = {"return": PortPins}
        pins.__doc__ = f"The stream pins its port {name} gives a channel: {child}'s {value.port}'s."
        pace = _served_pace(child, value.port)
        pace.__annotations__ = {"return": Pace | Inapplicable}
        pace.__doc__ = f"Its port {name}'s pace: {child}'s {value.port}'s (CONTRACT.md § 4)."
        advance = _served_advance(child, value.port)
        advance.__annotations__ = {"return": Advance}
        advance.__doc__ = (
            f"Its port {name}'s advance rule: {child}'s {value.port}'s (CONTRACT.md § 3)."
        )
        members[f"{name}_side"] = derived(side)
        members[f"{name}_stream_pins"] = view(pins)
        members[f"{name}_pace"] = derived(pace)
        members[f"{name}_advance"] = derived(advance)
        served[(child, value.port)] = name
        for key, member in members.items():
            if key in own:
                raise DefinitionError.broken(
                    "served-port-repeated",
                    "Kernel",
                    f"{where}: {cls.__qualname__} declares {key}, which its served port {name} "
                    "is lowered into",
                    declaration=where,
                )
            setattr(cls, key, member)
            getattr(member, "__set_name__")(cls, key)
    cls.served_names = MappingProxyType(served)


def _install(
    cls: type[Kernel],
    prefix: str,
    reading: Callable[[str, str], Callable[[Kernel], tuple[Any, ...]]],
    parts: tuple[str, ...],
) -> None:
    """The helpers ``<prefix>_<i>`` reading ``parts`` from the ``i``-th on (above)."""
    for index, part in enumerate(parts):
        following = f"{prefix}_{index + 1}" if index + 1 < len(parts) else "_module_end"
        setattr(cls, f"{prefix}_{index}", reading(part, following))
    if not parts:
        setattr(cls, f"{prefix}_0", Kernel._module_end)


class Kernel(Space):
    """A kernel: the design space of its hardware, binding a library module or placing
    kernels, never both (the module docstring). ``id`` names it, a nonempty string.

    The one base every kernel derives, and the class a check or a type admitting any
    kernel names. Its answers are Views: its netlist, the cycles a frame takes through it,
    the bits it holds and what it uses of the device. A native kernel states its netlist
    in ordinary serial source over typed model inputs and pure physical helpers, and its
    own cycle calculation; the base refuses an absent calculation rather than assuming
    zero. Buffering/resources read its own statements and its members' answers. A
    composed kernel reaches those answers through ``Members`` over the member each
    states. A kernel that counts no cycles of its own, a channel's FIFO
    (``finn.kernels.channels.ChannelFifo``), whose beats are counted where they are made,
    answers its ``cycles`` inapplicable, and its parent's ``member_cycles`` leaves it out.

    A reader of a kernel's design point types it ``Kernel``: its four answers (``netlist``,
    ``cycles``, ``buffering``, ``resources``) read as their accepted values, and raise where
    the kernel refuses them; ``point.query(type(point).netlist)`` gives the result,
    ``Available`` with the value or the refusal."""

    id: ClassVar[str] = ""

    module_spec: ClassVar[ModuleSpec | None] = None

    # The ports it serves, each by its child and the child's port: what its intake names
    # the child's (``intake``). Recorded when its class is made (``_serve_ports``).
    served_names: ClassVar[Mapping[tuple[str, str], str]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _supply_ports(cls)
        _check_advance(cls)
        _serve_ports(cls)
        super().__init_subclass__(**kwargs)
        buses = list(control_models(cls))
        if len(buses) > 1:
            raise DefinitionError.broken(
                "control-bus-repeated",
                "Kernel",
                f"{cls.__qualname__} declares the control buses {buses}: "
                "a kernel presents one at most",
                declaration=cls.__qualname__,
            )
        if buses and (not _declared(cls, "bus_control") or not _declared(cls, "declares_bus")):
            raise DefinitionError.broken(
                "control-bus-unaddressed",
                "Kernel",
                f"{cls.__qualname__}: an included control states parent bus_control "
                "and declares_bus answers",
                declaration=cls.__qualname__,
            )
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError.broken(
                "kernel-id-invalid",
                "Kernel",
                f"{cls.__qualname__} must declare a nonempty string id",
                declaration=cls.__qualname__,
            )
        members = _members(cls)
        specs = [(name, item) for name, item in members.items() if isinstance(item, ModuleSpec)]
        for name, spec in specs:
            spec.validate(cls.__qualname__, f"{cls.__qualname__}.{name}")
        if any(spec.constant_sources is None for _, spec in specs) and not _declared(
            cls, "module_sources"
        ):
            raise DefinitionError.broken(
                "library-module-sources-unstated",
                "Kernel",
                f"{cls.__qualname__}: a dynamic ModuleSpec states its module_sources View",
                declaration=cls.__qualname__,
            )
        if len(specs) > 1:
            raise DefinitionError.broken(
                "library-module-repeated",
                "Kernel",
                f"{cls.__qualname__} binds the library modules {[name for name, _ in specs]}: "
                "a kernel binds one at most",
                declaration=cls.__qualname__,
            )
        if specs:
            pins, holds = _module_parts(cls)
            _install(cls, _PINS, _pins_from, pins)
            _install(cls, _HELD, _held_from, holds)
            _install(cls, _PRESENTED, _presented_from, pins)
            beating = tuple(name for name, item in members.items() if _is_stream_port(item))
            _install(cls, _BEATS, _beats_from, beating)
        if unmet_required(cls) or not specs:
            return
        from finn.kernels.port_model import stream_ports  # noqa: PLC0415

        ports = list(stream_ports(cls))
        if not _declared(cls, "parameters"):
            raise DefinitionError.broken(
                "module-parameters-unstated",
                "Kernel",
                f"{cls.__qualname__} states no parameters: "
                "a kernel with a module states its module's parameters",
                declaration=cls.__qualname__,
            )
        if not ports:
            raise DefinitionError.broken(
                "module-ports-unplaced",
                "Kernel",
                f"{cls.__qualname__} declares no port: a kernel with a module states its ports",
                declaration=cls.__qualname__,
            )
        children = sorted(name for name, value in members.items() if isinstance(value, Kernel))
        if children:
            raise DefinitionError.broken(
                "module-places-kernels",
                "Kernel",
                f"{cls.__qualname__} places the kernels {children}: "
                "a kernel with a module places none",
                declaration=cls.__qualname__,
            )
        if not _declared(cls, "resource_use"):
            raise DefinitionError.broken(
                "module-resources-unstated",
                "Kernel",
                f"{cls.__qualname__} states no resource_use: "
                "a kernel with a module states what it uses of the device",
                declaration=cls.__qualname__,
            )

    # -- its clock ------------------------------------------------------------------------

    # The clock and reset (and its polarity) its ports and its module run on (L8): a class
    # constant, one of the library's two domains (``finn.kernels.laws``), chosen by its
    # module's family, never by its ports' transports. The AXIS family's by default; a
    # FinnLib native module states ``NATIVE_CLOCKING``.
    clock_domain: ClassVar[Clocking] = AXIS_CLOCKING

    @derived
    def clocking(self) -> Clocking:
        """Its module's clocking: its ``clock_domain``; a kernel with a doubled clock builds
        on it (``finn.kernels.laws.doubled(type(self).clock_domain, doubling)``)."""
        return type(self).clock_domain

    @derived
    def port_clocking(self) -> Clocking:
        """Its ports' clock and reset: its ``clock_domain`` (L8), never a doubled clock's
        choice. The base gives it to every port its body declares (``StreamPort``)."""
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

    # -- its schedule ----------------------------------------------------------------------

    @derived
    def schedule(self) -> Schedule | Inapplicable:
        """Its schedule: the iteration its scheduled ports are projections of
        (``StreamPort(schedule=schedule, ...)``), so that every one presents, at each
        iteration, the positions of one index point (their association, CONTRACT.md § 3)
        and moves its beats at its pace (§ 4). A kernel with scheduled ports states it, over
        its ``extents`` (``bound_schedule``); one with none has none."""
        return Inapplicable()

    # -- its intake ------------------------------------------------------------------------

    @derived
    def intake(self) -> Intake | Rejected:
        """How far it takes input ahead of delivering output (CONTRACT.md § 5): one
        ``finn.kernels.laws.PortIntake`` an input port it places.

        A kernel with a module states its own, read from its RTL (an HLS kernel's
        measured); one that does not answers ``intake-unstated``, by name, never zero, and
        no refusal of its design point: no rule reads it. A kernel with children derives
        its own from its members' (``member_intake``), never states it beside them: a port
        it serves takes the name it serves it under (``served_names``), any other member's
        port its path below the member (``adapter.input_gen.input``). A member that states
        none leaves its parent's unstated too."""
        if type(self).module_spec is not None:
            return reject(INTAKE_UNSTATED, f"{type(self).__qualname__} states no intake")
        served = type(self).served_names
        found = []
        for item in self.member_intake:
            node = str(item.node)
            child = node.split(".", 1)[0]
            for each in item.value.ports:
                name = served.get((child, each.port))
                found.append(each.under(node) if name is None else replace(each, port=name))
        return Intake(tuple(found))

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

    # -- its module's binding: the module (``module_spec``), its parameters, sources, data --

    @view
    def parameters(self) -> dict[str, int | str] | Inapplicable:
        """Its module's parameters, by name: a kernel with a module states them (the class
        definition check requires it); one binding no module has none."""
        return Inapplicable()

    @view
    def module_sources(self) -> tuple[Contribution, ...] | Rejected:
        """Explicit dynamic source product for a ModuleSpec with constant_sources=None.

        Constant-source modules do not read this View. A dynamic module declares
        its own implementation; the class definition check requires it.
        """
        return reject("library-module-sources-unstated", "a dynamic module states its sources")

    @view
    def module_data(self) -> tuple[GeneratedData, ...]:
        """The data files its module reads (a memory's initial contents): none; a kernel
        whose module reads them states them."""
        return ()

    # -- its answers, written once ---------------------------------------------------------

    @view
    def netlist(self) -> Fragment | Rejected:
        """Its netlist, labelled relative to itself: its module at the empty label (``under``
        names it by its node where it is held), or its members' under theirs.

        A kernel with a module is its one leaf, assembled from its binding and its parts
        (``finn.kernels.library.assemble_module``): its clocking, its parameters, its parts'
        pins, its sources (refused where they do not provide the module), its data, its
        parts' holds (L10) and which parts are presented."""
        if type(self).module_spec is not None:
            return self._module()
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

    def _module(self) -> Fragment | Rejected:
        """``netlist`` of a kernel with a module (above), its parts read in order."""
        spec = type(self).module_spec
        assert spec is not None
        owner = type(self).__qualname__
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        pin_parts = _each(self._module_pins_0())
        if isinstance(pin_parts, Rejected):
            return pin_parts
        sources = spec.constant_sources
        if sources is None:
            sources = self.module_sources
        refused = module_unprovided(spec, owner, sources)
        if isinstance(refused, Rejected):
            return refused
        data = self.module_data
        held_parts = _each(self._module_held_0())
        if isinstance(held_parts, Rejected):
            return held_parts
        leaf = assemble_module(
            spec,
            type(self).id,
            owner,
            clocking,
            parameters,
            tuple(pin_parts),
            sources,
            data,
            tuple(held_parts),
            self._module_presented_0(),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    # The heads of its module's readings, which a kernel with a module has installed for
    # it; the end of each.

    def _module_pins_0(self) -> tuple[tuple[Pin, ...] | Rejected, ...]:
        return ()

    def _module_held_0(self) -> tuple[Held | Rejected, ...]:
        return ()

    def _module_presented_0(self) -> tuple[bool, ...]:
        return ()

    def _module_beats_0(self) -> tuple[int | None, ...]:
        return ()

    def _module_end(self) -> tuple[Any, ...]:
        return ()

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

    @view
    def cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through it, at full rate: characterization,
        derived by the rate law (CONTRACT.md L6). A kernel with a module advances one
        iteration a cycle when nothing blocks it: its cycles a frame are its schedule's
        iterations (each of its scheduled ports' pace's span), or, a kernel with no
        schedule (a stage, a memory), advancing one beat a side a cycle, the most beats a
        frame any of its placed ports carries. A kernel with children takes the most any
        member takes, a member counting none (inapplicable) left out. A kernel states
        none of its own; one whose hardware stalls internally would state its exception."""
        return self._cycles()

    def _cycles(self) -> int | Rejected:
        """``cycles`` (above). A class restating ``cycles`` as the base's, to be counted by
        its own kind (a channel's members, ``finn.kernels.channels.Channel``), calls it."""
        if type(self).module_spec is not None:
            return self._rate_cycles()
        found = [item.value for item in self.member_cycles]
        if found:
            return max(found)
        return reject("kernel-cycles", f"no member of {type(self).__qualname__} states cycles")

    def _rate_cycles(self) -> int | Rejected:
        """A kernel with a module's cycles a frame by the rate law (``cycles``): its
        schedule's iterations where it states a schedule, else its placed ports' most beats
        a frame, each port read in its declaration order (``StreamPort.frame_beats``);
        refused where it places no port."""
        if _declared(type(self), "schedule"):
            schedule: Schedule = self.schedule
            return schedule.iterations
        beats = [each for each in self._module_beats_0() if each is not None]
        if not beats:
            return reject(
                "kernel-cycles", f"no port of {type(self).__qualname__} carries a stream's beats"
            )
        return max(beats)

    # -- what it reaches of its members: their answers, read by its views, and their intake,
    # read by its own (a derived value, contract) ------------------------------------------
    # Each ``Members`` names the member it reads, so it follows the members it names.

    netlists = Members(netlist)
    controls = Members(bus_control)
    member_buffering = Members(buffering)
    member_resources = Members(resources)
    member_cycles = Members(cycles)
    member_intake = Members(intake)


# -- a kernel's extents and schedule, by finn.dataflow's rules ------------------------------
#
# The computations are ``finn.dataflow.schedule``'s, over the tensors a kernel's ports read
# (each target with an index reads one, ``StreamPort.access``): ``bind_extents``,
# ``schedule_over`` and ``span``. A kernel with scheduled ports states two derived values by
# them, its ``extents`` (its facts: the indices' extents, bound from the accesses and any it
# gives itself, which its folding's domains read through ``extent_of``) and its
# ``schedule`` (over the extents, with its folding). These give dataflow's refusal as the
# kernel's own: ``kernel-schedule`` for a schedule its order or folding cannot build,
# ``kernel-extents`` for anything else, each with dataflow's message.


def bound_extents(
    accesses: Iterable[Access], given: Mapping[Index, int] | None = None
) -> dict[Index, int] | Rejected:
    """Each index's extent, bound from ``accesses``, its ports' reads in their order, and
    the extents the kernel ``given`` (``bind_extents``)."""
    try:
        return bind_extents(tuple(accesses), given)
    except Refused as error:
        return reject("kernel-extents", str(error))


def bound_schedule(
    extents: Mapping[Index, int],
    order: tuple[Index, ...],
    factors: Mapping[Index, int] | None = None,
) -> Schedule | Rejected:
    """The schedule in beat ``order`` (outer to inner) over the kernel's ``extents``, with
    folding ``factors`` (``schedule_over``)."""
    try:
        return schedule_over(extents, order, factors)
    except Refused as error:
        invalid = any(finding.code == "schedule-invalid" for finding in error.findings)
        return reject("kernel-schedule" if invalid else "kernel-extents", str(error))


def spanned(
    index: tuple[Index | Affine, ...], extents: Mapping[Index, int]
) -> tuple[int, ...] | Rejected:
    """The shape ``index`` spans over the kernel's ``extents`` (``span``): how a kernel
    states an output tensor's shape, from the extents its inputs bind."""
    try:
        return span(index, extents)
    except Refused as error:
        return reject("kernel-extents", str(error))


def extent_of(index: Index) -> int:
    """A derived member of a kernel: ``index``'s extent, read from its ``extents``, which
    the kernel states.

    Name it in the class body (``channels = extent_of(c)``) and read that name,
    in a folding factor's ``divisors_of`` domain for instance; used inline inside another
    declaration it is not a member of the class, and linking refuses it.
    """

    stated = "extents"  # the kernel's member, which the base does not declare

    def extent(self: Kernel) -> int | Rejected:
        extents: dict[Index, int] = getattr(self, stated)
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return extents[index]

    return derived(extent)


__all__ = [
    "Kernel",
    "PortPins",
    "bound_extents",
    "bound_schedule",
    "extent_of",
    "merged",
    "spanned",
]
