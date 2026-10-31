# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware design Spaces with actual model children and native outward source.

Kernel, the one base, states netlist/frame_cycles/buffering/resources; a channel's FIFO
answers its frame_cycles inapplicable. Real StreamPort children own intrinsic rules,
included ControlModel Groups share their real parent, and plain ModuleSpec values
describe sources.
A kernel binding a module states its binding (ModuleSpec, parameters, sources,
data) and its ports; the base assembles its netlist from them, serially, in the
order the module's pins and holds take (``Kernel.netlist``). The base derives its
cycles a frame by the rate law (``Kernel.frame_cycles``) and answers its intake unstated unless it
states its own (``Kernel.intake``). Parents compose resolved child Members in ordinary
source. Importing it registers FINN's value types (``finn.kernels.values.semantics``),
which a kernel's facts hold and the engine admits only registered. No setter, read-list
or flat callback table is installed on a kernel. Three things are installed when its class
is made, each a function reading by names its closure fixes, so that its source states its
reads: on a kernel with a module, one helper a part of its module; on a kernel with
members that may take the doubled clock or present a control bus, what it takes and
presents of theirs, ``takes_doubled_clock`` and ``control_buses``; on a kernel declaring a
control bus (``ControlModel``), what is read of it, ``declares_bus`` and ``bus_control``. A
kernel with children exposes a member's port by a node alias (``x = compute.x``), for
which nothing is installed.

Hardware configuration facts/Choices/rules remain inward model computations.
Outward Views keep selected owning non-refusal and actual unresolved inputs.
A reader types against Kernel and reads its answers; no timing/native/container
facility or computational verification is added. Physical source provision,
module ownership, required declarations and resource statements are checked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, ClassVar, get_args, get_type_hints

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
from finn.core.space._aliases import alias_table
from finn.core.space._nodes import (
    NodeChoice,
    NodeDecision,
    NodeDecl,
    assign,
    node_record,
    selected,
    unwrap,
)
from finn.core.space.declarations import Declaration, Derived, View, unmet_required
from finn.dataflow.schedule import (
    Access,
    Affine,
    Index,
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
from finn.kernels.laws import (
    AXIS_FAMILY,
    INTAKE_UNSTATED,
    RATE_INTERVAL,
    Intake,
    Interval,
    PinFamily,
    PortIntake,
    Standing,
    TransportKind,
    in_cycles,
)
from finn.kernels.library import ModuleSpec, _each, assemble_module, module_unprovided
from finn.kernels.transport import StreamPins
from finn.kernels.utilization import total
from finn.kernels.values import semantics as _value_types  # noqa: F401 (registers value types)
from finn.platform.capabilities import Resources


@dataclass(frozen=True)
class PortPins:
    """The stream pins a port gives a channel to wire to (``pins``,
    ``finn.kernels.transport.StreamPins``: its stream on its transport's pins) under its
    ``label``, the name the member that states it presents the port under (``x``, a module's
    port's and MatMul's alike, which exposes its core's; empty for pins the member presents
    itself, which are no port's), and the ``instance`` they are on, the leaf whose module
    has them, relative to the same member: empty for a module's own port, the selected core
    for MatMul's (``compute.packed``), the node an alias reaches where its holders place it
    (``finn.kernels.port.exposed_pins``), never parsed from a label; pins the member presents
    itself are on none. ``under`` gives both from the network (or kernel with children) that
    holds the member."""

    label: str
    pins: StreamPins
    instance: str = ""

    def under(self, member: str) -> PortPins:
        """The pins as seen from where ``member`` is held."""
        return PortPins(
            f"{member}.{self.label}" if self.label else member,
            self.pins,
            f"{member}.{self.instance}" if self.instance else (member if self.label else ""),
        )


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


def _refuse_restated_cycles(cls: type[Kernel]) -> None:
    """Refuse a kernel with a module that restates ``frame_cycles`` (``kernel-cycles-
    restated``): the base derives them by the rate law from its ports and its
    ``interval``, which it states where its hardware differs. A restatement as the base's
    own (``frame_cycles = view(Kernel._frame_cycles)``) changes nothing and stays, so that
    a channel counts its kind (MemStream's, ``finn.kernels.channels.Channel``), until the
    channel counts its members' cycles otherwise."""
    for base in cls.__mro__[: cls.__mro__.index(Kernel)]:
        item = vars(base).get("frame_cycles")
        if item is None or (isinstance(item, View) and item.function is Kernel._frame_cycles):
            continue
        where = f"{cls.__qualname__}.frame_cycles"
        raise DefinitionError.broken(
            "kernel-cycles-restated",
            "Kernel",
            f"{where}: a kernel with a module states no cycles of its own: the base derives "
            "them by the rate law (L6) from its ports and its interval; state its interval "
            "where its hardware differs",
            declaration=where,
        )


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


def _is_input_port(item: object) -> bool:
    from finn.kernels.port_model import InPort  # noqa: PLC0415

    return isinstance(item, InPort)


def _supply_ports(cls: type[Kernel]) -> None:
    """Supply each stream port declared in ``cls``'s own body with what is the kernel's,
    where the port does not state it: its ``label``, its name in the kernel, its
    ``clocking``, its kernel's pin family's clock and reset (``Kernel.port_clocking``; L8: a
    port's clock is its kernel's), its ``transport``, its kernel's pin family's
    (``Kernel.port_transport``), and its ``interval``, the kernel's (``Kernel.interval``,
    L6), which it carries with its pace. A port declared in a Group states its label,
    clocking and transport; only a kernel with no module declares one
    (``module-part-nested``), and such a kernel has no interval, so the port keeps L6's.

    And a selector of its body (``InPort.governs``, a literal naming another port of its
    body) with the beats a frame of the port it governs (``InPort.governed_beats``, from
    which its ``group`` follows), by a private derived member, ``_<selector>_governed``."""
    from finn.kernels.port_model import InPort, StreamPort  # noqa: PLC0415

    body = dict(vars(cls))
    ports = {name for name, value in body.items() if isinstance(value, StreamPort)}
    for name, value in body.items():
        record = node_record(value)
        if record is None or not issubclass(record.space_type, StreamPort):
            continue
        governs = record.bindings.get("governs")
        if (
            issubclass(record.space_type, InPort)
            and isinstance(governs, str)
            and governs in ports - {name}
            and "governed_beats" not in record.settings
        ):
            beats = derived(_governed_beats(governs))
            setattr(cls, f"_{name}_governed", beats)
            getattr(beats, "__set_name__")(cls, f"_{name}_governed")
            assign(value, "governed_beats", beats)
        if "label" not in record.settings:
            assign(value, "label", name)
        if "clocking" not in record.settings:
            assign(value, "clocking", Kernel.port_clocking)
        if "transport" not in record.settings:
            assign(value, "transport", Kernel.port_transport)
        if "interval" not in record.settings:
            assign(value, "interval", Kernel.interval)


def _governed_beats(governed: str) -> Callable[[Kernel], int | None]:
    """A selector's ``governed_beats`` (``_supply_ports``): the beats a frame of port
    ``governed``, None where it is idle."""

    def governed_beats(self: Kernel) -> int | None:
        if not getattr(self, governed).placed:
            return None
        beats: int = getattr(self, governed).stream.sequence.form.beats
        return beats

    governed_beats.__annotations__ = {"return": int | None}
    return governed_beats


def _exposed_as(exposed: Mapping[tuple[str, str], str], node: str, port: str) -> str:
    """The name a kernel with children gives its member ``node``'s port ``port`` in its
    intake: the name it exposes it under, else its path below the member."""
    return exposed.get((node.split(".", 1)[0], port), f"{node}.{port}")


# -- what a kernel with a module states of its intake ---------------------------------------
#
# A kernel with a module states its intake (a derived ``intake`` of its own body); the base
# reads it, as ``_stated_intake``, and installs its ``intake``: the stated one, refused where it
# names a port that is no placed input of the kernel (``intake-port-unknown``). Its placed
# inputs are read as its module's parts are, one helper each (``_placed_inputs_<i>``).

_INPUTS = "_placed_inputs"


def _inputs_from(part: str, following: str) -> Callable[[Kernel], tuple[Any, ...]]:
    def inputs_from(self: Kernel) -> tuple[Any, ...]:
        found = (getattr(self, part).label,) if getattr(self, part).placed else ()
        return (*found, *getattr(self, following)())

    return inputs_from


def _checked_intake(owner: str) -> Callable[[Kernel], Intake | Rejected]:
    def intake(self: Kernel) -> Intake | Rejected:
        stated: Intake = getattr(self, "_stated_intake")
        placed = self._placed_inputs_0()
        unknown = [name for name in stated.names if name not in placed]
        if unknown:
            return reject(
                "intake-port-unknown",
                f"{owner} states the intake of {unknown}, no input it places: an intake "
                "names its kernel's placed inputs",
            )
        return stated

    return intake


def _read_intake(cls: type[Kernel]) -> None:
    """Install, for a kernel with a module whose own body states ``intake``, the base's
    reading of it (above)."""
    item = vars(cls).get("intake")
    if getattr(cls, "module_spec", None) is None or not isinstance(item, Derived):
        return
    stated = derived(item.function)
    setattr(cls, "_stated_intake", stated)
    getattr(stated, "__set_name__")(cls, "_stated_intake")
    checked = _checked_intake(cls.__qualname__)
    checked.__doc__ = item.function.__doc__
    _set_aggregated(cls, "intake", checked, Intake | Rejected)


# -- its stream ports' declarations ---------------------------------------------------------
#
# What a constraint cannot see is refused when a kernel's class is made: a constraint reading
# an inapplicable value is inapplicable, not refused. Each stream port its body declares, in
# the body or in a Group of it, is an input or an output (its kind, InPort or OutPort, whose
# relation's parameters are its own), binds exactly one presentation and states one advance
# rule; a port of its body presents its kernel's one
# schedule, which is stated and never inapplicable, and governs a port its kernel has. A
# placed port's presentation at a design point is the port's own constraint
# (``StreamPort.presented``).

_PROJECTION = ("index", "lanes", "reduces", "holds", "closes", "reshaped")


def _declared_ports(cls: type[Kernel]) -> list[tuple[str, Any, bool]]:
    """Each stream port ``cls``'s own body declares, by its path (``group.port`` in a
    Group), with its declaration's record and whether it is in the body itself."""
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415

    def below(namespace: type[Any], prefix: str, own: bool) -> list[tuple[str, Any, bool]]:
        effective: dict[str, object] = {}
        for base in reversed(namespace.__mro__ if not own else (namespace,)):
            effective.update(vars(base))
        found: list[tuple[str, Any, bool]] = []
        for name, value in effective.items():
            record = node_record(value)
            if record is not None and issubclass(record.space_type, StreamPort):
                found.append((prefix + name, record, own))
            elif isinstance(value, Group):
                found += below(type(value), f"{prefix}{name}.", False)
        return found

    return below(cls, "", True)


def _check_ports(cls: type[Kernel]) -> None:
    """Refuse, when ``cls`` is made, a stream port its own body declares (above) that:

    - is a ``StreamPort`` of neither kind, ``InPort`` or ``OutPort`` (``port-direction``): a
      stream is an input or an output, never ``INOUT``, and its bus role follows
      (``finn.kernels.port.stream_endpoint``). What only an input states (``governs``,
      ``dtype``) or only an output (``sourced``) is its kind's, so no other port states it;
    - binds no presentation or both, its kernel's ``schedule`` and a given ``sequence``, or a
      given sequence with a projection's parameters (``index``, ``lanes``, ``reduces``,
      ``holds``, ``closes``, ``reshaped``) (``port-presentation``);
    - states more than one of what an advance rule follows from, its kernel's schedule, the
      port it governs or its memory, or states ``governs`` or ``sourced`` bound to other
      than a literal (``port-advance-ambiguous``; CONTRACT.md § 3: a port states what it
      is, one thing);
    - in its body, governs a port its kernel does not have (``port-governs-unknown``), or
      presents a schedule other than its kernel's one ``schedule``
      (``port-schedule-foreign``): its kernel's scheduled ports are projections of one
      schedule, so they present one index point at each iteration (their association) and
      their paces count the same iterations (§ 4). A port presents its kernel's schedule
      by binding it by name (``schedule=schedule``), which a subclass overriding
      ``schedule`` then gives it.

    And refuse ``cls`` where a port it places presents its schedule and it states none
    (``port-schedule-unstated``) or states one whose declared answer admits
    ``Inapplicable`` (``port-schedule-inapplicable``): a scheduled port's stream would be
    inapplicable, which no constraint refuses."""
    from finn.kernels.port_model import InPort, OutPort, StreamPort  # noqa: PLC0415

    ports = {name for name, item in _members(cls).items() if isinstance(item, StreamPort)}
    for name, record, own in _declared_ports(cls):
        where = f"{cls.__qualname__}.{name}"

        def refused(code: str, message: str, where: str = where) -> DefinitionError:
            return DefinitionError.broken(code, "Kernel", f"{where} {message}", declaration=where)

        bound = record.bindings
        if not issubclass(record.space_type, (InPort, OutPort)):
            raise refused(
                "port-direction",
                f"is a {record.space_type.__name__}, neither an input nor an output: a stream "
                "port is declared as one (InPort, OutPort)",
            )
        presentations = [key for key in ("schedule", "sequence") if bound.get(key) is not None]
        if len(presentations) != 1:
            raise refused(
                "port-presentation",
                ("binds " + " and ".join(presentations) if presentations else "binds none")
                + ": a port presents exactly one of its kernel's schedule's projection "
                "(schedule) or a given sequence (sequence)",
            )
        projecting = [key for key in _PROJECTION if key in bound]
        if presentations == ["sequence"] and projecting:
            raise refused(
                "port-presentation",
                f"binds {', '.join(projecting)} with a given sequence: a schedule's "
                "projection's parameters go with schedule",
            )
        stated = [key for key in ("schedule", "governs") if bound.get(key) is not None]
        stated += ["sourced"] if bound.get("sourced") is True else []
        if len(stated) > 1:
            raise refused(
                "port-advance-ambiguous",
                f"states {' and '.join(stated)}: a port's advance rule follows from one of "
                "its kernel's schedule, the port it governs or its memory (sourced)",
            )
        governs, sourced = bound.get("governs"), bound.get("sourced", False)
        for key, value, kind in (("governs", governs, str), ("sourced", sourced, bool)):
            if value is not None and not isinstance(value, kind):
                raise refused(
                    "port-advance-ambiguous",
                    f"binds {key} to {value!r}: its advance rule is its declaration's, a literal",
                )
        if not own:
            continue
        if isinstance(governs, str) and (governs == name or governs not in ports):
            raise refused(
                "port-governs-unknown",
                f"governs {governs!r}, which is no other stream port of {cls.__qualname__}",
            )
        schedule = bound.get("schedule")
        if schedule is not None and not _kernel_schedule(cls, schedule):
            named = schedule.name if isinstance(schedule, Declaration) else None
            raise refused(
                "port-schedule-foreign",
                "presents "
                + (f"{named!r}" if named else f"a {type(schedule).__name__} of its own")
                + f", not {cls.__qualname__}'s schedule: a scheduled port is a projection of "
                "its kernel's one schedule (schedule=schedule)",
            )
    _check_schedule(cls)


def _check_schedule(cls: type[Kernel]) -> None:
    """``port-schedule-unstated`` and ``port-schedule-inapplicable`` (``_check_ports``)."""
    from finn.kernels.port_model import StreamPort  # noqa: PLC0415

    members = _members(cls)
    records = {
        name: node_record(item) for name, item in members.items() if isinstance(item, StreamPort)
    }
    scheduled = [
        name
        for name, record in records.items()
        if record is not None and _kernel_schedule(cls, record.bindings.get("schedule"))
    ]
    if not scheduled:
        return
    if not _declared(cls, "schedule"):
        raise DefinitionError.broken(
            "port-schedule-unstated",
            "Kernel",
            f"{cls.__qualname__} presents its schedule on {scheduled} and states none: a "
            "kernel with scheduled ports states its schedule",
            declaration=cls.__qualname__,
        )
    item = members["schedule"]
    if not isinstance(item, Derived):
        return
    owner = next(base for base in cls.__mro__ if vars(base).get("schedule") is item)
    try:
        answer = get_type_hints(item.function, localns=dict(vars(owner))).get("return")
    except (NameError, TypeError, ValueError):
        return  # the engine refuses it, with its own text (``annotation-unresolved``)
    if answer is Inapplicable or Inapplicable in get_args(answer):
        raise DefinitionError.broken(
            "port-schedule-inapplicable",
            "Kernel",
            f"{cls.__qualname__} presents its schedule on {scheduled} and declares it "
            f"{answer}: a scheduled port's schedule is never inapplicable",
            declaration=cls.__qualname__,
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


# -- the ports a kernel with children exposes ----------------------------------------------
#
# A kernel with children exposes a member's stream port as its own by a node alias in its
# body (``x = compute.x``, ``finn.core.space``): the port stays its member's, and nothing is
# lowered for it. What its class records is the name each member's port is exposed under,
# by the member and the name the member presents it under, as the alias is written
# (``("compute", "x")``, ``("mm", "x")`` through MatMul's own: what its intake names the
# member's port, ``Kernel.intake``); a port is exposed under one name.


def _expose_ports(cls: type[Kernel]) -> None:
    """Record the name each member's port ``cls`` exposes is exposed under, in
    ``Kernel._exposed_names`` (above); and refuse a port it exposes:

    - whose nodes, one each selection, are not all inputs or all outputs, by their kind
      (``port-direction``): a port is an input or an output in each of its kernel's points.
      Their transports may differ, each case's carrier its own;
    - that another of its aliases exposes already (``port-exposed-twice``): one port, one
      name, so no two channels connect it under two."""
    from finn.kernels.port_model import InPort, OutPort, StreamPort  # noqa: PLC0415

    exposed: dict[tuple[str, str], str] = {}
    seen: dict[frozenset[object], str] = {}
    for name, entry in alias_table(cls).items():
        written = entry.declaration.written
        exposed.setdefault((written[0], ".".join(written[1:])), name)
        if not entry.ends or not all(issubclass(end.space_type, StreamPort) for end in entry.ends):
            continue
        where = f"{cls.__qualname__}.{name}"
        ends = frozenset(end.path for end in entry.ends)
        if ends in seen:
            raise DefinitionError.broken(
                "port-exposed-twice",
                "Kernel",
                f"{where} exposes the port {seen[ends]} exposes: a port is exposed under one name",
                declaration=where,
            )
        seen[ends] = name
        kinds = {end.path: end.space_type for end in entry.ends}
        if not (
            all(issubclass(kind, InPort) for kind in kinds.values())
            or all(issubclass(kind, OutPort) for kind in kinds.values())
        ):
            raise DefinitionError.broken(
                "port-direction",
                "Kernel",
                f"{where} exposes nodes not all inputs or all outputs ("
                + ", ".join(f"{path}: {kind.__name__}" for path, kind in kinds.items())
                + "): an exposed port is an input or an output in every selection",
                declaration=where,
            )
    cls._exposed_names = MappingProxyType(exposed)


# -- its control bus ------------------------------------------------------------------------
#
# A kernel declares its control bus once, a ``ControlModel`` in its own body, and the base
# derives what is read of it, ``declares_bus`` and ``bus_control``, when its class is made. A
# bus its model may present (its ``presented`` bound) is presented at the port the kernel's
# ``control`` names, with the writes its ``register_map`` states; a bus never presented is
# held, ``Control(None)``. As for the module's parts, it reads its model by the name its
# closure fixes, and its port and writes only once it is presented.


def _declares_bus() -> Callable[[Any], Any]:
    def declares_bus(self: Any) -> bool:
        return True

    return declares_bus


def _bus_held() -> Callable[[Any], Any]:
    def bus_control(self: Any) -> Any:
        return Control(None)

    return bus_control


def _bus_presented(model: str) -> Callable[[Any], Any]:
    def bus_control(self: Any) -> Any:
        if not getattr(self, model).presented:
            return Control(None)
        port = self.control
        if port is None:
            return reject(
                "control-bus-unaddressed", "a bus it presents names the port it is presented at"
            )
        return Control(getattr(self, model).bus, port, self.register_map)

    return bus_control


def _own_buses(model: str) -> Callable[[Any], tuple[str, ...]]:
    def own_buses(self: Any) -> tuple[str, ...]:
        return ("",) if getattr(self, model).presented else ()

    return own_buses


def _derive_control(cls: type[Kernel]) -> None:
    """Derive ``cls``'s ``declares_bus`` and ``bus_control`` from the control bus its own body
    declares (above); one a kernel it derives from declares, it inherits with that kernel's
    answers. Refused where it declares more than one (``control-bus-repeated``), one in a
    Group (``control-bus-nested``; a kernel with a module, ``module-part-nested`` with its
    module's other parts) or in a class it derives from that is not a kernel
    (``control-bus-nested``), states either answer itself
    (``control-bus-restated``), or may present its bus without stating ``control`` and
    ``register_map`` (``control-bus-unaddressed``)."""
    buses = list(control_models(cls))
    if len(buses) > 1:
        raise DefinitionError.broken(
            "control-bus-repeated",
            "Kernel",
            f"{cls.__qualname__} declares the control buses {buses}: a kernel presents one at most",
            declaration=cls.__qualname__,
        )
    if not buses:
        return
    (name,) = buses
    if "." in name:
        if any(isinstance(item, ModuleSpec) for item in _members(cls).values()):
            return  # a module's part, refused with its others (``module-part-nested``)
        raise DefinitionError.broken(
            "control-bus-nested",
            "Kernel",
            f"{cls.__qualname__} declares its control bus {name} in a Group: a kernel declares "
            "it in its own body, where the base derives what is read of it",
            declaration=cls.__qualname__,
        )
    stated = [key for key in ("declares_bus", "bus_control") if key in vars(cls)]
    if stated:
        raise DefinitionError.broken(
            "control-bus-restated",
            "Kernel",
            f"{cls.__qualname__} states {' and '.join(stated)}: the base derives them from "
            f"its control bus {name}",
            declaration=cls.__qualname__,
        )
    owner = next(base for base in cls.__mro__ if name in vars(base))
    if owner is not cls:
        if issubclass(owner, Kernel):
            return  # inherited, with what its class derived
        raise DefinitionError.broken(
            "control-bus-nested",
            "Kernel",
            f"{cls.__qualname__} declares its control bus {name} in {owner.__qualname__}, "
            "not a kernel: a kernel declares it in its own body, where the base derives what "
            "is read of it",
            declaration=cls.__qualname__,
        )
    record = node_record(vars(cls)[name])
    assert record is not None
    may_present = record.bindings.get("presented", False) is not False
    members = _members(cls)
    if may_present and not ("control" in members and "register_map" in members):
        raise DefinitionError.broken(
            "control-bus-unaddressed",
            "Kernel",
            f"{cls.__qualname__} may present its control bus {name}: it states the port it "
            "is presented at (control) and the writes its configuration takes (register_map)",
            declaration=cls.__qualname__,
        )
    if may_present:
        cls._presents_buses = True
        setattr(cls, "_own_buses", _own_buses(name))
    declares = _declares_bus()
    declares.__annotations__ = {"return": bool}
    declares.__doc__ = f"Whether it declares a control bus: it does, {name}."
    control = _bus_presented(name) if may_present else _bus_held()
    control.__annotations__ = {"return": Control | Rejected}
    control.__doc__ = (
        f"Its control bus, {name}: presented at the port ``control`` names, with the writes "
        "``register_map`` states, where its model presents it; else held, ``Control(None)``."
        if may_present
        else f"Its control bus, {name}, which it never presents: held, ``Control(None)``."
    )
    for key, member in (("declares_bus", derived(declares)), ("bus_control", view(control))):
        setattr(cls, key, member)
        getattr(member, "__set_name__")(cls, key)


# -- what its members require of its holder: the doubled clock and the control buses -------
#
# A kernel with members takes the doubled clock where it or any present member takes it, and
# presents the control buses it and its present members present, each under its member's
# path: the base derives both once, when its class is made, from the members whose class may
# take a doubled clock (one deriving its own ``clocking``, or a kernel with such members) or
# may present a bus. It reads those members only, one after another, each where it is
# present (its guard read by the name its closure fixes) and, a selector, its selected case
# only: a dormant case is never read. As for the module's parts, each member is read by its
# own helper, ``<prefix>_<i>``, so that each read names its member.

_DOUBLED, _BUSES = "_doubled", "_buses"
_AGGREGATED = "_finn_aggregated"


def _member_types(item: object) -> tuple[type[Any], ...]:
    """The kernel classes a member may place: a child's, or each case's of a selector."""
    record = unwrap(item)
    if isinstance(record, NodeDecision):
        return tuple(each.space_type for each in record.candidates.values() if each is not None)
    if isinstance(record, NodeDecl):
        return (record.space_type,)
    return ()


def _aggregated(cls: type[Kernel], name: str) -> bool:
    """Whether ``cls`` states ``name`` itself, below ``Kernel``, but for the base's own
    aggregation of it."""
    return any(
        name in vars(base) and not getattr(vars(base)[name], _AGGREGATED, False)
        for base in cls.__mro__[: cls.__mro__.index(Kernel)]
    )


def _may_double(space_type: type[Any]) -> bool:
    """Whether a kernel of ``space_type`` may take the doubled clock: it derives its own
    ``clocking``, or states (or the base derives) its ``takes_doubled_clock``."""
    return (
        isinstance(space_type, type)
        and issubclass(space_type, Kernel)
        and (_declared(space_type, "clocking") or _declared(space_type, "takes_doubled_clock"))
    )


def may_present_bus(space_type: type[Any]) -> bool:
    """Whether a kernel of ``space_type`` may present a control bus: its own (a model whose
    ``presented`` is bound, ``finn.kernels.control``) or a member's. What a shell counts
    (``finn.kernels.composer``)."""
    if not (isinstance(space_type, type) and issubclass(space_type, Kernel)):
        return False
    return space_type._presents_buses


def _guarded(cls: type[Kernel], item: object) -> str | None:
    """The member of ``cls`` that guards member ``item``'s presence, by its name; None where
    it has none."""
    record = unwrap(item)
    when = getattr(record, "when", None)
    if when is None:
        return None
    name = getattr(when, "name", None)
    if not isinstance(name, str) or _members(cls).get(name) is not when:
        raise DefinitionError.broken(
            "member-guard-unread",
            "Kernel",
            f"{cls.__qualname__}: a member whose presence is not a member of its own is not "
            "read for what it requires of its holder",
            declaration=cls.__qualname__,
        )
    return name


def _doubled_from(guard: str | None, member: str, following: str) -> Callable[[Kernel], bool]:
    def doubled_from(self: Kernel) -> bool:
        if guard is None or getattr(self, guard):
            if getattr(self, member).takes_doubled_clock:
                return True
        return bool(getattr(self, following)())

    return doubled_from


def _buses_from(
    guard: str | None, case: str | None, member: str, following: str
) -> Callable[[Kernel], tuple[str, ...]]:
    def buses_from(self: Kernel) -> tuple[str, ...]:
        found: tuple[str, ...] = ()
        if guard is None or getattr(self, guard):
            where = member if case is None else f"{member}.{getattr(self, case)}"
            buses: tuple[str, ...] = getattr(self, member).control_buses
            found = tuple(f"{where}.{path}" if path else where for path in buses)
        following_: tuple[str, ...] = getattr(self, following)()
        return (*found, *following_)

    return buses_from


def _case_member(cls: type[Kernel], member: str, item: object) -> str | None:
    """The name of ``cls``'s private member answering a selector member's case
    (``_<member>_case``), installed where it is not yet; None for a child."""
    if not isinstance(item, NodeChoice):
        return None
    case = f"_{member}_case"
    if case not in vars(cls):
        found = selected(item)
        setattr(cls, case, found)
        getattr(found, "__set_name__")(cls, case)
    return case


def _derive_doubled(cls: type[Kernel]) -> None:
    """Derive ``cls``'s ``takes_doubled_clock`` and ``control_buses`` from the members that
    may require them (above), where it has any and does not state its own."""
    members = [
        (name, item)
        for name, item in _members(cls).items()
        if isinstance(item, (Kernel, NodeChoice)) and _member_types(item)
    ]
    doubling = [
        (name, item) for name, item in members if any(map(_may_double, _member_types(item)))
    ]
    if doubling and not _aggregated(cls, "takes_doubled_clock"):
        _install_reading(
            cls,
            _DOUBLED,
            [(partial(_doubled_from, _guarded(cls, item)), name) for name, item in doubling],
        )

        def takes_doubled_clock(self: Kernel) -> bool:
            if self.clocking.doubling:
                return True
            return bool(self._doubled_0())

        takes_doubled_clock.__doc__ = vars(Kernel)["takes_doubled_clock"].function.__doc__
        _set_aggregated(cls, "takes_doubled_clock", takes_doubled_clock, bool)
    presenting = [
        (name, item)
        for name, item in members
        if any(may_present_bus(each) for each in _member_types(item))
    ]
    cls._presents_buses = cls._presents_buses or bool(presenting)
    if presenting:
        _install_reading(
            cls,
            _BUSES,
            [
                (partial(_buses_from, _guarded(cls, item), _case_member(cls, name, item)), name)
                for name, item in presenting
            ],
        )

        def control_buses(self: Kernel) -> tuple[str, ...]:
            found: tuple[str, ...] = (*self._own_buses(), *self._buses_0())
            return found

        control_buses.__doc__ = vars(Kernel)["control_buses"].function.__doc__
        _set_aggregated(cls, "control_buses", control_buses, tuple[str, ...])


def _install_reading(
    cls: type[Kernel],
    prefix: str,
    readers: list[tuple[Callable[[str, str], Callable[[Kernel], Any]], str]],
) -> None:
    """The helpers ``<prefix>_<i>``, each reading its member, by the reading made for it
    from its member's name and the next helper's, and then the next helper; the last
    reads none (``_module_end``)."""
    for index, (reading, member) in enumerate(readers):
        following = f"{prefix}_{index + 1}" if index + 1 < len(readers) else "_module_end"
        setattr(cls, f"{prefix}_{index}", reading(member, following))


def _set_aggregated(
    cls: type[Kernel], name: str, function: Callable[..., Any], answer: Any
) -> None:
    function.__annotations__ = {"return": answer}
    setattr(function, _AGGREGATED, True)
    member = derived(function)
    setattr(cls, name, member)
    getattr(member, "__set_name__")(cls, name)


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
    in ordinary serial source over typed model inputs and pure physical helpers; the base
    derives its cycles by the rate law, and a kernel restates them only to be counted by
    its kind (MemStream; a composite, StreamAdapter). Its contract's intake (``intake``)
    it states from its RTL, else the base answers it unstated, by name. Buffering and
    resources read its own statements and its members' answers. A composed kernel
    reaches those answers through ``Members`` over the member each states. A kernel that
    counts no cycles of its own, a channel's FIFO (``finn.kernels.channels.ChannelFifo``),
    whose beats are counted where they are made, answers its ``frame_cycles`` inapplicable,
    and its parent's ``member_frame_cycles`` leaves it out.

    A reader of a kernel's design point types it ``Kernel``: its four answers (``netlist``,
    ``frame_cycles``, ``buffering``, ``resources``) read as their accepted values, and raise where
    the kernel refuses them; ``point.query(type(point).netlist)`` gives the result,
    ``Available`` with the value or the refusal."""

    id: ClassVar[str] = ""

    module_spec: ClassVar[ModuleSpec | None] = None

    # The ports it exposes, each by its child and the child's port: what its intake names
    # the child's (``intake``). Recorded when its class is made (``_expose_ports``).
    _exposed_names: ClassVar[Mapping[tuple[str, str], str]] = MappingProxyType({})

    # Where its netlist places a member, by the member's name, where not at its name (a
    # channel's source, ``finn.kernels.channels``): the instance a port it exposes of that
    # member is on (``finn.kernels.port.exposed_pins``).
    placed_at: ClassVar[Mapping[str, str]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _supply_ports(cls)
        _read_intake(cls)
        _check_ports(cls)
        _expose_ports(cls)
        _derive_control(cls)
        _derive_doubled(cls)
        super().__init_subclass__(**kwargs)
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError.broken(
                "kernel-id-invalid",
                "Kernel",
                f"{cls.__qualname__} must declare a nonempty string id",
                declaration=cls.__qualname__,
            )
        members = _members(cls)
        readers = [
            name
            for name, item in members.items()
            if isinstance(item, Derived) and getattr(item.function, _READS_EXTENTS, False)
        ]
        if readers and not _declared(cls, "extents"):
            raise DefinitionError.broken(
                "kernel-extents-unstated",
                "Kernel",
                f"{cls.__qualname__} reads its extents ({', '.join(readers)}, extent_of) "
                "and states none: a kernel reading an extent states its extents",
                declaration=cls.__qualname__,
            )
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
            _refuse_restated_cycles(cls)
            pins, holds = _module_parts(cls)
            _install(cls, _PINS, _pins_from, pins)
            _install(cls, _HELD, _held_from, holds)
            _install(cls, _PRESENTED, _presented_from, pins)
            beating = tuple(name for name, item in members.items() if _is_stream_port(item))
            _install(cls, _BEATS, _beats_from, beating)
            inputs = tuple(name for name, item in members.items() if _is_input_port(item))
            _install(cls, _INPUTS, _inputs_from, inputs)
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

    # -- its clock and its pins' family ---------------------------------------------------

    # Its module's pin family (``finn.kernels.laws.PinFamily``): a class constant, one of the
    # library's two, chosen by its module's family: the clock and reset (and its polarity)
    # its ports and its module run on (L8), and the transport its ports take where a port
    # states none. The AXIS family by default; a FinnLib native module states
    # ``NATIVE_FAMILY``.
    pin_family: ClassVar[PinFamily] = AXIS_FAMILY

    @derived
    def clocking(self) -> Clocking:
        """Its module's clocking: its pin family's; a kernel with a doubled clock builds on
        it (``finn.kernels.laws.doubled(type(self).pin_family.clocking, doubling)``)."""
        return type(self).pin_family.clocking

    @derived
    def port_clocking(self) -> Clocking:
        """Its ports' clock and reset: its pin family's (L8), never a doubled clock's
        choice. The base gives it to every port its body declares (``StreamPort``)."""
        return type(self).pin_family.clocking

    @derived
    def port_transport(self) -> TransportKind:
        """Its ports' transport where a port states none: its pin family's. The base gives
        it to every port its body declares that states none (``StreamPort.transport``)."""
        return type(self).pin_family.transport

    @derived
    def takes_doubled_clock(self) -> bool:
        """Whether it takes the doubled clock: its own ``clocking``'s doubling, or any
        present member's (a MatMul's selected core, a channel's source): a parent's
        ``clocking`` is its own, never its members'. The base derives a kernel's with
        members that may take it when its class is made, reading those members only, each
        where it is present (``_derive_doubled``); a case not selected is never read. What
        a shell's clock rule reads (``finn.kernels.budgets``)."""
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

    @derived
    def interval(self) -> Interval | Inapplicable:
        """Its initiation interval (L6, ``finn.kernels.laws.Interval``): the cycles an
        iteration takes when nothing blocks it (a stage's: a beat a side). A kernel with a
        module states it: L6's 1 here (``RATE_INTERVAL``), its own, with its standing and
        source, where its hardware differs or its 1 rests on what its RTL does not show
        (Pool's, assumed). A kernel with children has none: it has no schedule; each of its
        members' ports carries its own kernel's (``StreamPort.interval``), and its cycles a
        frame are its members' (``frame_cycles``)."""
        if type(self).module_spec is None:
            return Inapplicable()
        return RATE_INTERVAL

    # -- its intake ------------------------------------------------------------------------

    @derived
    def intake(self) -> Intake | Rejected:
        """How far it takes input ahead of delivering output (CONTRACT.md § 5): one
        ``finn.kernels.laws.PortIntake`` an input port it places.

        A kernel with a module states its own, read from its RTL (an HLS kernel's
        measured), with its inputs' coupling (``Intake.joined``, ``PortIntake.ahead``); one
        that does not answers ``intake-unstated``, by name, never zero, and no refusal of
        its design point: no rule reads it. The base reads what it states, and refuses an
        intake naming a port that is no placed input of it (``intake-port-unknown``, its
        own answer's refusal, not its design point's). A kernel with children derives its
        own from its members' (``member_intake``), never states it beside them, each
        member's renamed, its ports, its groups and its partners alike: a member's port it
        exposes takes the name it exposes it under (``x = compute.x``: ``x``; one port, one
        name, however many aliases name it), any other member's port its path below the
        member (``adapter.input_gen.input_gen.input``). A member that states none leaves
        its parent's unstated too."""
        if type(self).module_spec is not None:
            return reject(INTAKE_UNSTATED, f"{type(self).__qualname__} states no intake")
        exposed = type(self)._exposed_names
        ports: list[PortIntake] = []
        joined: list[tuple[str, ...]] = []
        for item in self.member_intake:
            node = str(item.node)
            renamed = item.value.renamed(partial(_exposed_as, exposed, node))
            ports += renamed.ports
            joined += renamed.joined
        return Intake(tuple(ports), tuple(joined))

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
        names it by its node where it is held), or its members' under theirs (or where it
        places them, ``placed_at``).

        A kernel with a module is its one leaf, assembled from its binding and its parts
        (``finn.kernels.library.assemble_module``): its clocking, its parameters, its parts'
        pins, its sources (refused where they do not provide the module), its data, its
        parts' holds (L10) and which parts are presented."""
        if type(self).module_spec is not None:
            return self._module()
        placed = type(self).placed_at
        parts = [(placed.get(str(item.node), str(item.node)), item.value) for item in self.netlists]
        if not parts:
            return reject(
                "kernel-module",
                f"{type(self).__qualname__} declares no module and places no kernel",
            )
        buses = presented(
            (placed.get(str(item.node), str(item.node)), item.value) for item in self.controls
        )
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

    def _placed_inputs_0(self) -> tuple[str, ...]:
        return ()

    def _module_end(self) -> tuple[Any, ...]:
        return ()

    # What its members require of its holder (``_derive_doubled``): the head of each reading,
    # which the base installs for a kernel with such members, and its own control bus's part.

    def _doubled_0(self) -> bool:
        return False

    def _buses_0(self) -> tuple[str, ...]:
        return ()

    def _own_buses(self) -> tuple[str, ...]:
        return ()

    # Whether it or a member may present a control bus (``may_present_bus``).
    _presents_buses: ClassVar[bool] = False

    @derived
    def control_buses(self) -> tuple[str, ...]:
        """The control buses it presents, each by its path below it: its own (``""``), where
        its model presents it, and each present member's, under the member's path
        (``compute.packed``, a selector's case). What a shell's budget counts
        (``finn.kernels.budgets``); the base derives it for a kernel with members that may
        present one when its class is made (``_derive_doubled``)."""
        return self._own_buses()

    @derived
    def declares_bus(self) -> bool:
        """Whether it declares a control bus (presented or not): False here; the base derives
        True for a kernel declaring one (``ControlModel``) when its class is made."""
        return False

    @view
    def bus_control(self) -> Control | Rejected | Inapplicable:
        """The control bus it declares: a ``Control``, presenting nothing while the bus is
        not presented. A kernel declaring none answers inapplicable; for one declaring a
        bus the base derives it when its class is made, from the bus's model, the kernel's
        ``control`` and its ``register_map``."""
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
    def frame_cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through it, at full rate: characterization,
        derived by the rate law (CONTRACT.md L6). A kernel with a module advances one
        iteration an ``interval`` when nothing blocks it: its cycles a frame are its
        schedule's iterations (each of its scheduled ports' pace's span), or, a kernel with
        no schedule (a stage, a memory), advancing one beat a side an interval, the most
        beats a frame any of its placed ports carries, times its interval (``in_cycles``,
        the one conversion, which FIFO sizing reads too). A kernel with children takes the
        most any member takes, a member counting none (inapplicable) left out. A kernel with
        a module never restates it (``kernel-cycles-restated``, when its class is made): it
        states its ``interval`` where its hardware differs."""
        return self._frame_cycles()

    @view
    def assumed(self) -> tuple[str, ...]:
        """What its cycles a frame rest on that is assumed, not read or measured: its own
        ``interval`` where it stands assumed (``Standing.ASSUMED``, Pool's), and each of its
        members', below the member (``inner: interval 1, assumed: ...``). What a reader of
        its cycles reports beside them (``finn.kernels.explore.Cost.assumed``)."""
        own: tuple[str, ...] = ()
        if type(self).module_spec is not None:
            interval: Interval = self.interval
            if interval.standing is Standing.ASSUMED:
                own = (f"interval {interval}",)
        below = (f"{item.node}: {each}" for item in self.member_assumed for each in item.value)
        return (*own, *below)

    def _frame_cycles(self) -> int | Rejected:
        """``frame_cycles`` (above). A class restating ``frame_cycles`` as the base's, to be
        counted by its own kind (a channel's members, ``finn.kernels.channels.Channel``),
        calls it."""
        if type(self).module_spec is not None:
            return self._rate_frame_cycles()
        found = [item.value for item in self.member_frame_cycles]
        if found:
            return max(found)
        return reject("kernel-cycles", f"no member of {type(self).__qualname__} states cycles")

    def _rate_frame_cycles(self) -> int | Rejected:
        """A kernel with a module's cycles a frame by the rate law (``frame_cycles``): its
        schedule's iterations where it states a schedule, else its placed ports' most beats
        a frame, each port read in its declaration order (``StreamPort.frame_beats``), at
        its ``interval``; refused where it places no port."""
        interval: Interval = self.interval
        if _declared(type(self), "schedule"):
            schedule: Schedule = self.schedule
            return in_cycles(schedule.iterations, interval)
        beats = [each for each in self._module_beats_0() if each is not None]
        if not beats:
            return reject(
                "kernel-cycles", f"no port of {type(self).__qualname__} carries a stream's beats"
            )
        return in_cycles(max(beats), interval)

    # -- what it reaches of its members: their answers, read by its views, and their intake,
    # read by its own (a derived value, contract) ------------------------------------------
    # Each ``Members`` names the member it reads, so it follows the members it names.

    netlists = Members(netlist)
    controls = Members(bus_control)
    member_buffering = Members(buffering)
    member_resources = Members(resources)
    member_frame_cycles = Members(frame_cycles)
    member_assumed = Members(assumed)
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

    # Marked, so that a kernel reading it and stating no extents is refused when its class
    # is made (``kernel-extents-unstated``), not when it is evaluated.
    setattr(extent, _READS_EXTENTS, True)
    return derived(extent)


_READS_EXTENTS = "_finn_reads_extents"


__all__ = [
    "Kernel",
    "PortPins",
    "bound_extents",
    "bound_schedule",
    "extent_of",
    "merged",
    "spanned",
]
