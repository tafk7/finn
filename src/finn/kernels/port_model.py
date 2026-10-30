# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lean real port children and typed authoring surfaces.

PortModel is a constant-idle physical port: it declares no logical stream, width,
side, contract or tensor access. SequencePort adds an optional logical stream.
Both use ordinary core declarations, with zero Decisions and intrinsic element
admission at the port's real scope. Folding and composition remain on the parent.

Public physical Views characterize the child's contract. Module assembly (the
base's, ``Kernel.netlist``) calls each port's ordinary methods, ``pin_values``,
``held_values`` and ``pins_presented``, over literal model inputs and the pure
physical functions; it must not read these Views, so that it collects directly
returned refusals. ``OtherPins`` is the one raw pin declaration: a module's pins
outside its ports and its control bus.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar, cast

from finn.core.space import (
    Group,
    Inapplicable,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    reject,
    view,
)
from finn.core.space.declarations import Declaration, View, declared_path
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Access, Affine, Index, Refused, Schedule, plain_index
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Clocking, Endpoint, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import PortContract
from finn.kernels.laws import AXIS, AXIS_CLOCKING, TransportKind
from finn.kernels.port import (
    LoopEnds,
    PhysicalPort,
    Presenting,
    data_width,
    held,
    pins,
    stream_refusal,
)
from finn.kernels.port import contract as port_contract
from finn.kernels.sides import Side
from finn.kernels.values.domains import Integer, admit_element


class PortModel(Space):
    """A real physical port, constant idle unless a logical subtype adds presentation.

    ``label`` names the port in its kernel's ABI, not inferred from its model path.
    It declares its ``transport`` kind, not its pins: they follow from the library's
    defaults (``finn.kernels.laws``: ``AXIS``, ``NATIVE``) and the names its kernel states
    where its RTL differs (``Axis(bus=...)``, ``Native(prefix=...)``). A genuinely guarded
    absent child retains ordinary Space presence; an idle logical port remains present.
    """

    label: str = Param()
    endpoint: Endpoint = Param(default=Endpoint.TARGET)
    dtype: QONNXDataType | None = Param(default=None)
    admits: Integer | None = Param(default=None)
    transport: TransportKind = Param(default=AXIS)
    loop_ends: LoopEnds | None = Param(default=None)
    lanes: tuple[Index, ...] = Param(default=())
    clocking: Clocking = Param(default=AXIS_CLOCKING)

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        dtype = self.dtype
        if dtype is not None:
            if self.endpoint is Endpoint.INITIATOR:
                return reject("port-element", f"{self.label}: a producer's element is its tensor's")
            return admit_element(dtype, None)
        return reject("port-element", f"{self.label}: an idle port states its dtype")

    @constraint
    def element_taken(self) -> bool | Rejected:
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    def physical_spec(self, *, idle: bool) -> PhysicalPort:
        """Resolved physical fields in existing read order; placed ports skip idle lanes.

        An undecorated native method: no receiver escapes as a model value and
        no generated names or manually declared reads participate.
        """
        return PhysicalPort(
            name=self.label,
            endpoint=self.endpoint,
            transport=self.transport,
            loop_ends=self.loop_ends,
            lanes=self.lanes if idle else (),
        )

    def presenting(self) -> Presenting:
        """Model input to pure pin assembly; a subtype supplies its placed stream."""
        return self.element

    # -- what its kernel's module takes of it (``Kernel.netlist``) -------------------------

    def pin_values(self) -> tuple[Pin, ...] | Rejected:
        """Its pins of the module, from its transport (``finn.kernels.port.pins``): idle,
        one lane of its element."""
        presenting = self.presenting()
        return pins(
            self.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.clocking,
        )

    def held_values(self) -> Held | Rejected:
        """Its inputs held while idle (L10): every one, for a port that is never placed."""
        return held(self.physical_spec(idle=True), self.element, self.clocking)

    def pins_presented(self) -> bool:
        """Whether its pins are presented, so that its module's inputs need no hold: never,
        for a port that is never placed."""
        return False

    @view
    def physical_pins(self) -> tuple[Pin, ...] | Rejected:
        return self.pin_values()

    @view
    def idle_pins(self) -> Held | Rejected:
        return self.held_values()


class SequencePort(PortModel):
    """A given sequence over an optional tensor; absence skips the sequence entirely."""

    tensor: Tensor | None = Param(default=None)
    sequence: BeatSequence | None = Param(default=None)

    @derived
    def placed(self) -> bool:
        return self.tensor is not None

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        dtype = self.dtype
        if dtype is not None:
            if self.endpoint is Endpoint.INITIATOR:
                return reject("port-element", f"{self.label}: a producer's element is its tensor's")
            return admit_element(dtype, None)
        tensor = self.tensor
        if tensor is None:
            return reject("port-element", f"{self.label}: an idle port states its dtype")
        return tensor.element

    @derived
    def stream(self) -> Stream | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        sequence = self.sequence
        if sequence is None:
            return reject(
                "port-presentation", f"{self.label}: exactly one of a schedule or a sequence"
            )
        stream = Stream(self.element, sequence)
        unsupported = stream_refusal(stream)
        return unsupported if unsupported is not None else stream

    @derived
    def width(self) -> int | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        return data_width(self.physical_spec(idle=False), stream)

    @derived
    def side(self) -> Side | Inapplicable:
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        label = self.label
        if self.endpoint is Endpoint.INITIATOR:
            return Side.kernel(stream, label, self.width)
        return Side.kernel(stream, label)

    def presenting(self) -> Presenting:
        return self.stream if self.placed else self.element

    def held_values(self) -> Held | Rejected:
        """Placed, none; idle, its inputs held (L10)."""
        if self.placed:
            return Held()
        return held(self.physical_spec(idle=True), self.element, self.clocking)

    def pins_presented(self) -> bool:
        """Placed, its pins are presented (a channel connects them)."""
        placed: bool = self.placed
        return placed

    @view
    def contract(self) -> PortContract | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        found = port_contract(self.physical_spec(idle=False), stream, self.clocking)
        return found if isinstance(found, Rejected) else PortContract(self.label, found)


class ScheduledPort(SequencePort):
    """A scheduled tensor projection, with inward target-only index access.

    Schedule/index/folding inputs are model values. Public physical contracts retain
    pace; serial module assembly still uses undecorated model/native methods.
    """

    schedule: Schedule | None = Param(default=None)
    index: tuple[Index | Affine, ...] = Param(default=())
    reduces: tuple[Index, ...] = Param(default=())
    holds: tuple[Index, ...] = Param(default=())
    closes: tuple[Index, ...] = Param(default=())
    reshaped: bool = Param(default=False)

    @derived
    def stream(self) -> Stream | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        label = self.label
        schedule, given = self.schedule, self.sequence
        if (schedule is None) == (given is None):
            return reject("port-presentation", f"{label}: exactly one of a schedule or a sequence")
        if given is not None:
            sequence = given
        else:
            assert schedule is not None
            index = self.index
            try:
                shaped = None
                if self.reshaped:
                    plain = [each for each in map(plain_index, index) if each is not None]
                    if len(plain) != len(index):
                        raise Refused(
                            "reshape-not-plain",
                            "AxiStreamPort.presented",
                            "a reshaped port reads plain indices",
                        )
                    shaped = tuple(schedule.extent(axis) for axis in plain)
                tensor = self.tensor
                assert tensor is not None
                form = schedule.present(
                    tensor.shape,
                    index,
                    lanes=self.lanes,
                    reduces=self.reduces,
                    holds=self.holds,
                    reshape=shaped,
                )
                closes = self.closes
                markers = (schedule.closing(closes),) if closes else ()
                sequence = BeatSequence(form, markers=markers)
            except Refused as error:
                return reject("port-schedule", f"{label}: {error}")
        stream = Stream(self.element, sequence)
        unsupported = stream_refusal(stream)
        return unsupported if unsupported is not None else stream

    @view
    def contract(self) -> PortContract | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        physical = self.physical_spec(idle=False)
        paced = replace(physical, schedule=self.schedule, reduces=self.reduces, holds=self.holds)
        found = port_contract(paced, stream, self.clocking)
        return found if isinstance(found, Rejected) else PortContract(self.label, found)


class IndexedPort(ScheduledPort):
    """A potentially binding scheduled target; constant nonbinding uses ScheduledPort."""

    @derived
    def binds(self) -> bool:
        if not self.placed:
            return False
        return self.endpoint is Endpoint.TARGET and bool(self.index)

    @derived
    def access(self) -> Access | Inapplicable:
        """Only a placed target with indices reads its tensor to bind parent extents."""
        if not self.binds:
            return Inapplicable()
        tensor = self.tensor
        assert tensor is not None
        return Access(self.label, tensor.shape, self.index, self.reshaped)


class OtherPins(Group):
    """Pins of a module that are no stream port's and no control bus's (a mode pin), and
    what holds its inputs: the one raw pin declaration a kernel keeps (CONTRACT.md § 7).
    Its pins go before its kernel's control bus and ports, its holds after its ports'."""

    signals: tuple[Pin, ...] = Param()
    holding: Held = Param(default=Held())

    def pin_values(self) -> tuple[Pin, ...]:
        return self.signals

    def held_values(self) -> Held:
        return self.holding

    def pins_presented(self) -> bool:
        """Never presented: its inputs are held, or the module is refused (``kernel-pins``)."""
        return False


_SurfaceValue = TypeVar("_SurfaceValue")


@dataclass(frozen=True, init=False)
class PortSurface:
    """Actual authored references for a logical ABI key, never an allowed-read list.

    Construct only from node declarations. Fields are typed as their values,
    like ordinary core authored references; they are queried against the owning
    real Space. Fields are None when that authored form omits them; logical fields are
    absent for structurally idle ports and physical fields for parent forwarding. This is
    authoring/driver metadata, not a model Param or a validity certificate.
    """

    element: ScalarEncoding | None
    endpoint: Endpoint | None
    physical_pins: tuple[Pin, ...] | None
    idle_pins: Held | None
    stream: Stream | None = None
    width: int | None = None
    side: Side | None = None
    _contract: PortContract | View[PortContract] | None = None

    def __init__(
        self,
        element: ScalarEncoding | None,
        endpoint: Endpoint | None,
        physical_pins: tuple[Pin, ...] | None,
        idle_pins: Held | None,
        stream: Stream | None = None,
        width: int | None = None,
        side: Side | None = None,
        contract: PortContract | View[PortContract] | None = None,
    ) -> None:
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "physical_pins", physical_pins)
        object.__setattr__(self, "idle_pins", idle_pins)
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "_contract", contract)

    def contract_reference(self) -> PortContract | View[PortContract] | None:
        # A View is stored as metadata, not installed as a descriptor of this
        # dataclass. Read its storage directly; its __get__ belongs to Space.
        return cast(
            PortContract | View[PortContract] | None, object.__getattribute__(self, "_contract")
        )

    def at(self, namespace: type[Space] | type[Group] | Space | Group) -> PortSurface:
        """Rebase explicitly authored parent references for a driver/composer.

        Plain metadata only; no method or read-list callback is introduced on a
        Space. Actual native model/View declarations remain on their parent.
        """
        if not isinstance(namespace, type) and declared_path(namespace) is None:
            raise TypeError("PortSurface.at requires a class or node declaration")

        def bind(value: _SurfaceValue) -> _SurfaceValue:
            if isinstance(value, Declaration):
                assert value.name is not None
                return cast(_SurfaceValue, getattr(namespace, value.name))
            return value

        return PortSurface(
            bind(self.element),
            bind(self.endpoint),
            bind(self.physical_pins),
            bind(self.idle_pins),
            bind(self.stream),
            bind(self.width),
            bind(self.side),
            bind(self.contract_reference()),
        )

    @classmethod
    def of(cls, port: PortModel) -> PortSurface:
        if not isinstance(port, PortModel) or declared_path(port) is None:
            raise TypeError("PortSurface requires a PortModel node declaration")
        if isinstance(port, SequencePort):
            return cls(
                port.element,
                port.endpoint,
                port.physical_pins,
                port.idle_pins,
                port.stream,
                port.width,
                port.side,
                port.contract,
            )
        return cls(port.element, port.endpoint, port.physical_pins, port.idle_pins)


def port_surfaces(space_type: type[Space]) -> dict[str, PortSurface]:
    """Effective port declarations in inherited declaration order, through Group frames.

    Stops at real Space boundaries: grandchildren of hardware kernels are not
    this kernel's ports. Group paths are explicit dotted keys, with lexical
    order retained; this does not claim Members(Group.member) support.
    """

    def below(
        namespace: type[Space] | type[Group], placed: Space | Group | None, prefix: str
    ) -> dict[str, PortSurface]:
        effective: dict[str, object] = {}
        for base in reversed(namespace.__mro__):
            effective.update(vars(base))
        result: dict[str, PortSurface] = {}
        for name, declaration in effective.items():
            if isinstance(declaration, PortModel):
                port = declaration if placed is None else getattr(placed, name)
                result[prefix + name] = PortSurface.of(port)
            elif isinstance(declaration, PortSurface):
                result[prefix + name] = declaration.at(namespace if placed is None else placed)
            elif isinstance(declaration, Group):
                group = declaration if placed is None else getattr(placed, name)
                result.update(below(type(declaration), group, prefix + name + "."))
        return result

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise TypeError("port_surfaces requires a real Space class")
    return below(space_type, None, "")


__all__ = [
    "OtherPins",
    "PortModel",
    "PortSurface",
    "SequencePort",
    "ScheduledPort",
    "IndexedPort",
    "port_surfaces",
]
