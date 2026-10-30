# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's ports: real child Spaces, and typed authoring surfaces.

``StreamPort`` is the one stream-port construct: a child Space with zero Decisions and
intrinsic element admission at the port's real scope, presenting a given sequence or its
kernel's schedule's projection, or idle. Folding and composition remain on the parent.

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
from finn.kernels.laws import Advance, TransportKind
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


class StreamPort(Space):
    """A stream port: where one tensor leaves or enters its kernel as one stream, the one
    construct every port of a kernel with a module is declared by (CONTRACT.md § 7).

    It declares its role (``endpoint``), its tensor, how the tensor is presented and its
    ``transport`` kind (``finn.kernels.laws``: ``AXIS``, ``NATIVE``, or the names its RTL
    states, ``Axis(bus=...)``, ``Native(prefix=...)``), never its pins: they, its idle holds
    and its packing follow from the library. Its presentation is a given ``sequence`` (a
    stage's side) or its kernel's ``schedule`` projected through its ``index`` (``lanes``,
    ``reduces``, ``holds``, ``closes``, ``reshaped``); a port with no tensor is idle, and
    states its element (``dtype``). Its advance rule follows from how it is declared
    (``advance``). A target with an index reads its tensor to bind its kernel's extents
    (``access``).

    ``label`` is the port's name in its kernel and ``clocking`` its kernel's clock domain
    (L8: the kernel's, never the port's): a kernel supplies both for a port declared in its
    own body that does not state them (``finn.kernels.base.Kernel``). Its ``transport`` has
    no default (R1 F4): a native port taken for AXIS would be a bus the RTL does not have.
    """

    label: str = Param()
    endpoint: Endpoint = Param(default=Endpoint.TARGET)
    transport: TransportKind = Param()
    clocking: Clocking = Param()
    tensor: Tensor | None = Param(default=None)
    dtype: QONNXDataType | None = Param(default=None)
    admits: Integer | None = Param(default=None)
    loop_ends: LoopEnds | None = Param(default=None)
    # A given sequence, or the schedule's projection.
    sequence: BeatSequence | None = Param(default=None)
    schedule: Schedule | None = Param(default=None)
    index: tuple[Index | Affine, ...] = Param(default=())
    lanes: tuple[Index, ...] = Param(default=())
    reduces: tuple[Index, ...] = Param(default=())
    holds: tuple[Index, ...] = Param(default=())
    closes: tuple[Index, ...] = Param(default=())
    reshaped: bool = Param(default=False)
    # What makes a port of a given sequence selected or sourced (``advance``).
    governs: str | None = Param(default=None)
    stored: bool = Param(default=False)

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

    @constraint
    def element_taken(self) -> bool | Rejected:
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    @derived
    def advance(self) -> Advance:
        """Its advance rule, by how it is declared: scheduled, a projection of its kernel's
        schedule; selected, a selector port, which ``governs`` the port whose beats it goes
        with; sourced, a memory's output, ``stored``; else buffered, a stage's side."""
        if self.schedule is not None:
            return Advance.SCHEDULED
        if self.governs is not None:
            return Advance.SELECTED
        if self.stored:
            return Advance.SOURCED
        return Advance.BUFFERED

    @derived
    def stream(self) -> Stream | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        schedule, given = self.schedule, self.sequence
        if (schedule is None) == (given is None):
            return reject(
                "port-presentation", f"{self.label}: exactly one of a schedule or a sequence"
            )
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
                return reject("port-schedule", f"{self.label}: {error}")
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
        """The side it gives a channel: its stream, its label and, an output, its width."""
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        label = self.label
        if self.endpoint is Endpoint.INITIATOR:
            return Side.kernel(stream, label, self.width)
        return Side.kernel(stream, label)

    @derived
    def binds(self) -> bool:
        """Whether it reads its tensor to bind its kernel's extents: a placed target with
        an index."""
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
        """Model input to pure pin assembly: its stream, placed; idle, its element."""
        return self.stream if self.placed else self.element

    # -- what its kernel's module takes of it (``Kernel.netlist``) -------------------------

    def pin_values(self) -> tuple[Pin, ...] | Rejected:
        """Its pins of the module, from its transport (``finn.kernels.port.pins``): placed,
        its stream's; idle, one lane of its element."""
        presenting = self.presenting()
        return pins(
            self.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.clocking,
        )

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
    def physical_pins(self) -> tuple[Pin, ...] | Rejected:
        return self.pin_values()

    @view
    def idle_pins(self) -> Held | Rejected:
        return self.held_values()

    @view
    def contract(self) -> PortContract | Rejected | Inapplicable:
        """What it states to a channel: its stream on its transport, and, scheduled, its
        pace (``finn.kernels.port.contract``)."""
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        physical = self.physical_spec(idle=False)
        paced = replace(physical, schedule=self.schedule, reduces=self.reduces, holds=self.holds)
        found = port_contract(paced, stream, self.clocking)
        return found if isinstance(found, Rejected) else PortContract(self.label, found)


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
    def of(cls, port: StreamPort) -> PortSurface:
        if not isinstance(port, StreamPort) or declared_path(port) is None:
            raise TypeError("PortSurface requires a StreamPort node declaration")
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
            if isinstance(declaration, StreamPort):
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
    "PortSurface",
    "StreamPort",
    "port_surfaces",
]
