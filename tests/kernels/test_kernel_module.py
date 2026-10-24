# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel contract: its bases, and what a kernel binding a module states.

A kernel binding a module (``KernelSpace`` with a ``LibraryModule``) states its
parameters, sources and resources, and its ports; it places no kernel (checked when the
class is made: ``tests/kernels/test_kernel_space.py``). Its control bus is a declaration
(``ControlBus``): held idle while it is not presented, presented (read as
``KernelSpace.bus_control``) while it is, so a bus is accounted for by construction; its
pins come before every port's, its hold after theirs. Every port runs on its kernel's clock domain
(``clock_domain``). Its module is the ``Leaf`` at the empty label of its netlist
(``library_module``). A kernel with children presents a child's port as its own
(``served``: ``<name>_contract``, labelled below it), and its netlist merges its members'.
Every kernel is a ``KernelSpace``.
"""

from __future__ import annotations

from dataclasses import replace

from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, OtherPins, by_hand, codes
from support.refusals import refused

from finn.core.space import (
    Available,
    DefinitionError,
    Param,
    Rejected,
    derived,
    design_space,
    reject,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import (
    Bus,
    Clocking,
    Direction,
    Endpoint,
    Member,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Held, Leaf, RegisterMap
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.configure import commit
from finn.kernels.control import Control, ControlBus, held_bus
from finn.kernels.library import LibraryModule, library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.port import AxiStreamPort, served
from finn.kernels.sides import Side
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import Resources

INT4 = DataType["INT4"]
TENSOR = Tensor((4,), ScalarEncoding(INT4))


PROBE = CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",))


class Probe(KernelSpace):
    """A model-only kernel: one input port, a mode pin held, and a doubled clock."""

    id = "test.probe"

    tensor: Tensor | None = Param(default=None)

    x = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        tensor=tensor,
        sequence=BeatSequence(vector_major((4,), 1)),
    )
    # The output is no one's to drive: left open.
    mode = OtherPins(
        (Signal("mode", Direction.IN, 2), Signal("busy", Direction.OUT, 1)), Held((("mode", 1),))
    )

    @derived
    def clocking(self) -> Clocking:
        return replace(type(self).clock_domain, doubled="ap_clk2x", doubling=True)

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")

    module = LibraryModule("probe", parameters=parameters, sources=(PROBE,))


def probe(**facts: object) -> Probe:
    network = by_hand(
        {
            "edge": channel(
                Side.boundary("in0_V"), Port("kernel", "x"), tensor=TENSOR, platform=FULL_DSP48E2
            ),
            "kernel": Facts(Probe, {"tensor": TENSOR, **facts}),  # type: ignore[arg-type]
        }
    )
    kernel: Probe = network.member("kernel")
    return kernel


def test_a_kernel_s_other_pins_go_before_its_ports_and_their_hold_after() -> None:
    module = library_module(probe())
    assert isinstance(module, Leaf) and module.held.inputs == (("mode", 1),)
    names = [pin.name for pin in module.abi.pins]
    assert names == ["ap_clk", "ap_clk2x", "ap_rst_n", "mode", "busy", "s_axis"]


def test_a_port_runs_on_its_kernels_clock_domain() -> None:
    bus = library_module(probe()).abi.pins[-1]
    assert (bus.associated_clock, bus.associated_reset) == ("ap_clk", "ap_rst_n")

    class Native(Probe):
        id = "test.probe.native"
        clock_domain = NATIVE_CLOCKING

    point = design_space(Native(tensor=TENSOR))
    assert isinstance(point.query(KernelSpace.netlist), Available)
    pins = library_module(point).abi.pins
    assert [pin.name for pin in pins[:3]] == ["clk", "ap_clk2x", "rst"]
    bus = pins[-1]
    assert (bus.associated_clock, bus.associated_reset) == ("clk", "rst")


BUS = Bus(
    "s_axilite",
    StandardProtocol.AXILITE,
    (Member("awvalid", "s_axilite_awvalid", 1), Member("awready", "s_axilite_awready", 1)),
)


class Controlled(Probe):
    """The probe with a control bus it presents where ``writable``, at the port ``control``."""

    id = "test.probe.controlled"

    writable: bool = Param(default=False)
    control: str | None = Param(default=None)

    @derived
    def config_bus(self) -> Bus:
        return BUS

    @view
    def register_map(self) -> RegisterMap:
        return RegisterMap(((0, 7),))

    config = ControlBus(config_bus, presented=writable, port=control, registers=register_map)


def controlled(**facts: object) -> Controlled:
    return design_space(Controlled(tensor=TENSOR, **facts))  # type: ignore[arg-type]


def test_a_control_bus_held_holds_its_inputs_after_every_port_s_hold() -> None:
    point = controlled()
    assert point.config_presented is False
    assert point.config_control == Control(None)
    assert point.query(Controlled.bus_control) == Available(Control(None))
    module = library_module(point)
    names = [pin.name for pin in module.abi.pins]
    # Its pins before every port's, after the probe's other pins (declared before it).
    assert names == ["ap_clk", "ap_clk2x", "ap_rst_n", "mode", "busy", "s_axilite", "s_axis"]
    expected = held_bus(BUS)
    assert module.held.inputs == (("mode", 1), *expected.inputs)
    assert module.held.unused == expected.unused


def test_a_control_bus_presented_is_read_with_its_port_and_registers() -> None:
    point = controlled(writable=True, control="first_s_axilite")
    assert point.config_presented is True
    control = Control(BUS, "first_s_axilite", RegisterMap(((0, 7),)))
    assert point.query(Controlled.bus_control) == Available(control)
    assert library_module(point).held.inputs == (("mode", 1),)


def test_a_control_bus_presented_without_its_port_is_refused() -> None:
    # A bus it presents names the port it is presented at: presented with no port (the
    # optional ``control`` left out, None), its control view refuses rather than present a
    # bus at port None.
    point = controlled(writable=True)
    assert point.config_presented is True
    answer = point.query(Controlled.config_control)
    assert isinstance(answer, Rejected)
    assert codes(answer) == {"control-bus-unaddressed"}


def test_a_memstream_s_bus_is_held_by_construction() -> None:
    point = design_space(
        MemStreamKernel(
            platform=FULL_DSP48E2,
            dtype=INT4,
            form=vector_major((4,), 2),
            contents=IntegerTensorValue.of((1, 2, 3, 4)),
        )
    )
    memory = commit(point, {"ram_style": "auto", "pumped_memory": False})
    assert memory.config_presented is False and memory.config_control == Control(None)
    assert set(dict(held_bus(memory.config_bus).inputs)) <= set(
        dict(library_module(memory).held.inputs)
    )


def test_a_kernel_presents_one_control_bus_at_most() -> None:
    with refused("control-bus-repeated", error=DefinitionError, message="Twice declares"):

        class Twice(Controlled):
            id = "test.probe.twice"
            other = ControlBus(Controlled.config_bus)


def test_two_control_buses_in_one_body_are_refused_as_repeated_before_lowering() -> None:
    """Counted before any declaration is lowered: each bus lowers to ``declares_bus`` and
    ``bus_control``, so lowering first would answer ``declaration-member-taken`` for the
    second and hide the refusal. (A second bus from a subclass is the case above.)"""
    with refused("control-bus-repeated", error=DefinitionError, message="Both declares"):

        class Both(Probe):
            id = "test.probe.both"
            writable: bool = Param(default=False)
            control: str | None = Param(default=None)

            @derived
            def config_bus(self) -> Bus:
                return BUS

            @view
            def register_map(self) -> RegisterMap:
                return RegisterMap(((0, 7),))

            first = ControlBus(config_bus, presented=writable, port=control, registers=register_map)
            second = ControlBus(
                config_bus, presented=writable, port=control, registers=register_map
            )


def test_a_control_bus_it_may_present_names_its_port_and_registers() -> None:
    with refused("control-bus-unaddressed", error=DefinitionError):

        class Unaddressed(Probe):
            id = "test.probe.unaddressed"
            writable: bool = Param(default=False)

            @derived
            def config_bus(self) -> Bus:
                return BUS

            config = ControlBus(config_bus, presented=writable)


def test_a_kernel_s_cycles_are_the_most_beats_a_port_carries() -> None:
    assert probe().cycles == 4
    assert codes(probe().query(KernelSpace.resources)) == {"kernel-resources"}


def test_a_kernel_with_children_presents_a_childs_port_as_its_own() -> None:
    class Presenting(KernelSpace):
        id = "test.presenting"
        kernel = Probe(tensor=TENSOR)
        x = served(kernel, "x")

    point = design_space(Presenting())
    assert point.x_contract.label == "kernel.x"
    assert point.x_contract.contract == point.kernel.x_contract.contract
    assert point.x_stream == point.kernel.x_stream

    with refused("served-child-unknown", error=DefinitionError, message="serves y of no child"):

        class Unserved(KernelSpace):
            id = "test.unserved"
            kernel = Probe(tensor=TENSOR)
            y = served(kernel, "y")
