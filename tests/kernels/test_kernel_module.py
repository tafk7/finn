# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel contract: two bases, and what each must state.

A leaf (``LeafKernel``) binds one FinnLib module. It states its module, parameters,
sources and resources (``required()``: a class that leaves one unmet is not placed),
and its ports, which are checked when the class is made; it places no kernel. Every
input among its other pins is held or presented as a control bus (``pins_accounted``),
and every port runs on its clock (``clocked``): constraints, counted in its validity,
refused when the kernel is created, not when it is composed. Its ``module`` is the
``Leaf``. A kernel with children (``ComposedKernel``) states the interfaces it
presents, each a view ``<name>_end``, and merges its members' netlists into one
``Composed`` module. A kernel is one of the two.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, codes
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
from finn.kernels.artifacts.abi import Clocking, Direction, Endpoint, Pin, Signal
from finn.kernels.artifacts.contributions import Contribution
from finn.kernels.artifacts.module import Held, Leaf
from finn.kernels.base import (
    PORT_END,
    ComposedKernel,
    Kernel,
    LeafKernel,
    PortEnd,
    interface,
)
from finn.kernels.configure import commit
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.network import Interface, Network, wire
from finn.kernels.port import AxiStreamPort
from finn.kernels.utilization import RESOURCES_SEMANTICS
from finn.platform.capabilities import Resources

INT4 = DataType["INT4"]
TENSOR = Tensor((4,), ScalarEncoding(INT4))


class Probe(LeafKernel):
    """A model-only leaf: one input port, a mode pin, and a clocking of its choice."""

    id = "test.probe"
    rtl_module = "probe"

    tensor: Tensor = Param(required=False)
    hold_mode: bool = Param(default=True)
    port_clock: str = Param(default="ap_clk")

    x = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        tensor=tensor,
        sequence=BeatSequence(vector_major((4,), 1)),
        clock=port_clock,
    )
    x_end = interface("x")

    @derived
    def clocking(self) -> Clocking:
        return Clocking(doubled="ap_clk2x", doubling=True)

    @derived
    def other_pins(self) -> tuple[Pin, ...]:
        return (Signal("mode", Direction.IN, 2), Signal("busy", Direction.OUT, 1))

    @derived
    def held(self) -> Held:
        return Held((("mode", 1),)) if self.hold_mode else Held()

    @derived
    def parameters(self) -> dict[str, int | str]:
        return {}

    @derived
    def sources(self) -> tuple[Contribution, ...]:
        return ()

    @derived(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources | Rejected:
        return reject("kernel-resources", "a model only")


def probe(**facts: object) -> Probe:
    network = Network(
        {
            "edge": wire(
                tensor=TENSOR,
                port="in0_V",
                consumer=Interface("kernel", "x_end"),
                platform=FULL_DSP48E2,
            ),
            "kernel": Probe(tensor=TENSOR, **facts),  # type: ignore[arg-type]
        }
    )
    kernel: Probe = network.member("kernel")
    return kernel


def test_a_leaf_holds_or_presents_every_input_that_is_no_ports() -> None:
    held = probe().module
    assert isinstance(held, Leaf) and held.held.inputs == (("mode", 1),)
    # The output is no one's to drive: left open, not accounted.
    assert codes(probe(hold_mode=False).query(LeafKernel.module)) == {"kernel-pins"}


def test_a_port_runs_on_its_kernels_clock() -> None:
    assert codes(probe(port_clock="ap_clk2x").query(LeafKernel.module)) == {"kernel-clock"}


def test_a_held_bus_is_accounted_for() -> None:
    point = design_space(
        MemStreamKernel(
            platform=FULL_DSP48E2, dtype=INT4, form=vector_major((4,), 2), contents=(1, 2, 3, 4)
        )
    )
    memory = commit(point, {"ram_style": "auto", "pumped_memory": False})
    assert memory.query(LeafKernel.pins_accounted) == Available(True)


def test_a_leafs_cycles_are_the_most_beats_a_port_carries() -> None:
    assert probe().cycles == 4
    assert codes(probe().query(LeafKernel.resources)) == {"kernel-resources"}


def test_a_leaf_that_leaves_a_required_member_unmet_is_not_placed() -> None:
    class Unsourced(LeafKernel):
        id = "test.unsourced"
        rtl_module = "probe"
        x = AxiStreamPort(name="s_axis", endpoint=Endpoint.TARGET, sequence=Probe.x.sequence)

        @derived
        def parameters(self) -> dict[str, int | str]:
            return {}

    with refused(
        "required-unmet",
        error=DefinitionError,
        message=r"LeafKernel\.sources, LeafKernel\.resource_use",
    ):
        Unsourced()


def stated(name: str, **members: object) -> type[LeafKernel]:
    """A leaf named ``name`` stating its module, parameters, sources and resources, and
    ``members``."""

    def parameters(self: LeafKernel) -> dict[str, int | str]:
        return {}

    def sources(self: LeafKernel) -> tuple[Contribution, ...]:
        return ()

    def resource_use(self: LeafKernel) -> Resources:
        return Resources()

    body = {
        "id": f"test.{name}",
        "rtl_module": "probe",
        "parameters": derived(parameters),
        "sources": derived(sources),
        "resource_use": derived(semantics=RESOURCES_SEMANTICS)(resource_use),
        **members,
    }
    return type(name, (LeafKernel,), body)


def port() -> AxiStreamPort:
    return AxiStreamPort(name="s_axis", endpoint=Endpoint.TARGET, sequence=Probe.x.sequence)


def test_a_leaf_states_its_ports_and_places_no_kernel() -> None:
    stated("Complete", x=port())
    with refused("leaf-ports-unplaced", error=DefinitionError, message="Portless places no port"):
        stated("Portless")
    with refused(
        "leaf-places-kernels",
        error=DefinitionError,
        message=r"Nesting places the kernels \['inner'\]",
    ):
        stated("Nesting", x=port(), inner=Probe(tensor=TENSOR))


def test_a_kernel_with_children_names_the_end_of_each_interface() -> None:
    with refused(
        "composed-end-absent",
        error=DefinitionError,
        message=r"the interfaces \['y'\] without their ends",
    ):

        class Unended(ComposedKernel):
            id = "test.unended"
            interfaces = ("x", "y")
            kernel = Probe(tensor=TENSOR)

            @view(semantics=PORT_END)
            def x_end(self) -> PortEnd:
                served: PortEnd = self.kernel.x_end
                return served.under("kernel")


def test_a_kernel_is_a_leaf_or_a_kernel_with_children() -> None:
    with refused("kernel-kind-invalid", error=DefinitionError, message="a kernel is a LeafKernel"):

        class Neither(Kernel):
            id = "test.neither"
