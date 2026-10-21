# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel base, ``KernelSpace``: its four answers, its library module, its checks.

A kernel answers four questions of its hardware, each a view written once in the base:
its netlist (its library module at the empty label, its members' netlists under their
nodes), its cycles, its buffering and its resources, its own part and its members'. Each
class's views read its declarations by the names their lowering fixes, so none reads
dynamically. A kernel binds at most one library module (``LibraryModule``), whose
sources must provide it: constant sources when the class is made, a member's when the
module is built. A kernel with a module states its ports and its resources and places
no kernel; one whose module names a member still ``required()`` is a base of kernels. A
declaration's argument is a member of its kernel (a declaration of the engine, or one
``required()``), or a constant, of whatever type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, OtherPins, codes, fifo_node
from support.refusals import refused

from finn.core.space import (
    Available,
    DefinitionError,
    Param,
    Rejected,
    Unresolved,
    View,
    default_semantics,
    design_space,
    divisors_of,
    required,
    view,
)
from finn.core.space._reads import infer, summary
from finn.core.space._structure import expand
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction, Endpoint, Signal
from finn.kernels.artifacts.contributions import Contribution, CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import (
    BUFFERING,
    CYCLES,
    NATIVE_CLOCKING,
    NETLIST,
    RESOURCES,
    KernelDeclaration,
    KernelSpace,
    is_member,
)
from finn.kernels.channels import ChannelFifo
from finn.kernels.fifo import FifoKernel, fifo_resources
from finn.kernels.library import LibraryModule, library_module
from finn.kernels.port import native_stream
from finn.kernels.utilization import RESOURCES_SEMANTICS
from finn.platform.capabilities import Resources

TENSOR = Tensor((4, 1), ScalarEncoding(DataType["INT4"]))
ARRIVING = BeatSequence(vector_major((4, 1), 1))
PROBE = CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",))


# -- the four answers, on the FIFO ------------------------------------------------------------


def test_a_fifo_answers_its_netlist_cycles_buffering_and_resources() -> None:
    point = design_space(fifo_node(word_bits=13, depth=8)).with_choices(ram_style="auto")
    module = library_module(point)
    assert module.name == "fifo"
    assert point.netlist == Fragment((("", module),))
    # Six words of its tensor, a word a beat; its depth of words; its own resources.
    assert point.cycles == 6
    assert point.buffering == 8 * 13
    assert point.resources == fifo_resources(8, 13, "auto", fabric=FULL_DSP48E2.fabric)
    for key, value in (
        (NETLIST, point.netlist),
        (CYCLES, 6),
        (BUFFERING, 8 * 13),
        (RESOURCES, point.resources),
    ):
        assert point.query(FifoKernel.exports[key]) == Available(value)


def test_a_fifo_s_answers_wait_on_its_choices_and_refuse_with_it() -> None:
    undecided = design_space(fifo_node(word_bits=13, depth=8))
    assert isinstance(undecided.query(FifoKernel.netlist), Unresolved)
    # Its cycles read its ports' streams alone, which no choice decides.
    assert undecided.query(FifoKernel.cycles) == Available(6)
    refused_point = design_space(fifo_node(word_bits=13, depth=1)).with_choices(ram_style="auto")
    for answer in (FifoKernel.netlist, FifoKernel.buffering, FifoKernel.resources):
        assert "fifo-geometry" in codes(refused_point.query(answer))


def test_a_fifo_s_own_statements_and_answers_are_views_read_by_fixed_names() -> None:
    for member in ("parameters", "held_bits", "resource_use", "netlist", "cycles"):
        assert isinstance(getattr(FifoKernel, member), View)
    # Its netlist reads its library module's members and its ports' by name.
    reads = cast(Any, FifoKernel.netlist).function.__reads__
    assert {"parameters", "clocking", "input_stream", "output_stream"} <= set(reads)
    for space_type in (FifoKernel, ChannelFifo):
        table = expand(space_type)
        assert summary(table, infer(table))["dynamic"] == 0


# -- a kernel with members --------------------------------------------------------------------


class Stage(KernelSpace):
    """A model-only stage on a library module: one input port, three LUTs."""

    id = "test.kernel_space.stage"
    clock_domain = NATIVE_CLOCKING

    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    input = native_stream("input", Endpoint.TARGET, input_tensor, arriving)

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"WIDTH": 4}

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        return Resources(lut=3)

    @view
    def held_bits(self) -> int:
        return 5

    module = LibraryModule("probe", parameters=parameters, sources=(PROBE,))


class Pair(KernelSpace):
    """Two stages, no module of its own."""

    id = "test.kernel_space.pair"
    first = Stage(input_tensor=TENSOR, arriving=ARRIVING)
    second = Stage(input_tensor=TENSOR, arriving=ARRIVING)


class Empty(KernelSpace):
    """Neither a module nor a member."""

    id = "test.kernel_space.empty"


def test_a_kernel_with_members_answers_theirs_under_their_nodes() -> None:
    point = design_space(Pair())
    leaf = library_module(design_space(Stage(input_tensor=TENSOR, arriving=ARRIVING)))
    assert point.netlist == Fragment((("first", leaf), ("second", leaf)))
    assert point.cycles == 4
    assert point.buffering == 10
    assert point.resources == Resources(lut=6)
    with pytest.raises(ValueError, match="Pair binds no library module"):
        library_module(point)


def test_a_kernel_with_neither_a_module_nor_a_member_is_refused_its_answers() -> None:
    point = design_space(Empty())
    assert codes(point.query(Empty.netlist)) == {"kernel-module"}
    found = point.query(Empty.cycles)
    assert isinstance(found, Rejected)
    assert found.findings[0].message == "no member of Empty states cycles"
    assert point.buffering == 0 and point.resources == Resources()


# -- the library module's sources -------------------------------------------------------------


def test_constant_sources_must_provide_the_module_when_the_class_is_made() -> None:
    with refused(
        "library-module-unprovided",
        error=DefinitionError,
        message=r"Unprovided: its sources provide \['module:other'\], not the library module "
        r"probe \(module:probe\)",
    ):

        class Unprovided(Stage):
            id = "test.kernel_space.unprovided"
            module = LibraryModule(
                "probe",
                parameters=Stage.parameters,
                sources=(CopiedSource("finnlib", "rtl/other.sv", provides=("module:other",)),),
            )


class Referenced(Stage):
    """Its sources a member, which provides another module."""

    id = "test.kernel_space.referenced"

    @view
    def sources(self) -> tuple[Contribution, ...]:
        return (CopiedSource("finnlib", "rtl/other.sv", provides=("module:other",)),)

    module = LibraryModule("probe", parameters=Stage.parameters, sources=sources)


def test_a_member_s_sources_must_provide_the_module_when_it_is_built() -> None:
    point = design_space(Referenced(input_tensor=TENSOR, arriving=ARRIVING))
    assert codes(point.query(Referenced.netlist)) == {"library-module-unprovided"}


# -- the checks when a class is made ----------------------------------------------------------


def test_a_kernel_binds_one_library_module_at_most() -> None:
    with refused("library-module-repeated", error=DefinitionError, message="Twice binds"):

        class Twice(Stage):
            id = "test.kernel_space.twice"
            other = LibraryModule("probe", parameters=Stage.parameters, sources=(PROBE,))


def test_a_kernel_with_a_module_states_its_ports_and_resources_and_places_no_kernel() -> None:
    def parameters(self: KernelSpace) -> dict[str, int | str]:
        return {}

    def resource_use(self: KernelSpace) -> Resources:
        return Resources()

    def kernel(name: str, **members: object) -> type[KernelSpace]:
        stated = view(parameters)
        body = {
            "id": f"test.{name}",
            "parameters": stated,
            "module": LibraryModule("probe", parameters=stated, sources=(PROBE,)),
            **members,
        }
        return type(name, (KernelSpace,), body)

    port = native_stream("input", Endpoint.TARGET, TENSOR, ARRIVING)
    kernel("Complete", input=port, resource_use=view(semantics=RESOURCES_SEMANTICS)(resource_use))
    with refused("module-ports-unplaced", error=DefinitionError, message="Portless declares no"):
        kernel("Portless", resource_use=view(semantics=RESOURCES_SEMANTICS)(resource_use))
    with refused(
        "module-places-kernels",
        error=DefinitionError,
        message=r"Nesting places the kernels \['inner'\]",
    ):
        kernel(
            "Nesting",
            input=native_stream("input", Endpoint.TARGET, TENSOR, ARRIVING),
            resource_use=view(semantics=RESOURCES_SEMANTICS)(resource_use),
            inner=Stage(input_tensor=TENSOR, arriving=ARRIVING),
        )
    with refused("module-resources-unstated", error=DefinitionError, message="states no"):
        kernel("Unstated", input=native_stream("input", Endpoint.TARGET, TENSOR, ARRIVING))
    with refused("kernel-id-invalid", error=DefinitionError):

        class Unnamed(KernelSpace):
            pass


class Based(KernelSpace):
    """A base of kernels: its module names its sources, which it leaves required."""

    id = "test.kernel_space.based"
    clock_domain = NATIVE_CLOCKING
    sources = cast("tuple[Contribution, ...]", required(tuple))

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    module = LibraryModule("probe", parameters=parameters, sources=sources)


class Sourced(Based):
    """A kernel on ``Based``: its sources, its port and its resources stated."""

    id = "test.kernel_space.sourced"
    input_tensor: Tensor = Param()
    input = native_stream("input", Endpoint.TARGET, input_tensor, ARRIVING)

    @view
    def sources(self) -> tuple[Contribution, ...]:
        return (PROBE,)

    @view(semantics=RESOURCES_SEMANTICS)
    def resource_use(self) -> Resources:
        return Resources()


def test_a_module_naming_a_required_member_makes_a_base_of_kernels() -> None:
    # The base states no port and no resources, and is not placed.
    with refused("required-unmet", error=DefinitionError):
        design_space(Based())
    point = design_space(Sourced(input_tensor=TENSOR))
    assert library_module(point).sources == (PROBE,)


# -- the module's inputs: each a port's, a presented control bus's, or held -----------------

LOOSE = Signal("mode", Direction.IN, 1)


class Loose(Stage):
    """A stage with a pin that is no port's and no control bus's, held where ``holding``."""

    id = "test.kernel_space.loose"
    holding: bool = Param(default=False)

    @view
    def held_pins(self) -> Held:
        return Held(((LOOSE.name, 0),)) if self.holding else Held()

    other = OtherPins((LOOSE, Signal("busy", Direction.OUT, 1)), held_pins)


def test_an_input_neither_held_nor_presented_refuses_the_module() -> None:
    loose = design_space(Loose(input_tensor=TENSOR, arriving=ARRIVING))
    found = loose.query(Loose.netlist)
    assert codes(found) == {"kernel-pins"}
    assert isinstance(found, Rejected)
    assert found.findings[0].message == "Loose: the inputs ['mode'] are neither held nor presented"
    held = design_space(Loose(input_tensor=TENSOR, arriving=ARRIVING, holding=True))
    assert library_module(held).held.inputs == (("mode", 0),)


# -- a declaration's arguments ----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Given(KernelDeclaration):
    """A declaration of one argument, read as its constant or its member."""

    given: Any


def test_a_constant_of_an_engine_type_is_a_constant_not_a_member() -> None:
    for constant in (default_semantics(int), divisors_of(4), 3, "x"):
        assert not is_member(constant)
        declaration = Given(constant)
        assert declaration.references() == {}
        point = design_space(Stage(input_tensor=TENSOR, arriving=ARRIVING))
        assert declaration.value(point, "given") is constant
    assert is_member(Stage.parameters) and Given(Stage.parameters).references() == {
        "given": "parameters"
    }
    assert is_member(Based.sources)  # required(): a member a subclass states
