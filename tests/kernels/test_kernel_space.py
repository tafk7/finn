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
no kernel, and a kernel stating a stream port binds a module; one leaving a member
``required()`` is a base of kernels, which these checks pass by. A declaration's
argument is a member of its kernel (a declaration of the engine, or one ``required()``),
or a constant, of whatever type.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, cast

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, OtherPins, codes, fifo_node
from support.refusals import refused

import finn.kernels
from finn.core.space import (
    Available,
    DefinitionError,
    Param,
    Rejected,
    Unresolved,
    derived,
    design_space,
    divisors_of,
    inspection,
    required,
    view,
)
from finn.core.space._reads import summary
from finn.core.space.compiler import compile_model
from finn.core.space.declarations import Declaration, Required, View
from finn.core.space.semantics import default_semantics
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Direction, Endpoint, Pin, Signal
from finn.kernels.artifacts.contributions import Contribution, CopiedSource
from finn.kernels.artifacts.module import Fragment, Held
from finn.kernels.base import NATIVE_CLOCKING, KernelSpace, PortContract, Timed, UntimedKernel
from finn.kernels.channels import ChannelFifo
from finn.kernels.fifo import FifoKernel, fifo_resources
from finn.kernels.library import (
    ModuleSpec,
    _each,
    assemble_module,
    library_module,
    module_unprovided,
)
from finn.kernels.port import pins
from finn.kernels.port_model import native_port
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
    for member, value in (
        (FifoKernel.netlist, point.netlist),
        (FifoKernel.cycles, 6),
        (FifoKernel.buffering, 8 * 13),
        (FifoKernel.resources, point.resources),
    ):
        assert point.query(member) == Available(value)


def test_a_fifo_s_answers_wait_on_its_choices_and_refuse_with_it() -> None:
    undecided = design_space(fifo_node(word_bits=13, depth=8))
    assert isinstance(undecided.query(FifoKernel.netlist), Unresolved)
    # Its cycles read its ports' streams alone, which no choice decides.
    assert undecided.query(FifoKernel.cycles) == Available(6)
    refused_point = design_space(fifo_node(word_bits=13, depth=1)).with_choices(ram_style="auto")
    for answer in (FifoKernel.netlist, FifoKernel.buffering, FifoKernel.resources):
        assert "fifo-geometry" in codes(refused_point.query(answer))


def test_a_fifo_s_own_statements_and_answers_use_inferred_fixed_reads() -> None:
    for member in ("parameters", "held_bits", "resource_use", "netlist", "cycles"):
        assert isinstance(getattr(FifoKernel, member), View)
    # Native assembly infers its actual child/model dependencies from source.
    point = design_space(fifo_node(word_bits=13, depth=8)).with_choices(ram_style="auto")
    evidence = inspection.explain(point, FifoKernel.netlist)
    assert isinstance(evidence.result, Available)
    reads = {node.declaration.key for node in evidence.nodes}
    assert {"parameters", "clocking", "input.stream", "output.stream"} <= reads
    for space_type in (FifoKernel, ChannelFifo):
        model = compile_model(space_type)
        assert summary(model.scopes, model.per_class.reads)["dynamic"] == 0


# -- a kernel with members --------------------------------------------------------------------


class Stage(KernelSpace):
    """A model-only stage on a library module: one input port, three LUTs."""

    id = "test.kernel_space.stage"
    clock_domain = NATIVE_CLOCKING

    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    input = native_port(
        "input", Endpoint.TARGET, input_tensor, arriving, clocking=NATIVE_CLOCKING, label="input"
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"WIDTH": 4}

    @view
    def resource_use(self) -> Resources:
        return Resources(lut=3)

    @view
    def held_bits(self) -> int:
        return 5

    module_spec = ModuleSpec("probe", (PROBE,))

    def _extra_pins(self) -> tuple[Pin, ...]:
        return ()

    def _extra_held(self) -> Held:
        return Held()

    @view
    def netlist(self) -> Fragment | Rejected:
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters.items()))
        presenting = self.input.presenting()
        parts = _each(
            (
                self._extra_pins(),
                pins(self.input.physical_spec(idle=False), presenting, self.input.clocking),
            )
        )
        if isinstance(parts, Rejected):
            return parts
        spec = type(self).module_spec
        sources = spec.constant_sources
        if sources is None:
            sources = self.module_sources
        refused = module_unprovided(spec, type(self).__qualname__, sources)
        if isinstance(refused, Rejected):
            return refused
        holding = _each((Held(), self._extra_held()))
        if isinstance(holding, Rejected):
            return holding
        leaf = assemble_module(
            spec,
            type(self).id,
            type(self).__qualname__,
            clocking,
            parameters,
            tuple(parts),
            sources,
            (),
            tuple(holding),
            (False, self.input.placed),
        )
        return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

    @view
    def cycles(self) -> int:
        return self.input.stream.sequence.form.beats


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


class Untimed(UntimedKernel):
    """A kernel that counts no cycles of its own, placing one stage."""

    id = "test.kernel_space.untimed"
    stage = Stage(input_tensor=TENSOR, arriving=ARRIVING)


class Beside(KernelSpace):
    """A stage, and beside it a kernel that counts no cycles."""

    id = "test.kernel_space.beside"
    first = Stage(input_tensor=TENSOR, arriving=ARRIVING)
    held = Untimed()


def test_a_kernel_counting_no_cycles_is_a_kernel_but_for_its_cycles() -> None:
    """It derives the untimed base alone: no ``Timed``, no ``cycles``, so its parent's
    cycles are its other members'; its netlist, buffering and resources count as any
    kernel's."""
    assert not issubclass(Untimed, Timed)
    for member in ("cycles", "member_cycles"):
        assert not hasattr(Untimed, member)
    assert design_space(Untimed()).buffering == 5
    point = design_space(Beside())
    assert point.cycles == 4
    assert point.buffering == 10
    assert point.resources == Resources(lut=6)


def test_a_channel_s_fifo_is_the_one_kernel_of_the_library_counting_no_cycles() -> None:
    """A channel's FIFO presents what arrives, whose beats are counted where they are made:
    it derives ``UntimedKernel`` alone; every other kernel class derives ``KernelSpace``."""
    for module in pkgutil.iter_modules(finn.kernels.__path__):
        importlib.import_module(f"finn.kernels.{module.name}")
    found: list[type[UntimedKernel]] = []
    pending: list[type[UntimedKernel]] = [UntimedKernel]
    while pending:
        for each in pending.pop().__subclasses__():
            pending.append(each)
            if each.__module__.startswith("finn.kernels.") and not issubclass(each, KernelSpace):
                found.append(each)
    assert found == [ChannelFifo]
    assert not issubclass(ChannelFifo, Timed)
    for member in ("cycles", "counts_cycles", "member_cycles"):
        assert not hasattr(ChannelFifo, member)


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
            module_spec = ModuleSpec(
                "probe",
                constant_sources=(
                    CopiedSource("finnlib", "rtl/other.sv", provides=("module:other",)),
                ),
            )


class Referenced(Stage):
    """Its sources a member, which provides another module."""

    id = "test.kernel_space.referenced"

    @view
    def module_sources(self) -> tuple[Contribution, ...]:
        return (CopiedSource("finnlib", "rtl/other.sv", provides=("module:other",)),)

    module_spec = ModuleSpec("probe")


def test_a_member_s_sources_must_provide_the_module_when_it_is_built() -> None:
    point = design_space(Referenced(input_tensor=TENSOR, arriving=ARRIVING))
    assert codes(point.query(Referenced.netlist)) == {"library-module-unprovided"}


# -- the checks when a class is made ----------------------------------------------------------


def test_a_kernel_binds_one_library_module_at_most() -> None:
    with refused("library-module-repeated", error=DefinitionError, message="Twice binds"):

        class Twice(Stage):
            id = "test.kernel_space.twice"
            other = ModuleSpec("probe", (PROBE,))


def test_a_kernel_with_a_module_states_its_ports_and_resources_and_places_no_kernel() -> None:
    def parameters(self: KernelSpace) -> dict[str, int | str]:
        return {}

    def resource_use(self: KernelSpace) -> Resources:
        return Resources()

    def netlist(self: KernelSpace) -> Fragment:
        return Fragment()

    def cycles(self: KernelSpace) -> int:
        return 1

    def kernel(name: str, **members: object) -> type[KernelSpace]:
        body = {
            "id": f"test.{name}",
            "parameters": view(parameters),
            "netlist": view(netlist),
            "cycles": view(cycles),
            "module_spec": ModuleSpec("probe", (PROBE,)),
            **members,
        }
        return type(name, (KernelSpace,), body)

    port = native_port(
        "input", Endpoint.TARGET, TENSOR, ARRIVING, clocking=NATIVE_CLOCKING, label="input"
    )
    kernel("Complete", input=port, resource_use=view(resource_use))
    with refused("module-ports-unplaced", error=DefinitionError, message="Portless declares no"):
        kernel("Portless", resource_use=view(resource_use))
    with refused(
        "module-places-kernels",
        error=DefinitionError,
        message=r"Nesting places the kernels \['inner'\]",
    ):
        kernel(
            "Nesting",
            input=native_port(
                "input", Endpoint.TARGET, TENSOR, ARRIVING, clocking=NATIVE_CLOCKING, label="input"
            ),
            resource_use=view(resource_use),
            inner=Stage(input_tensor=TENSOR, arriving=ARRIVING),
        )
    with refused("module-resources-unstated", error=DefinitionError, message="states no"):
        kernel(
            "Unstated",
            input=native_port(
                "input", Endpoint.TARGET, TENSOR, ARRIVING, clocking=NATIVE_CLOCKING, label="input"
            ),
        )
    with refused("kernel-id-invalid", error=DefinitionError):

        class Unnamed(KernelSpace):
            pass


def test_a_kernel_stating_a_stream_port_binds_a_module() -> None:
    # A stream port's pins are a module's: on a kernel with children and no module of its
    # own, it is refused, naming the class and the port.
    class Unbound(KernelSpace):
        id = "test.kernel_space.unbound"
        input = native_port(
            "input", Endpoint.TARGET, TENSOR, ARRIVING, clocking=NATIVE_CLOCKING, label="input"
        )

    point = design_space(Unbound())
    assert point.input.stream == Stream(TENSOR.element, ARRIVING)
    assert codes(point.query(Unbound.netlist)) == {"kernel-module"}

    # A kernel with children presents its children's ports.
    class Serving(KernelSpace):
        id = "test.kernel_space.serving"
        inner = Stage(input_tensor=TENSOR, arriving=ARRIVING)

        @view
        def input_contract(self) -> PortContract:
            return self.inner.input.contract.under("inner")

    serving: Any = design_space(Serving())  # its lowered members are untyped
    assert serving.input_contract.label == "inner.input"

    # A base of kernels, leaving a required() member unmet, states its ports for the
    # kernels on it, which bind the module.
    class Ported(KernelSpace):
        id = "test.kernel_space.ported"
        clock_domain = NATIVE_CLOCKING
        input_tensor: Tensor = Param()
        input = native_port(
            "input",
            Endpoint.TARGET,
            input_tensor,
            ARRIVING,
            clocking=NATIVE_CLOCKING,
            label="input",
        )
        parameters = cast("dict[str, int | str]", required(dict))

    class Bound(Ported):
        id = "test.kernel_space.bound"

        @view
        def parameters(self) -> dict[str, int | str]:
            return {}

        @view
        def resource_use(self) -> Resources:
            return Resources()

        module_spec = ModuleSpec("probe", (PROBE,))

        def _extra_pins(self) -> tuple[Pin, ...]:
            return ()

        def _extra_held(self) -> Held:
            return Held()

        @view
        def netlist(self) -> Fragment | Rejected:
            clocking = self.clocking
            parameters = tuple(sorted(self.parameters.items()))
            presenting = self.input.presenting()
            parts = _each(
                (
                    self._extra_pins(),
                    pins(self.input.physical_spec(idle=False), presenting, self.input.clocking),
                )
            )
            if isinstance(parts, Rejected):
                return parts
            spec = type(self).module_spec
            sources = spec.constant_sources
            if sources is None:
                sources = self.module_sources
            refused = module_unprovided(spec, type(self).__qualname__, sources)
            if isinstance(refused, Rejected):
                return refused
            holding = _each((Held(), self._extra_held()))
            if isinstance(holding, Rejected):
                return holding
            leaf = assemble_module(
                spec,
                type(self).id,
                type(self).__qualname__,
                clocking,
                parameters,
                tuple(parts),
                sources,
                (),
                tuple(holding),
                (False, self.input.placed),
            )
            return leaf if isinstance(leaf, Rejected) else Fragment((("", leaf),))

        @view
        def cycles(self) -> int:
            return self.input.stream.sequence.form.beats

    assert library_module(design_space(Bound(input_tensor=TENSOR))).name == "probe"


class Based(Stage):
    """A base of kernels: its module names its sources, which it leaves required."""

    id = "test.kernel_space.based"
    clock_domain = NATIVE_CLOCKING
    source_values = cast("tuple[Contribution, ...]", required(tuple))

    @view
    def module_sources(self) -> tuple[Contribution, ...]:
        return self.source_values

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    module_spec = ModuleSpec("probe")


class Sourced(Based):
    """A kernel on ``Based``: its sources, its port and its resources stated."""

    id = "test.kernel_space.sourced"
    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param(default=ARRIVING)
    input = native_port(
        "input", Endpoint.TARGET, input_tensor, ARRIVING, clocking=NATIVE_CLOCKING, label="input"
    )

    @derived
    def source_values(self) -> tuple[Contribution, ...]:
        return (PROBE,)

    @view
    def resource_use(self) -> Resources:
        return Resources()


def test_a_module_naming_a_required_member_makes_a_base_of_kernels() -> None:
    # The base states no port and no resources, and is not placed.
    with refused("required-unmet", error=DefinitionError):
        design_space(Based())
    point = design_space(Sourced(input_tensor=TENSOR))
    assert library_module(point).sources == (PROBE,)


class Deep(KernelSpace):
    """A stand-in base of kernels: every kernel on it states its ``depth``."""

    id = "test.kernel_space.deep"
    clock_domain = NATIVE_CLOCKING
    depth = required(int)


class Forgetful(Deep):
    """A kernel on ``Deep`` with its module, port and resources, that forgets ``depth``."""

    id = "test.kernel_space.forgetful"
    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param(default=ARRIVING)
    input = native_port(
        "input", Endpoint.TARGET, input_tensor, ARRIVING, clocking=NATIVE_CLOCKING, label="input"
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources:
        return Resources()

    module_spec = ModuleSpec("probe", (PROBE,))


def test_a_kernel_forgetting_a_required_member_is_refused_where_placed() -> None:
    # Leaving a required() member unmet makes a base of kernels, which the module checks
    # pass by; placing it, as the top or as a child, is refused, naming the member.
    with refused(
        "required-unmet",
        error=DefinitionError,
        message=r"Forgetful.* leaves required members unmet: Deep\.depth",
    ):
        design_space(Forgetful(input_tensor=TENSOR))
    with refused("required-unmet", error=DefinitionError, message=r"Deep\.depth"):

        class Holder(KernelSpace):
            id = "test.kernel_space.holder"
            inner = Forgetful(input_tensor=TENSOR)


# -- the module's inputs: each a port's, a presented control bus's, or held -----------------

LOOSE = Signal("mode", Direction.IN, 1)


class Loose(Stage):
    """A stage with a pin that is no port's and no control bus's, held where ``holding``."""

    id = "test.kernel_space.loose"
    holding: bool = Param(default=False)

    @view
    def held_pins(self) -> Held:
        return Held(((LOOSE.name, 0),)) if self.holding else Held()

    other = OtherPins(signals=(LOOSE, Signal("busy", Direction.OUT, 1)))

    def _extra_pins(self) -> tuple[Pin, ...]:
        return self.other.pin_values()

    def _extra_held(self) -> Held:
        return self.held_pins


def test_an_input_neither_held_nor_presented_refuses_the_module() -> None:
    loose = design_space(Loose(input_tensor=TENSOR, arriving=ARRIVING))
    found = loose.query(Loose.netlist)
    assert codes(found) == {"kernel-pins"}
    assert isinstance(found, Rejected)
    assert found.findings[0].message == "Loose: the inputs ['mode'] are neither held nor presented"
    held = design_space(Loose(input_tensor=TENSOR, arriving=ARRIVING, holding=True))
    assert library_module(held).held.inputs == (("mode", 0),)


# -- a declaration's arguments ----------------------------------------------------------------


def test_a_constant_of_an_engine_type_is_a_constant_not_a_member() -> None:
    for constant in (default_semantics(int), divisors_of(4), 3, "x"):
        assert not isinstance(constant, (Declaration, Required))
    assert isinstance(Stage.parameters, View)
    assert isinstance(Based.source_values, Required)
