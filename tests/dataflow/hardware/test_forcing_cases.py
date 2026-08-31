# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 gate: the five shapes the physical Kernel surface has to support.

The migration plan names five forcing cases, and they exist because the
one-Kernel-one-Region assumption is the easy mistake and the expensive one.
Each is built here over synthetic semantics, so the surface is proved before
MVAU depends on it:

1. one Region, one Kernel, one component;
2. one Region, one Kernel, several components;
3. two Regions, two independent Kernels;
4. two Regions and their edge, one fused Kernel;
5. two Kernel alternatives over one Region, with the Region value unchanged
   whichever is chosen.

The semantics are deliberately not MVAU's.  A surface that only works for the
operation it was extracted from has not been extracted.

The two Regions here carry *identical* schedules, requirements, availability,
and beat maps, and differ only in operand naming, while the two computation
contracts over them differ outright.  That is on purpose: it makes every
coverage check that could have been satisfied by shape fail, so what remains
passing is the part that reads declarations.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.design.region import QONNX_DATATYPE_VALUE_SEMANTICS

from dataclasses import dataclass
from typing import cast

import pytest

import finn.dataflow.hardware as hardware
from finn.dataflow.authoring import (
    AuthoringError,
    OpDesign,
    Ref,
    assemble_specs,
    divisors_of,
    finite,
    reject,
)
from finn.dataflow.authoring.scope import semantics_for
from finn.dataflow.design import (
    Answer,
    Decided,
    DependencyKind,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.hardware import (
    BoundRegion,
    ComputationContract,
    HardwareDesign,
    HardwareKernel,
    HardwareKernelSelection,
    PhysicalComponent,
    bind_hardware_kernel,
    bound_regions,
    check_declared_references,
    declare_hardware_kernel,
    hardware_namespace,
    scalar_parameters,
)
from finn.dataflow.hardware.kernel import HardwareKernelDeclaration
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    element_width as element_width_of,
)
from finn.dataflow.spec_algebra import SpecAuthoringError

INT8 = DataType["INT8"]
OWNER = "example"

#: The node ids the synthetic assembly uses.  A Kernel never sees these -- it
#: declares roles, and the assembly says which node fills each one.
PRODUCER_NODE = "upstream"
CONSUMER_NODE = "downstream"
LINK_EDGE = "producer_to_consumer"

#: Two contracts over identically shaped traffic.  Nothing in a Region tells
#: them apart, which is exactly why they are declared.
SCALED = ComputationContract("scaled_copy")
REDUCED = ComputationContract("running_maximum")


# -- synthetic semantics -----------------------------------------------------


def _region(extent: int, operand: str) -> DataflowRegion:
    """One tiny Region, parameterized only by what the folding fixes."""

    schedule = LogicalSchedule((ScheduleLevel("element", extent),))
    source = Operand(f"{operand}_in", INT8, (extent,))
    result = Operand(f"{operand}_out", INT8, (extent,))
    beats = BeatSequence(1, tuple(((index,),) for index in range(extent)))
    requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        ((index,), (index,)): 1 for index in range(extent)
    }
    availability: dict[tuple[int, ...], tuple[int, ...]] = {
        (index,): (index,) for index in range(extent)
    }
    return DataflowRegion(
        schedule,
        (InputInterface(Port("input", source, beats), ScheduledInputRequirements(requirements)),),
        (
            OutputInterface(
                Port("output", result, beats), ScheduledOutputAvailability(availability)
            ),
        ),
    )


def _network(producer: DataflowRegion, consumer: DataflowRegion) -> DataflowNetwork:
    """The producer feeding the consumer, with one named edge between them."""

    produced = producer.output_interface("output").port
    return DataflowNetwork(
        (NetworkNode(PRODUCER_NODE, producer), NetworkNode(CONSUMER_NODE, consumer)),
        (
            Edge(
                LINK_EDGE,
                RegionEndpoint(PRODUCER_NODE, "output"),
                (
                    SinkContract(
                        RegionEndpoint(CONSUMER_NODE, "input"),
                        PositionMap.identity(produced.beat_sequence.image),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "in",
                RegionEndpoint(PRODUCER_NODE, "input"),
                producer.input_interface("input").port.beat_sequence,
            ),
            BoundaryContract(
                "out",
                RegionEndpoint(CONSUMER_NODE, "output"),
                consumer.output_interface("output").port.beat_sequence,
            ),
        ),
    )


@dataclass(frozen=True)
class HardwareInputs:
    """What a physical Kernel in this example is allowed to read.

    ``lanes`` is the Region-affecting decision, declared once by the semantics
    and imported here.  A Kernel that declared its own would be choosing a fold
    the Region was already built from.

    The Regions, their computation contracts, and the Network arrive the same
    way: as handles, so a Kernel names a declaration rather than a shape.
    """

    extent: Ref[int]
    element_width: Ref[int]
    target_family: Ref[str]
    lanes: Ref[int]
    producer: Ref[DataflowRegion]
    consumer: Ref[DataflowRegion]
    producer_computation: Ref[ComputationContract]
    consumer_computation: Ref[ComputationContract]
    network: Ref[DataflowNetwork]


def _semantics() -> tuple[OpDesign, HardwareInputs, QualifiedPath]:
    """The logical half, plus the one problem path the caller has to supply.

    ``element_type`` is returned as a path rather than folded into
    ``HardwareInputs`` because no Kernel reads it: what a Kernel needs is the
    width, and the width is a declared property derived from it.
    """

    design = OpDesign("example.op", problem_namespace=OWNER)
    extent = design.graph_fact("extent", int)
    element_type = design.graph_fact("element_type", QONNX_DATATYPE_VALUE_SEMANTICS)
    target_family = design.target_fact("family", str, required=False)
    lanes = design.decision("lanes", int, domain=divisors_of(extent))
    # The width is a declared property, not a projection applied on the way out
    # to the RTL.  A parameter that needs bits names this.
    element_width = design.derived(
        "element_width",
        int,
        dependencies={"element_type": element_type},
        evaluate=element_width_of,
    )
    producer = design.derived(
        "producer_region",
        DataflowRegion,
        dependencies={"extent": extent},
        evaluate=lambda extent: _region(extent, "produced"),
    )
    consumer = design.derived(
        "consumer_region",
        DataflowRegion,
        dependencies={"extent": extent},
        evaluate=lambda extent: _region(extent, "consumed"),
    )
    producer_computation = design.derived(
        "producer_computation",
        ComputationContract,
        dependencies={},
        evaluate=lambda: SCALED,
    )
    consumer_computation = design.derived(
        "consumer_computation",
        ComputationContract,
        dependencies={},
        evaluate=lambda: REDUCED,
    )
    network = design.derived(
        "network",
        DataflowNetwork,
        dependencies={"producer": producer, "consumer": consumer},
        evaluate=_network,
    )
    return (
        design,
        HardwareInputs(
            extent,
            element_width,
            target_family,
            lanes,
            producer,
            consumer,
            producer_computation,
            consumer_computation,
            network,
        ),
        element_type.path,
    )


# -- the Kernels -------------------------------------------------------------


class SingleComponentKernel(HardwareKernel):
    """Case 1: one covered Region, one instantiated module."""

    id = "single"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "compute",
            region=facts.consumer,
            computation=facts.consumer_computation,
            implements=REDUCED,
        )
        design.source("example", "rtl/single.sv")
        design.parameter("LANES", cast("Ref[object]", facts.lanes))
        design.parameter("WIDTH", cast("Ref[object]", facts.element_width))
        design.constant("MODE", 0, why="this core has one mode; the parameter is vestigial")

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (
            PhysicalComponent(
                "single.core", "example.single", scalar_parameters(dict(binding.parameters))
            ),
        )


class MultiComponentKernel(HardwareKernel):
    """Case 2: one covered Region, a shell around a core and a buffer.

    Also the pool alternative for case 5: it covers exactly what
    ``SingleComponentKernel`` covers, and picking it changes no Region.
    """

    id = "multi"
    version = "3"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "compute",
            region=facts.consumer,
            computation=facts.consumer_computation,
            implements=REDUCED,
        )
        design.source("example", "rtl/multi_pkg.sv", "rtl/multi_core.sv", "rtl/multi.sv")
        pipelined = design.choice("pipelined", bool, domain=finite((False, True)))
        depth = design.derived(
            "buffer_depth",
            int,
            dependencies={"lanes": facts.lanes, "pipelined": pipelined},
            evaluate=lambda lanes, pipelined: lanes * 2 if pipelined else lanes,
        )
        design.parameter("LANES", cast("Ref[object]", facts.lanes))
        design.parameter("DEPTH", cast("Ref[object]", depth))
        design.parameter("PIPELINED", cast("Ref[object]", pipelined))
        design.coverage_constraint(
            "target_supported",
            dependencies={"family": facts.target_family.allow_absent()},
            evaluate=lambda family: family == "wide",
        )

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        values = scalar_parameters(dict(binding.parameters))
        shell = PhysicalComponent("multi.shell", "example.multi", values)
        return (
            shell,
            PhysicalComponent("multi.core", "example.multi_core", values, shell.id),
            PhysicalComponent("multi.buffer", "example.multi_buffer", values, shell.id),
        )


class UpstreamKernel(HardwareKernel):
    """Case 3, first half: covers the producer Region and nothing else."""

    id = "upstream"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "producer",
            region=facts.producer,
            computation=facts.producer_computation,
            implements=SCALED,
        )
        design.source("example", "rtl/upstream.sv")
        design.parameter("LANES", cast("Ref[object]", facts.lanes))

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("upstream.core", "example.upstream"),)


class DownstreamKernel(HardwareKernel):
    """Case 3, second half: covers the consumer Region and nothing else."""

    id = "downstream"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "consumer",
            region=facts.consumer,
            computation=facts.consumer_computation,
            implements=REDUCED,
        )
        design.source("example", "rtl/downstream.sv")
        design.parameter("LANES", cast("Ref[object]", facts.lanes))

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("downstream.core", "example.downstream"),)


class FusedKernel(HardwareKernel):
    """Case 4: both Regions and the edge between them, as one module."""

    id = "fused"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "producer",
            region=facts.producer,
            computation=facts.producer_computation,
            implements=SCALED,
        )
        design.covers_region(
            "consumer",
            region=facts.consumer,
            computation=facts.consumer_computation,
            implements=REDUCED,
        )
        design.absorbs_edge(
            "link", network=facts.network, source_role="producer", sink_role="consumer"
        )
        design.source("example", "rtl/fused.sv")
        design.parameter("LANES", cast("Ref[object]", facts.lanes))
        design.coverage_constraint(
            "extent_supported",
            dependencies={"extent": facts.extent},
            evaluate=lambda extent: (
                True if extent <= 8 else reject("extent-too-wide", "the fused core stops at 8")
            ),
        )

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("fused.core", "example.fused"),)


class MiscomputingKernel(HardwareKernel):
    """Claims the consumer Region while implementing the producer's arithmetic.

    Nothing about its Region coverage is wrong -- it names the right
    declaration.  Only the contract is, which is the failure no amount of
    schedule checking would find.
    """

    id = "miscomputing"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "compute",
            region=facts.consumer,
            computation=facts.consumer_computation,
            implements=SCALED,
        )
        design.source("example", "rtl/miscomputing.sv")

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("miscomputing.core", "example.miscomputing"),)


class ElsewhereKernel(HardwareKernel):
    """Uses the ``compute`` role name for the *other* Region.

    Nothing about it is malformed on its own.  It is only wrong as an
    alternative to something that means a different Region by the same word.
    """

    id = "elsewhere"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "compute",
            region=facts.producer,
            computation=facts.producer_computation,
            implements=SCALED,
        )
        design.source("example", "rtl/elsewhere.sv")

    @classmethod
    def elaborate(cls, binding: HardwareKernel) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("elsewhere.core", "example.elsewhere"),)


# -- assembly ----------------------------------------------------------------


@dataclass(frozen=True)
class Placed:
    """One resolved synthetic design, ready to bind against."""

    engine: Engine
    point: DesignPoint
    inputs: HardwareInputs
    declarations: dict[str, HardwareKernelDeclaration]
    selection: HardwareKernelSelection
    specification: DesignSpaceSpec

    def value(self, handle: Ref[object]) -> object:
        answer = self.engine.query_property(self.point, handle.path)
        assert isinstance(answer, Decided)
        return answer.value

    def region(self, handle: Ref[DataflowRegion]) -> DataflowRegion:
        return cast(DataflowRegion, self.value(cast("Ref[object]", handle)))

    @property
    def producer(self) -> DataflowRegion:
        return self.region(self.inputs.producer)

    @property
    def consumer(self) -> DataflowRegion:
        return self.region(self.inputs.consumer)

    def bind(
        self,
        name: str,
        regions: dict[str, BoundRegion],
        edges: dict[str, str] | None = None,
    ) -> Answer[HardwareKernel]:
        return bind_hardware_kernel(
            self.engine, self.declarations[name], self.point, regions, edges
        )


def _place(
    extent: int = 4,
    lanes: int = 2,
    *,
    family: str = "wide",
    pipelined: bool = False,
    hardware_kernel: str | None = "single",
) -> Placed:
    design, inputs, element_type = _semantics()
    kernels = {
        kernel.id: declare_hardware_kernel(kernel, hardware_namespace(OWNER, kernel.id), inputs)[0]
        for kernel in (UpstreamKernel, DownstreamKernel, FusedKernel, MiscomputingKernel)
    }
    # Case 5's two alternatives live behind a selection instead, because
    # choosing between them is a real choice; the others have exactly one
    # coverer each and so get no decision at all.
    selection = HardwareKernelSelection(
        f"{OWNER}.compute",
        tuple(
            declare_hardware_kernel(kernel, hardware_namespace(OWNER, kernel.id), inputs)[0]
            for kernel in (SingleComponentKernel, MultiComponentKernel)
        ),
    )
    specification = assemble_specs(
        (design.spec(), *(item.spec for item in kernels.values()), selection.build_spec())
    )
    check_declared_references(specification, (*kernels.values(), *selection.kernels))
    engine = Engine()
    space = engine.validate(specification)
    point = engine.start(
        space,
        {
            inputs.extent.path: extent,
            element_type: INT8,
            inputs.target_family.path: family,
        },
    )
    assignments: dict[QualifiedPath, object] = {inputs.lanes.path: lanes}
    if hardware_kernel is not None:
        # ``None`` leaves the physical choice open, which is the state a policy
        # asks ``supported_kernels`` in.
        assignments[selection.kernel_path] = hardware_kernel
        assignments[QualifiedPath(f"{hardware_namespace(OWNER, 'multi')}.pipelined")] = pipelined
    point = engine.commit_assignments(point, assignments).point
    return Placed(engine, point, inputs, kernels, selection, specification)


def _compute_role(placed: Placed) -> dict[str, BoundRegion]:
    return bound_regions((("compute", CONSUMER_NODE, placed.consumer),))


def _both_roles(placed: Placed) -> dict[str, BoundRegion]:
    return bound_regions(
        (
            ("producer", PRODUCER_NODE, placed.producer),
            ("consumer", CONSUMER_NODE, placed.consumer),
        )
    )


# -- case 1 ------------------------------------------------------------------


def test_one_region_one_kernel_one_component() -> None:
    placed = _place(hardware_kernel="single")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    binding = bound.value

    assert binding.kernel_id == "single"
    assert binding.node_ids == (CONSUMER_NODE,)
    assert binding.edge_ids == ()
    assert len(binding.components()) == 1
    assert binding.components()[0].module == "example.single"


def test_a_bound_kernel_is_an_instance_of_the_class_that_declared_it() -> None:
    placed = _place(hardware_kernel="single")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    assert isinstance(bound.value, SingleComponentKernel)


def test_no_public_kernel_binding_wrapper_remains() -> None:
    assert "KernelBinding" not in hardware.__all__
    assert not hasattr(hardware, "KernelBinding")


def test_every_declared_parameter_resolves_from_the_point() -> None:
    """A decision, a declared width property, and a constant."""

    placed = _place(lanes=2, hardware_kernel="single")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    assert dict(bound.value.parameters) == {"LANES": 2, "WIDTH": 8, "MODE": 0}


# -- case 2 ------------------------------------------------------------------


def test_one_region_one_kernel_several_components() -> None:
    placed = _place(hardware_kernel="multi")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    components = bound.value.components()

    assert len(components) == 3
    assert {item.id for item in components} == {"multi.shell", "multi.core", "multi.buffer"}
    # The shell is the one with no parent; the other two hang off it.
    assert {item.parent for item in components} == {None, "multi.shell"}


def test_a_kernel_local_choice_reaches_its_derived_parameter() -> None:
    """``pipelined`` is physical: it doubles the buffer and moves no beat."""

    plain = _place(lanes=2, pipelined=False, hardware_kernel="multi")
    piped = _place(lanes=2, pipelined=True, hardware_kernel="multi")
    plain_bound = plain.selection.bind(plain.engine, plain.point, _compute_role(plain))
    piped_bound = piped.selection.bind(piped.engine, piped.point, _compute_role(piped))
    assert isinstance(plain_bound, Decided) and isinstance(piped_bound, Decided)

    assert dict(plain_bound.value.parameters)["DEPTH"] == 2
    assert dict(piped_bound.value.parameters)["DEPTH"] == 4
    # ...and the Region is untouched by it.
    assert plain.consumer == piped.consumer


# -- case 3 ------------------------------------------------------------------


def test_two_regions_two_independent_kernels() -> None:
    placed = _place()
    upstream = placed.bind(
        "upstream", bound_regions((("producer", PRODUCER_NODE, placed.producer),))
    )
    downstream = placed.bind(
        "downstream", bound_regions((("consumer", CONSUMER_NODE, placed.consumer),))
    )
    assert isinstance(upstream, Decided) and isinstance(downstream, Decided)

    assert upstream.value.node_ids == (PRODUCER_NODE,)
    assert downstream.value.node_ids == (CONSUMER_NODE,)
    # Neither absorbs the edge, so it remains a connection between them.
    assert upstream.value.edge_ids == () and downstream.value.edge_ids == ()


def test_neither_kernel_needs_a_selection_when_it_is_the_only_coverer() -> None:
    """No gratuitous decision: the upstream binding is derived, not chosen."""

    placed = _place()
    decisions = {str(item.path) for item in placed.specification.decisions}
    assert not any("upstream" in path or "downstream" in path for path in decisions)
    # The one pool that does have a choice still has exactly one.
    assert sum("hardware_kernel" in path for path in decisions) == 1


# -- case 4 ------------------------------------------------------------------


def test_two_regions_and_their_edge_covered_by_one_fused_kernel() -> None:
    placed = _place()
    fused = placed.bind("fused", _both_roles(placed), {"link": LINK_EDGE})
    assert isinstance(fused, Decided)

    assert fused.value.node_ids == (CONSUMER_NODE, PRODUCER_NODE)
    assert fused.value.edge_ids == (LINK_EDGE,)
    assert len(fused.value.components()) == 1


def test_the_fused_kernel_covers_the_same_regions_the_separate_ones_do() -> None:
    """One semantic result, two physical readings of it."""

    placed = _place()
    separate = tuple(
        cast(Decided[HardwareKernel], placed.bind(name, bound_regions((role,)))).value
        for name, role in (
            ("upstream", ("producer", PRODUCER_NODE, placed.producer)),
            ("downstream", ("consumer", CONSUMER_NODE, placed.consumer)),
        )
    )
    fused = cast(
        Decided[HardwareKernel], placed.bind("fused", _both_roles(placed), {"link": LINK_EDGE})
    ).value

    assert tuple(sorted(item.node_ids[0] for item in separate)) == tuple(sorted(fused.node_ids))
    assert tuple(item.region for item in fused.regions.values()) == (
        placed.producer,
        placed.consumer,
    )


def test_a_fused_kernel_validates_the_edge_against_the_selected_network() -> None:
    """An edge id is a claim until the Network is asked."""

    placed = _place()
    absent = placed.bind("fused", _both_roles(placed), {"link": "no_such_edge"})
    assert isinstance(absent, Unresolved)
    assert any(item.code == "hardware-covered-edge-absent" for item in absent.findings)


def test_a_fused_kernel_rejects_an_edge_that_runs_the_wrong_way() -> None:
    """The roles are swapped, so the real edge no longer connects them."""

    placed = _place()
    swapped = bound_regions(
        (
            ("producer", CONSUMER_NODE, placed.consumer),
            ("consumer", PRODUCER_NODE, placed.producer),
        )
    )
    reversed_binding = placed.bind("fused", swapped, {"link": LINK_EDGE})
    assert isinstance(reversed_binding, Unresolved)
    codes = {item.code for item in reversed_binding.findings}
    assert "hardware-coverage-region-not-the-declared-one" in codes
    assert "hardware-covered-edge-misconnected" in codes


# -- case 5 ------------------------------------------------------------------


def test_two_alternatives_over_one_region_leave_the_region_unchanged() -> None:
    single = _place(hardware_kernel="single")
    multi = _place(hardware_kernel="multi")

    assert single.consumer == multi.consumer
    assert single.producer == multi.producer


def test_the_committed_alternative_is_the_one_that_binds() -> None:
    for kernel_id, components in (("single", 1), ("multi", 3)):
        placed = _place(hardware_kernel=kernel_id)
        bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
        assert isinstance(bound, Decided)
        assert bound.value.kernel_id == kernel_id
        assert len(bound.value.components()) == components


def _pool(*kernels: type[HardwareKernel]) -> tuple[str, ...]:
    """Build a pool from these Kernels and return why it was refused, if it was."""

    _, inputs, _ = _semantics()
    declarations = tuple(
        declare_hardware_kernel(kernel, hardware_namespace(OWNER, kernel.id), inputs)[0]
        for kernel in kernels
    )
    try:
        HardwareKernelSelection("example.pool", declarations)
    except SpecAuthoringError as error:
        return tuple(item.message for item in error.issues)
    return ()


def test_a_pool_refuses_members_that_cover_different_roles() -> None:
    """Two Kernels behind one decision must be alternatives, not two bindings."""

    reasons = _pool(SingleComponentKernel, FusedKernel)
    assert reasons
    assert any("covered by only one of them" in item for item in reasons)


def test_a_pool_refuses_members_that_implement_different_arithmetic() -> None:
    """The defect matching role names alone let through.

    ``MiscomputingKernel`` covers the same role over the same Region
    declaration as ``SingleComponentKernel``.  Only the contract differs, and
    admitting it would let the physical choice change what the design computes
    -- which is precisely what selecting a Kernel may never do.
    """

    reasons = _pool(SingleComponentKernel, MiscomputingKernel)
    assert reasons
    assert any("implements running_maximum:1 versus scaled_copy:1" in item for item in reasons)


def test_a_pool_refuses_members_that_cover_different_region_declarations() -> None:
    """Same role name, different Region: still not alternatives."""

    reasons = _pool(SingleComponentKernel, ElsewhereKernel)
    assert reasons
    assert any("producer_region" in item and "consumer_region" in item for item in reasons)


def test_a_pool_accepts_genuine_alternatives() -> None:
    """The positive case, so the check above is not merely refusing everything."""

    assert _pool(SingleComponentKernel, MultiComponentKernel) == ()


def test_policy_can_eliminate_unsupported_kernels_before_binding() -> None:
    """The named coverage set answers "what can this target build" without binding."""

    wide = _place(family="wide", hardware_kernel=None)
    narrow = _place(family="narrow", hardware_kernel=None)

    assert wide.selection.supported_kernels(wide.engine, wide.point) == ("single", "multi")
    # ``multi`` requires the wide family; ``single`` declares no condition.
    assert narrow.selection.supported_kernels(narrow.engine, narrow.point) == ("single",)


def test_once_the_choice_is_committed_there_is_nothing_left_to_eliminate() -> None:
    """Reporting rejected peers to a policy past switching would mislead it."""

    placed = _place(family="wide", hardware_kernel="multi")
    assert placed.selection.supported_kernels(placed.engine, placed.point) == ("multi",)


def test_the_coverage_constraint_set_is_named_in_the_design_space() -> None:
    placed = _place()
    names = {item.name for item in placed.specification.constraint_sets}
    assert placed.selection.coverage_constraint_set in names


# -- coverage is a declaration, not a shape ----------------------------------


def test_an_equal_looking_region_in_the_wrong_role_is_rejected() -> None:
    """The two Regions are schedule-identical; only the declaration separates them."""

    placed = _place()
    assert placed.producer.schedule == placed.consumer.schedule
    assert placed.producer != placed.consumer

    misrouted = placed.bind(
        "upstream", bound_regions((("producer", CONSUMER_NODE, placed.consumer),))
    )
    assert isinstance(misrouted, Unresolved)
    assert any(
        item.code == "hardware-coverage-region-not-the-declared-one" for item in misrouted.findings
    )


def test_the_explicitly_declared_region_binds() -> None:
    placed = _place()
    correct = placed.bind(
        "upstream", bound_regions((("producer", PRODUCER_NODE, placed.producer),))
    )
    assert isinstance(correct, Decided)
    assert correct.value.regions["producer"].region == placed.producer


def test_the_computation_contract_is_checked_rather_than_inferred() -> None:
    """Right Region, right schedule, wrong arithmetic.

    ``MiscomputingKernel`` covers the declared consumer Region, so every
    structural check passes.  What refuses it is the contract it says it
    implements.
    """

    placed = _place()
    wrong = placed.bind("miscomputing", _compute_role(placed))
    assert isinstance(wrong, Unresolved)
    assert any(item.code == "hardware-computation-contract-mismatch" for item in wrong.findings)


def test_a_binding_must_fill_every_covered_role_and_no_other() -> None:
    placed = _place()
    missing = placed.bind(
        "fused",
        bound_regions((("producer", PRODUCER_NODE, placed.producer),)),
        {"link": LINK_EDGE},
    )
    assert isinstance(missing, Unresolved)
    assert any(item.code == "hardware-coverage-region-roles-mismatch" for item in missing.findings)


def test_a_binding_must_fill_every_covered_edge() -> None:
    """A fused Kernel bound without its edge would silently become two."""

    placed = _place()
    without = placed.bind("fused", _both_roles(placed))
    assert isinstance(without, Unresolved)
    assert any(item.code == "hardware-coverage-edge-roles-mismatch" for item in without.findings)


def test_an_absorbed_edge_must_run_between_covered_roles() -> None:
    _, inputs, _ = _semantics()

    class Disconnected(HardwareKernel):
        id = "disconnected"

        @classmethod
        def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
            facts = design.inputs
            design.covers_region(
                "producer",
                region=facts.producer,
                computation=facts.producer_computation,
                implements=SCALED,
            )
            design.absorbs_edge(
                "link", network=facts.network, source_role="producer", sink_role="elsewhere"
            )

    with pytest.raises(SpecAuthoringError) as raised:
        declare_hardware_kernel(Disconnected, "example.disconnected", inputs)
    assert any(item.code == "coverage-edge-role-unknown" for item in raised.value.issues)


# -- refusal reasons survive -------------------------------------------------


def test_a_kernel_that_does_not_cover_the_point_refuses_with_its_own_reason() -> None:
    placed = _place(extent=16, lanes=2)
    refused = placed.bind("fused", _both_roles(placed), {"link": LINK_EDGE})
    assert isinstance(refused, Unresolved)
    codes = {item.code for item in refused.findings}

    assert "hardware-coverage-refused" in codes
    # The reason the author wrote is what makes the refusal actionable, so it
    # travels alongside the summary rather than being replaced by it.
    assert "extent-too-wide" in codes
    assert any("stops at 8" in item.message for item in refused.findings)


# -- parameters are declared, and only declared ------------------------------


def test_an_undeclared_parameter_reference_is_caught_when_the_space_is_assembled() -> None:
    """``Engine.validate`` cannot see this: the path is read, not declared."""

    design, inputs, _ = _semantics()
    stray: Ref[object] = cast(
        "Ref[object]", design.derived("stray", int, dependencies={}, evaluate=lambda: 1)
    )

    class Reaching(HardwareKernel):
        id = "reaching"

        @classmethod
        def define_design(cls, kernel_design: HardwareDesign[HardwareInputs]) -> None:
            facts = kernel_design.inputs
            kernel_design.covers_region(
                "compute",
                region=facts.consumer,
                computation=facts.consumer_computation,
                implements=REDUCED,
            )
            kernel_design.parameter("STRAY", stray)

    declaration = declare_hardware_kernel(Reaching, "example.reaching", inputs)[0]
    # Assemble *without* the scope that declares ``stray``.
    specification = assemble_specs((declaration.spec,))
    with pytest.raises(SpecAuthoringError) as raised:
        check_declared_references(specification, (declaration,))
    assert any(item.code == "hardware-reference-not-assembled" for item in raised.value.issues)


def test_binding_reports_an_undeclared_reference_rather_than_raising() -> None:
    """Belt and braces: if the assembly check was skipped, binding still answers."""

    design, inputs, _ = _semantics()
    stray: Ref[object] = cast(
        "Ref[object]", design.derived("stray", int, dependencies={}, evaluate=lambda: 1)
    )

    class Reaching(HardwareKernel):
        id = "reaching"

        @classmethod
        def define_design(cls, kernel_design: HardwareDesign[HardwareInputs]) -> None:
            facts = kernel_design.inputs
            kernel_design.covers_region(
                "compute",
                region=facts.consumer,
                computation=facts.consumer_computation,
                implements=REDUCED,
            )
            kernel_design.parameter("STRAY", stray)

    declaration = declare_hardware_kernel(Reaching, "example.reaching", inputs)[0]
    placed = _place()
    answer = bind_hardware_kernel(placed.engine, declaration, placed.point, _compute_role(placed))
    assert isinstance(answer, Unresolved)
    assert any(item.code == "hardware-reference-not-declared" for item in answer.findings)


def _reaching(source: Ref[object], inputs: HardwareInputs) -> HardwareKernelDeclaration:
    """One Kernel whose only interesting feature is the reference it makes."""

    class Reaching(HardwareKernel):
        id = "reaching"

        @classmethod
        def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
            facts = design.inputs
            design.covers_region(
                "compute",
                region=facts.consumer,
                computation=facts.consumer_computation,
                implements=REDUCED,
            )
            design.parameter("VALUE", source)

    return declare_hardware_kernel(Reaching, "example.reaching", inputs)[0]


def test_a_reference_naming_the_wrong_dependency_kind_is_refused() -> None:
    """A DECISION handle onto a derived property.

    The path exists, so a path-only check accepts it -- and then the value looks
    like a decision nobody assigned, which is a symptom with no obvious cause.
    """

    design, inputs, _ = _semantics()
    mislabelled: Ref[object] = Ref(
        inputs.element_width.path, DependencyKind.DECISION, inputs.element_width.semantics
    )
    declaration = _reaching(mislabelled, inputs)
    specification = assemble_specs((design.spec(), declaration.spec))

    with pytest.raises(SpecAuthoringError) as raised:
        check_declared_references(specification, (declaration,))
    issue = next(
        item for item in raised.value.issues if item.code == "hardware-reference-wrong-kind"
    )
    assert "derived_property" in issue.message


def test_a_reference_under_a_false_type_contract_is_refused() -> None:
    """A ``Ref[str]`` onto an integer property would deliver an int as a str."""

    design, inputs, _ = _semantics()
    mistyped: Ref[object] = Ref(
        inputs.element_width.path, DependencyKind.PROPERTY, semantics_for(str)
    )
    declaration = _reaching(mistyped, inputs)
    specification = assemble_specs((design.spec(), declaration.spec))

    with pytest.raises(SpecAuthoringError) as raised:
        check_declared_references(specification, (declaration,))
    assert any(item.code == "hardware-reference-wrong-type" for item in raised.value.issues)


def test_a_well_formed_reference_passes_the_same_check() -> None:
    design, inputs, _ = _semantics()
    declaration = _reaching(cast("Ref[object]", inputs.element_width), inputs)
    check_declared_references(assemble_specs((design.spec(), declaration.spec)), (declaration,))


def test_coverage_references_are_checked_too_not_only_parameters() -> None:
    """A Kernel covering a Region the assembly never declared is as broken."""

    _, inputs, _ = _semantics()
    declaration = _reaching(cast("Ref[object]", inputs.element_width), inputs)
    # Assemble the Kernel alone: its covered Region and computation handles now
    # name nothing.
    with pytest.raises(SpecAuthoringError) as raised:
        check_declared_references(assemble_specs((declaration.spec,)), (declaration,))
    missing = {item.path for item in raised.value.issues if "coverage" in item.message}
    assert missing == {
        "semantic.example.op.consumer_region",
        "semantic.example.op.consumer_computation",
    }


def test_a_non_scalar_parameter_is_refused_rather_than_stringified() -> None:
    """An element type reaching a parameter means a width property was owed."""

    with pytest.raises(SpecAuthoringError) as raised:
        scalar_parameters({"WIDTH": INT8})
    assert any(item.code == "hardware-parameter-not-scalar" for item in raised.value.issues)


def test_a_parameter_ownership_is_read_off_its_handle() -> None:
    """Nothing restates where a value comes from, so nothing can misstate it."""

    _, inputs, _ = _semantics()
    declaration = declare_hardware_kernel(
        SingleComponentKernel, hardware_namespace(OWNER, "single"), inputs
    )[0]
    ownership = {item.name: item.ownership for item in declaration.parameters}
    assert ownership == {
        "LANES": "decision",
        "WIDTH": "derived_property",
        "MODE": "constant",
    }


def test_a_constant_parameter_must_say_why_it_is_one() -> None:
    _, inputs, _ = _semantics()

    class Nameless(HardwareKernel):
        id = "nameless"

        @classmethod
        def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
            facts = design.inputs
            design.covers_region(
                "compute",
                region=facts.consumer,
                computation=facts.consumer_computation,
                implements=REDUCED,
            )
            design.constant("MODE", 0, why="")

    with pytest.raises(AuthoringError):
        declare_hardware_kernel(Nameless, "example.nameless", inputs)


# -- what a bound Kernel may see ---------------------------------------------


def test_a_bound_kernel_does_not_carry_the_design_point() -> None:
    """Elaboration that could read an undeclared field would decide out of band."""

    placed = _place(hardware_kernel="multi")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    kernel = bound.value

    assert not hasattr(kernel, "point")
    assert not hasattr(bound.value, "point")
    assert not hasattr(kernel, "engine")


def test_a_bound_kernel_sees_only_its_own_committed_choices() -> None:
    placed = _place(hardware_kernel="multi", lanes=2)
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    paths = {str(path) for path in bound.value.assignments}

    # Its own physical choice, and nothing of the semantics that configured it.
    assert paths == {f"{hardware_namespace(OWNER, 'multi')}.pipelined"}
    assert not any(path.endswith(".lanes") for path in paths)


def test_an_imported_value_reaches_elaboration_only_as_a_declared_parameter() -> None:
    placed = _place(hardware_kernel="multi", lanes=2)
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)

    assert dict(bound.value.parameters)["LANES"] == 2
    assert set(bound.value.parameters) == {"LANES", "DEPTH", "PIPELINED"}


def test_a_physical_kernel_cannot_declare_a_region() -> None:
    """The scope simply has no way to; this pins that it stays that way."""

    assert not hasattr(HardwareDesign, "region")
    assert not hasattr(HardwareDesign, "demand")
    assert not hasattr(HardwareDesign, "export")


# -- evidence ----------------------------------------------------------------


def test_the_origin_records_what_the_binding_is_without_the_graph() -> None:
    placed = _place(hardware_kernel="multi")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    origin = bound.value.origin()

    assert origin.kernel_id == "multi"
    assert origin.kernel_version == "3"
    assert origin.covered_nodes == (CONSUMER_NODE,)
    assert origin.computations == (("compute", "running_maximum:1"),)
    assert dict(origin.parameters)["LANES"] == 2
    assert ("example", "rtl/multi.sv") in origin.sources
