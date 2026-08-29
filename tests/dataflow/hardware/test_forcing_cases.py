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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.dataflow.authoring import (
    AuthoringError,
    OpDesign,
    Ref,
    assemble_specs,
    divisors_of,
    finite,
    reject,
)
from finn.dataflow.design import (
    Decided,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.hardware import (
    HardwareDesign,
    HardwareKernel,
    HardwareKernelDeclaration,
    HardwareKernelSelection,
    KernelBinding,
    PhysicalComponent,
    bind_hardware_kernel,
    bound_regions,
    declare_hardware_kernel,
    hardware_namespace,
    scalar_parameters,
)
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.spec_algebra import SpecAuthoringError

INT8 = NumericElementType("int", 8)
OWNER = "example"

#: The two node ids the synthetic assembly uses.  A Kernel never sees these --
#: it declares roles, and the assembly says which node fills each one.
PRODUCER_NODE = "upstream"
CONSUMER_NODE = "downstream"
LINK_EDGE = "producer_to_consumer"


# -- synthetic semantics -----------------------------------------------------


def _region(extent: int, operand: str) -> DataflowRegion:
    """One tiny Region, parameterized only by what the folding fixes."""

    schedule = LogicalSchedule((ScheduleLevel("element", extent),))
    source = Operand(f"{operand}_in", INT8, (extent,))
    result = Operand(f"{operand}_out", INT8, (extent,))
    beats = BeatSequence(1, tuple(((index,),) for index in range(extent)))
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("input", source, beats),
                ScheduledInputRequirements({((index,), (index,)): 1 for index in range(extent)}),
            ),
        ),
        (
            OutputInterface(
                Port("output", result, beats),
                ScheduledOutputAvailability({(index,): (index,) for index in range(extent)}),
            ),
        ),
    )


@dataclass(frozen=True)
class HardwareInputs:
    """What a physical Kernel in this example is allowed to read.

    ``lanes`` is the Region-affecting decision, declared once by the semantics
    and imported here.  A Kernel that declared its own would be choosing a fold
    the Region was already built from.
    """

    extent: Ref[int]
    element_type: Ref[NumericElementType]
    target_family: Ref[str]
    lanes: Ref[int]


@dataclass(frozen=True)
class Semantics:
    """The logical half: two Regions and the values they were built from."""

    design: OpDesign
    inputs: HardwareInputs
    producer: Ref[DataflowRegion]
    consumer: Ref[DataflowRegion]


def _semantics() -> Semantics:
    design = OpDesign("example.op", problem_namespace=OWNER)
    extent = design.graph_fact("extent", int)
    element_type = design.graph_fact("element_type", NumericElementType)
    target_family = design.target_fact("family", str, required=False)
    lanes = design.decision("lanes", int, domain=divisors_of(extent))
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
    return Semantics(
        design,
        HardwareInputs(extent, element_type, target_family, lanes),
        producer,
        consumer,
    )


# -- the Kernels -------------------------------------------------------------


class SingleComponentKernel(HardwareKernel):
    """Case 1: one covered Region, one instantiated module."""

    id = "single"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        facts = design.inputs
        design.covers("compute")
        design.source("example", "rtl/single.sv")
        design.parameter("LANES", cast("Ref[object]", facts.lanes))
        design.parameter("WIDTH", cast("Ref[object]", facts.element_type))
        design.constant("MODE", 0, why="this core has one mode; the parameter is vestigial")

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
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
        design.covers("compute")
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
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
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
        design.covers("producer")
        design.source("example", "rtl/upstream.sv")
        design.parameter("LANES", cast("Ref[object]", design.inputs.lanes))

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("upstream.core", "example.upstream"),)


class DownstreamKernel(HardwareKernel):
    """Case 3, second half: covers the consumer Region and nothing else."""

    id = "downstream"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        design.covers("consumer")
        design.source("example", "rtl/downstream.sv")
        design.parameter("LANES", cast("Ref[object]", design.inputs.lanes))

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("downstream.core", "example.downstream"),)


class FusedKernel(HardwareKernel):
    """Case 4: both Regions and the edge between them, as one module.

    The edge coverage is the whole point.  Two adjacent Kernels leave the
    connection as a physical interface pair; this absorbs it, and the only
    thing that says so is the declaration.
    """

    id = "fused"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
        design.covers("producer", "consumer", edges=("link",))
        design.source("example", "rtl/fused.sv")
        design.parameter("LANES", cast("Ref[object]", design.inputs.lanes))
        design.coverage_constraint(
            "extent_supported",
            dependencies={"extent": design.inputs.extent},
            evaluate=lambda extent: (
                True if extent <= 8 else reject("extent-too-wide", "the fused core stops at 8")
            ),
        )

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        return (PhysicalComponent("fused.core", "example.fused"),)


# -- assembly ----------------------------------------------------------------


@dataclass(frozen=True)
class Placed:
    """One resolved synthetic design, ready to bind against."""

    engine: Engine
    point: DesignPoint
    semantics: Semantics
    declarations: dict[str, HardwareKernelDeclaration]
    selection: HardwareKernelSelection
    specification: DesignSpaceSpec

    def region(self, handle: Ref[DataflowRegion]) -> DataflowRegion:
        answer = self.engine.query_property(self.point, handle.path)
        assert isinstance(answer, Decided)
        return cast(DataflowRegion, answer.value)


def _place(
    extent: int = 4,
    lanes: int = 2,
    *,
    family: str = "wide",
    pipelined: bool = False,
    hardware_kernel: str = "single",
) -> Placed:
    semantics = _semantics()
    kernels = {
        kernel.id: declare_hardware_kernel(
            kernel, hardware_namespace(OWNER, kernel.id), semantics.inputs
        )[0]
        for kernel in (UpstreamKernel, DownstreamKernel, FusedKernel)
    }
    # Case 5's two alternatives live behind a selection instead, because
    # choosing between them is a real choice; the other three have exactly one
    # coverer each and so get no decision at all.
    selection = HardwareKernelSelection(
        f"{OWNER}.compute",
        tuple(
            declare_hardware_kernel(kernel, hardware_namespace(OWNER, kernel.id), semantics.inputs)[
                0
            ]
            for kernel in (SingleComponentKernel, MultiComponentKernel)
        ),
    )
    engine = Engine()
    specification = assemble_specs(
        (
            semantics.design.spec(),
            *(item.spec for item in kernels.values()),
            selection.build_spec(),
        )
    )
    space = engine.validate(specification)
    point = engine.start(
        space,
        {
            semantics.inputs.extent.path: extent,
            semantics.inputs.element_type.path: INT8,
            semantics.inputs.target_family.path: family,
        },
    )
    point = engine.commit_assignments(
        point,
        {
            semantics.inputs.lanes.path: lanes,
            selection.kernel_path: hardware_kernel,
            QualifiedPath(f"{hardware_namespace(OWNER, 'multi')}.pipelined"): pipelined,
        },
    ).point
    return Placed(engine, point, semantics, kernels, selection, specification)


def _compute_role(placed: Placed) -> dict[str, object]:
    return bound_regions((("compute", CONSUMER_NODE, placed.region(placed.semantics.consumer)),))


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
    assert isinstance(bound.value.kernel, SingleComponentKernel)


def test_every_declared_parameter_resolves_from_the_point() -> None:
    """A decision, a problem field projected to its width, and a constant."""

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
    assert plain.region(plain.semantics.consumer) == piped.region(piped.semantics.consumer)


# -- case 3 ------------------------------------------------------------------


def test_two_regions_two_independent_kernels() -> None:
    placed = _place()
    upstream = bind_hardware_kernel(
        placed.engine,
        placed.declarations["upstream"],
        placed.point,
        bound_regions((("producer", PRODUCER_NODE, placed.region(placed.semantics.producer)),)),
    )
    downstream = bind_hardware_kernel(
        placed.engine,
        placed.declarations["downstream"],
        placed.point,
        bound_regions((("consumer", CONSUMER_NODE, placed.region(placed.semantics.consumer)),)),
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
    fused = bind_hardware_kernel(
        placed.engine,
        placed.declarations["fused"],
        placed.point,
        bound_regions(
            (
                ("producer", PRODUCER_NODE, placed.region(placed.semantics.producer)),
                ("consumer", CONSUMER_NODE, placed.region(placed.semantics.consumer)),
            )
        ),
        {"link": LINK_EDGE},
    )
    assert isinstance(fused, Decided)

    assert fused.value.node_ids == (CONSUMER_NODE, PRODUCER_NODE)
    assert fused.value.edge_ids == (LINK_EDGE,)
    assert len(fused.value.components()) == 1


def test_the_fused_kernel_covers_the_same_regions_the_separate_ones_do() -> None:
    """One semantic result, two physical readings of it."""

    placed = _place()
    producer = placed.region(placed.semantics.producer)
    consumer = placed.region(placed.semantics.consumer)
    separate = tuple(
        cast(
            Decided[KernelBinding],
            bind_hardware_kernel(
                placed.engine, placed.declarations[name], placed.point, bound_regions((role,))
            ),
        ).value
        for name, role in (
            ("upstream", ("producer", PRODUCER_NODE, producer)),
            ("downstream", ("consumer", CONSUMER_NODE, consumer)),
        )
    )
    fused = cast(
        Decided[KernelBinding],
        bind_hardware_kernel(
            placed.engine,
            placed.declarations["fused"],
            placed.point,
            bound_regions(
                (
                    ("producer", PRODUCER_NODE, producer),
                    ("consumer", CONSUMER_NODE, consumer),
                )
            ),
            {"link": LINK_EDGE},
        ),
    ).value

    covered_separately = tuple(sorted(item.node_ids[0] for item in separate))
    assert covered_separately == tuple(sorted(fused.node_ids))
    assert tuple(item.region for item in fused.regions) == (producer, consumer)


# -- case 5 ------------------------------------------------------------------


def test_two_alternatives_over_one_region_leave_the_region_unchanged() -> None:
    single = _place(hardware_kernel="single")
    multi = _place(hardware_kernel="multi")

    assert single.region(single.semantics.consumer) == multi.region(multi.semantics.consumer)
    assert single.region(single.semantics.producer) == multi.region(multi.semantics.producer)


def test_the_committed_alternative_is_the_one_that_binds() -> None:
    for kernel_id, components in (("single", 1), ("multi", 3)):
        placed = _place(hardware_kernel=kernel_id)
        bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
        assert isinstance(bound, Decided)
        assert bound.value.kernel_id == kernel_id
        assert len(bound.value.components()) == components


def test_a_pool_refuses_members_that_cover_different_shapes() -> None:
    """Two Kernels behind one decision must be alternatives, not two bindings."""

    semantics = _semantics()
    declarations = tuple(
        declare_hardware_kernel(kernel, hardware_namespace(OWNER, kernel.id), semantics.inputs)[0]
        for kernel in (SingleComponentKernel, FusedKernel)
    )
    with pytest.raises(SpecAuthoringError) as raised:
        HardwareKernelSelection("example.mixed", declarations)
    assert any(item.code == "hardware-selection-coverage-differs" for item in raised.value.issues)


# -- the contract itself -----------------------------------------------------


def test_a_binding_must_fill_every_covered_role_and_no_other() -> None:
    placed = _place()
    missing = bind_hardware_kernel(
        placed.engine,
        placed.declarations["fused"],
        placed.point,
        bound_regions((("producer", PRODUCER_NODE, placed.region(placed.semantics.producer)),)),
        {"link": LINK_EDGE},
    )
    assert isinstance(missing, Unresolved)
    assert any(item.code == "hardware-coverage-region-roles-mismatch" for item in missing.findings)


def test_a_binding_must_fill_every_covered_edge() -> None:
    """A fused Kernel bound without its edge would silently become two."""

    placed = _place()
    without = bind_hardware_kernel(
        placed.engine,
        placed.declarations["fused"],
        placed.point,
        bound_regions(
            (
                ("producer", PRODUCER_NODE, placed.region(placed.semantics.producer)),
                ("consumer", CONSUMER_NODE, placed.region(placed.semantics.consumer)),
            )
        ),
    )
    assert isinstance(without, Unresolved)
    assert any(item.code == "hardware-coverage-edge-roles-mismatch" for item in without.findings)


def test_a_kernel_that_does_not_cover_the_point_refuses_to_bind() -> None:
    placed = _place(extent=16, lanes=2)
    refused = bind_hardware_kernel(
        placed.engine,
        placed.declarations["fused"],
        placed.point,
        bound_regions(
            (
                ("producer", PRODUCER_NODE, placed.region(placed.semantics.producer)),
                ("consumer", CONSUMER_NODE, placed.region(placed.semantics.consumer)),
            )
        ),
        {"link": LINK_EDGE},
    )
    assert isinstance(refused, Unresolved)
    assert any(item.code == "hardware-coverage-refused" for item in refused.findings)


def test_coverage_is_stated_rather_than_inferred_from_equal_shapes() -> None:
    """The two synthetic Regions differ only in operand names.

    Binding the wrong one is therefore not a shape error, and nothing in the
    Kernel could detect it -- which is exactly why the assembly states the
    association instead of the Kernel guessing it.
    """

    placed = _place()
    producer = placed.region(placed.semantics.producer)
    consumer = placed.region(placed.semantics.consumer)
    assert producer != consumer
    assert producer.schedule == consumer.schedule

    bound = bind_hardware_kernel(
        placed.engine,
        placed.declarations["upstream"],
        placed.point,
        bound_regions((("producer", CONSUMER_NODE, consumer),)),
    )
    # It binds: the Kernel was told this node holds the producer role, and it
    # has no standing to disagree.  What it records is exactly what it was told.
    assert isinstance(bound, Decided)
    assert bound.value.regions[0].node_id == CONSUMER_NODE


def test_a_physical_kernel_cannot_declare_a_region() -> None:
    """The scope simply has no way to; this pins that it stays that way."""

    assert not hasattr(HardwareDesign, "region")
    assert not hasattr(HardwareDesign, "demand")
    assert not hasattr(HardwareDesign, "export")


def test_a_constant_parameter_must_say_why_it_is_one() -> None:
    semantics = _semantics()

    class Nameless(HardwareKernel):
        id = "nameless"

        @classmethod
        def define_design(cls, design: HardwareDesign[HardwareInputs]) -> None:
            design.covers("compute")
            design.constant("MODE", 0, why="")

    with pytest.raises(AuthoringError):
        declare_hardware_kernel(Nameless, "example.nameless", semantics.inputs)


def test_a_parameter_ownership_is_read_off_its_handle() -> None:
    """Nothing restates where a value comes from, so nothing can misstate it."""

    semantics = _semantics()
    declaration = declare_hardware_kernel(
        SingleComponentKernel, hardware_namespace(OWNER, "single"), semantics.inputs
    )[0]
    ownership = {item.name: item.ownership for item in declaration.parameters}
    assert ownership == {"LANES": "decision", "WIDTH": "problem_field", "MODE": "constant"}
