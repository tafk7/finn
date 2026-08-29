# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 gate: a Kernel is a class its author subclasses.

Two subclasses derive equal Regions from the same operation facts and stay
distinct identities.  Resolving a pool returns an instance of the subclass that
was selected, so a consumer gets the type it asked for without a second noun
between the family and the selected thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.dataflow.authoring import (
    FEASIBILITY,
    SOURCE_ADMISSION,
    AuthoringError,
    Kernel,
    KernelDesign,
    KernelSelection,
    OpDesign,
    Provenance,
    Ref,
    assemble_specs,
    bind_kernel,
    declare_kernel,
    divisors_of,
    finite,
    kernel_namespace,
    reject,
)
import finn.dataflow.authoring as authoring
from finn.dataflow.design import Decided, DesignPoint, Engine, QualifiedPath
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

INT8 = NumericElementType("int", 8)
POOL = "example.compute"


# -- what the operation wires into its pool ----------------------------------


@dataclass(frozen=True)
class ComputeInputs:
    """The typed facts this pool's Kernels are allowed to read."""

    extent: Ref[int]
    element_type: Ref[NumericElementType]
    target_family: Ref[str]


def _region(extent: int, lanes: int) -> DataflowRegion:
    """One Region shape both Kernels derive, so equality is testable."""

    del lanes
    schedule = LogicalSchedule((ScheduleLevel("element", extent),))
    source = Operand("x", INT8, (extent,))
    result = Operand("y", INT8, (extent,))
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


# -- two Kernel subclasses ---------------------------------------------------


class _ExampleComputeKernel(Kernel):
    """Shared authoring for the pool, so the subclasses state only what differs.

    Sharing is a plain base class with a helper, not an inheritance hierarchy
    the engine knows about: each subclass still runs its own ``define_design``
    and owns its own namespace.
    """

    @classmethod
    def _common(cls, design: KernelDesign[ComputeInputs]) -> Ref[int]:
        lanes = design.choice("lanes", int, domain=divisors_of(design.inputs.extent))
        design.region(
            dependencies={"extent": design.inputs.extent, "lanes": lanes},
            evaluate=_region,
        )
        # A pool presents one path per export, so an export is a property even
        # when the value happens to be one member's choice verbatim.
        design.export(
            "lane_count",
            cast(
                Ref[object],
                design.derived(
                    "lane_count", int, dependencies={"lanes": lanes}, evaluate=lambda lanes: lanes
                ),
            ),
        )
        return lanes


class EvenComputeKernel(_ExampleComputeKernel):
    """Serves only even extents, and needs a parameter supplied."""

    id = "even"
    version = "2"

    @classmethod
    def define_design(cls, design: KernelDesign[ComputeInputs]) -> None:
        lanes = cls._common(design)

        def even_extent(extent: int) -> object:
            if extent % 2:
                return reject("extent-is-odd", "this Kernel serves even extents only")
            return True

        design.source_constraint(
            "extent_admitted",
            dependencies={"extent": design.inputs.extent},
            evaluate=even_extent,
        )
        design.feasibility_constraint(
            "lanes_fit",
            dependencies={"extent": design.inputs.extent, "lanes": lanes},
            evaluate=lambda extent, lanes: extent % lanes == 0,
        )
        design.demand(
            "parameter",
            dependencies={"extent": design.inputs.extent, "element": design.inputs.element_type},
            evaluate=lambda extent, element: Port(
                "parameter",
                Operand("w", element, (extent,)),
                BeatSequence(1, tuple(((index,),) for index in range(extent))),
            ),
        )
        design.provider("example.rtl.even")


class AnyComputeKernel(_ExampleComputeKernel):
    """Serves any extent, but only on one target family."""

    id = "any"
    version = "1"

    @classmethod
    def define_design(cls, design: KernelDesign[ComputeInputs]) -> None:
        cls._common(design)
        design.feasibility_constraint(
            "target_supported",
            dependencies={"target": design.inputs.target_family.allow_absent()},
            evaluate=lambda target: target == "wide",
        )
        design.provider("example.rtl.any")
        design.provider("example.hls.any")


# -- the operation that places them ------------------------------------------


def _pool() -> tuple[OpDesign, ComputeInputs, KernelSelection]:
    op = OpDesign("example.op", problem_namespace="example")
    inputs = ComputeInputs(
        extent=op.graph_fact("extent", int),
        element_type=op.graph_fact("element_type", NumericElementType),
        target_family=op.target_fact("family", str, required=False),
    )
    provenance = op.provenance()
    kernels = tuple(
        declare_kernel(kernel, kernel_namespace(POOL, kernel.id), inputs, provenance=provenance)
        for kernel in (EvenComputeKernel, AnyComputeKernel)
    )
    return op, inputs, KernelSelection(POOL, kernels)


def _resolved(
    extent: int, kernel_id: str, lanes: int, *, family: str = "wide"
) -> tuple[Engine, KernelSelection, DesignPoint]:
    op, inputs, selection = _pool()
    engine = Engine()
    space = engine.validate(assemble_specs((op.spec(), selection.build_spec())))
    point = engine.start(
        space,
        {
            inputs.extent.path: extent,
            inputs.element_type.path: INT8,
            inputs.target_family.path: family,
        },
    )
    point = engine.commit_assignments(
        point,
        {
            selection.paths.kernel: kernel_id,
            QualifiedPath(f"{kernel_namespace(POOL, kernel_id)}.lanes"): lanes,
        },
    ).point
    return engine, selection, point


# -- the gate ----------------------------------------------------------------


def test_selecting_a_subclass_returns_an_instance_of_that_subclass() -> None:
    engine, selection, point = _resolved(4, "even", 2)
    bound = bind_kernel(engine, selection, point)
    assert isinstance(bound, Decided)
    assert isinstance(bound.value, EvenComputeKernel)
    assert not isinstance(bound.value, AnyComputeKernel)

    engine, selection, point = _resolved(4, "any", 2)
    other = bind_kernel(engine, selection, point)
    assert isinstance(other, Decided)
    assert isinstance(other.value, AnyComputeKernel)


def test_equal_regions_do_not_collapse_kernel_identity() -> None:
    engine, selection, point = _resolved(4, "even", 2)
    even = cast(Decided[Kernel], bind_kernel(engine, selection, point)).value
    engine, selection, point = _resolved(4, "any", 2)
    any_kernel = cast(Decided[Kernel], bind_kernel(engine, selection, point)).value

    assert even.region == any_kernel.region
    assert even.id != any_kernel.id
    assert even.identity != any_kernel.identity
    assert even.identity.version == "2"


def test_the_instance_carries_its_choices_region_demands_exports_and_providers() -> None:
    engine, selection, point = _resolved(6, "even", 3)
    even = cast(Decided[EvenComputeKernel], bind_kernel(engine, selection, point)).value

    assert list(even.assignments.values()) == [3]
    assert isinstance(even.region, DataflowRegion)
    assert set(even.demands) == {"parameter"}
    assert even.demands["parameter"].operand.shape == (6,)
    assert even.exports["lane_count"] == 3
    assert [item.id for item in even.providers] == ["example.rtl.even"]
    assert all(item.kernel_id == "even" for item in even.providers)


def test_the_other_subclass_demands_nothing_and_lists_its_own_providers() -> None:
    engine, selection, point = _resolved(6, "any", 3)
    bound = cast(Decided[AnyComputeKernel], bind_kernel(engine, selection, point)).value

    assert bound.demands == {}
    assert [item.id for item in bound.providers] == ["example.rtl.any", "example.hls.any"]


def test_there_is_no_second_public_kernel_noun() -> None:
    """The rule is one Kernel noun, not the absence of three particular names.

    A contributor sees ``Kernel``; anything else ending in "Kernel" on the
    authoring surface is a second thing they would have to learn, whatever it
    happens to be called.  ``KernelDeclaration`` is the assembly record the
    pool machinery passes around and is deliberately not offered here.
    """

    # The pool and the authoring scope are their own concepts, not Kernels.
    machinery = {"KernelDesign", "KernelSelection", "KernelSelectionPaths"}
    exported = set(authoring.__all__)
    nouns = {name for name in exported if name.startswith("Kernel")} - machinery
    assert nouns == {"Kernel", "KernelDemand", "KernelExport", "KernelProvider"}, nouns
    assert "KernelDeclaration" not in exported
    assert not hasattr(authoring, "KernelInstance")
    assert set(authoring.__all__) == {name for name in exported if hasattr(authoring, name)}


# -- the scope's own rules ---------------------------------------------------


def test_a_source_constraint_registers_into_both_sets() -> None:
    _op, inputs, _selection = _pool()
    design: KernelDesign[ComputeInputs] = KernelDesign("probe", inputs)
    design.region(
        dependencies={"extent": inputs.extent}, evaluate=lambda extent: _region(extent, 1)
    )
    design.source_constraint(
        "graph_only",
        dependencies={"extent": inputs.extent},
        evaluate=lambda extent: extent > 0,
    )

    assert design.constraint_paths(SOURCE_ADMISSION) == (
        QualifiedPath("constraint.probe.graph_only"),
    )
    assert design.constraint_paths(FEASIBILITY) == (QualifiedPath("constraint.probe.graph_only"),)


def test_a_source_constraint_may_not_read_a_local_decision() -> None:
    """Admission is asked before anything is decided, so this can never hold."""

    _op, inputs, _selection = _pool()
    design: KernelDesign[ComputeInputs] = KernelDesign("probe", inputs)
    lanes = design.choice("lanes", int, domain=finite((1, 2)))

    with pytest.raises(AuthoringError, match="source admission"):
        design.source_constraint(
            "reads_a_decision",
            dependencies={"lanes": lanes},
            evaluate=lambda lanes: lanes > 0,
        )


def test_a_source_constraint_may_not_read_a_target_fact() -> None:
    op, inputs, _selection = _pool()
    design: KernelDesign[ComputeInputs] = KernelDesign("probe", inputs, provenance=op.provenance())

    with pytest.raises(AuthoringError, match="is target"):
        design.source_constraint(
            "reads_the_target",
            dependencies={"target": inputs.target_family.allow_absent()},
            evaluate=lambda target: target == "wide",
        )
    assert op.provenance().kind_of(inputs.target_family.path) is Provenance.TARGET


def test_placing_one_subclass_twice_produces_non_colliding_paths() -> None:
    _op, inputs, _selection = _pool()
    left = declare_kernel(EvenComputeKernel, "left.compute.even", inputs)
    right = declare_kernel(EvenComputeKernel, "right.compute.even", inputs)

    assert left.id == right.id == "even"
    assert left.region_path != right.region_path
    assert {item.path for item in left.spec.decisions} & {
        item.path for item in right.spec.decisions
    } == set()


def test_a_subclass_without_an_id_is_refused() -> None:
    class Nameless(Kernel):
        @classmethod
        def define_design(cls, design: KernelDesign[ComputeInputs]) -> None:
            design.region(
                dependencies={"extent": design.inputs.extent},
                evaluate=lambda extent: _region(extent, 1),
            )

    _op, inputs, _selection = _pool()
    with pytest.raises(AuthoringError, match="must set a Kernel id"):
        declare_kernel(Nameless, "probe", inputs)


def test_a_kernel_that_declares_no_region_is_refused() -> None:
    class Regionless(Kernel):
        id = "regionless"

        @classmethod
        def define_design(cls, design: KernelDesign[ComputeInputs]) -> None:
            design.choice("lanes", int, domain=finite((1,)))

    _op, inputs, _selection = _pool()
    with pytest.raises(AuthoringError, match="declares no Region"):
        declare_kernel(Regionless, "probe", inputs)


def test_binding_propagates_an_unresolved_demand_rather_than_dropping_it() -> None:
    """A missing demand must not read as "this Kernel has no such requirement".

    ``EvenComputeKernel`` demands a parameter port derived from the element
    type.  Withhold that fact and the demand cannot resolve; binding has to say
    so, because a consumer seeing ``demands == {}`` would conclude the Kernel
    never needed one.
    """

    class TargetDemandingKernel(_ExampleComputeKernel):
        id = "target_demanding"

        @classmethod
        def define_design(cls, design: KernelDesign[ComputeInputs]) -> None:
            cls._common(design)
            design.demand(
                "parameter",
                dependencies={
                    "extent": design.inputs.extent,
                    "family": design.inputs.target_family,
                },
                evaluate=lambda extent, family: Port(
                    f"parameter_{family}",
                    Operand("w", INT8, (extent,)),
                    BeatSequence(1, tuple(((index,),) for index in range(extent))),
                ),
            )

    op = OpDesign("example.op", problem_namespace="example")
    inputs = ComputeInputs(
        extent=op.graph_fact("extent", int),
        element_type=op.graph_fact("element_type", NumericElementType),
        target_family=op.target_fact("family", str, required=False),
    )
    pool = "probe.compute"
    selection = KernelSelection(
        pool,
        (
            declare_kernel(
                TargetDemandingKernel, kernel_namespace(pool, "target_demanding"), inputs
            ),
        ),
    )
    engine = Engine()
    space = engine.validate(assemble_specs((op.spec(), selection.build_spec())))
    # The optional target fact is withheld, so the demanded port cannot be
    # derived even though the problem is perfectly valid.
    point = engine.start(space, {inputs.extent.path: 4, inputs.element_type.path: INT8})
    point = engine.commit_assignments(
        point,
        {
            selection.paths.kernel: "target_demanding",
            QualifiedPath(f"{kernel_namespace(pool, 'target_demanding')}.lanes"): 2,
        },
    ).point

    assert not isinstance(
        engine.query_property(point, selection.paths.demand("parameter")), Decided
    )
    assert not isinstance(bind_kernel(engine, selection, point), Decided)


def test_binding_ignores_a_pool_path_another_member_owns() -> None:
    """Requiredness is per declaration: ``any`` never demanded a parameter."""

    engine, selection, point = _resolved(4, "any", 2)
    bound = bind_kernel(engine, selection, point)

    assert isinstance(bound, Decided)
    assert bound.value.demands == {}
