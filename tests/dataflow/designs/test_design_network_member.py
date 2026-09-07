# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Design's Network is a declared member, not a string-keyed projection.

Before this, a Network was reachable only by asking the occurrence layer for a
projection called ``"dataflow"``.  That made a Design the one layer a containing
Space could not compose the way a Design composes a Kernel: a Kernel exports its
Region as an ordinary value, and an operation could not name a Design's Network
at all.  These tests pin the replacement -- a declared ``SelectedNetwork``
placeholder whose evaluator lowering substitutes by member identity -- and each
way that substitution could silently do nothing.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from finn.dataflow._engine import DependencyKind, Engine
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.designs.design import (
    RESERVED_DESIGN_NAMES,
    NetworkBoundary,
    DataflowDesign,
    KernelChoice,
    SelectedNetwork,
    design_dataflow,
)
from finn.dataflow.space.compiler import _compile_space
from finn.dataflow.space.declarations import (
    AuthoringError,
    ConstraintGroup,
    Derived,
    Input,
    Problem,
    Projection,
    Readiness,
    Rejected,
    Space,
    Subspace,
    constraint,
    declared_members,
    reject,
    reject_all,
)
from finn.dataflow.space.dataflow_value_semantics import DATAFLOW_NETWORK_SEMANTICS
from finn.dataflow.kernels.kernel import Kernel, RegionDeclaration
from finn.dataflow.model.network import DataflowNetwork
from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.space.spec_algebra import assemble_specs

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]


def _region(extent: int) -> DataflowRegion:
    schedule = LogicalSchedule((ScheduleLevel("beat", extent),))
    operand = Operand("value", DataType["INT8"], (extent,))
    sequence = BeatSequence(1, tuple(((index,),) for index in range(extent)))
    availability = ScheduledOutputAvailability({(index,): (index,) for index in range(extent)})
    return DataflowRegion(
        schedule,
        (),
        (OutputInterface(Port("output", operand, sequence), availability),),
    )


class EmitKernel(Kernel):
    id = "emit"
    version = "1"

    extent = Input(int)
    region = RegionDeclaration(family="test.emit", version="1", construct=_region, extent=extent)

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("emit", (), ())


class Harness(Space):
    extent = Problem(int)


class OneSegment(DataflowDesign):
    id = "one_segment"
    version = "1"

    extent = Input(int)
    only = KernelChoice(Subspace(EmitKernel, extent=extent))
    result = NetworkBoundary(only.output("output"))


def _compiled(design_type: type[DataflowDesign]):
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    design = _compile_space(
        design_type,
        "root.design",
        {"extent": harness.member("extent")},
        _allow_problem=False,
    )
    return harness, design


def _started(design_type: type[DataflowDesign], extent: int = 4):
    harness, design = _compiled(design_type)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, design.spec))),
        {"problem.root.extent": extent},
    )
    return engine, point, design


# -- the member itself ---------------------------------------------------------


def test_the_network_is_an_ordinary_declared_member() -> None:
    declarations = dict(declared_members(OneSegment))

    network = declarations["network"]

    assert isinstance(network, SelectedNetwork)
    assert isinstance(network, Derived)
    assert network.value_semantics.type_token is DATAFLOW_NETWORK_SEMANTICS.type_token


def test_the_network_is_exported_so_a_containing_space_can_name_it() -> None:
    """The point of the change: a Design composes upward like a Kernel does."""

    _harness, design = _compiled(OneSegment)

    exported = design.exported("network")

    assert exported.kind is DependencyKind.PROPERTY
    assert exported == design.member("network")


def test_the_projection_is_a_declaration_not_a_generated_record() -> None:
    declarations = dict(declared_members(OneSegment))

    assert isinstance(declarations["dataflow"], Projection)
    assert isinstance(declarations["dataflow_ready"], Readiness)
    assert isinstance(declarations["dataflow_accepts"], ConstraintGroup)
    assert declarations["dataflow"].output is declarations["network"]


def test_the_substituted_evaluator_produces_the_network() -> None:
    engine, point, design = _started(OneSegment)

    assessment = design_dataflow(engine, design, point)

    assert isinstance(assessment.accepted_answer.value, DataflowNetwork)
    assert tuple(node.id for node in assessment.accepted_answer.value.nodes) == ("only",)


def test_the_placeholder_evaluator_is_never_reached() -> None:
    """A compiled Design that still holds the placeholder is a defect here.

    Asserted by calling it directly: if lowering ever stopped substituting, the
    failure should name this module rather than surface as an unresolved
    Network somewhere downstream.
    """

    placeholder = SelectedNetwork()

    with pytest.raises(AuthoringError, match="never lowered"):
        placeholder.evaluate()


# -- the ways substitution could silently do nothing ---------------------------


def test_a_class_body_may_not_replace_the_network() -> None:
    with pytest.raises(AuthoringError, match="synthesizes"):

        class OwnNetwork(DataflowDesign):
            id = "own_network"
            version = "1"
            extent = Input(int)
            network = SelectedNetwork()
            only = KernelChoice(Subspace(EmitKernel, extent=extent))


@pytest.mark.parametrize("name", sorted(RESERVED_DESIGN_NAMES - {"network"}))
def test_a_class_body_may_not_shadow_a_synthesized_member(name: str) -> None:
    @constraint()
    def anything() -> bool:
        return True

    with pytest.raises(AuthoringError, match="synthesizes"):
        type(
            "Shadowing",
            (DataflowDesign,),
            {"id": "shadowing", "version": "1", name: anything},
        )


def test_an_abstract_intermediate_with_no_segments_is_left_alone() -> None:
    """``WeightedDotProductDesign`` is real: a base that declares no segments.

    Synthesis must tolerate it and refuse only where a class is actually
    compiled, exactly as ``Kernel.__init_subclass__`` returns early for a
    subclass with no Region yet.
    """

    class Abstract(DataflowDesign):
        extent = Input(int)

    assert "dataflow" not in Abstract.__dict__
    assert "network_structurally_valid" not in Abstract.__dict__

    class Concrete(Abstract):
        id = "concrete"
        version = "1"
        only = KernelChoice(Subspace(EmitKernel, extent=Abstract.extent))
        result = NetworkBoundary(only.output("output"))

    assert isinstance(Concrete.__dict__["dataflow"], Projection)
    engine, point, design = _started(Concrete)
    assert isinstance(design_dataflow(engine, design, point).accepted_answer.value, DataflowNetwork)


def test_a_design_with_no_segment_is_still_refused_at_compilation() -> None:
    class NoSegment(DataflowDesign):
        id = "no_segment"
        version = "1"
        extent = Input(int)

    with pytest.raises(AuthoringError, match="at least one Kernel segment"):
        _compiled(NoSegment)


# -- several reasons, not one --------------------------------------------------


def test_a_network_refusal_reports_every_issue() -> None:
    """``validate_network`` returns a list, and the constraint must not truncate it.

    Reporting the first issue only turns one report into a queue of apparent
    regressions: each fix reveals the next as though it were new.
    """

    combined = reject_all(
        (reject("first", "one thing"), reject("second", "another"), reject("third", "a third"))
    )

    assert tuple(item.code for item in combined.findings) == ("first", "second", "third")


def test_a_rejection_must_carry_a_reason() -> None:
    with pytest.raises(AuthoringError, match="at least one reason"):
        Rejected(())
