# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD3: ``DataflowDesign`` owns semantics and ``Kernels`` specializes ``OneOf``."""

from __future__ import annotations

from typing import cast

from typing_extensions import Self

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Engine
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    Case,
    Decision,
    Input,
    OneOf,
    Problem,
    Space,
    divisors_of,
)
from finn.dataflow.model.design import Boundary, DataflowDesign, Kernels, configure_design
from finn.dataflow.model.kernel import Kernel, Parameter, Region
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
)
from finn.dataflow.spec_algebra import assemble_specs

COPY = ComputationContract("test.copy")
SCALE = ComputationContract("test.scale")


def _region(extent: int, lanes: int) -> DataflowRegion:
    folds = extent // lanes
    schedule = LogicalSchedule((ScheduleLevel("fold", folds),))
    operand = Operand("value", DataType["INT8"], (extent,))
    beats = tuple(tuple((fold * lanes + lane,) for lane in range(lanes)) for fold in range(folds))
    sequence = BeatSequence(lanes, beats)
    requirements = ScheduledInputRequirements(
        {((fold,), (fold * lanes + lane,)): 1 for fold in range(folds) for lane in range(lanes)}
    )
    availability = ScheduledOutputAvailability(
        {(fold * lanes + lane,): (fold,) for fold in range(folds) for lane in range(lanes)}
    )
    return DataflowRegion(
        schedule,
        (InputInterface(Port("input", operand, sequence), requirements),),
        (OutputInterface(Port("output", operand, sequence), availability),),
    )


class CopyKernel(Kernel):
    id = "copy"
    version = "1"
    computation = COPY

    extent = Input(int)
    lanes = Input(int)
    pumped = Decision(bool, values=(False, True))

    region = Region(
        family="test.copy",
        version="1",
        construct=_region,
        extent=extent,
        lanes=lanes,
    )

    LANES = Parameter(lanes)

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI("copy", (), (("LANES", str(configured.LANES)),))


class WideCopyKernel(Kernel):
    """Same computation, different Input vocabulary, one physical decision."""

    id = "wide_copy"
    version = "1"
    computation = COPY

    width = Input(int)
    parallel_lanes = Input(int)
    stages = Decision(int, values=(1, 2))

    region = Region(
        family="test.copy",
        version="1",
        construct=_region,
        extent=width,
        lanes=parallel_lanes,
    )

    STAGES = Parameter(stages)

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI("wide_copy", (), (("STAGES", str(configured.STAGES)),))


class ScaleKernel(Kernel):
    id = "scale"
    version = "1"
    computation = SCALE

    extent = Input(int)
    lanes = Input(int)

    region = Region(
        family="test.scale",
        version="1",
        construct=_region,
        extent=extent,
        lanes=lanes,
    )

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI("scale", ())


class TwoSegments(DataflowDesign):
    """Two singleton segments sharing one Design-owned semantic decision."""

    id = "two_segments"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    first = Kernels(Case(CopyKernel, extent=extent, lanes=lanes), computation=COPY)
    second = Kernels(Case(ScaleKernel, extent=extent, lanes=lanes), computation=SCALE)


class Alternatives(DataflowDesign):
    """Two candidates with unrelated Input names at one segment."""

    id = "alternatives"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    compute = Kernels(
        Case(CopyKernel, extent=extent, lanes=lanes),
        Case(WideCopyKernel, width=extent, parallel_lanes=lanes),
        computation=COPY,
    )


class Harness(Space):
    """The supplier of every Region-visible fact a standalone Design needs."""

    extent = Problem(int)
    lanes = Decision(int, domain=divisors_of(extent))


def _compiled(design_type: type[DataflowDesign], namespace: str = "root.design"):
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    design = _compile_space(
        design_type,
        namespace,
        {name: cast("_Ref[object]", harness.member(name)) for name in ("extent", "lanes")},
        _allow_problem=False,
    )
    return harness, design


def _started(design_type: type[DataflowDesign], extent: int = 8, lanes: int = 2):
    harness, design = _compiled(design_type)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, design.spec))),
        {"problem.root.extent": extent},
    )
    return engine, engine.commit_assignments(point, {"root.lanes": lanes}).point, design


# -- segment lowering ---------------------------------------------------------


def test_two_singleton_segments_share_one_design_decision() -> None:
    _harness, design = _compiled(TwoSegments)
    assert [str(item.path) for item in design.spec.decisions] == ["root.design.first.copy.pumped"]
    assert "semantic.root.design.first.region" in {
        str(item.path) for item in design.spec.properties
    }
    assert "semantic.root.design.second.region" in {
        str(item.path) for item in design.spec.properties
    }
    engine, point, _design = _started(TwoSegments)
    first = engine.query_property(point, "semantic.root.design.first.region")
    second = engine.query_property(point, "semantic.root.design.second.region")
    assert first == Decided(_region(8, 2))
    assert second == Decided(_region(8, 2))


def test_a_singleton_segment_adds_no_selector() -> None:
    _harness, design = _compiled(TwoSegments)
    assert not any(item.path.value.endswith(".kernel") for item in design.spec.decisions)
    assert design.catalog.branch("root.design.first").selector is None


def test_alternatives_use_the_generic_selector_and_candidate_namespaces() -> None:
    _harness, design = _compiled(Alternatives)
    assert [str(item.path) for item in design.spec.decisions] == [
        "root.design.compute.kernel",
        "root.design.compute.copy.pumped",
        "root.design.compute.wide_copy.stages",
    ]
    branch = design.catalog.branch("root.design.compute")
    assert str(branch.selector) == "root.design.compute.kernel"
    assert tuple(case.id for case in branch.cases) == ("copy", "wide_copy")
    assert tuple(case.namespace for case in branch.cases) == (
        "root.design.compute.copy",
        "root.design.compute.wide_copy",
    )


def test_a_candidate_namespace_does_not_move_when_an_alternative_is_added() -> None:
    _harness, singleton = _compiled(TwoSegments)
    _other, alternatives = _compiled(Alternatives)
    assert singleton.catalog.branch("root.design.first").cases[0].namespace == (
        "root.design.first.copy"
    )
    assert alternatives.catalog.branch("root.design.compute").cases[0].namespace == (
        "root.design.compute.copy"
    )


def test_the_segment_forwards_the_exact_selected_region() -> None:
    engine, point, _design = _started(Alternatives)
    chosen = engine.commit_assignments(point, {"root.design.compute.kernel": "wide_copy"}).point
    assert engine.query_property(chosen, "semantic.root.design.compute.region") == Decided(
        _region(8, 2)
    )
    assert engine.query_property(chosen, "semantic.root.design.compute.copy.region") != Decided(
        _region(8, 2)
    )


def test_region_is_the_only_selected_segment_output() -> None:
    _harness, design = _compiled(Alternatives)
    branch = design.catalog.branch("root.design.compute")
    assert tuple(item.name for item in branch.outputs) == ("region",)
    assert not any(item.path.value.endswith(".compute.pumped") for item in design.spec.properties)


def test_a_repeated_kernel_class_rebases_independently() -> None:
    class Twice(DataflowDesign):
        id = "twice"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        left = Kernels(Case(CopyKernel, extent=extent, lanes=lanes), computation=COPY)
        right = Kernels(Case(CopyKernel, extent=extent, lanes=lanes), computation=COPY)

    _harness, design = _compiled(Twice)
    assert [str(item.path) for item in design.spec.decisions] == [
        "root.design.left.copy.pumped",
        "root.design.right.copy.pumped",
    ]


def test_a_segment_may_override_its_role_and_node_id() -> None:
    class Renamed(DataflowDesign):
        id = "renamed"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        member = Kernels(
            Case(CopyKernel, extent=extent, lanes=lanes),
            computation=COPY,
            role="compute",
            node_id="compute_node",
        )

    _harness, design = _compiled(Renamed)
    assert "semantic.root.design.compute.region" in {
        str(item.path) for item in design.spec.properties
    }


# -- authoring refusals -------------------------------------------------------


def test_a_design_may_not_declare_a_problem() -> None:
    class OwnsProblem(DataflowDesign):
        id = "owns_problem"
        version = "1"
        extent = Problem(int)
        lanes = Input(int)
        compute = Kernels(Case(CopyKernel, extent=extent, lanes=lanes), computation=COPY)

    with pytest.raises(AuthoringError, match="external facts through Input"):
        _compile_space(
            OwnsProblem,
            "root.design",
            {"lanes": cast("_Ref[object]", _compiled(TwoSegments)[0].member("lanes"))},
            problem_namespace="problem.root",
        )


def test_a_design_needs_an_id_a_version_and_a_segment() -> None:
    class NoSegment(DataflowDesign):
        id = "no_segment"
        version = "1"
        extent = Input(int)
        lanes = Input(int)

    with pytest.raises(AuthoringError, match="at least one Kernel segment"):
        _compiled(NoSegment)

    class NoId(DataflowDesign):
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(Case(CopyKernel, extent=extent, lanes=lanes), computation=COPY)

    with pytest.raises(AuthoringError, match="non-empty id"):
        _compiled(NoId)


def test_a_non_kernel_case_is_rejected() -> None:
    class Helper(Space):
        extent = Input(int)

    class NotAKernel(DataflowDesign):
        id = "not_a_kernel"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(Case(Helper, name="helper", extent=extent), computation=COPY)

    with pytest.raises(AuthoringError, match="is not a Kernel"):
        _compiled(NotAKernel)


def test_a_candidate_computation_must_equal_the_segment_requirement() -> None:
    class Mismatched(DataflowDesign):
        id = "mismatched"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(
            Case(CopyKernel, extent=extent, lanes=lanes),
            Case(ScaleKernel, extent=extent, lanes=lanes),
            computation=COPY,
        )

    with pytest.raises(AuthoringError, match="requires computation test.copy"):
        _compiled(Mismatched)


def test_a_segment_needs_a_computation_contract() -> None:
    with pytest.raises(AuthoringError, match="one ComputationContract"):
        Kernels(Case(CopyKernel), computation="test.copy")  # type: ignore[arg-type]


def test_duplicate_kernel_ids_in_one_segment_are_rejected() -> None:
    class Duplicated(DataflowDesign):
        id = "duplicated"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(
            Case(CopyKernel, extent=extent, lanes=lanes),
            Case(CopyKernel, extent=extent, lanes=lanes),
            computation=COPY,
        )

    with pytest.raises(AuthoringError, match="case id 'copy' twice"):
        _compiled(Duplicated)


def test_duplicate_segment_roles_and_node_ids_are_rejected() -> None:
    class SameNode(DataflowDesign):
        id = "same_node"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        first = Kernels(
            Case(CopyKernel, extent=extent, lanes=lanes), computation=COPY, node_id="shared"
        )
        second = Kernels(
            Case(ScaleKernel, extent=extent, lanes=lanes), computation=SCALE, node_id="shared"
        )

    with pytest.raises(AuthoringError, match="segment node id 'shared' twice"):
        _compiled(SameNode)


def test_case_input_binding_is_exact() -> None:
    class Incomplete(DataflowDesign):
        id = "incomplete"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(Case(CopyKernel, extent=extent), computation=COPY)

    with pytest.raises(AuthoringError, match="Input binding is not exact"):
        _compiled(Incomplete)


# -- the seam stays generic ---------------------------------------------------


def test_a_kernel_segment_is_visible_through_the_generic_branch_catalog() -> None:
    _harness, design = _compiled(Alternatives)
    catalog = design.catalog
    assert catalog.namespaces == ("root.design.compute",)
    branch = catalog.branch("root.design.compute")
    assert branch.case("wide_copy").decision_paths == tuple(
        item.path for item in design.spec.decisions if "wide_copy" in item.path.value
    )


def test_one_external_algorithm_selects_a_kernel_through_generic_branch_info() -> None:
    engine, point, design = _started(Alternatives)
    branch = design.catalog.branch("root.design.compute")
    assert branch.selector is not None
    candidates = engine.enumerate_candidates(point, branch.selector)
    assert isinstance(candidates, Decided)
    chosen = engine.commit_assignments(point, {branch.selector: candidates.value[-1]}).point
    assert engine.query_property(chosen, "semantic.root.design.compute.region") == Decided(
        _region(8, 2)
    )


def test_kernels_reuses_generic_branch_compilation() -> None:
    """No second candidate catalog, no `Use`, no coverage or binding object."""

    assert issubclass(Kernels, OneOf)
    _harness, design = _compiled(Alternatives)
    assert not hasattr(design, "coverage")
    assert not hasattr(design, "bindings")
    extension = design.extension
    assert extension is not None
    assert not hasattr(extension, "coverage")


def test_an_aliased_kernel_case_configures_under_its_alias() -> None:
    """One Kernel class may fill a segment twice, so case id is not Kernel id."""

    class Aliased(DataflowDesign):
        id = "aliased"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(
            Case(CopyKernel, name="fast", extent=extent, lanes=lanes),
            Case(CopyKernel, name="slow", extent=extent, lanes=lanes),
            computation=COPY,
        )
        source = Boundary(compute.input("input"))
        result = Boundary(compute.output("output"))

    _harness, design = _compiled(Aliased)
    branch = design.catalog.branch("root.design.compute")
    assert tuple(case.id for case in branch.cases) == ("fast", "slow")
    assert tuple(case.namespace for case in branch.cases) == (
        "root.design.compute.fast",
        "root.design.compute.slow",
    )
    engine, point, _design = _started(Aliased)
    for alias in ("fast", "slow"):
        chosen = engine.commit_assignments(
            point,
            {
                "root.design.compute.kernel": alias,
                f"root.design.compute.{alias}.pumped": False,
            },
        ).point
        answer = configure_design(engine, design, chosen)
        assert isinstance(answer, Decided), answer
        assert answer.value.selected_candidates == {"compute": alias}
        assert isinstance(answer.value.compute, CopyKernel)


def test_a_singleton_aliased_case_configures() -> None:
    class Solo(DataflowDesign):
        id = "solo"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(
            Case(CopyKernel, name="only", extent=extent, lanes=lanes), computation=COPY
        )
        source = Boundary(compute.input("input"))
        result = Boundary(compute.output("output"))

    engine, point, design = _started(Solo)
    chosen = engine.commit_assignments(point, {"root.design.compute.only.pumped": False}).point
    answer = configure_design(engine, design, chosen)
    assert isinstance(answer, Decided), answer
    assert answer.value.selected_candidates == {"compute": "only"}


def test_roles_node_ids_and_case_ids_must_be_atomic_path_segments() -> None:
    with pytest.raises(AuthoringError, match="must be one path segment"):
        Kernels(Case(CopyKernel), computation=COPY, role="outer.inner")
    with pytest.raises(AuthoringError, match="must be one path segment"):
        Kernels(Case(CopyKernel), computation=COPY, node_id="outer.inner")

    class DottedCase(DataflowDesign):
        id = "dotted"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        compute = Kernels(
            Case(CopyKernel, name="a.b", extent=extent, lanes=lanes), computation=COPY
        )

    with pytest.raises(AuthoringError, match="contains a dot"):
        _compiled(DottedCase)
