# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The operation authoring scope and its problem provenance."""

from __future__ import annotations

import pytest

from finn.dataflow.authoring.op_design import (
    BUILD_OWNED,
    GRAPH_OWNED,
    OpDesign,
    Provenance,
)
from finn.dataflow.authoring.scope import AuthoringError
from finn.dataflow.design import Decided, Engine, QualifiedPath
from finn.dataflow.spec_algebra import assemble_specs


def _design() -> OpDesign:
    return OpDesign("example.op", problem_namespace="example")


def test_a_fact_lands_under_the_operation_namespace_and_records_its_owner() -> None:
    design = _design()
    width = design.graph_fact("width", int)
    owner = design.build_fact("analysis_owner", str)

    assert width.path == QualifiedPath("problem.example.width")
    assert owner.path == QualifiedPath("problem.example.analysis_owner")
    provenance = design.provenance()
    assert provenance.kind_of(width.path) is Provenance.GRAPH
    assert provenance.kind_of(owner.path) is Provenance.BUILD


def test_target_facts_share_one_root_across_operations() -> None:
    design = _design()
    part = design.target_fact("fpga_part", str, required=False)

    assert part.path == QualifiedPath("problem.target.fpga_part")
    assert design.provenance().kind_of(part.path) is Provenance.TARGET


def test_an_explicit_path_keeps_its_provenance_recorded() -> None:
    """A fact shared with another family is still owned by someone."""

    design = _design()
    shared = design.build_fact(
        "runtime_writable", bool, path=QualifiedPath("problem.other_family.runtime_writable")
    )

    assert shared.path == QualifiedPath("problem.other_family.runtime_writable")
    assert design.provenance().kind_of(shared.path) is Provenance.BUILD


def test_declaring_one_path_twice_is_an_authoring_error() -> None:
    design = _design()
    design.graph_fact("width", int)

    with pytest.raises(AuthoringError, match="declared twice"):
        design.analysis_fact("width", int)


def test_paths_for_selects_by_provenance() -> None:
    design = _design()
    width = design.graph_fact("width", int)
    accumulator = design.analysis_fact("accumulator", int)
    part = design.target_fact("fpga_part", str, required=False)
    provenance = design.provenance()

    assert provenance.paths_for(GRAPH_OWNED) == frozenset({width.path, accumulator.path})
    assert provenance.paths_for(BUILD_OWNED) == frozenset({part.path})
    assert provenance.paths_for(Provenance.GRAPH) == frozenset({width.path})


def test_a_graph_projection_may_not_supply_a_build_fact() -> None:
    design = _design()
    width = design.graph_fact("width", int)
    owner = design.build_fact("analysis_owner", str)
    provenance = design.provenance()

    provenance.check_graph_projection({width.path: 4})
    with pytest.raises(AuthoringError, match="analysis_owner is build"):
        provenance.check_graph_projection({width.path: 4, owner.path: "someone"})


def test_a_build_projection_may_not_fabricate_a_graph_fact() -> None:
    design = _design()
    width = design.graph_fact("width", int)
    part = design.target_fact("fpga_part", str, required=False)
    provenance = design.provenance()

    provenance.check_build_projection({part.path: "xcvc1902"})
    with pytest.raises(AuthoringError, match="width is graph"):
        provenance.check_build_projection({part.path: "xcvc1902", width.path: 4})


def test_an_undeclared_path_is_reported_separately_from_a_misowned_one() -> None:
    design = _design()
    design.graph_fact("width", int)

    with pytest.raises(AuthoringError, match=r"undeclared \['problem.example.height'\]"):
        design.provenance().check_graph_projection({QualifiedPath("problem.example.height"): 4})


def test_the_scope_still_produces_an_ordinary_engine_specification() -> None:
    """Provenance is authoring metadata; the engine never sees it."""

    design = _design()
    width = design.graph_fact("width", int, validate=lambda value: isinstance(value, int))
    doubled = design.derived(
        "doubled", int, dependencies={"width": width}, evaluate=lambda width: width * 2
    )

    engine = Engine()
    space = engine.validate(assemble_specs((design.spec(),)))
    point = engine.start(space, {width.path: 21})

    assert doubled.path == QualifiedPath("semantic.example.op.doubled")
    assert engine.query_property(point, doubled.path) == Decided(42)
