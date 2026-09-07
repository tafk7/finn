# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U2: the two Kernel projections, on the two reusable Kernels.

``test_kernel`` proves the mechanism on a toy.  This proves the separation
where it has to hold -- on DotpAxi and ReplayBuffer, the Kernels that carry the
hardware evidence -- and it proves the negative direction too: a Region that
resolves while every physical fact is still open, and a physical result that is
refused while the Region it would have realized stays exactly what it was.
"""

from __future__ import annotations

from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Decided, Engine, Unresolved
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.kernels.kernel import (
    ModuleBuildSpec,
    _KernelCompilation,
    kernel_dataflow,
    kernel_physical,
)
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
)
from finn.dataflow.space.compiler import _Ref, _compile_space
from finn.dataflow.space.declarations import Decision, Input, Problem, Space, divisors_of
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.space.spec_algebra import assemble_specs


class Supplier(Space):
    """A stand-in Design: it owns every fact both Kernels read."""

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Problem(bool)
    target_dsp = Problem(DspBlock)
    clock_period_ns = Problem(float)
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))


PROBLEM = {
    "problem.u2.repetitions": 2,
    "problem.u2.matrix_width": 8,
    "problem.u2.matrix_height": 4,
    "problem.u2.activation_type": DataType["INT8"],
    "problem.u2.weight_type": DataType["INT8"],
    "problem.u2.accumulator_type": DataType["INT32"],
    "problem.u2.output_type": DataType["INT32"],
    "problem.u2.narrow_weights": False,
    "problem.u2.target_dsp": DspBlock.DSP58,
    "problem.u2.clock_period_ns": 5.0,
}


def _compiled(kernel_type: type[Space], namespace: str) -> tuple[object, object]:
    supplier = _compile_space(Supplier, "u2", problem_namespace="problem.u2")
    names = {
        name for name, declaration in vars(kernel_type).items() if isinstance(declaration, Input)
    }
    bindings = {
        name: cast("_Ref[object]", supplier.member(name))
        for name in names
        if name in dict(supplier.members)
    }
    return supplier, _compile_space(kernel_type, namespace, bindings, _allow_problem=False)


def test_the_dotp_region_resolves_before_any_physical_choice_is_made() -> None:
    supplier, compiled = _compiled(DotpAxiKernel, "u2.compute")
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((supplier.spec, compiled.spec))),  # type: ignore[attr-defined]
        PROBLEM,
    )
    point = engine.commit_assignments(point, {"u2.pe": 2, "u2.simd": 4}).point

    # `compute_pumping` is the Kernel's own physical Decision and is uncommitted.
    dataflow = kernel_dataflow(engine, compiled, point)  # type: ignore[arg-type]
    assert dataflow.readiness.ready is True
    assert dataflow.accepted_answer == Decided(
        construct_dot_product_region(
            repetitions=2,
            matrix_width=8,
            matrix_height=4,
            activation_element_type=DataType["INT8"],
            weight_element_type=DataType["INT8"],
            output_element_type=DataType["INT32"],
            pe=2,
            simd=4,
        )
    )
    physical = kernel_physical(engine, compiled, point)  # type: ignore[arg-type]
    assert isinstance(physical.accepted_answer, Unresolved)


def test_the_replay_region_resolves_with_no_physical_decision_at_all() -> None:
    supplier, compiled = _compiled(ReplayBufferKernel, "u2.replay")
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((supplier.spec, compiled.spec))),  # type: ignore[attr-defined]
        PROBLEM,
    )
    point = engine.commit_assignments(point, {"u2.pe": 2, "u2.simd": 4}).point

    dataflow = kernel_dataflow(engine, compiled, point)  # type: ignore[arg-type]
    assert dataflow.accepted_answer == Decided(
        construct_activation_replay_region(
            repetitions=2,
            matrix_width=8,
            matrix_height=4,
            activation_element_type=DataType["INT8"],
            pe=2,
            simd=4,
        )
    )
    # The buffer owns no Decision, so its build unit is available at once.
    physical = kernel_physical(engine, compiled, point).accepted_answer  # type: ignore[arg-type]
    assert isinstance(physical, Decided)
    assert physical.value.parameters == {"LEN": 2, "REP": 2, "W": 32}


def test_an_unsupported_target_refuses_the_build_unit_and_not_the_region() -> None:
    """Target support is a physical question, and asking it does not move the Region."""

    supplier, compiled = _compiled(DotpAxiKernel, "u2.compute")
    engine = Engine()
    # A 27-bit weight is a perfectly ordinary logical operand and does not pack
    # into a DSP58 lane alongside an 8-bit activation.
    point = engine.start(
        engine.validate(assemble_specs((supplier.spec, compiled.spec))),  # type: ignore[attr-defined]
        {**PROBLEM, "problem.u2.weight_type": DataType["INT27"]},
    )
    point = engine.commit_assignments(
        point, {"u2.pe": 2, "u2.simd": 4, "u2.compute.compute_pumping": False}
    ).point

    region = kernel_dataflow(engine, compiled, point).accepted_answer  # type: ignore[arg-type]
    assert isinstance(region, Decided)

    physical = kernel_physical(engine, compiled, point)  # type: ignore[arg-type]
    assert physical.readiness.ready is True
    assert isinstance(physical.accepted_answer, Absent)
    # And the Region is untouched by having asked.
    assert kernel_dataflow(engine, compiled, point).accepted_answer == region  # type: ignore[arg-type]


def test_the_detached_build_unit_holds_no_handle_into_the_design_space() -> None:
    supplier, compiled = _compiled(ReplayBufferKernel, "u2.replay")
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((supplier.spec, compiled.spec))),  # type: ignore[attr-defined]
        PROBLEM,
    )
    point = engine.commit_assignments(point, {"u2.pe": 2, "u2.simd": 4}).point
    answer = kernel_physical(engine, compiled, point).accepted_answer  # type: ignore[arg-type]
    assert isinstance(answer, Decided)
    result = answer.value

    for name in ModuleBuildSpec.__slots__:
        value = getattr(result, name)
        assert not isinstance(value, (Engine, Space))
        assert not hasattr(value, "design_space")
        assert not hasattr(value, "_occurrence_state")
    assert result.abi.entry_point == "replay_buffer"


def test_the_kernel_layer_classifies_its_decisions_without_relaxing_ownership() -> None:
    """The Region closure is proven empty, not merely assumed to be."""

    _supplier, compiled = _compiled(DotpAxiKernel, "u2.compute")
    metadata = compiled.extension  # type: ignore[attr-defined]
    assert isinstance(metadata, _KernelCompilation)
    assert metadata.region_closure_decisions == ()
    assert [path.value for path in metadata.physical_decisions] == ["u2.compute.compute_pumping"]
    assert {path.value for path in metadata.imported_decisions} == {"u2.pe", "u2.simd"}
