# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""K5: DotpAxi is independently authored and configured through model.Space."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Decided, Engine, Unresolved
from finn.dataflow.artifacts.rtl import Declined, check_abi
from finn.dataflow.design.region import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import Problem, Space
from finn.dataflow.model.dotp_axi import (
    DspBlock,
    DotpAxiKernel,
    FINNLIB_SOURCES,
    construct_dot_product_region,
)
from finn.dataflow.model.kernel import configure_kernel
from finn.dataflow.ops.mvau.regions import construct_dot_product_region as baseline_region
from finn.dataflow.spec_algebra import assemble_specs


class Harness(Space):
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


def _compile():
    harness = _compile_space(Harness, "dotp_test", problem_namespace="problem.dotp")
    kernel = _compile_space(
        DotpAxiKernel,
        "dotp_test.kernel",
        {
            name: cast("_Ref[object]", harness.member(name))
            for name in (
                "repetitions",
                "matrix_width",
                "matrix_height",
                "activation_type",
                "weight_type",
                "accumulator_type",
                "output_type",
                "narrow_weights",
                "target_dsp",
                "clock_period_ns",
            )
        },
        _allow_problem=False,
    )
    return harness, kernel


def _problem(
    *,
    repetitions: int = 2,
    matrix_width: int = 8,
    matrix_height: int = 4,
    activation: str = "INT8",
    weight: str = "INT8",
    accumulator: str = "INT32",
    output: str | None = None,
    narrow: bool = False,
    target: DspBlock = DspBlock.DSP58,
    clock: float = 4.0,
) -> dict[str, object]:
    return {
        "problem.dotp.repetitions": repetitions,
        "problem.dotp.matrix_width": matrix_width,
        "problem.dotp.matrix_height": matrix_height,
        "problem.dotp.activation_type": DataType[activation],
        "problem.dotp.weight_type": DataType[weight],
        "problem.dotp.accumulator_type": DataType[accumulator],
        "problem.dotp.output_type": DataType[output or accumulator],
        "problem.dotp.narrow_weights": narrow,
        "problem.dotp.target_dsp": target,
        "problem.dotp.clock_period_ns": clock,
    }


def _configure(
    *,
    pe: int = 2,
    simd: int = 2,
    pumping: bool = False,
    **problem: object,
):
    harness, kernel = _compile()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, kernel.spec))),
        _problem(**problem),  # type: ignore[arg-type]
    )
    point = engine.commit_assignments(
        point,
        {
            "dotp_test.kernel.pe": pe,
            "dotp_test.kernel.simd": simd,
            "dotp_test.kernel.compute_pumping": pumping,
        },
    ).point
    return configure_kernel(engine, kernel, point)


def test_dotp_region_matches_the_previous_authority() -> None:
    configured = _configure(pe=2, simd=4)
    assert isinstance(configured, Decided)
    expected = baseline_region(
        2,
        8,
        4,
        DataType["INT8"],
        DataType["INT8"],
        DataType["INT32"],
        2,
        4,
    )
    assert configured.value.resolved_region == expected
    assert configured.value.resolved_region == construct_dot_product_region(
        2,
        8,
        4,
        DataType["INT8"],
        DataType["INT8"],
        DataType["INT32"],
        2,
        4,
    )


def test_dotp_parameter_table_remains_exact() -> None:
    configured = _configure(pe=2, simd=2, target=DspBlock.DSP48E2)
    assert isinstance(configured, Decided)
    assert dict(configured.value.parameters) == {
        "PE": 2,
        "SIMD": 2,
        "PUMPED_COMPUTE": False,
        "ACTIVATION_WIDTH": 8,
        "WEIGHT_WIDTH": 8,
        "ACCU_WIDTH": 32,
        "VERSION": 2,
        "SIGNED_ACTIVATIONS": True,
        "SEGMENTLEN": 1,
        "NARROW_WEIGHTS": False,
        "ACTIVATION_BROADCASTING": 1,
        "FORCE_BEHAVIORAL": 0,
    }


def test_dotp_owns_all_three_decisions() -> None:
    configured = _configure(pe=2, simd=4, pumping=True)
    assert isinstance(configured, Decided)
    assert {path.value for path in configured.value.assignments} == {
        "dotp_test.kernel.pe",
        "dotp_test.kernel.simd",
        "dotp_test.kernel.compute_pumping",
    }
    assert configured.value.imported_decisions == ()


def test_dotp_feasibility_rejects_numeric_pumping_and_packing_failures() -> None:
    bad_output = _configure(output="INT16")
    assert isinstance(bad_output, Unresolved)
    assert "dotp-axi-numeric-types-unsupported" in {finding.code for finding in bad_output.findings}

    bad_pumping = _configure(simd=1, pumping=True)
    assert isinstance(bad_pumping, Unresolved)

    bad_packing = _configure(weight="INT27", target=DspBlock.DSP58, narrow=False)
    assert isinstance(bad_packing, Unresolved)
    assert "dotp-axi-weights-do-not-pack" in {finding.code for finding in bad_packing.findings}


def test_dotp_sources_and_abi_are_exact() -> None:
    configured = _configure(pe=2, simd=4, pumping=True)
    assert isinstance(configured, Decided)
    kernel = configured.value
    assert tuple(source.path for source in kernel.source_contributions) == FINNLIB_SOURCES
    assert kernel.abi.entry_point == "dotp_axi"
    assert set(kernel.abi.physical_names()) == {
        "ap_clk",
        "ap_clk2x",
        "ap_rst_n",
        "s_axis_weights_tdata",
        "s_axis_weights_tvalid",
        "s_axis_weights_tready",
        "s_axis_input_tdata",
        "s_axis_input_tvalid",
        "s_axis_input_tlast",
        "s_axis_input_tready",
        "m_axis_output_tdata",
        "m_axis_output_tvalid",
        "m_axis_output_tready",
    }
    widths = {
        member.physical: member.width
        for port in kernel.abi.ports
        if hasattr(port, "signals")
        for member in port.signals  # type: ignore[union-attr]
    }
    assert widths["s_axis_weights_tdata"] == 64
    assert widths["s_axis_input_tdata"] == 32
    assert widths["m_axis_output_tdata"] == 64


def test_dotp_abi_agrees_with_pinned_finnlib() -> None:
    configured = _configure(pe=2, simd=4, pumping=True)
    assert isinstance(configured, Decided)
    root = Path(__file__).parents[3] / "deps/finnlib"
    sources = tuple(root / path for path in FINNLIB_SOURCES)
    if any(not path.is_file() for path in sources):
        return
    result = check_abi(
        configured.value.abi,
        sources,
        "dotp_axi",
        configured.value.abi.parameters,
    )
    assert not isinstance(result, Declined)
    assert result == ()


def test_new_dotp_kernel_has_no_old_stack_imports() -> None:
    path = Path(__file__).parents[3] / "src/finn/dataflow/model/dotp_axi.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden = (
        "finn.dataflow.authoring",
        "finn.dataflow.kernels",
        "finn.dataflow.ops",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported
        for prefix in forbidden
    )
