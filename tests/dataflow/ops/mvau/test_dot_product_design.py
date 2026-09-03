# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD7: the real replay-to-DotpAxi Design against the retained MVAU authority."""

from __future__ import annotations

from typing import cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Answer, Decided, Engine, QualifiedPath, Unresolved
from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.model.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import Problem, Space
from finn.dataflow.designs.design import configure_design
from finn.dataflow.ops.mvau.designs.dot_product import DESIGN_INPUTS, DotProductDesign
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.network import DirectConnection, FanoutMode, PassCorrespondence
from finn.dataflow.network_validation import validate_network
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region as baseline_replay,
)
from finn.dataflow.ops.mvau.regions import (
    construct_dot_product_region as baseline_dot_product,
)
from finn.dataflow.ops.mvau.networks import construct_decomposed_mvau_network
from finn.dataflow.model.spec_algebra import assemble_specs

CLOCK_PERIOD_NS = 4.0


class Problem_(Space):
    """The graph-side facts; a future DataflowOp supplies these."""

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


def _compile(namespace: str = "mvau.dot_product"):
    root = _compile_space(Problem_, "mvau", problem_namespace="problem.mvau")
    design = _compile_space(
        DotProductDesign,
        namespace,
        {name: cast("_Ref[object]", root.member(name)) for name in DESIGN_INPUTS},
        _allow_problem=False,
    )
    return root, design


def _configure(
    *,
    repetitions: int = 2,
    matrix_width: int = 8,
    matrix_height: int = 4,
    activation: str = "INT8",
    weight: str = "INT8",
    accumulator: str = "INT32",
    narrow: bool = False,
    target: DspBlock = DspBlock.DSP58,
    pe: int = 2,
    simd: int = 2,
    pumping: bool = False,
) -> Answer[DotProductDesign]:
    root, design = _compile()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((root.spec, design.spec))),
        {
            "problem.mvau.repetitions": repetitions,
            "problem.mvau.matrix_width": matrix_width,
            "problem.mvau.matrix_height": matrix_height,
            "problem.mvau.activation_type": DataType[activation],
            "problem.mvau.weight_type": DataType[weight],
            "problem.mvau.accumulator_type": DataType[accumulator],
            "problem.mvau.output_type": DataType[accumulator],
            "problem.mvau.narrow_weights": narrow,
            "problem.mvau.target_dsp": target,
            "problem.mvau.clock_period_ns": CLOCK_PERIOD_NS,
        },
    )
    point = engine.commit_assignments(
        point,
        {
            "mvau.dot_product.pe": pe,
            "mvau.dot_product.simd": simd,
            "mvau.dot_product.compute.dotp_axi.compute_pumping": pumping,
        },
    ).point
    return configure_design(engine, design, point)


#: PE and SIMD at one and above one, one and several repetitions, signed and
#: unsigned activations, every permitted DSP generation, pumped and unpumped,
#: and several legal divisor combinations.
MATRIX = (
    ("pe1_simd1", 1, 4, 4, "INT8", "INT8", "INT32", DspBlock.DSP58, 1, 1, False),
    ("pe1_simd_many", 2, 8, 4, "INT8", "INT8", "INT32", DspBlock.DSP58, 1, 4, False),
    ("pe_many_simd1", 2, 8, 4, "INT8", "INT8", "INT32", DspBlock.DSP58, 4, 1, False),
    ("square", 3, 8, 8, "INT8", "INT8", "INT32", DspBlock.DSP58, 2, 2, False),
    ("unsigned", 2, 8, 4, "UINT8", "INT8", "INT32", DspBlock.DSP58, 2, 2, False),
    ("dsp48e1", 2, 8, 4, "INT8", "INT8", "INT32", DspBlock.DSP48E1, 2, 2, False),
    ("dsp48e2", 2, 8, 6, "INT8", "INT8", "INT32", DspBlock.DSP48E2, 3, 2, False),
    ("pumped", 2, 8, 4, "INT8", "INT8", "INT32", DspBlock.DSP58, 2, 4, True),
    ("odd_folds", 1, 12, 6, "INT4", "INT8", "INT16", DspBlock.DSP58, 3, 4, False),
)


@pytest.mark.parametrize(
    (
        "label",
        "repetitions",
        "matrix_width",
        "matrix_height",
        "activation",
        "weight",
        "accumulator",
        "target",
        "pe",
        "simd",
        "pumping",
    ),
    MATRIX,
)
def test_the_design_matches_the_retained_decomposed_authority(
    label: str,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation: str,
    weight: str,
    accumulator: str,
    target: DspBlock,
    pe: int,
    simd: int,
    pumping: bool,
) -> None:
    del label
    answer = _configure(
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation=activation,
        weight=weight,
        accumulator=accumulator,
        target=target,
        pe=pe,
        simd=simd,
        pumping=pumping,
    )
    assert isinstance(answer, Decided), answer
    configured = answer.value

    expected_replay = baseline_replay(
        repetitions, matrix_width, matrix_height, DataType[activation], pe, simd
    )
    expected_compute = baseline_dot_product(
        repetitions,
        matrix_width,
        matrix_height,
        DataType[activation],
        DataType[weight],
        DataType[accumulator],
        pe,
        simd,
    )
    assert configured.kernels["replay"].resolved_region == expected_replay
    assert configured.kernels["compute"].resolved_region == expected_compute
    assert configured.resolved_network == construct_decomposed_mvau_network(
        expected_replay, expected_compute
    )
    assert not validate_network(configured.resolved_network).issues


def test_the_selected_network_has_exactly_two_nodes_one_edge_three_boundaries() -> None:
    answer = _configure()
    assert isinstance(answer, Decided)
    network = answer.value.resolved_network
    assert tuple(node.id for node in network.nodes) == ("compute", "replay")
    assert tuple(edge.id for edge in network.edges) == ("activation_replay",)
    assert tuple(item.id for item in network.boundaries) == ("activation", "output", "weight")
    edge = network.edges[0]
    assert edge.fanout is FanoutMode.REPLICATE
    assert edge.pass_correspondence is PassCorrespondence.ONE_TO_ONE
    assert edge.transport == DirectConnection()
    source = network.node("replay").region.output_interface("activation_out").port
    entries = edge.sinks[0].position_map.entries
    assert all(left == right for left, right in entries)
    assert frozenset(left for left, _right in entries) == source.beat_sequence.image


def test_the_configured_design_contains_exactly_two_kernels() -> None:
    answer = _configure()
    assert isinstance(answer, Decided)
    configured = answer.value
    assert set(configured.kernels) == {"replay", "compute"}
    assert isinstance(configured.replay, ReplayBufferKernel)
    assert isinstance(configured.compute, DotpAxiKernel)
    assert configured.selected_candidates == {
        "replay": "replay_buffer",
        "compute": "dotp_axi",
    }
    assert configured.replay.computation == ACTIVATION_REPLAY_COMPUTATION
    assert configured.compute.computation == DOT_PRODUCT_COMPUTATION
    assert configured.region_family("replay") == ("mvau.activation_replay", "1")
    assert configured.region_family("compute") == ("mvau.dot_product", "1")


def test_pe_and_simd_appear_only_at_design_scope() -> None:
    _root, design = _compile()
    paths = {str(item.path) for item in design.spec.decisions}
    assert paths == {
        "mvau.dot_product.pe",
        "mvau.dot_product.simd",
        "mvau.dot_product.compute.dotp_axi.compute_pumping",
    }
    for path in paths:
        assert ".replay." not in path


def test_the_design_owns_pe_and_simd_and_the_kernels_import_them() -> None:
    answer = _configure(pe=2, simd=4)
    assert isinstance(answer, Decided)
    configured = answer.value
    assert dict(configured.assignments) == {
        QualifiedPath("mvau.dot_product.pe"): 2,
        QualifiedPath("mvau.dot_product.simd"): 4,
    }
    for kernel in configured.kernels.values():
        assert {path.value for path in kernel.imported_decisions} >= {
            "mvau.dot_product.pe",
            "mvau.dot_product.simd",
        }
    assert dict(configured.compute.assignments) == {
        QualifiedPath("mvau.dot_product.compute.dotp_axi.compute_pumping"): False
    }
    assert dict(configured.replay.assignments) == {}


def test_the_physical_parameter_tables_stay_kernel_local() -> None:
    answer = _configure(pe=2, simd=2, target=DspBlock.DSP48E2)
    assert isinstance(answer, Decided)
    configured = answer.value
    assert dict(configured.replay.parameters) == {"LEN": 4, "REP": 2, "W": 16}
    assert dict(configured.compute.parameters) == {
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
    assert not hasattr(configured, "parameters")


def test_an_incomplete_or_infeasible_point_refuses() -> None:
    root, design = _compile()
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((root.spec, design.spec))),
        {
            "problem.mvau.repetitions": 2,
            "problem.mvau.matrix_width": 8,
            "problem.mvau.matrix_height": 4,
            "problem.mvau.activation_type": DataType["INT8"],
            "problem.mvau.weight_type": DataType["INT8"],
            "problem.mvau.accumulator_type": DataType["INT32"],
            "problem.mvau.output_type": DataType["INT32"],
            "problem.mvau.narrow_weights": False,
            "problem.mvau.target_dsp": DspBlock.DSP58,
            "problem.mvau.clock_period_ns": CLOCK_PERIOD_NS,
        },
    )
    assert isinstance(configure_design(engine, design, point), Unresolved)

    # DotpAxi drives its accumulator straight out, so a narrower output refuses.
    mismatched = _configure(accumulator="INT32")
    assert isinstance(mismatched, Decided)
    refused = _configure(simd=1, pumping=True)
    assert isinstance(refused, Unresolved)


def test_no_kernel_export_coverage_or_binding_object_appears() -> None:
    answer = _configure()
    assert isinstance(answer, Decided)
    configured = answer.value
    for name in ("coverage", "bindings", "regions", "covers"):
        assert not hasattr(configured, name)
    assert DotpAxiKernel.exports == ()
    assert ReplayBufferKernel.exports == ()


def test_two_occurrences_of_the_design_stay_independent() -> None:
    root = _compile_space(Problem_, "mvau", problem_namespace="problem.mvau")
    bindings = {name: cast("_Ref[object]", root.member(name)) for name in DESIGN_INPUTS}
    left = _compile_space(DotProductDesign, "mvau.left", bindings, _allow_problem=False)
    right = _compile_space(DotProductDesign, "mvau.right", bindings, _allow_problem=False)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((root.spec, left.spec, right.spec))),
        {
            "problem.mvau.repetitions": 2,
            "problem.mvau.matrix_width": 8,
            "problem.mvau.matrix_height": 4,
            "problem.mvau.activation_type": DataType["INT8"],
            "problem.mvau.weight_type": DataType["INT8"],
            "problem.mvau.accumulator_type": DataType["INT32"],
            "problem.mvau.output_type": DataType["INT32"],
            "problem.mvau.narrow_weights": False,
            "problem.mvau.target_dsp": DspBlock.DSP58,
            "problem.mvau.clock_period_ns": CLOCK_PERIOD_NS,
        },
    )
    point = engine.commit_assignments(
        point,
        {
            "mvau.left.pe": 2,
            "mvau.left.simd": 2,
            "mvau.left.compute.dotp_axi.compute_pumping": False,
            "mvau.right.pe": 4,
            "mvau.right.simd": 4,
            "mvau.right.compute.dotp_axi.compute_pumping": False,
        },
    ).point
    first = configure_design(engine, left, point)
    second = configure_design(engine, right, point)
    assert isinstance(first, Decided) and isinstance(second, Decided)
    assert first.value.resolved_network != second.value.resolved_network
    assert first.value.compute.parameters["PE"] == 2
    assert second.value.compute.parameters["PE"] == 4
