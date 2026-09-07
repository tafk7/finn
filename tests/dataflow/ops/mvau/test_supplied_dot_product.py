# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U3c: the three weight-supply modes, as three resolved Networks.

The forcing case.  Every claim below is about *dataflow*: which nodes exist,
which edges connect them, which boundaries the Design presents.  Two of the
three modes have no build unit at all at this phase, and every one of them
resolves a complete Network anyway -- which is the thing U3 is for.
"""

from __future__ import annotations

from typing import cast

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, DspBlock
from finn.dataflow.kernels.memstream import MemstreamKernel
from finn.dataflow.space.declarations import Problem, Space, Subspace
from finn.dataflow.ops.mvau.computation import (
    AccumulationMode,
    ActivationMode,
    MvauComputationProfile,
)
from finn.dataflow.space.dataflow_value_semantics import (
    QONNX_DATATYPE_CODEC,
    QONNX_DATATYPE_VALUE_SEMANTICS,
)
from finn.dataflow.ops.mvau.designs.supplied_dot_product import (
    SuppliedDotProductDesign,
    WeightSupply,
)

CLOCK_PERIOD_NS = 4.0


class Source(Space):
    """The graph-side facts, including whether the matrix is a constant."""

    repetitions = Problem(int)
    matrix_width = Problem(int)
    matrix_height = Problem(int)
    activation_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    weight_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    accumulator_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    output_type = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    narrow_weights = Problem(bool)
    computation_profile = Problem(MvauComputationProfile)
    target_dsp = Problem(DspBlock)
    clock_period_ns = Problem(float)
    initializer_present = Problem(bool)

    design = Subspace(
        SuppliedDotProductDesign,
        name="supplied",
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation_type,
        weight_type=weight_type,
        accumulator_type=accumulator_type,
        output_type=output_type,
        narrow_weights=narrow_weights,
        computation_profile=computation_profile,
        target_dsp=target_dsp,
        clock_period_ns=clock_period_ns,
        initializer_present=initializer_present,
    )


_CANDIDATE = {
    WeightSupply.EXTERNAL: "dotp_axi",
    WeightSupply.EMBEDDED: "dotp_axi_embedded",
    WeightSupply.DECOUPLED: "dotp_axi",
}


def _occurrence(
    supply: WeightSupply,
    *,
    initializer: bool = True,
    pe: int = 2,
    simd: int = 2,
) -> SuppliedDotProductDesign:
    root = Source.start(
        {
            Source.repetitions: 2,
            Source.matrix_width: 8,
            Source.matrix_height: 4,
            Source.activation_type: DataType["INT8"],
            Source.weight_type: DataType["INT8"],
            Source.accumulator_type: DataType["INT32"],
            Source.output_type: DataType["INT32"],
            Source.narrow_weights: False,
            Source.computation_profile: MvauComputationProfile(
                AccumulationMode.INTEGER, ActivationMode.NONE
            ),
            Source.target_dsp: DspBlock.DSP58,
            Source.clock_period_ns: CLOCK_PERIOD_NS,
            Source.initializer_present: initializer,
        },
        namespace="mvau",
    )
    design = cast(SuppliedDotProductDesign, root.design)
    design = (
        design.assign(SuppliedDotProductDesign.pe, pe)
        .assign(SuppliedDotProductDesign.simd, simd)
        .assign(SuppliedDotProductDesign.weight_supply, supply)
    )
    view = design.compute
    design = cast(SuppliedDotProductDesign, view.select(_CANDIDATE[supply]).root.design)
    # Pumping is the compute Kernel's own physical Decision either way; the
    # embedded candidate inherits it along with the rest of DotpAxi.
    compute = cast(DotpAxiKernel, design.compute.alternative(_CANDIDATE[supply]))
    return cast(
        SuppliedDotProductDesign,
        compute.assign(DotpAxiKernel.compute_pumping, False).root.design,
    )


def _network(design: SuppliedDotProductDesign):
    answer = design.dataflow.accepted_answer
    assert isinstance(answer, Decided), answer
    return answer.value


def test_external_streaming_gives_the_matrix_a_boundary_and_no_supplier() -> None:
    network = _network(_occurrence(WeightSupply.EXTERNAL))
    assert {node.id for node in network.nodes} == {"replay", "compute"}
    assert {edge.id for edge in network.edges} == {"activation_replay"}
    assert {item.id for item in network.boundaries} == {"activation", "weight", "output"}
    weight = next(item for item in network.boundaries if item.id == "weight")
    assert weight.endpoint.node_id == "compute"


def test_embedded_supply_gives_the_compute_region_no_weight_port_at_all() -> None:
    design = _occurrence(WeightSupply.EMBEDDED)
    network = _network(design)
    assert {node.id for node in network.nodes} == {"replay", "compute"}
    assert {edge.id for edge in network.edges} == {"activation_replay"}
    # No boundary to substitute, because there is nothing to substitute for.
    assert {item.id for item in network.boundaries} == {"activation", "output"}
    compute = network.node("compute")
    assert {item.port.id for item in compute.region.inputs} == {"activation"}
    assert design.selected("compute") == Decided("dotp_axi_embedded")
    assert design.region_family("compute") == Decided(("mvau.dot_product.embedded", "1"))


def test_decoupled_supply_gives_the_matrix_its_own_node_and_edge() -> None:
    design = _occurrence(WeightSupply.DECOUPLED)
    network = _network(design)
    assert {node.id for node in network.nodes} == {"replay", "compute", "memory"}
    assert {edge.id for edge in network.edges} == {"activation_replay", "weight_supply_edge"}
    # The matrix no longer crosses the Design's boundary; it is produced inside.
    assert {item.id for item in network.boundaries} == {"activation", "output"}
    edge = next(item for item in network.edges if item.id == "weight_supply_edge")
    assert edge.source.node_id == "memory"
    assert edge.sinks[0].endpoint.node_id == "compute"
    memory = design.kernel("memory")
    assert isinstance(memory, Decided)
    assert isinstance(memory.value, MemstreamKernel)


def test_the_compute_region_is_identical_in_external_and_decoupled_supply() -> None:
    """Where the matrix comes from does not change what the arithmetic means."""

    external = _network(_occurrence(WeightSupply.EXTERNAL)).node("compute").region
    decoupled = _network(_occurrence(WeightSupply.DECOUPLED)).node("compute").region
    assert external == decoupled
    # And the embedded one genuinely differs, in the one interface.
    embedded = _network(_occurrence(WeightSupply.EMBEDDED)).node("compute").region
    assert embedded != external
    assert embedded.schedule == external.schedule
    assert embedded.outputs == external.outputs


def test_the_supplier_produces_exactly_what_the_consumer_requires() -> None:
    """The edge is checked position by position, not by a shared import."""

    design = _occurrence(WeightSupply.DECOUPLED)
    network = _network(design)
    produced = network.node("memory").region.output_interface("weight").port
    consumed = network.node("compute").region.input_interface("weight").port
    assert produced.beat_sequence == consumed.beat_sequence
    assert produced.operand.shape == consumed.operand.shape


def test_every_mode_resolves_its_network_with_no_build_unit_available() -> None:
    """Two of the three suppliers cannot be built yet, and it changes nothing."""

    for supply in (WeightSupply.EMBEDDED, WeightSupply.DECOUPLED):
        design = _occurrence(supply)
        assessment = design.dataflow
        assert assessment.readiness.ready is True
        assert isinstance(assessment.accepted_answer, Decided)

    embedded = _occurrence(WeightSupply.EMBEDDED).kernel("compute")
    assert isinstance(embedded, Decided)
    unavailable = embedded.value.physical.accepted_answer
    assert isinstance(unavailable, Absent)
    assert any(finding.code == "kernel-physically-unsupported" for finding in unavailable.findings)

    memory = _occurrence(WeightSupply.DECOUPLED).kernel("memory")
    assert isinstance(memory, Decided)
    assert isinstance(memory.value.physical.accepted_answer, Absent)


def test_a_mode_that_keeps_the_matrix_locally_needs_one_to_keep() -> None:
    for supply in (WeightSupply.EMBEDDED, WeightSupply.DECOUPLED):
        answer = _occurrence(supply, initializer=False).dataflow.accepted_answer
        assert isinstance(answer, Absent), supply
        assert "mvau-local-weights-need-an-initializer" in {
            finding.code for finding in answer.findings
        }


def test_an_initializer_neither_forces_nor_forbids_external_streaming() -> None:
    """The fact narrows what is available; it never picks."""

    for initializer in (True, False):
        design = _occurrence(WeightSupply.EXTERNAL, initializer=initializer)
        assert isinstance(design.dataflow.accepted_answer, Decided)
    # And with an initializer present, all three remain available.
    for supply in WeightSupply:
        assert isinstance(_occurrence(supply).dataflow.accepted_answer, Decided)


def test_the_mode_is_uncommitted_until_it_is_chosen() -> None:
    root = Source.start(
        {
            Source.repetitions: 2,
            Source.matrix_width: 8,
            Source.matrix_height: 4,
            Source.activation_type: DataType["INT8"],
            Source.weight_type: DataType["INT8"],
            Source.accumulator_type: DataType["INT32"],
            Source.output_type: DataType["INT32"],
            Source.narrow_weights: False,
            Source.computation_profile: MvauComputationProfile(
                AccumulationMode.INTEGER, ActivationMode.NONE
            ),
            Source.target_dsp: DspBlock.DSP58,
            Source.clock_period_ns: CLOCK_PERIOD_NS,
            Source.initializer_present: True,
        },
        namespace="mvau",
    )
    design = cast(SuppliedDotProductDesign, root.design)
    design = design.assign(SuppliedDotProductDesign.pe, 2).assign(SuppliedDotProductDesign.simd, 2)
    assert isinstance(design.dataflow.accepted_answer, Unresolved)
    assert design.is_active("memory") == Decided(False) or isinstance(
        design.is_active("memory"), Unresolved
    )


@pytest.mark.parametrize("supply", list(WeightSupply))
def test_the_memory_role_is_present_exactly_when_the_mode_says(supply: WeightSupply) -> None:
    design = _occurrence(supply)
    expected = supply is WeightSupply.DECOUPLED
    assert design.is_active("memory") == Decided(expected)
    assert design.is_active("compute") == Decided(True)
    assert ("memory" in {node.id for node in _network(design).nodes}) is expected
