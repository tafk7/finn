# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3.5 gate: the two Regions the standard streamed MVAU decomposes into.

``ActivationReplayKernel`` expands the compact activation sequence to one
presentation per neuron fold; ``DotProductKernel`` is the former monolithic
Region reading that expanded sequence.  Both must validate, and the
decomposition must be invisible at the outer boundary.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

import pytest

from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.region import BeatSequence, DataflowRegion
from finn.dataflow.region_validation import validate_region

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]

#: ``(R, MW, MH, PE, SIMD)``, including the degenerate folds that break naive
#: constructions: one neuron fold, one synapse fold, and PE spanning MH.
GEOMETRIES = [
    (2, 4, 6, 2, 2),
    (1, 8, 8, 4, 4),
    (3, 6, 4, 1, 3),
    (2, 4, 4, 4, 4),  # NF = 1
    (2, 4, 6, 2, 4),  # SF = 1
    (1, 3, 5, 5, 3),  # PE = MH and SF = 1
]


def _replay(geometry: tuple[int, int, int, int, int]) -> DataflowRegion:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    return construct_activation_replay_region(
        repetitions, matrix_width, matrix_height, INT8, pe, simd
    )


def _dot_product(geometry: tuple[int, int, int, int, int]) -> DataflowRegion:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    return construct_dot_product_region(
        repetitions, matrix_width, matrix_height, INT8, INT8, INT16, pe, simd
    )


def _monolithic(geometry: tuple[int, int, int, int, int]) -> DataflowRegion:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    return construct_standard_streamed_mvau_region(
        repetitions, matrix_width, matrix_height, INT8, INT8, INT16, pe, simd
    )


def _image(sequence: BeatSequence) -> set[tuple[int, ...]]:
    return {position for beat in sequence.beats for position in beat}


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_both_regions_are_structurally_well_formed(
    geometry: tuple[int, int, int, int, int],
) -> None:
    assert list(validate_region(_replay(geometry))) == []
    assert list(validate_region(_dot_product(geometry))) == []


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_availability_domain_equals_the_output_beat_image(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """The canon obligation, stated directly rather than left to the validator."""

    for region in (_replay(geometry), _dot_product(geometry)):
        for interface in region.outputs:
            domain = {position for position, _ in interface.availability.entries}
            assert domain == _image(interface.port.beat_sequence)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_replay_multiplies_occurrences_without_changing_the_image(
    geometry: tuple[int, int, int, int, int],
) -> None:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    region = _replay(geometry)
    incoming = region.inputs[0].port.beat_sequence
    outgoing = region.outputs[0].port.beat_sequence

    assert len(incoming.beats) == repetitions * synapse_folds
    assert len(outgoing.beats) == repetitions * neuron_folds * synapse_folds
    assert incoming.elements_per_beat == outgoing.elements_per_beat == simd
    assert _image(incoming) == _image(outgoing)
    assert region.inputs[0].port.operand == region.outputs[0].port.operand


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_internal_edge_matches_exactly(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """Replay's output is the dot-product Region's activation input, verbatim."""

    produced = _replay(geometry).outputs[0].port
    consumed = _dot_product(geometry).inputs[0].port

    assert consumed.id == "activation"
    assert produced.beat_sequence == consumed.beat_sequence
    assert produced.operand == consumed.operand


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_outer_boundary_is_unchanged_by_the_decomposition(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """The primary acceptance criterion: equal ``BeatSequence`` values outside.

    Padding, bit order, and wire layout are binding-owned and not claimed here.
    """

    monolithic = _monolithic(geometry)
    replay = _replay(geometry)
    dot_product = _dot_product(geometry)

    monolithic_ports = {item.port.id: item.port for item in monolithic.inputs}
    assert replay.inputs[0].port.beat_sequence == monolithic_ports["activation"].beat_sequence
    assert replay.inputs[0].port.operand == monolithic_ports["activation"].operand

    weight = next(item.port for item in dot_product.inputs if item.port.id == "weight")
    assert weight.beat_sequence == monolithic_ports["weight"].beat_sequence
    assert weight.operand == monolithic_ports["weight"].operand

    assert dot_product.outputs[0].port.beat_sequence == monolithic.outputs[0].port.beat_sequence
    assert dot_product.outputs[0].port.operand == monolithic.outputs[0].port.operand


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_only_the_activation_boundary_differs_from_the_monolithic_region(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """Requirements, weights, output, and schedule are untouched."""

    _repetitions, _matrix_width, matrix_height, pe, _simd = geometry
    monolithic = _monolithic(geometry)
    dot_product = _dot_product(geometry)

    assert dot_product.schedule == monolithic.schedule
    assert dot_product.outputs == monolithic.outputs
    for left, right in zip(dot_product.inputs, monolithic.inputs, strict=True):
        assert left.requirements == right.requirements

    differs = dot_product.inputs[0].port.beat_sequence != monolithic.inputs[0].port.beat_sequence
    assert differs is (matrix_height // pe > 1)


@pytest.mark.parametrize("geometry", [g for g in GEOMETRIES if g[2] // g[3] == 1])
def test_replay_is_an_identity_at_one_neuron_fold(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """With one neuron fold there is nothing to replay, and that is not a bug.

    The semantic node is retained as an identity so the Network shape does not
    depend on the folding; eliding the physical buffer is a provider option.
    """

    region = _replay(geometry)
    assert region.inputs[0].port.beat_sequence == region.outputs[0].port.beat_sequence
    assert (
        _dot_product(geometry).inputs[0].port.beat_sequence
        == _monolithic(geometry).inputs[0].port.beat_sequence
    )


def test_a_geometry_the_folds_do_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="SIMD must divide matrix_width"):
        construct_activation_replay_region(2, 5, 4, INT8, 2, 2)
    with pytest.raises(ValueError, match="PE must divide matrix_height"):
        construct_dot_product_region(2, 4, 5, INT8, INT8, INT16, 2, 2)
