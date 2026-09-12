# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.model.maps import (
    CoordinateSet,
    MaterializationLimitError,
    RectangularDomain,
    encode_coordinate_map,
)
from finn.dataflow.model.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.model.presentation import (
    boundary_presented_position_set,
    unpresented_position_set,
)
from finn.dataflow.model.refs import RegionInputRef
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.model.region_validation import RegionValidationReport, validate_region
from finn.dataflow.ops.mvau.networks import construct_decomposed_mvau_network
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
)
from finn.dataflow.space.dataflow_value_semantics import (
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
)


def test_large_replay_and_connected_network_never_enter_expansion(monkeypatch) -> None:
    repetitions = 2
    matrix_width = 3
    matrix_height = 1_048_576
    pe = 1
    simd = 3
    replay = construct_activation_replay_region(
        repetitions,
        matrix_width,
        matrix_height,
        DataType["INT8"],
        pe,
        simd,
    )
    replay_twin = construct_activation_replay_region(
        repetitions,
        matrix_width,
        matrix_height,
        DataType["INT8"],
        pe,
        simd,
    )
    dot_product = construct_dot_product_region(
        repetitions,
        matrix_width,
        matrix_height,
        DataType["INT8"],
        DataType["INT8"],
        DataType["INT32"],
        pe,
        simd,
    )
    network = construct_decomposed_mvau_network(replay, dot_product)

    def refuse(*_args, **_kwargs):
        raise AssertionError("compact operation entered an expansion iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)

    assert validate_region(replay) == RegionValidationReport()
    assert validate_network(network) == NetworkValidationReport()
    assert replay == replay_twin
    assert hash(replay) == hash(replay_twin)
    assert DATAFLOW_REGION_SEMANTICS.freeze(replay) is replay
    assert DATAFLOW_NETWORK_SEMANTICS.freeze(network) is network

    input_sequence = replay.input_interface("activation_in").port.beat_sequence
    output_sequence = replay.output_interface("activation_out").port.beat_sequence
    assert input_sequence.position_at(1, 2) == (1, 2)
    middle = output_sequence.beat_count // 2
    assert output_sequence.position_at(middle, 1) == (1, 1)
    assert input_sequence.image_set.cardinality == repetitions * matrix_width
    assert (
        replay.input("X").requirements.required_position_set.cardinality
        == repetitions * matrix_width
    )
    assert boundary_presented_position_set(network, RegionInputRef("replay", "X")).is_full
    assert unpresented_position_set(network, RegionInputRef("replay", "X")).is_empty
    encoded = encode_coordinate_map(output_sequence.affine_map)
    assert encoded["kind"] == "affine_rank"
    assert len(str(encoded)) < 512


def test_materialization_limit_is_checked_before_iteration(monkeypatch) -> None:
    value = CoordinateSet.full(RectangularDomain((1_000_001,)))

    monkeypatch.setattr(
        CoordinateSet,
        "iter_coordinates",
        lambda *_args, **_kwargs: pytest.fail("iterator entered before budget refusal"),
    )
    with pytest.raises(MaterializationLimitError) as error:
        value.materialize(max_points=1_000_000)
    assert error.value.required == 1_000_001
    assert error.value.limit == 1_000_000
    assert error.value.unit == "points"
