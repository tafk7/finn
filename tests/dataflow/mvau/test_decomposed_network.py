# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4 gate: the decomposed MVAU as an assembled, resolvable Network.

Fixture 1 -- the outer boundary equals the monolithic Region's, across a
geometry matrix including the degenerate folds.
Fixture 2 -- Region and Network validation.
Fixture 3 -- the pools resolve from one problem and one set of choices.
Fixture 4 -- walking the declared beats and requirements reproduces ``X @ W``.

Fixture 4 is what stops the first three from being a tautology: it reads the
Region's own requirement map as the specification of what the hardware must
consume, and checks that specification computes a matrix product.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from typing import cast

import numpy as np  # type: ignore[import-not-found]
import pytest

from dataflow.mvau_op_facts import compute_pool_context
from finn.dataflow.authoring import assemble_specs
from finn.dataflow.design import Decided, Engine
from finn.dataflow.kernels import bind_kernel
from finn.dataflow.mvau.compute_kernels import (
    DECOMPOSED_MVAU_KERNELS,
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
)
from finn.dataflow.mvau.decomposed import (
    ACTIVATION_EDGE,
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
    REPLAY_POOL,
    ActivationReplayKernel,
    DotProductKernel,
    construct_decomposed_mvau_network,
)
from finn.dataflow.mvau.regions import construct_standard_streamed_mvau_region
from finn.dataflow.mvau_problem import MVAUComputationProfile, MVAUProblemPaths
from finn.dataflow.network_validation import validate_network
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import validate_region

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]

#: ``(R, MW, MH, PE, SIMD)``.
GEOMETRIES = [
    (2, 4, 6, 2, 2),
    (1, 8, 8, 4, 4),
    (3, 6, 4, 1, 3),
    (2, 4, 4, 4, 4),  # NF = 1
    (2, 4, 6, 2, 4),  # SF = 1
    (1, 3, 5, 5, 3),  # PE = MH and SF = 1
]


def _bound(
    geometry: tuple[int, int, int, int, int],
) -> tuple[ActivationReplayKernel, DotProductKernel]:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    pools = DECOMPOSED_MVAU_KERNELS
    engine = Engine()
    space = engine.validate(
        assemble_specs(
            (
                compute_pool_context(),
                MVAU_COMPUTE_SELECTION.build_spec(),
                MVAU_REPLAY_SELECTION.build_spec(),
            )
        )
    )
    point = engine.start(
        space,
        {
            MVAUProblemPaths.REPETITIONS: repetitions,
            MVAUProblemPaths.MATRIX_WIDTH: matrix_width,
            MVAUProblemPaths.MATRIX_HEIGHT: matrix_height,
            MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE: INT8,
            MVAUProblemPaths.WEIGHT_ELEMENT_TYPE: INT8,
            MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE: INT16,
            MVAUProblemPaths.OUTPUT_ELEMENT_TYPE: INT16,
            MVAUProblemPaths.COMPUTATION_PROFILE: MVAUComputationProfile.ACCUMULATOR_INTEGER,
            MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: True,
            MVAUProblemPaths.RUNTIME_WRITABLE: False,
        },
    )
    point = engine.commit_assignments(
        point,
        {
            MVAU_COMPUTE_SELECTION.paths.kernel: DotProductKernel.id,
            MVAU_REPLAY_SELECTION.paths.kernel: ActivationReplayKernel.id,
            pools.pe.path: pe,
            pools.simd.path: simd,
        },
    ).point
    replay = bind_kernel(engine, MVAU_REPLAY_SELECTION, point)
    compute = bind_kernel(engine, MVAU_COMPUTE_SELECTION, point)
    assert isinstance(replay, Decided) and isinstance(compute, Decided)
    return (
        cast(ActivationReplayKernel, replay.value),
        cast(DotProductKernel, compute.value),
    )


def _monolithic(geometry: tuple[int, int, int, int, int]) -> DataflowRegion:
    repetitions, matrix_width, matrix_height, pe, simd = geometry
    return construct_standard_streamed_mvau_region(
        repetitions, matrix_width, matrix_height, INT8, INT8, INT16, pe, simd
    )


# -- fixture 3: the pools resolve together -----------------------------------


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_both_pools_resolve_from_one_problem_and_one_folding(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """One ``pe`` and one ``simd``, owned by the consumer, reach both Kernels."""

    replay, compute = _bound(geometry)

    assert isinstance(replay, ActivationReplayKernel)
    assert isinstance(compute, DotProductKernel)
    assert set(compute.demands) == {"weight"}
    assert replay.demands == {}
    # Replay owns no choices; it was told the folding.
    assert replay.assignments == {}
    assert sorted(cast(int, value) for value in compute.assignments.values()) == sorted(
        geometry[3:5]
    )


# -- fixture 2: validation ---------------------------------------------------


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_assembled_network_is_structurally_well_formed(
    geometry: tuple[int, int, int, int, int],
) -> None:
    replay, compute = _bound(geometry)
    assert list(validate_region(replay.region)) == []
    assert list(validate_region(compute.region)) == []

    network = construct_decomposed_mvau_network(replay.region, compute.region)
    assert list(validate_network(network)) == []
    assert {node.id for node in network.nodes} == {REPLAY_NODE, DOT_PRODUCT_NODE}
    assert [edge.id for edge in network.edges] == [ACTIVATION_EDGE]


# -- fixture 1: the boundary is unchanged ------------------------------------


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_network_boundary_equals_the_monolithic_region_boundary(
    geometry: tuple[int, int, int, int, int],
) -> None:
    """The decomposition is invisible from outside, at every geometry."""

    replay, compute = _bound(geometry)
    network = construct_decomposed_mvau_network(replay.region, compute.region)
    boundaries = {item.id: item for item in network.boundaries}
    monolithic = _monolithic(geometry)
    inputs = {item.port.id: item.port for item in monolithic.inputs}

    assert boundaries["activation"].external_beat_sequence == inputs["activation"].beat_sequence
    assert boundaries["weight"].external_beat_sequence == inputs["weight"].beat_sequence
    assert boundaries["output"].external_beat_sequence == monolithic.outputs[0].port.beat_sequence


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_demanded_weight_port_is_the_monolithic_weight_port(
    geometry: tuple[int, int, int, int, int],
) -> None:
    _replay, compute = _bound(geometry)
    monolithic = _monolithic(geometry)
    expected = next(item.port for item in monolithic.inputs if item.port.id == "weight")

    assert compute.demands["weight"].beat_sequence == expected.beat_sequence
    assert compute.demands["weight"].operand == expected.operand


# -- fixture 4: the declared contract computes a matrix product --------------


def _evaluate(region: DataflowRegion, activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Run the Region's own requirement map as if it were the hardware.

    For each scheduled iteration, take exactly the activation and weight
    positions the Region says it consumes, accumulate their products, and emit
    each output position at the iteration its availability names.  Nothing here
    knows it is an MVAU; if the declared contract is wrong, the result is wrong.
    """

    activations = _by_iteration(region, "activation")
    weights = _by_iteration(region, "weight")
    availability = dict(region.outputs[0].availability.entries)
    repetitions, matrix_height = activation.shape[0], weight.shape[0]

    accumulator: dict[tuple[int, ...], int] = {}
    for iteration, activation_positions in activations.items():
        columns = {position[1] for position in activation_positions}
        rows = {position[0] for position in activation_positions}
        assert len(rows) == 1, "one iteration reads one repetition"
        row = rows.pop()
        for neuron, column in weights[iteration]:
            assert column in columns, "a weight column with no activation to meet it"
            key = (row, neuron)
            accumulator[key] = accumulator.get(key, 0) + int(activation[row, column]) * int(
                weight[neuron, column]
            )

    result = np.zeros((repetitions, matrix_height), dtype=np.int64)
    for position in availability:
        result[position[0], position[1]] = accumulator[position]
    return result


def _by_iteration(
    region: DataflowRegion, interface: str
) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    grouped: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for (iteration, position), _count in region.input_interface(interface).requirements.entries:
        grouped.setdefault(iteration, []).append(position)
    return grouped


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_declared_contract_computes_the_matrix_product(
    geometry: tuple[int, int, int, int, int],
) -> None:
    repetitions, matrix_width, matrix_height, _pe, _simd = geometry
    generator = np.random.default_rng(seed=repetitions * 1000 + matrix_width * 10 + matrix_height)
    activation = generator.integers(-8, 8, size=(repetitions, matrix_width))
    weight = generator.integers(-8, 8, size=(matrix_height, matrix_width))

    _replay, compute = _bound(geometry)
    assert np.array_equal(_evaluate(compute.region, activation, weight), activation @ weight.T)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_the_decomposed_and_monolithic_regions_compute_the_same_thing(
    geometry: tuple[int, int, int, int, int],
) -> None:
    repetitions, matrix_width, matrix_height, _pe, _simd = geometry
    generator = np.random.default_rng(seed=7)
    activation = generator.integers(-8, 8, size=(repetitions, matrix_width))
    weight = generator.integers(-8, 8, size=(matrix_height, matrix_width))

    _replay, compute = _bound(geometry)
    assert np.array_equal(
        _evaluate(compute.region, activation, weight),
        _evaluate(_monolithic(geometry), activation, weight),
    )


# -- pool identity -----------------------------------------------------------


def test_the_two_pools_are_separate_choices() -> None:
    assert MVAU_COMPUTE_SELECTION.paths.kernel is not None
    assert MVAU_REPLAY_SELECTION.name == REPLAY_POOL
    assert MVAU_COMPUTE_SELECTION.paths.kernel != MVAU_REPLAY_SELECTION.paths.kernel


def test_the_folding_is_owned_by_the_consumer_not_the_producer() -> None:
    """Replay declares no decisions; the paths it reads belong to the dot product."""

    pools = DECOMPOSED_MVAU_KERNELS
    replay = MVAU_REPLAY_SELECTION.kernels[0]
    assert replay.spec.decisions == ()
    assert str(pools.pe.path).startswith(MVAU_COMPUTE_SELECTION.name)
    assert str(pools.simd.path).startswith(MVAU_COMPUTE_SELECTION.name)

    read = {
        dependency.path
        for item in replay.spec.properties
        for dependency in item.evaluator.dependencies
    }
    assert {pools.pe.path, pools.simd.path} <= read


def test_a_folding_disagreement_is_unrepresentable() -> None:
    """There is one ``pe`` path and one ``simd`` path, so the two cannot differ."""

    decisions = {
        item.path
        for kernel in (
            MVAU_COMPUTE_SELECTION.kernel(DotProductKernel.id),
            MVAU_REPLAY_SELECTION.kernels[0],
        )
        for item in kernel.spec.decisions
    }
    assert sum(str(path).endswith(".pe") for path in decisions) == 1
    assert sum(str(path).endswith(".simd") for path in decisions) == 1
