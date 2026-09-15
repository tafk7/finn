# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.model.composition import (
    ImplementationPath,
    NetworkResult,
    ParentBoundary,
    ParentConnection,
    RegionResult,
    compose_network,
    qualify_network,
)
from finn.dataflow.model.network import PassCorrespondence
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.ops.mvau.networks import construct_decomposed_mvau_network
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
)


def _mvau() -> NetworkResult:
    replay = construct_activation_replay_region(1, 4, 4, DataType["INT3"], 2, 2)
    dotp = construct_dot_product_region(
        1,
        4,
        4,
        DataType["INT3"],
        DataType["INT3"],
        DataType["INT3"],
        2,
        2,
    )
    return NetworkResult(construct_decomposed_mvau_network(replay, dotp))


def test_three_level_qualification_preserves_flat_contracts_and_rewires_boundaries() -> None:
    base = _mvau()
    left = qualify_network(ImplementationPath(("left",)), base)
    right = qualify_network(ImplementationPath(("right",)), base)
    pipeline = compose_network(
        children=(left, right),
        connections=(ParentConnection("between", "left/output", ("right/activation",)),),
        boundaries=(
            ParentBoundary("activation", "left/activation"),
            ParentBoundary("left_weight", "left/weight"),
            ParentBoundary("right_weight", "right/weight"),
            ParentBoundary("output", "right/output"),
        ),
    )
    top = qualify_network(ImplementationPath(("outer",)), pipeline)

    assert validate_network(pipeline.network).issues == ()
    assert validate_network(top.network).issues == ()
    assert {node.id for node in top.network.nodes} == {
        "outer/left/replay",
        "outer/left/compute",
        "outer/right/replay",
        "outer/right/compute",
    }
    assert {edge.id for edge in top.network.edges} == {
        "outer/left/activation_replay",
        "outer/right/activation_replay",
        "outer/between",
    }
    connection = next(edge for edge in top.network.edges if edge.id == "outer/between")
    assert connection.source.node_id == "outer/left/compute"
    assert connection.sinks[0].endpoint.node_id == "outer/right/replay"
    assert connection.pass_correspondence is PassCorrespondence.ONE_TO_ONE
    original = base.network.edges[0]
    qualified = next(
        edge for edge in top.network.edges if edge.id == "outer/left/activation_replay"
    )
    assert qualified.sinks[0].position_map == original.sinks[0].position_map
    assert qualified.transport == original.transport
    assert qualified.pass_correspondence == original.pass_correspondence
    assert {boundary.id for boundary in top.network.boundaries} == {
        "outer/activation",
        "outer/left_weight",
        "outer/right_weight",
        "outer/output",
    }


def test_reused_composites_and_leaves_keep_distinct_occurrence_paths() -> None:
    base = _mvau()
    left = qualify_network(ImplementationPath(("left",)), base)
    right = qualify_network(ImplementationPath(("right",)), base)
    assert left.network.node("left/replay").region == right.network.node("right/replay").region
    assert left.network.node("left/replay").id != right.network.node("right/replay").id
    assert left.use_path != right.use_path


def test_region_result_is_the_typed_leaf_logical_value() -> None:
    region = _mvau().network.node("replay").region
    assert RegionResult(region).region is region
