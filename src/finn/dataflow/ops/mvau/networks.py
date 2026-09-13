# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical Networks over the normalized MVAU Regions.

The Region constructors next door say what each half of a decomposed MVAU
means; this module says how the two halves are wired when they are placed
together.  It is the reference answer a Design's own Network projection is
compared against, so it constructs the value directly and consults no
declaration, decision or Design.

The three names below are the canonical node and edge ids of the decomposed
form.  They are values, not display names: a Network that used different ones
would not be comparable to this one.
"""

from __future__ import annotations

from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.region import DataflowRegion, Port

REPLAY_NODE = "replay"
DOT_PRODUCT_NODE = "compute"
MEMORY_NODE = "memory"
ACTIVATION_EDGE = "activation_replay"
WEIGHT_EDGE = "weight_supply_edge"


def construct_decomposed_mvau_network(
    replay_region: DataflowRegion, dot_product_region: DataflowRegion
) -> DataflowNetwork:
    """Assemble activation replay and dot product into one flat Network."""

    produced: Port = replay_region.output_interface("activation_out").port
    return DataflowNetwork(
        (
            NetworkNode(REPLAY_NODE, replay_region),
            NetworkNode(DOT_PRODUCT_NODE, dot_product_region),
        ),
        (
            Edge(
                ACTIVATION_EDGE,
                RegionEndpoint(REPLAY_NODE, "activation_out"),
                (
                    SinkContract(
                        RegionEndpoint(DOT_PRODUCT_NODE, "activation"),
                        PositionMap.identity(produced.beat_sequence.image_set),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint(REPLAY_NODE, "activation_in"),
                replay_region.input_interface("activation_in").port.beat_sequence,
            ),
            BoundaryContract(
                "weight",
                RegionEndpoint(DOT_PRODUCT_NODE, "weight"),
                dot_product_region.input_interface("weight").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint(DOT_PRODUCT_NODE, "output"),
                dot_product_region.output_interface("output").port.beat_sequence,
            ),
        ),
    )


def construct_embedded_mvau_network(
    replay_region: DataflowRegion, dot_product_region: DataflowRegion
) -> DataflowNetwork:
    """Assemble Replay with a dot product whose weight is an InternalInput."""

    produced: Port = replay_region.output_interface("activation_out").port
    return DataflowNetwork(
        (
            NetworkNode(REPLAY_NODE, replay_region),
            NetworkNode(DOT_PRODUCT_NODE, dot_product_region),
        ),
        (
            Edge(
                ACTIVATION_EDGE,
                RegionEndpoint(REPLAY_NODE, "activation_out"),
                (
                    SinkContract(
                        RegionEndpoint(DOT_PRODUCT_NODE, "activation"),
                        PositionMap.identity(produced.beat_sequence.image_set),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint(REPLAY_NODE, "activation_in"),
                replay_region.input_interface("activation_in").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint(DOT_PRODUCT_NODE, "output"),
                dot_product_region.output_interface("output").port.beat_sequence,
            ),
        ),
    )


def construct_decoupled_mvau_network(
    replay_region: DataflowRegion,
    dot_product_region: DataflowRegion,
    memory_region: DataflowRegion,
) -> DataflowNetwork:
    """Assemble Replay, dot product, and the cyclic weight source."""

    activation: Port = replay_region.output_interface("activation_out").port
    weight: Port = memory_region.output_interface("weight").port
    return DataflowNetwork(
        (
            NetworkNode(REPLAY_NODE, replay_region),
            NetworkNode(DOT_PRODUCT_NODE, dot_product_region),
            NetworkNode(MEMORY_NODE, memory_region),
        ),
        (
            Edge(
                ACTIVATION_EDGE,
                RegionEndpoint(REPLAY_NODE, "activation_out"),
                (
                    SinkContract(
                        RegionEndpoint(DOT_PRODUCT_NODE, "activation"),
                        PositionMap.identity(activation.beat_sequence.image_set),
                    ),
                ),
            ),
            Edge(
                WEIGHT_EDGE,
                RegionEndpoint(MEMORY_NODE, "weight"),
                (
                    SinkContract(
                        RegionEndpoint(DOT_PRODUCT_NODE, "weight"),
                        PositionMap.identity(weight.beat_sequence.image_set),
                    ),
                ),
            ),
        ),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint(REPLAY_NODE, "activation_in"),
                replay_region.input_interface("activation_in").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint(DOT_PRODUCT_NODE, "output"),
                dot_product_region.output_interface("output").port.beat_sequence,
            ),
        ),
    )


__all__ = [
    "ACTIVATION_EDGE",
    "DOT_PRODUCT_NODE",
    "MEMORY_NODE",
    "REPLAY_NODE",
    "WEIGHT_EDGE",
    "construct_decoupled_mvau_network",
    "construct_decomposed_mvau_network",
    "construct_embedded_mvau_network",
]
