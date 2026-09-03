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

from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.region import DataflowRegion, Port

REPLAY_NODE = "replay"
DOT_PRODUCT_NODE = "compute"
ACTIVATION_EDGE = "activation_replay"


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
                        PositionMap.identity(produced.beat_sequence.image),
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


__all__ = [
    "ACTIVATION_EDGE",
    "DOT_PRODUCT_NODE",
    "REPLAY_NODE",
    "construct_decomposed_mvau_network",
]
