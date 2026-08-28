# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU standard streamed form as two Kernels and a Network.

``ActivationReplayKernel`` presents each activation row once per neuron fold;
``DotProductKernel`` does the arithmetic.  The monolithic Region did both, with
the replay hidden inside its schedule.

Both are authored through ``KernelDesign``, so they read only what this module
wires into them.  ``DotProductKernel`` owns ``pe`` and ``simd`` because it is
the thing those numbers fold; ``ActivationReplayKernel`` owns nothing and
derives its geometry from the folding it must feed.  That direction is the
supply waterfall: a producer presents what its consumer's configuration
requires, and never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.authoring.kernel_design import (
    KernelDesign,
    declare_kernel_design,
    kernel_namespace,
)
from finn.dataflow.authoring.op_design import ProblemProvenance
from finn.dataflow.authoring.scope import Ref, divisors_of
from finn.dataflow.kernels import Kernel, KernelDeclaration, KernelSelection
from finn.dataflow.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.mvau_problem import MVAU_PROBLEM, MVAUProblem
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.region import DataflowRegion, NumericElementType, Port

#: The two pools this decomposition adds.  Each has one member today; they are
#: separate pools because they are separate choices, not because either is
#: currently plural.
REPLAY_POOL = "mvau.replay"
DOT_PRODUCT_POOL = "mvau.dotp"

#: The one parameter interface the dot-product half needs supplied.
WEIGHT_INTERFACE = "weight"

#: Node ids inside the assembled Network.
REPLAY_NODE = "replay"
DOT_PRODUCT_NODE = "compute"

#: The single internal edge.
ACTIVATION_EDGE = "activation_replay"


@dataclass(frozen=True)
class DotProductInputs:
    """What the dot-product Kernel is allowed to read."""

    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]


@dataclass(frozen=True)
class ActivationReplayInputs:
    """What the replay Kernel is allowed to read.

    ``pe`` and ``simd`` arrive from the dot-product Kernel's choices, wired in
    by the operation.  Replay does not decide them and does not name their
    paths; it is told what folding it has to feed.
    """

    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    pe: Ref[int]
    simd: Ref[int]


class DotProductKernel(Kernel):
    """Folded multiply-accumulate over an already-expanded activation stream."""

    id = "dot_product"
    version = "1"

    @classmethod
    def define_design(cls, design: KernelDesign[DotProductInputs]) -> None:
        facts = design.inputs
        pe = design.choice("pe", int, domain=divisors_of(facts.matrix_height))
        simd = design.choice("simd", int, domain=divisors_of(facts.matrix_width))
        design.region(
            dependencies={
                "repetitions": facts.repetitions,
                "matrix_width": facts.matrix_width,
                "matrix_height": facts.matrix_height,
                "activation_element_type": facts.activation_element_type,
                "weight_element_type": facts.weight_element_type,
                "output_element_type": facts.output_element_type,
                "pe": pe,
                "simd": simd,
            },
            evaluate=construct_dot_product_region,
        )
        design.demand(
            WEIGHT_INTERFACE,
            dependencies={
                "repetitions": facts.repetitions,
                "matrix_width": facts.matrix_width,
                "matrix_height": facts.matrix_height,
                "weight_element_type": facts.weight_element_type,
                "pe": pe,
                "simd": simd,
            },
            evaluate=construct_standard_mvau_weight_port,
        )


class ActivationReplayKernel(Kernel):
    """Present each activation row once per neuron fold.

    Retained even at one neuron fold, where it is an identity: the Network
    shape should not depend on the matrix geometry, and eliding the physical
    buffer is a provider's decision, not a semantic one.
    """

    id = "activation_replay"
    version = "1"

    @classmethod
    def define_design(cls, design: KernelDesign[ActivationReplayInputs]) -> None:
        facts = design.inputs
        design.region(
            dependencies={
                "repetitions": facts.repetitions,
                "matrix_width": facts.matrix_width,
                "matrix_height": facts.matrix_height,
                "activation_element_type": facts.activation_element_type,
                "pe": facts.pe,
                "simd": facts.simd,
            },
            evaluate=construct_activation_replay_region,
        )


@dataclass(frozen=True)
class DecomposedMVAUPools:
    """The two pools and the folding handles the operation wired between them."""

    dot_product: KernelSelection
    activation_replay: KernelSelection
    pe: Ref[int]
    simd: Ref[int]


def build_decomposed_mvau_pools(
    problem: MVAUProblem = MVAU_PROBLEM,
    *,
    provenance: ProblemProvenance | None = None,
) -> DecomposedMVAUPools:
    """Declare both Kernels and wire the folding from consumer to producer.

    The dot-product Kernel is declared first because it owns the folding; the
    replay Kernel is then told what that folding is.  Declaration order here is
    the waterfall made literal.
    """

    dot_product, dot_product_design = declare_kernel_design(
        DotProductKernel,
        kernel_namespace(DOT_PRODUCT_POOL, DotProductKernel.id),
        DotProductInputs(
            repetitions=problem.repetitions,
            matrix_width=problem.matrix_width,
            matrix_height=problem.matrix_height,
            activation_element_type=problem.activation_element_type,
            weight_element_type=problem.weight_element_type,
            output_element_type=problem.output_element_type,
        ),
        provenance=provenance,
    )
    pe = dot_product_design.handle("pe", int)
    simd = dot_product_design.handle("simd", int)
    replay: KernelDeclaration = declare_kernel_design(
        ActivationReplayKernel,
        kernel_namespace(REPLAY_POOL, ActivationReplayKernel.id),
        ActivationReplayInputs(
            repetitions=problem.repetitions,
            matrix_width=problem.matrix_width,
            matrix_height=problem.matrix_height,
            activation_element_type=problem.activation_element_type,
            pe=pe,
            simd=simd,
        ),
        provenance=provenance,
    )[0]
    return DecomposedMVAUPools(
        KernelSelection(DOT_PRODUCT_POOL, (dot_product,)),
        KernelSelection(REPLAY_POOL, (replay,)),
        pe,
        simd,
    )


def construct_decomposed_mvau_network(
    replay_region: DataflowRegion, dot_product_region: DataflowRegion
) -> DataflowNetwork:
    """Assemble replay and dot product into one Network.

    One internal edge under an identity position map over the expanded image.
    The three external boundaries are the same ``BeatSequence`` values the
    monolithic standard streamed Region presented, which is what makes the
    decomposition invisible from outside.
    """

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
    "DOT_PRODUCT_POOL",
    "REPLAY_NODE",
    "REPLAY_POOL",
    "WEIGHT_INTERFACE",
    "ActivationReplayInputs",
    "ActivationReplayKernel",
    "DecomposedMVAUPools",
    "DotProductInputs",
    "DotProductKernel",
    "build_decomposed_mvau_pools",
    "construct_decomposed_mvau_network",
]
