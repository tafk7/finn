# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three ways a matrix can reach the production dot-product Design.

```text
external    activation -> replay -> compute -> output
                          weight boundary --^

embedded    activation -> replay -> compute -> output
                          (weight is an InternalInput; there is no weight port)

decoupled   activation -> replay -> compute -> output
                          memory --weight-->  ^
```

All three differences are semantic. External and decoupled place the same
compute Region and differ in what presents its weight input. Embedded places a
different compute Region whose weight remains required as an ``InternalInput``
but has no dataflow port. Physical availability is separate: the embedded core
and memstream currently have no standalone module, while their Networks still
resolve.

Initializer presence does not choose the mode. ``initializer_present`` is a
source fact used only by this Design's admission policy: modes that keep the
matrix locally require an initializer. External supply remains available with
or without one.
"""

from __future__ import annotations

from enum import Enum

from finn.dataflow.designs.design import (
    EdgeSink,
    KernelChoice,
    NetworkBoundary,
    NetworkEdge,
)
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel, EmbeddedDotpAxiKernel
from finn.dataflow.kernels.memstream import MemstreamKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.ops.mvau.designs.base import SHARED_INPUTS, WeightedDotProductDesign
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Subspace,
    constraint,
    derived,
    reject,
)


class WeightSupply(str, Enum):
    """How the matrix reaches the compute Region."""

    EXTERNAL = "external"
    EMBEDDED = "embedded"
    DECOUPLED = "decoupled"


class DotProductDesign(WeightedDotProductDesign):
    """Replay and dot product with an explicitly selected weight supply."""

    id = "dot_product"
    version = "2"

    repetitions = WeightedDotProductDesign.repetitions
    matrix_width = WeightedDotProductDesign.matrix_width
    matrix_height = WeightedDotProductDesign.matrix_height
    activation_type = WeightedDotProductDesign.activation_type
    weight_type = WeightedDotProductDesign.weight_type
    accumulator_type = WeightedDotProductDesign.accumulator_type
    output_type = WeightedDotProductDesign.output_type
    narrow_weights = WeightedDotProductDesign.narrow_weights
    target_dsp = WeightedDotProductDesign.target_dsp
    clock_period_ns = WeightedDotProductDesign.clock_period_ns
    pe = WeightedDotProductDesign.pe
    simd = WeightedDotProductDesign.simd

    initializer_present = Input(bool)
    weight_supply = Decision(WeightSupply, values=tuple(WeightSupply))

    @derived(bool, supply=weight_supply)
    def streams_weights(*, supply: WeightSupply) -> bool:
        return supply is WeightSupply.EXTERNAL

    @derived(bool, supply=weight_supply)
    def decouples_weights(*, supply: WeightSupply) -> bool:
        return supply is WeightSupply.DECOUPLED

    replay = KernelChoice(
        Subspace(
            ReplayBufferKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            pe=pe,
            simd=simd,
        ),
    )

    compute = KernelChoice(
        Subspace(
            DotpAxiKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            weight_type=weight_type,
            accumulator_type=accumulator_type,
            output_type=output_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
        ),
        Subspace(
            EmbeddedDotpAxiKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            weight_type=weight_type,
            accumulator_type=accumulator_type,
            output_type=output_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
        ),
    )

    memory = KernelChoice(
        Subspace(
            MemstreamKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            weight_type=weight_type,
            pe=pe,
            simd=simd,
        ),
        when=decouples_weights,
    )

    activation_replay = NetworkEdge(
        replay.output("activation_out"),
        EdgeSink(compute.input("activation")),
    )
    weight_supply_edge = NetworkEdge(
        memory.output("weight"),
        EdgeSink(compute.input("weight")),
        when=decouples_weights,
    )

    activation = NetworkBoundary(replay.input("activation_in"))
    weight = NetworkBoundary(compute.input("weight"), when=streams_weights)
    output = NetworkBoundary(compute.output("output"))

    @constraint(supply=weight_supply, present=initializer_present)
    def local_weights_need_an_initializer(*, supply: WeightSupply, present: bool) -> object:
        """Embedded and decoupled supply require a source initializer."""

        if supply is not WeightSupply.EXTERNAL and not present:
            return reject(
                "mvau-local-weights-need-an-initializer",
                f"{supply.value} weight supply holds the matrix locally, "
                "and this source node supplies none",
                values={"supply": supply.value},
            )
        return True

    dataflow_support = ConstraintGroup(
        *WeightedDotProductDesign.dataflow_support.constraints,
        local_weights_need_an_initializer,
        name="supply_available",
    )


DESIGN_INPUTS = (*SHARED_INPUTS, "initializer_present")

__all__ = ["DESIGN_INPUTS", "DotProductDesign", "WeightSupply"]
