# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three ways a matrix can reach the production dot-product Kernel.

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
source fact used only by this Kernel's admission policy: modes that keep the
matrix locally require an initializer. External supply remains available with
or without one.
"""

from __future__ import annotations

from finn.dataflow.analysis.integer_dot import IntegerSupportReport
from finn.kernels.artifacts.build import ModuleBuildRequirements
from finn.dataflow.kernels.matmul.physical import (
    DECOMPOSED_PRODUCER,
    DECOMPOSED_WRAPPER_TEMPLATE,
    compose_decomposed,
    top_boundary_layout,
)
from finn.kernels.physical.lowering import lower_module_structure
from finn.dataflow.model.physical.interface import (
    PhysicalResult,
    PhysicalPort,
    KernelRealizationFacts,
    KernelStreamBinding,
)
from finn.dataflow.model.children import KernelChoice
from finn.dataflow.model.logical.authoring import EdgeSink, NetworkBoundary, NetworkEdge
from finn.dataflow.kernels.dotp_axi import (
    DotpAxiKernel,
    EmbeddedDotpAxiKernel,
    require_dotp_axi_numerical_support,
)
from finn.dataflow.kernels.memstream import MemstreamKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.kernels.matmul.base import (
    SHARED_INPUTS,
    MvauComputationProfile,
    WeightedDotProductKernel,
)
from finn.dataflow.kernels.matmul.supply import WeightSupply
from finn.kernels.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Subspace,
    allow_absent,
    constraint,
    derived,
    reject,
)


class DotProductKernel(WeightedDotProductKernel):
    """Replay and dot product with an explicitly selected weight supply."""

    id = "dot_product"
    version = "3"

    repetitions = WeightedDotProductKernel.repetitions
    matrix_width = WeightedDotProductKernel.matrix_width
    matrix_height = WeightedDotProductKernel.matrix_height
    activation_type = WeightedDotProductKernel.activation_type
    weight_type = WeightedDotProductKernel.weight_type
    accumulator_type = WeightedDotProductKernel.accumulator_type
    output_type = WeightedDotProductKernel.output_type
    computation_profile = WeightedDotProductKernel.computation_profile
    numerical_support = WeightedDotProductKernel.numerical_support
    narrow_weights = WeightedDotProductKernel.narrow_weights
    target_dsp = WeightedDotProductKernel.target_dsp
    clock_period_ns = WeightedDotProductKernel.clock_period_ns
    pe = WeightedDotProductKernel.pe
    simd = WeightedDotProductKernel.simd

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
        outputs=("logical_result", "physical_result", "physical_streams"),
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
            output_type=WeightedDotProductKernel.result_type,
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
            output_type=WeightedDotProductKernel.result_type,
            narrow_weights=narrow_weights,
            target_dsp=target_dsp,
            clock_period_ns=clock_period_ns,
            pe=pe,
            simd=simd,
        ),
        outputs=("logical_result", "physical_result", "physical_streams"),
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

    logical_support = ConstraintGroup(
        *WeightedDotProductKernel.logical_support.constraints,
        local_weights_need_an_initializer,
        name="supply_available",
    )

    @derived(
        PhysicalResult,
        profile=computation_profile,
        numerical=allow_absent(numerical_support),
        supply=weight_supply,
        replay_requirements=replay.physical_result,
        replay_streams=replay.physical_streams,
        compute_requirements=compute.physical_result,
        compute_streams=compute.physical_streams,
    )
    def physical_result(
        *,
        profile: MvauComputationProfile,
        numerical: object,
        supply: WeightSupply,
        replay_requirements: ModuleBuildRequirements,
        replay_streams: tuple[KernelStreamBinding, ...],
        compute_requirements: ModuleBuildRequirements,
        compute_streams: tuple[KernelStreamBinding, ...],
    ) -> object:
        """Build the selected external Replay/Dotp composition only."""

        report = numerical if isinstance(numerical, IntegerSupportReport) else None
        try:
            require_dotp_axi_numerical_support(report, profile)
        except ValueError as error:
            return reject("kernel-physically-unsupported", str(error))

        if supply is not WeightSupply.EXTERNAL:
            return reject(
                "kernel-physically-unsupported",
                f"{supply.value} weight supply has no supported composed module",
            )

        replay = KernelRealizationFacts(replay_requirements, replay_streams)
        compute = KernelRealizationFacts(compute_requirements, compute_streams)

        try:
            structure = compose_decomposed(replay=replay, compute=compute)
            requirements = lower_module_structure(
                structure,
                producer=DECOMPOSED_PRODUCER,
                wrapper_template=DECOMPOSED_WRAPPER_TEMPLATE,
            )
            replay_activation = next(
                item for item in replay.streams if item.region_port_id == "activation_in"
            )
            compute_weight = next(
                item for item in compute.streams if item.region_port_id == "weight"
            )
            compute_output = next(
                item for item in compute.streams if item.region_port_id == "output"
            )
            facts = PhysicalResult(
                requirements,
                (
                    PhysicalPort(
                        "activation",
                        "in0_V",
                        top_boundary_layout(structure.top_abi, "in0_V", replay_activation),
                    ),
                    PhysicalPort(
                        "weights",
                        "in1_V",
                        top_boundary_layout(structure.top_abi, "in1_V", compute_weight),
                    ),
                    PhysicalPort(
                        "result",
                        "out0_V",
                        top_boundary_layout(structure.top_abi, "out0_V", compute_output),
                    ),
                ),
                structure=structure,
            )
        except (KeyError, StopIteration, ValueError) as error:
            return reject("kernel-physically-unsupported", str(error))
        return facts


KERNEL_INPUTS = (*SHARED_INPUTS, "initializer_present")

__all__ = ["KERNEL_INPUTS", "DotProductKernel", "WeightSupply"]
