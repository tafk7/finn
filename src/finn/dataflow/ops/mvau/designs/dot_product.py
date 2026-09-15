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

from dataclasses import replace
from typing import cast

from finn.dataflow._engine import Answer, Decided
from finn.dataflow.designs.physical import (
    BoundaryBinding,
    DECOMPOSED_PRODUCER,
    DECOMPOSED_WRAPPER_TEMPLATE,
    DesignPhysicalFacts,
    EdgeBinding,
    SemanticPortBinding,
    compose_decomposed,
    design_physical_refusal,
    lower_module_structure,
    selected_kernel_realization,
    top_boundary_layout,
    validate_design_physical_facts,
)
from finn.dataflow.designs.design import (
    EdgeSink,
    KernelChoice,
    NetworkBoundary,
    NetworkEdge,
    SelectedGraph,
)
from finn.dataflow.kernels.dotp_axi import (
    DotpAxiKernel,
    EmbeddedDotpAxiKernel,
    require_dotp_axi_numerical_support,
)
from finn.dataflow.kernels.memstream import MemstreamKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.ops.mvau.designs.base import SHARED_INPUTS, WeightedDotProductDesign
from finn.dataflow.ops.mvau.designs.supply import WeightSupply
from finn.dataflow.ops.mvau.selected import (
    MVAU_SELECTED_CONSTRUCTION,
    WEIGHT_KEY,
    MvauSelectionParameters,
)
from finn.dataflow.ops.selected import SelectedInitializerInput, SelectionFacts
from finn.dataflow.ops.tensor_summary import FrozenInitializer
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    Subspace,
    constraint,
    derived,
    reject,
)


class DotProductDesign(WeightedDotProductDesign):
    """Replay and dot product with an explicitly selected weight supply."""

    id = "dot_product"
    version = "3"

    repetitions = WeightedDotProductDesign.repetitions
    matrix_width = WeightedDotProductDesign.matrix_width
    matrix_height = WeightedDotProductDesign.matrix_height
    activation_type = WeightedDotProductDesign.activation_type
    weight_type = WeightedDotProductDesign.weight_type
    accumulator_type = WeightedDotProductDesign.accumulator_type
    output_type = WeightedDotProductDesign.output_type
    computation_profile = WeightedDotProductDesign.computation_profile
    numerical_support = WeightedDotProductDesign.numerical_support
    narrow_weights = WeightedDotProductDesign.narrow_weights
    target_dsp = WeightedDotProductDesign.target_dsp
    clock_period_ns = WeightedDotProductDesign.clock_period_ns
    pe = WeightedDotProductDesign.pe
    simd = WeightedDotProductDesign.simd

    initializer_present = Input(bool)
    weight_initializer = Input(FrozenInitializer, allow_absent=True)
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

    def physical_implementation(self) -> Answer[DesignPhysicalFacts]:
        """Build the selected external Replay/Dotp composition only."""

        profile = self.answer(type(self).computation_profile)
        if not isinstance(profile, Decided):
            return cast("Answer[DesignPhysicalFacts]", profile)
        numerical = self.answer(type(self).numerical_support)
        report = numerical.value if isinstance(numerical, Decided) else None
        try:
            require_dotp_axi_numerical_support(report, profile.value)
        except ValueError as error:
            return cast(
                "Answer[DesignPhysicalFacts]",
                design_physical_refusal(self, str(error)),
            )

        supply = self.answer(type(self).weight_supply)
        if not isinstance(supply, Decided):
            return cast("Answer[DesignPhysicalFacts]", supply)
        if supply.value is not WeightSupply.EXTERNAL:
            return cast(
                "Answer[DesignPhysicalFacts]",
                design_physical_refusal(
                    self,
                    f"{supply.value.value} weight supply has no supported composed module",
                ),
            )

        replay = selected_kernel_realization(self, "replay")
        if not isinstance(replay, Decided):
            return cast("Answer[DesignPhysicalFacts]", replay)
        compute = selected_kernel_realization(self, "compute")
        if not isinstance(compute, Decided):
            return cast("Answer[DesignPhysicalFacts]", compute)

        try:
            structure = compose_decomposed(replay=replay.value, compute=compute.value)
            requirements = lower_module_structure(
                structure,
                producer=DECOMPOSED_PRODUCER,
                wrapper_template=DECOMPOSED_WRAPPER_TEMPLATE,
            )
            port_bindings = (
                *(
                    SemanticPortBinding("replay", "u_replay", binding)
                    for binding in replay.value.streams
                ),
                *(
                    SemanticPortBinding("compute", "u_compute", binding)
                    for binding in compute.value.streams
                ),
            )
            replay_activation = next(
                binding.local
                for binding in port_bindings
                if binding.node_id == "replay" and binding.local.region_port_id == "activation_in"
            )
            compute_weight = next(
                binding.local
                for binding in port_bindings
                if binding.node_id == "compute" and binding.local.region_port_id == "weight"
            )
            compute_output = next(
                binding.local
                for binding in port_bindings
                if binding.node_id == "compute" and binding.local.region_port_id == "output"
            )
            facts = DesignPhysicalFacts(
                requirements,
                port_bindings,
                (
                    BoundaryBinding(
                        "activation",
                        "in0_V",
                        top_boundary_layout(structure.top_abi, "in0_V", replay_activation),
                        "u_replay",
                        replay_activation.abi_bus_id,
                    ),
                    BoundaryBinding(
                        "weight",
                        "in1_V",
                        top_boundary_layout(structure.top_abi, "in1_V", compute_weight),
                        "u_compute",
                        compute_weight.abi_bus_id,
                    ),
                    BoundaryBinding(
                        "output",
                        "out0_V",
                        top_boundary_layout(structure.top_abi, "out0_V", compute_output),
                        "u_compute",
                        compute_output.abi_bus_id,
                    ),
                ),
                (
                    EdgeBinding(
                        "activation_replay",
                        "u_replay",
                        next(
                            binding.local.abi_bus_id
                            for binding in port_bindings
                            if binding.node_id == "replay"
                            and binding.local.region_port_id == "activation_out"
                        ),
                        "u_compute",
                        next(
                            binding.local.abi_bus_id
                            for binding in port_bindings
                            if binding.node_id == "compute"
                            and binding.local.region_port_id == "activation"
                        ),
                    ),
                ),
            )
            network = self.dataflow.accepted_answer
            if not isinstance(network, Decided):  # guarded by DataflowDesign.physical
                return cast("Answer[DesignPhysicalFacts]", network)
            validate_design_physical_facts(network.value, structure, facts)
        except (KeyError, StopIteration, ValueError) as error:
            return cast(
                "Answer[DesignPhysicalFacts]",
                design_physical_refusal(self, str(error)),
            )
        return Decided(facts)


DESIGN_INPUTS = (*SHARED_INPUTS, "initializer_present", "weight_initializer")


def _local_weight_required(facts: SelectionFacts[object, object]) -> bool:
    if not isinstance(facts.parameters, MvauSelectionParameters):
        raise TypeError("MVAU initializer predicate received the wrong parameters")
    return facts.parameters.weight_supply is not WeightSupply.EXTERNAL


DotProductDesign.selected_graph = SelectedGraph(
    replace(
        MVAU_SELECTED_CONSTRUCTION,
        initializer_inputs=(
            SelectedInitializerInput(
                WEIGHT_KEY,
                DotProductDesign.weight_initializer,
                _local_weight_required,
            ),
        ),
    )
)

__all__ = ["DESIGN_INPUTS", "DotProductDesign", "WeightSupply"]
