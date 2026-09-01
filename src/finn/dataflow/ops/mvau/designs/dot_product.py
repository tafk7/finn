# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The production replay-plus-dot-product ``DataflowDesign``."""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignScope,
)
from finn.dataflow.authoring.inventory import (
    DataflowDesignDeclaration,
    DataflowDesignEntry,
    DataflowDesignInventory,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.design import DesignSpaceSpec
from finn.dataflow.kernels.dotp_axi import DotpAxiHandles, DotpAxiKernel
from finn.dataflow.kernels.dotp_axi import DotProductKernelInputs
from finn.dataflow.kernels.replay_buffer import ReplayBufferInputs
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.ops.mvau.input_supply import (
    MVAUInputSupply,
    declare_mvau_input_supply,
    declare_supplied_source_association,
)
from finn.dataflow.ops.mvau.semantics import (
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
    MVAUDotProductSemantics,
    declare_dot_product_semantics,
)
from finn.dataflow.ops.mvau.problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUProblem,
)
from finn.dataflow.ops.mvau.associations import MVAUSourceAssociation


@dataclass(frozen=True)
class DotProductDesignInputs:
    """Shared MVAU declarations imported by ``DotProductDesign``."""

    problem: MVAUProblem
    semantics: MVAUDotProductSemantics
    narrow_weights: Ref[bool]


class DotProductDesign(DataflowDesign):
    """Activation replay and dot product in two independent Kernel placements."""

    id = "dot_product"
    version = "1"

    @classmethod
    def define(cls, design: DataflowDesignScope[DotProductDesignInputs]) -> None:
        inputs = design.inputs
        semantics = inputs.semantics
        replay = design.node(
            "replay",
            node_id=REPLAY_NODE,
            region=semantics.replay_region,
            computation=semantics.replay_computation,
        )
        compute = design.node(
            "compute",
            node_id=DOT_PRODUCT_NODE,
            region=semantics.dot_product_region,
            computation=semantics.dot_product_computation,
        )
        design.use_network(semantics.network)
        weight_interface = design.input_interface("compute.weight_interface", compute, "weight")
        design.map_input("weight", boundary_id="weight", consumer=weight_interface)
        design.kernels(
            "compute",
            covers=(compute,),
            candidates=(DotpAxiKernel,),
            inputs=DotProductKernelInputs(
                region=semantics.dot_product_region,
                computation=semantics.dot_product_computation,
                pe=semantics.pe,
                simd=semantics.simd,
                activation_element_type=inputs.problem.activation_element_type,
                weight_element_type=inputs.problem.weight_element_type,
                output_element_type=inputs.problem.output_element_type,
                accumulator_element_type=inputs.problem.accumulator_element_type,
                narrow_weights=inputs.narrow_weights,
                target_dsp_block=inputs.problem.target_dsp_block,
                target_clock_period_ns=inputs.problem.target_clock_period_ns,
            ),
        )
        design.kernels(
            "replay",
            covers=(replay,),
            candidates=(ReplayBufferKernel,),
            inputs=ReplayBufferInputs(
                region=semantics.replay_region,
                computation=semantics.replay_computation,
                matrix_width=inputs.problem.matrix_width,
                matrix_height=inputs.problem.matrix_height,
                pe=semantics.pe,
                simd=semantics.simd,
                activation_element_type=inputs.problem.activation_element_type,
            ),
        )


@dataclass(frozen=True)
class DotProductDesignAssembly:
    """The shared semantics and compiled design declared together once."""

    semantics: MVAUDotProductSemantics
    input_supply: MVAUInputSupply
    inventory: DataflowDesignInventory
    design: DataflowDesignDeclaration
    compute_pumping: Ref[bool]
    source_association: Ref[MVAUSourceAssociation]

    @property
    def specification(self) -> DesignSpaceSpec:
        return self.inventory.specification


def declare_dot_product_design(
    problem: MVAUProblem = MVAU_PROBLEM,
    *,
    narrow_weights: Ref[bool] = MVAU_EFFECTIVE_NARROW_WEIGHTS,
) -> DotProductDesignAssembly:
    """Declare DotProduct over the shared semantic handles without selecting it."""

    semantics = declare_dot_product_semantics(problem)
    supply = declare_mvau_input_supply(problem)
    source_association, association_spec = declare_supplied_source_association(
        "mvau.design.dot_product",
        semantics.source_association,
        supply.declaration,
    )
    inventory = declare_dataflow_design_inventory(
        "mvau",
        (
            DataflowDesignEntry(
                DotProductDesign,
                DotProductDesignInputs(problem, semantics, narrow_weights),
                (semantics.spec, association_spec),
                (semantics.pe, semantics.simd),
                semantics.feasibility_constraints,
            ),
        ),
        input_supplies=(supply.declaration,),
        shared_specs=(MVAU_PROBLEM_SPEC,),
    )
    declaration = inventory.declarations[0]
    compute = declaration.placement("compute").candidates[0]
    return DotProductDesignAssembly(
        semantics,
        supply,
        inventory,
        declaration,
        compute.typed_handles(DotpAxiHandles).compute_pumping,
        source_association,
    )


MVAU_DOT_PRODUCT_DESIGN = declare_dot_product_design()


__all__ = [
    "DotProductDesign",
    "DotProductDesignAssembly",
    "DotProductDesignInputs",
    "MVAU_DOT_PRODUCT_DESIGN",
    "declare_dot_product_design",
]
