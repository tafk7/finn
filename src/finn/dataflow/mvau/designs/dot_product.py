# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The production-intended replay-plus-dot-product ``DataflowDesign``.

D3 declares and validates this path beside the legacy semantic-Kernel flow. It
does not switch ``MVAUDataflowOp`` selection or persistence; that remains D7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from finn.dataflow.authoring.design import (
    DataflowDesign,
    DataflowDesignDeclaration,
    DataflowDesignEntry,
    DataflowDesignInventory,
    DataflowDesignScope,
    DesignRealization,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.design import DependencyKind, DesignSpaceSpec
from finn.dataflow.hardware.kernel import KernelBinding
from finn.dataflow.mvau.hardware.binding import DecomposedBindings
from finn.dataflow.mvau.hardware.composition import compose
from finn.dataflow.mvau.hardware.dotp_axi import DotpAxiKernel
from finn.dataflow.mvau.hardware.inputs import (
    ActivationReplayHardwareInputs,
    DotProductHardwareInputs,
)
from finn.dataflow.mvau.hardware.replay_buffer import ReplayBufferKernel
from finn.dataflow.mvau.semantics import (
    DOT_PRODUCT_NODE,
    REPLAY_NODE,
    MVAUDotProductSemantics,
    declare_dot_product_semantics,
)
from finn.dataflow.mvau_problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUProblem,
)

if TYPE_CHECKING:
    from finn.dataflow.mvau.elaboration import MVAUPhysicalElaboration
    from finn.dataflow.mvau.source import MVAUResolvedDesign


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
        design.kernels(
            "compute",
            covers=(compute,),
            candidates=(DotpAxiKernel,),
            inputs=DotProductHardwareInputs(
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
            inputs=ActivationReplayHardwareInputs(
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
    inventory: DataflowDesignInventory
    design: DataflowDesignDeclaration
    compute_pumping: Ref[bool]

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
    inventory = declare_dataflow_design_inventory(
        "mvau",
        (
            DataflowDesignEntry(
                DotProductDesign,
                DotProductDesignInputs(problem, semantics, narrow_weights),
            ),
        ),
        shared_specs=(MVAU_PROBLEM_SPEC, semantics.spec),
    )
    declaration = inventory.declarations[0]
    compute = declaration.placement("compute").candidates[0]
    pumping = tuple(
        item for item in compute.spec.decisions if item.path.value.endswith(".compute_pumping")
    )
    if len(pumping) != 1:
        raise AssertionError("DotpAxiKernel must declare exactly one compute-pumping choice")
    return DotProductDesignAssembly(
        semantics,
        inventory,
        declaration,
        Ref(pumping[0].path, DependencyKind.DECISION, pumping[0].value_semantics),
    )


def decomposed_bindings(realization: DesignRealization) -> DecomposedBindings:
    """Present a validated DotProduct realization to the existing composer."""

    if realization.design_id != DotProductDesign.id:
        raise ValueError("only DotProductDesign has decomposed MVAU bindings")
    by_kernel: dict[str, KernelBinding] = {
        binding.kernel_id: binding for binding in realization.bindings
    }
    try:
        compute = by_kernel[DotpAxiKernel.id]
        replay = by_kernel[ReplayBufferKernel.id]
    except KeyError as error:
        raise ValueError("DotProductDesign requires compute and replay Kernels") from error
    return DecomposedBindings(realization.network, compute, replay)


def compose_dot_product_design(
    resolved_source: MVAUResolvedDesign,
    realization: DesignRealization,
) -> MVAUPhysicalElaboration:
    """Use the proven decomposed composer with the new design's bindings.

    During the side-by-side phases the source envelope still comes from the
    legacy Operation path.  D7 replaces that envelope after persistence cuts
    over; the physical composition itself is already shared here.
    """

    if getattr(resolved_source.result, "network", None) != realization.network:
        raise ValueError("the source envelope and DotProduct realization name different Networks")
    return compose(resolved_source, decomposed_bindings(realization))


MVAU_DOT_PRODUCT_DESIGN = declare_dot_product_design()


__all__ = [
    "DotProductDesign",
    "DotProductDesignAssembly",
    "DotProductDesignInputs",
    "MVAU_DOT_PRODUCT_DESIGN",
    "compose_dot_product_design",
    "declare_dot_product_design",
    "decomposed_bindings",
]
