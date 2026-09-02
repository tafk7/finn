# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The class-authored replay-plus-dot-product ``DataflowDesign``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from finn.dataflow.authoring import (
    Choice,
    DataflowDesign,
    Imported,
    Kernels,
    Network,
    Readiness,
    Region,
    SourceInput,
    class_divisors_of,
    constraint,
    derived,
)
from finn.dataflow.authoring.inventory import (
    DataflowDesignDeclaration,
    DataflowDesignEntry,
    DataflowDesignInventory,
    declare_dataflow_design_inventory,
)
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
    ComputationContract,
)
from finn.dataflow.design import (
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    DesignSpaceSpec,
)
from finn.dataflow.kernels.dotp_axi import DotProductKernelInputs, DotpAxiHandles, DotpAxiKernel
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.kernels.replay_buffer import ReplayBufferInputs, ReplayBufferKernel
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.ops.mvau.associations import (
    MVAUParameterTopology,
    MVAUSourceAssociation,
)
from finn.dataflow.ops.mvau.input_supply import (
    EXTERNAL_SUPPLY,
    MVAUInputSupply,
    declare_mvau_input_supply,
)
from finn.dataflow.ops.mvau.problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUComputationProfile,
    MVAUProblem,
    MVAUSourceDescription,
)
from finn.dataflow.ops.mvau.regions import (
    MVAURegionDeclaration,
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.ops.mvau.semantics import (
    DOT_PRODUCT_DESIGN_NAMESPACE,
    DOT_PRODUCT_NODE,
    DOT_PRODUCT_SEMANTIC_READINESS,
    REGION_FORM_EXPORT,
    REPLAY_NODE,
    WEIGHT_INTERFACE,
    MVAUDotProductSemantics,
    MVAUSemanticDemand,
    MVAUSemanticExport,
    accumulator_output_type_supported,
    construct_decomposed_mvau_network,
    construct_external_dot_product_source_association,
    dot_product_computation_supported,
)
from finn.dataflow.region import (
    DataflowRegion,
    NumericElementType,
    Port,
    element_width,
)


@dataclass(frozen=True)
class DotProductDesignInputs:
    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    computation_profile: Ref[MVAUComputationProfile]
    source_description: Ref[MVAUSourceDescription]
    narrow_weights: Ref[bool]
    target_dsp_block: Ref[DspBlock]
    target_clock_period_ns: Ref[float]
    weight_supply: Ref[str]


def _region_form(matrix_width: int) -> MVAURegionDeclaration:
    del matrix_width
    return MVAURegionDeclaration.DOT_PRODUCT_STREAMED


def _network_validation(network: DataflowNetwork) -> NetworkValidationReport:
    return validate_network(network)


def _network_valid(report: NetworkValidationReport) -> bool:
    return not report


def _source_association(
    description: MVAUSourceDescription,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    supply: str,
) -> MVAUSourceAssociation:
    association = construct_external_dot_product_source_association(
        description,
        repetitions,
        matrix_width,
        matrix_height,
    )
    if supply == EXTERNAL_SUPPLY:
        return association
    return replace(association, parameter_topology=MVAUParameterTopology.CYCLIC)


class DotProductDesign(DataflowDesign):
    """Activation replay and dot product in two independent placements."""

    id = "dot_product"
    version = "1"
    uses_class_authoring = True

    repetitions = Imported(int)
    matrix_width = Imported(int)
    matrix_height = Imported(int)
    activation_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    computation_profile = Imported(MVAUComputationProfile)
    source_description = Imported(MVAUSourceDescription)
    narrow_weights = Imported(bool)
    target_dsp_block = Imported(DspBlock)
    target_clock_period_ns = Imported(float)
    weight_supply = Imported(str)

    pe = Choice(int, domain=class_divisors_of(matrix_height))
    simd = Choice(int, domain=class_divisors_of(matrix_width))

    computation_supported = constraint(
        computation_profile,
        sets=(
            f"{DOT_PRODUCT_DESIGN_NAMESPACE}.source_admission",
            f"{DOT_PRODUCT_DESIGN_NAMESPACE}.feasibility",
        ),
    )(dot_product_computation_supported)
    accumulator_output_type_supported = constraint(
        accumulator_element_type,
        output_element_type,
        sets=(
            f"{DOT_PRODUCT_DESIGN_NAMESPACE}.source_admission",
            f"{DOT_PRODUCT_DESIGN_NAMESPACE}.feasibility",
        ),
    )(accumulator_output_type_supported)

    compute_node = Region(
        role="compute",
        node_id=DOT_PRODUCT_NODE,
        construct=construct_dot_product_region,
        dependencies=(
            repetitions,
            matrix_width,
            matrix_height,
            activation_element_type,
            weight_element_type,
            output_element_type,
            pe,
            simd,
        ),
        computation=DOT_PRODUCT_COMPUTATION,
    )
    replay_node = Region(
        role="replay",
        node_id=REPLAY_NODE,
        construct=construct_activation_replay_region,
        dependencies=(
            repetitions,
            matrix_width,
            matrix_height,
            activation_element_type,
            pe,
            simd,
        ),
        computation=ACTIVATION_REPLAY_COMPUTATION,
    )
    weight_port = derived(
        repetitions,
        matrix_width,
        matrix_height,
        weight_element_type,
        pe,
        simd,
        value_type=Port,
        stable_name="compute.weight_port",
    )(construct_standard_mvau_weight_port)
    region_form = derived(
        matrix_width,
        value_type=MVAURegionDeclaration,
        stable_name="compute.region_form",
    )(_region_form)
    network = Network(
        replay_node,
        compute_node,
        construct=construct_decomposed_mvau_network,
    )
    network_validation = derived(
        network,
        value_type=NETWORK_VALIDATION_REPORT_SEMANTICS,
    )(_network_validation)
    network_structurally_well_formed = constraint(
        network_validation,
        sets=(f"{DOT_PRODUCT_DESIGN_NAMESPACE}.feasibility",),
    )(_network_valid)
    source_association = derived(
        source_description,
        repetitions,
        matrix_width,
        matrix_height,
        weight_supply,
        value_type=MVAUSourceAssociation,
    )(_source_association)

    weight = SourceInput(WEIGHT_INTERFACE, compute_node.input("weight"), "weight")
    compute = Kernels(
        name="compute",
        covers=(compute_node,),
        candidates=(DotpAxiKernel,),
        inputs=DotProductKernelInputs(
            role="compute",
            region=cast("Ref[DataflowRegion]", compute_node.region),
            computation=cast("Ref[ComputationContract]", compute_node.computation),
            pe=cast("Ref[int]", pe),
            simd=cast("Ref[int]", simd),
            activation_element_type=cast("Ref[NumericElementType]", activation_element_type),
            weight_element_type=cast("Ref[NumericElementType]", weight_element_type),
            output_element_type=cast("Ref[NumericElementType]", output_element_type),
            accumulator_element_type=cast("Ref[NumericElementType]", accumulator_element_type),
            narrow_weights=cast("Ref[bool]", narrow_weights),
            target_dsp_block=cast("Ref[DspBlock]", target_dsp_block),
            target_clock_period_ns=cast("Ref[float]", target_clock_period_ns),
        ),
    )
    replay_length = derived(matrix_width, simd, value_type=int, stable_name="replay.length")(
        lambda matrix_width, simd: matrix_width // simd
    )
    replay_repetitions = derived(
        matrix_height,
        pe,
        value_type=int,
        stable_name="replay.repetitions",
    )(lambda matrix_height, pe: matrix_height // pe)
    replay_width = derived(
        activation_element_type,
        simd,
        value_type=int,
        stable_name="replay.width",
    )(lambda activation_type, simd: simd * element_width(activation_type))
    replay = Kernels(
        name="replay",
        covers=(replay_node,),
        candidates=(ReplayBufferKernel,),
        inputs=ReplayBufferInputs(
            role="replay",
            region=cast("Ref[DataflowRegion]", replay_node.region),
            computation=cast("Ref[ComputationContract]", replay_node.computation),
            length=cast("Ref[int]", replay_length),
            repetitions=cast("Ref[int]", replay_repetitions),
            width=cast("Ref[int]", replay_width),
        ),
    )
    semantic_readiness = Readiness(
        DOT_PRODUCT_SEMANTIC_READINESS,
        decisions=(pe, simd),
        properties=(
            replay_node.region,
            replay_node.computation,
            compute_node.region,
            compute_node.computation,
            weight_port,
            region_form,
            network,
            network_validation,
            source_association,
        ),
        constraints=(
            computation_supported,
            accumulator_output_type_supported,
            network_structurally_well_formed,
        ),
    )


@dataclass(frozen=True)
class DotProductDesignAssembly:
    semantics: MVAUDotProductSemantics
    input_supply: MVAUInputSupply
    inventory: DataflowDesignInventory
    design: DataflowDesignDeclaration
    compute_pumping: Ref[bool]
    source_association: Ref[MVAUSourceAssociation]

    @property
    def specification(self) -> DesignSpaceSpec:
        return self.inventory.specification


def _inputs(
    problem: MVAUProblem,
    narrow_weights: Ref[bool],
    supply: Ref[str],
) -> DotProductDesignInputs:
    return DotProductDesignInputs(
        problem.repetitions,
        problem.matrix_width,
        problem.matrix_height,
        problem.activation_element_type,
        problem.weight_element_type,
        problem.accumulator_element_type,
        problem.output_element_type,
        problem.computation_profile,
        problem.source_description,
        narrow_weights,
        problem.target_dsp_block,
        problem.target_clock_period_ns,
        supply,
    )


def _semantics(declaration: DataflowDesignDeclaration) -> MVAUDotProductSemantics:
    exported = declaration.exports
    constraints = {
        item.path.value.rsplit(".", 1)[-1]: item for item in declaration.constraint_handles
    }
    source = tuple(
        constraints[name] for name in ("computation_supported", "accumulator_output_type_supported")
    )
    feasibility = (*source, constraints["network_structurally_well_formed"])
    return MVAUDotProductSemantics(
        declaration.spec,
        cast("Ref[int]", exported["pe"]),
        cast("Ref[int]", exported["simd"]),
        cast("Ref[DataflowRegion]", exported["replay_node.region"]),
        cast("Ref[ComputationContract]", exported["replay_node.computation"]),
        cast("Ref[DataflowRegion]", exported["compute_node.region"]),
        cast("Ref[ComputationContract]", exported["compute_node.computation"]),
        cast("Ref[Port]", exported["weight_port"]),
        cast("Ref[MVAURegionDeclaration]", exported["region_form"]),
        declaration.network,
        cast("Ref[NetworkValidationReport]", exported["network_validation"]),
        cast("Ref[MVAUSourceAssociation]", exported["source_association"]),
        source,
        feasibility,
        f"{DOT_PRODUCT_DESIGN_NAMESPACE}.source_admission",
        f"{DOT_PRODUCT_DESIGN_NAMESPACE}.feasibility",
        (MVAUSemanticDemand(WEIGHT_INTERFACE, cast("Ref[Port]", exported["weight_port"])),),
        (
            MVAUSemanticExport(
                REGION_FORM_EXPORT,
                exported["region_form"],
            ),
        ),
        DOT_PRODUCT_SEMANTIC_READINESS,
    )


def declare_dot_product_design(
    problem: MVAUProblem = MVAU_PROBLEM,
    *,
    narrow_weights: Ref[bool] = MVAU_EFFECTIVE_NARROW_WEIGHTS,
) -> DotProductDesignAssembly:
    supply = declare_mvau_input_supply(problem)
    inventory = declare_dataflow_design_inventory(
        "mvau",
        (
            DataflowDesignEntry(
                DotProductDesign, _inputs(problem, narrow_weights, supply.declaration.choice)
            ),
        ),
        input_supplies=(supply.declaration,),
        shared_specs=(MVAU_PROBLEM_SPEC,),
    )
    declaration = inventory.declarations[0]
    compute = declaration.placement("compute").candidates[0]
    return DotProductDesignAssembly(
        _semantics(declaration),
        supply,
        inventory,
        declaration,
        compute.typed_handles(DotpAxiHandles).compute_pumping,
        cast("Ref[MVAUSourceAssociation]", declaration.exports["source_association"]),
    )


MVAU_DOT_PRODUCT_DESIGN = declare_dot_product_design()


__all__ = [
    "DotProductDesign",
    "DotProductDesignAssembly",
    "DotProductDesignInputs",
    "MVAU_DOT_PRODUCT_DESIGN",
    "declare_dot_product_design",
]
