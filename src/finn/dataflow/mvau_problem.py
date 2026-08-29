# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU operation's problem: every fact it owns, declared once.

This module is the answer to "who owns ``problem.mvau.*``".  Previously the
compute Kernel pool carried a ``MVAUComputeProblemPaths`` table and the supply
pool carried another, so a Kernel module was the definition site for facts no
Kernel projects, and every consumer -- source projection, elaboration,
artifacts -- reached into a Kernel module to name them.

The operation declares them here through one ``OpDesign``, each with a
provenance saying who may supply it, and hands out typed ``Ref`` handles.  A
Kernel reads a fact through the handle it was given; it does not name a path.

This module deliberately imports nothing from ``finn.dataflow.mvau`` beyond
the leaf computation profile, so the Kernel modules can depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.authoring.op_design import OpDesign, ProblemProvenance
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.design import DesignSpaceSpec, QualifiedPath
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterKernelPaths,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.region import BeatSequence, NumericElementType


class MVAUComputationProfile(str, Enum):
    """Evidenced source-computation profiles supported by MVAU bindings."""

    ACCUMULATOR_INTEGER = "accumulator_integer"
    BIPOLAR_XNOR_ACCUMULATOR = "bipolar_xnor_accumulator"
    FUSED_THRESHOLD = "fused_threshold"


class MVAUDspBlock(str, Enum):
    """Target DSP capability consumed only by Kernel feasibility constraints."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


@dataclass(frozen=True)
class MVAUSourceDescription:
    """Compiler-owned source identities and shapes for one MVAU scope."""

    source_node_id: str
    activation_operand_id: str
    weight_operand_id: str
    output_operand_id: str
    leading_shape: tuple[int, ...]
    threshold_operand_id: str | None = None
    fused_source_node_ids: tuple[str, ...] = ()
    threshold_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "leading_shape", tuple(self.leading_shape))
        object.__setattr__(self, "fused_source_node_ids", tuple(self.fused_source_node_ids))
        if self.threshold_shape is not None:
            object.__setattr__(self, "threshold_shape", tuple(self.threshold_shape))


# -- field validators --------------------------------------------------------


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _positive_float(value: object) -> bool:
    return type(value) is float and value > 0


def _non_empty_string(value: object) -> bool:
    return type(value) is str and bool(value)


def _complete_numeric_element_type(value: object) -> bool:
    return type(value) is NumericElementType and bool(value.type_id) and value.bit_width > 0


def _source_description_valid(value: object) -> bool:
    if type(value) is not MVAUSourceDescription:
        return False
    description = value
    names: tuple[object, ...] = (
        description.source_node_id,
        description.activation_operand_id,
        description.weight_operand_id,
        description.output_operand_id,
        *description.fused_source_node_ids,
    )
    threshold_valid = description.threshold_operand_id is None or (
        isinstance(description.threshold_operand_id, str) and bool(description.threshold_operand_id)
    )
    threshold_shape_valid = description.threshold_shape is None or all(
        type(extent) is int and extent > 0 for extent in description.threshold_shape
    )
    # A threshold operand and its shape are one fact in two halves: having
    # exactly one of them means the projection lost information.
    threshold_pair_valid = (description.threshold_operand_id is None) == (
        description.threshold_shape is None
    )
    return (
        all(isinstance(name, str) and bool(name) for name in names)
        and threshold_valid
        and threshold_shape_valid
        and threshold_pair_valid
        and all(type(extent) is int and extent > 0 for extent in description.leading_shape)
    )


# -- the declared problem ----------------------------------------------------


@dataclass(frozen=True)
class MVAUProblem:
    """Typed handles to every fact the MVAU operation owns.

    Kernel modules, source projection, and elaboration all read facts through
    these handles rather than through a path table, so the declaration site and
    the use site cannot drift apart.
    """

    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    threshold_element_type: Ref[NumericElementType]
    threshold_initializer_available: Ref[bool]
    computation_profile: Ref[MVAUComputationProfile]
    weight_initializer_available: Ref[bool]
    weight_initializer_fingerprint: Ref[str]
    threshold_initializer_fingerprint: Ref[str]
    source_description: Ref[MVAUSourceDescription]
    initializer_excludes_minimum: Ref[bool]
    runtime_weight_range_contract: Ref[bool]
    runtime_writable: Ref[bool]
    external_weight_sequence: Ref[BeatSequence]
    accumulator_type_analysis_owner: Ref[str]
    target_dsp_block: Ref[MVAUDspBlock]
    target_fpga_part: Ref[str]
    target_clock_period_ns: Ref[float]
    target_memory_capabilities: Ref[CyclicTargetMemoryCapabilities]


def build_mvau_problem(design: OpDesign) -> MVAUProblem:
    """Declare every MVAU problem field on ``design`` and return its handles."""

    def extent(name: str) -> Ref[int]:
        return design.graph_fact(
            name,
            int,
            validate=_positive_integer,
            description="must be a positive integer",
        )

    def numeric(name: str, *, required: bool = True) -> Ref[NumericElementType]:
        return design.graph_fact(
            name,
            NumericElementType,
            required=required,
            validate=_complete_numeric_element_type,
            description="must be a complete numeric element type",
        )

    def fingerprint(name: str) -> Ref[str]:
        return design.graph_fact(
            name,
            str,
            required=False,
            validate=_non_empty_string,
            description="must be a non-empty initializer fingerprint",
        )

    return MVAUProblem(
        repetitions=extent("r"),
        matrix_width=extent("mw"),
        matrix_height=extent("mh"),
        activation_element_type=numeric("activation_element_type"),
        weight_element_type=numeric("weight_element_type"),
        # The accumulator width is recovered by MinimizeAccumulatorWidth, not
        # read off a tensor, which is why it is an analysis fact rather than a
        # graph fact even though both come from the model.
        accumulator_element_type=design.analysis_fact(
            "accumulator_element_type",
            NumericElementType,
            validate=_complete_numeric_element_type,
            description="must be a complete numeric element type",
        ),
        output_element_type=numeric("output_element_type"),
        threshold_element_type=numeric("threshold_element_type", required=False),
        threshold_initializer_available=design.graph_fact(
            "threshold_initializer_available", bool, required=False
        ),
        computation_profile=design.graph_fact("computation_profile", MVAUComputationProfile),
        weight_initializer_available=design.graph_fact("weight_initializer_available", bool),
        weight_initializer_fingerprint=fingerprint("weight_initializer_fingerprint"),
        threshold_initializer_fingerprint=fingerprint("threshold_initializer_fingerprint"),
        source_description=design.graph_fact(
            "source_description",
            MVAUSourceDescription,
            validate=_source_description_valid,
            description="must contain complete source identities and extents",
        ),
        # "The initializer that governs the delivered weight values excludes
        # the minimum".  Both halves are graph facts -- the initializer's range
        # and, for an imported legacy node, whether its delivery mode makes
        # that initializer the value authority at all -- so the analysis
        # answers the whole question and simply declines when it cannot.
        initializer_excludes_minimum=design.analysis_fact(
            "initializer_excludes_minimum", bool, required=False
        ),
        # A promise about values written at run time.  Only the caller can make
        # it, so it is a build fact and never a graph fact.
        runtime_weight_range_contract=design.build_fact(
            "runtime_weight_range_contract", bool, required=False
        ),
        runtime_writable=design.build_fact(
            "runtime_writable", bool, path=CyclicParameterKernelPaths.RUNTIME_WRITABLE
        ),
        external_weight_sequence=design.build_fact(
            "external_weight_sequence", BeatSequence, required=False
        ),
        accumulator_type_analysis_owner=design.build_fact(
            "accumulator_type_analysis_owner",
            str,
            required=False,
            validate=_non_empty_string,
            description="must identify the accumulator analysis owner",
        ),
        target_dsp_block=design.target_fact("dsp_block", MVAUDspBlock, required=False),
        target_fpga_part=design.target_fact(
            "fpga_part",
            str,
            required=False,
            validate=_non_empty_string,
            description="must be a non-empty FPGA part identifier",
        ),
        target_clock_period_ns=design.target_fact(
            "clock_period_ns",
            float,
            required=False,
            validate=_positive_float,
            description="must be a positive clock period",
        ),
        target_memory_capabilities=design.target_fact(
            "memory_capabilities",
            CyclicTargetMemoryCapabilities,
            required=False,
            path=CyclicParameterKernelPaths.TARGET_MEMORY_CAPABILITIES,
        ),
    )


# -- effective narrow weights ------------------------------------------------

#: The name of the operation-owned derived property replacing the former
#: ``problem.mvau.weights_narrow`` field.
EFFECTIVE_NARROW_WEIGHTS = "effective_narrow_weights"


def _effective_narrow_weights(
    initializer_excludes_minimum: object,
    runtime_weight_range_contract: object,
    runtime_writable: object,
) -> object:
    """Decide whether the minimum weight value is excluded, from its authority.

    ``NARROW_WEIGHTS`` is not a property of the weight datatype; it is a
    promise about the values that reach the implementation.  Which fact carries
    that promise therefore depends on who owns the values:

    * runtime-written weights are governed by the caller's contract, if any;
    * otherwise the initializer governs, when the analysis found one.

    Anything else is ``False``.  That is conservative -- it can cost a tighter
    DSP packing -- but it can never promise a range the hardware will not see.
    """

    if bool(runtime_writable):
        return _optional_flag(runtime_weight_range_contract)
    return _optional_flag(initializer_excludes_minimum)


def _optional_flag(value: object) -> bool:
    return isinstance(value, bool) and value


def declare_effective_narrow_weights(design: OpDesign, problem: MVAUProblem) -> Ref[bool]:
    """Declare the operation-owned narrow-weight property."""

    return design.derived(
        EFFECTIVE_NARROW_WEIGHTS,
        bool,
        dependencies={
            "initializer_excludes_minimum": problem.initializer_excludes_minimum.allow_absent(),
            "runtime_weight_range_contract": problem.runtime_weight_range_contract.allow_absent(),
            "runtime_writable": problem.runtime_writable,
        },
        evaluate=_effective_narrow_weights,
    )


# -- the operation's declared problem, built once -----------------------------

MVAU_PROBLEM_DESIGN = OpDesign("mvau.op", problem_namespace="mvau")
MVAU_PROBLEM: MVAUProblem = build_mvau_problem(MVAU_PROBLEM_DESIGN)
MVAU_EFFECTIVE_NARROW_WEIGHTS: Ref[bool] = declare_effective_narrow_weights(
    MVAU_PROBLEM_DESIGN, MVAU_PROBLEM
)

#: Every MVAU problem field and the narrow-weight property, as one ordinary
#: engine specification the operation assembles with its Kernel pools.
MVAU_PROBLEM_SPEC: DesignSpaceSpec = MVAU_PROBLEM_DESIGN.spec()

#: Who may supply each field.  The operation checks its two projections
#: against this rather than trusting them to stay in their lanes.
MVAU_PROBLEM_PROVENANCE: ProblemProvenance = MVAU_PROBLEM_DESIGN.provenance()


class MVAUProblemPaths:
    """The MVAU problem paths, for consumers that hold values not handles.

    Elaboration, artifacts, and persistence read a resolved point's problem
    mapping, which is keyed by path.  They are given the paths here rather than
    restating the strings, and every path comes from the same declaration the
    engine saw.
    """

    REPETITIONS = MVAU_PROBLEM.repetitions.path
    MATRIX_WIDTH = MVAU_PROBLEM.matrix_width.path
    MATRIX_HEIGHT = MVAU_PROBLEM.matrix_height.path
    ACTIVATION_ELEMENT_TYPE = MVAU_PROBLEM.activation_element_type.path
    WEIGHT_ELEMENT_TYPE = MVAU_PROBLEM.weight_element_type.path
    ACCUMULATOR_ELEMENT_TYPE = MVAU_PROBLEM.accumulator_element_type.path
    OUTPUT_ELEMENT_TYPE = MVAU_PROBLEM.output_element_type.path
    THRESHOLD_ELEMENT_TYPE = MVAU_PROBLEM.threshold_element_type.path
    THRESHOLD_INITIALIZER_AVAILABLE = MVAU_PROBLEM.threshold_initializer_available.path
    COMPUTATION_PROFILE = MVAU_PROBLEM.computation_profile.path
    WEIGHT_INITIALIZER_AVAILABLE = MVAU_PROBLEM.weight_initializer_available.path
    WEIGHT_INITIALIZER_FINGERPRINT = MVAU_PROBLEM.weight_initializer_fingerprint.path
    THRESHOLD_INITIALIZER_FINGERPRINT = MVAU_PROBLEM.threshold_initializer_fingerprint.path
    SOURCE_DESCRIPTION = MVAU_PROBLEM.source_description.path
    INITIALIZER_EXCLUDES_MINIMUM = MVAU_PROBLEM.initializer_excludes_minimum.path
    RUNTIME_WEIGHT_RANGE_CONTRACT = MVAU_PROBLEM.runtime_weight_range_contract.path
    RUNTIME_WRITABLE = MVAU_PROBLEM.runtime_writable.path
    EXTERNAL_WEIGHT_SEQUENCE = MVAU_PROBLEM.external_weight_sequence.path
    ACCUMULATOR_TYPE_ANALYSIS_OWNER = MVAU_PROBLEM.accumulator_type_analysis_owner.path
    TARGET_DSP_BLOCK = MVAU_PROBLEM.target_dsp_block.path
    TARGET_FPGA_PART = MVAU_PROBLEM.target_fpga_part.path
    TARGET_CLOCK_PERIOD_NS = MVAU_PROBLEM.target_clock_period_ns.path
    TARGET_MEMORY_CAPABILITIES = MVAU_PROBLEM.target_memory_capabilities.path

    EFFECTIVE_NARROW_WEIGHTS: QualifiedPath = MVAU_EFFECTIVE_NARROW_WEIGHTS.path


__all__ = [
    "EFFECTIVE_NARROW_WEIGHTS",
    "MVAU_EFFECTIVE_NARROW_WEIGHTS",
    "MVAU_PROBLEM",
    "MVAU_PROBLEM_DESIGN",
    "MVAU_PROBLEM_PROVENANCE",
    "MVAU_PROBLEM_SPEC",
    "MVAUComputationProfile",
    "MVAUDspBlock",
    "MVAUProblem",
    "MVAUProblemPaths",
    "MVAUSourceDescription",
    "build_mvau_problem",
    "declare_effective_narrow_weights",
]
