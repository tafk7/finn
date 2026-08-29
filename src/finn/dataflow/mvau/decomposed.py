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
from typing import cast
from enum import Enum

from finn.dataflow.authoring.kernel_design import (
    KernelDesign,
    declare_kernel_design,
    kernel_namespace,
)
from finn.dataflow.authoring.op_design import ProblemProvenance
from finn.dataflow.authoring.scope import Ref, divisors_of, finite
from finn.dataflow.kernels import (
    KERNEL_ID_SEMANTICS,
    Kernel,
    KernelDeclaration,
    KernelSelection,
)
from finn.dataflow.mvau.compute_pool import (
    REGION_FORM_EXPORT,
    WEIGHT_INTERFACE,
    MVAUComputeKernelId,
)
from finn.dataflow.mvau.rtl_parameters import (
    DSP_VERSION,
    dsp_version,
    segment_length,
    signed_activations,
)
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.design import (
    AbsenceMode,
    Answer,
    Decided,
    DependencyRef,
    DependencyView,
    EvaluatorSpec,
    QualifiedPath,
)
from finn.dataflow.mvau_problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAUComputationProfile,
    MVAUDspBlock,
    MVAUProblem,
)
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

#: Replay is its own pool because it is a second node, not a second compute
#: choice: it is present exactly when the decomposed compute member is selected.
#: The dot product has no pool of its own -- it is a member of the operation's
#: one compute pool, beside the fused Kernels it replaces.
REPLAY_POOL = "mvau.replay"

#: Node ids inside the assembled Network.
REPLAY_NODE = "replay"
DOT_PRODUCT_NODE = "compute"

#: The single internal edge.
ACTIVATION_EDGE = "activation_replay"

#: The two providers this slice targets.
REPLAY_PROVIDER = "finn.rtl.replay_buffer"
DOT_PRODUCT_PROVIDER = "finnlib.rtl.dotp_axi"


class ParameterOwnership(str, Enum):
    """The four things a provider parameter is allowed to be.

    Design note section 9.1: anything else is an undeclared derivation, which
    is how ``SEGMENTLEN`` used to reach an artifact without ever appearing in
    the design point.
    """

    PROBLEM = "projected problem data"
    DECISION = "committed decision"
    DERIVED = "derived property"
    CONSTANT = "provider constant"


@dataclass(frozen=True)
class ProviderParameter:
    """One RTL parameter and where its value is entitled to come from."""

    name: str
    ownership: ParameterOwnership
    source: QualifiedPath | None = None
    value: object | None = None
    why: str = ""

    def __post_init__(self) -> None:
        constant = self.ownership is ParameterOwnership.CONSTANT
        if constant and self.source is not None:
            raise ValueError(f"{self.name} is a constant and cannot name a source path")
        if not constant and self.source is None:
            raise ValueError(f"{self.name} is {self.ownership.value} and must name a source path")
        if constant and not self.why:
            raise ValueError(f"{self.name} is a provider constant and must say why")


@dataclass(frozen=True)
class DotProductInputs:
    """What the dot-product Kernel is allowed to read."""

    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    computation_profile: Ref[MVAUComputationProfile]
    target_dsp_block: Ref[MVAUDspBlock]
    target_clock_period_ns: Ref[float]
    narrow_weights: Ref[bool]


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

    id = MVAUComputeKernelId.DOT_PRODUCT.value
    version = "1"

    @classmethod
    def define_design(cls, design: KernelDesign[DotProductInputs]) -> None:
        facts = design.inputs
        pe = design.choice("pe", int, domain=divisors_of(facts.matrix_height))
        simd = design.choice("simd", int, domain=divisors_of(facts.matrix_width))
        pumping = design.choice("compute_pumping", bool, domain=finite((False, True)))

        # Everything the provider needs that is neither projected nor decided
        # is declared here, so no value reaches an artifact without appearing
        # in the design point first.
        # The operation reads these back by name when it assembles the provider
        # table, so they are declared rather than bound to locals.
        design.derived(
            "dsp_version",
            int,
            dependencies={"target": facts.target_dsp_block},
            evaluate=dsp_version,
        )
        design.derived(
            "signed_activations",
            bool,
            dependencies={"activation": facts.activation_element_type},
            evaluate=signed_activations,
        )
        design.derived(
            "segment_length",
            int,
            dependencies={
                "clock_period_ns": facts.target_clock_period_ns,
                "pumping": pumping,
                "simd": simd,
            },
            evaluate=segment_length,
        )
        # -- what the source must be for this Kernel to serve it at all ----
        # Answerable from graph facts alone, so they gate inference.
        design.source_constraint(
            "computation_supported",
            dependencies={"profile": facts.computation_profile},
            evaluate=lambda profile: profile is MVAUComputationProfile.ACCUMULATOR_INTEGER,
        )
        design.source_constraint(
            "numeric_supported",
            dependencies={
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
            },
            evaluate=_numeric_supported,
        )
        design.source_constraint(
            "accumulator_output_type_supported",
            dependencies={
                "accumulator": facts.accumulator_element_type,
                "output": facts.output_element_type,
            },
            evaluate=lambda accumulator, output: accumulator == output,
        )

        # -- what the provider can actually build --------------------------
        # These read the target, a decision, or a derived property, so they are
        # coverage questions rather than admission ones: a source this Kernel
        # serves may still be unbuildable on a given board.
        design.feasibility_constraint(
            "target_supported",
            dependencies={"target": facts.target_dsp_block},
            evaluate=lambda target: target in DSP_VERSION,
        )
        design.feasibility_constraint(
            "width_supported",
            dependencies={
                "target": facts.target_dsp_block,
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
                "accumulator": facts.accumulator_element_type,
                "output": facts.output_element_type,
            },
            evaluate=_width_supported,
        )
        design.feasibility_constraint(
            "narrow_weights_supported",
            dependencies={"target": facts.target_dsp_block, "narrow": facts.narrow_weights},
            evaluate=_narrow_weights_supported,
        )
        design.feasibility_constraint(
            "pumping_supported",
            dependencies={"pumping": pumping, "simd": simd},
            evaluate=lambda pumping, simd: simd >= 2 if pumping else True,
        )
        design.provider(DOT_PRODUCT_PROVIDER)
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
        # The pool presents one region-form path across its members, and the
        # operation reads it to decide the parameter topology and, now, whether
        # a replay node belongs in the assembly.
        design.export(
            REGION_FORM_EXPORT,
            cast(
                "Ref[object]",
                design.derived(
                    "region_form",
                    MVAURegionDeclaration,
                    dependencies={"matrix_width": facts.matrix_width},
                    evaluate=lambda matrix_width: MVAURegionDeclaration.DOT_PRODUCT_STREAMED,
                ),
            ),
        )

    @classmethod
    def provider_parameters(
        cls, design_paths: DotProductPaths, problem: MVAUProblem
    ) -> tuple[ProviderParameter, ...]:
        """Every ``dotp_axi`` parameter, audited against section 9.1."""

        return (
            ProviderParameter("PE", ParameterOwnership.DECISION, design_paths.pe),
            ProviderParameter("SIMD", ParameterOwnership.DECISION, design_paths.simd),
            ProviderParameter(
                "PUMPED_COMPUTE", ParameterOwnership.DECISION, design_paths.compute_pumping
            ),
            ProviderParameter(
                "ACTIVATION_WIDTH",
                ParameterOwnership.PROBLEM,
                problem.activation_element_type.path,
            ),
            ProviderParameter(
                "WEIGHT_WIDTH", ParameterOwnership.PROBLEM, problem.weight_element_type.path
            ),
            ProviderParameter(
                "ACCU_WIDTH", ParameterOwnership.PROBLEM, problem.accumulator_element_type.path
            ),
            # No MW or MH: dotp_axi does not take them.  The fused wrapper did,
            # only to derive SF and NF for the replay it contained -- which is
            # now the replay Kernel's LEN and REP.  That absence is the
            # decomposition showing up in the parameter list.
            ProviderParameter("VERSION", ParameterOwnership.DERIVED, design_paths.dsp_version),
            ProviderParameter(
                "SIGNED_ACTIVATIONS",
                ParameterOwnership.DERIVED,
                design_paths.signed_activations,
            ),
            ProviderParameter(
                "SEGMENTLEN", ParameterOwnership.DERIVED, design_paths.segment_length
            ),
            ProviderParameter(
                "NARROW_WEIGHTS",
                ParameterOwnership.DERIVED,
                MVAU_EFFECTIVE_NARROW_WEIGHTS.path,
            ),
            ProviderParameter(
                "ACTIVATION_BROADCASTING",
                ParameterOwnership.CONSTANT,
                value=1,
                why=(
                    "this slice covers the MVU form only; the VVU form is a separate "
                    "question that must not be decided from the Boolean alone"
                ),
            ),
            ProviderParameter(
                "FORCE_BEHAVIORAL",
                ParameterOwnership.CONSTANT,
                value=0,
                why="synthesis uses the inferred implementation; behavioural is a debug aid",
            ),
        )


#: Multiplier operand and accumulator widths each DSP generation offers.
_DSP_WIDTHS = {
    MVAUDspBlock.DSP48E1: (25, 18, 48),
    MVAUDspBlock.DSP48E2: (27, 18, 48),
    MVAUDspBlock.DSP58: (27, 24, 58),
}


def _numeric_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    """The dot-product core multiplies integers, and needs at least two bits."""

    return (
        activation.type_id in {"int", "uint"}
        and weight.type_id == "int"
        and activation.bit_width >= 2
        and weight.bit_width >= 2
    )


def _width_supported(
    target: MVAUDspBlock,
    activation: NumericElementType,
    weight: NumericElementType,
    accumulator: NumericElementType,
    output: NumericElementType,
) -> object:
    """Whether the operands and accumulator fit the target's DSP datapath."""

    a_width, b_width, p_width = _DSP_WIDTHS[target]
    return (
        2 <= weight.bit_width <= a_width
        and 2 <= activation.bit_width <= b_width
        and accumulator.bit_width <= p_width
        and output.bit_width <= p_width
    )


def _narrow_weights_supported(target: MVAUDspBlock, narrow: bool) -> object:
    """DSP48E1's narrower A port needs the minimum-value promise to pack.

    This is coverage, not admission: the source is perfectly expressible, the
    board just cannot build it without the stronger contract on the weights.
    """

    return narrow if target is MVAUDspBlock.DSP48E1 else True


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
        # The buffer's three parameters are the folding restated in the RTL's
        # own vocabulary, so they are derived, never decided.
        design.derived(
            "buffer_length",
            int,
            dependencies={"matrix_width": facts.matrix_width, "simd": facts.simd},
            evaluate=lambda matrix_width, simd: matrix_width // simd,
        )
        design.derived(
            "buffer_repetitions",
            int,
            dependencies={"matrix_height": facts.matrix_height, "pe": facts.pe},
            evaluate=lambda matrix_height, pe: matrix_height // pe,
        )
        design.derived(
            "buffer_width",
            int,
            dependencies={
                "activation_element_type": facts.activation_element_type,
                "simd": facts.simd,
            },
            evaluate=lambda activation_element_type, simd: simd * activation_element_type.bit_width,
        )
        design.provider(REPLAY_PROVIDER)

    @classmethod
    def provider_parameters(cls, design_paths: ReplayPaths) -> tuple[ProviderParameter, ...]:
        """Every ``replay_buffer`` parameter, audited against section 9.1."""

        return (
            ProviderParameter("LEN", ParameterOwnership.DERIVED, design_paths.buffer_length),
            ProviderParameter("REP", ParameterOwnership.DERIVED, design_paths.buffer_repetitions),
            ProviderParameter("W", ParameterOwnership.DERIVED, design_paths.buffer_width),
        )


@dataclass(frozen=True)
class DotProductPaths:
    """Where each dot-product value lives, for the provider audit."""

    pe: QualifiedPath
    simd: QualifiedPath
    compute_pumping: QualifiedPath
    dsp_version: QualifiedPath
    signed_activations: QualifiedPath
    segment_length: QualifiedPath


@dataclass(frozen=True)
class ReplayPaths:
    """Where each replay value lives, for the provider audit."""

    buffer_length: QualifiedPath
    buffer_repetitions: QualifiedPath
    buffer_width: QualifiedPath


@dataclass(frozen=True)
class DecomposedMVAUKernels:
    """The dot-product member, the replay pool, and the folding between them.

    The dot product is a *declaration*, not a pool: it belongs to the operation's
    one compute pool alongside the Kernels it is replacing.  Replay is its own
    optional pool because it is a second node, present only when the compute
    choice is the decomposed one.
    """

    dot_product: KernelDeclaration
    activation_replay: KernelSelection
    pe: Ref[int]
    simd: Ref[int]
    compute_pumping: Ref[bool]
    dot_product_paths: DotProductPaths
    replay_paths: ReplayPaths

    def provider_parameters(
        self, problem: MVAUProblem = MVAU_PROBLEM
    ) -> tuple[ProviderParameter, ...]:
        """Every parameter both providers consume, with its declared owner."""

        return (
            *DotProductKernel.provider_parameters(self.dot_product_paths, problem),
            *ActivationReplayKernel.provider_parameters(self.replay_paths),
        )


def _replay_applies(compute_pool: str) -> EvaluatorSpec[Answer[bool]]:
    """Replay belongs in the assembly exactly when the compute is decomposed.

    Gated on the compute *decision* rather than on the region form, because the
    question is which Kernel was chosen, and reading the choice directly says
    that without a derivation in between.
    """

    selected = DependencyRef.decision(
        "compute_kernel",
        QualifiedPath(f"{compute_pool}.kernel"),
        KERNEL_ID_SEMANTICS,
        absence=AbsenceMode.ALLOWS_ABSENT,
    )

    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["compute_kernel"] == DotProductKernel.id)

    return EvaluatorSpec((selected,), evaluate)


def build_decomposed_mvau_kernels(
    compute_pool: str,
    problem: MVAUProblem = MVAU_PROBLEM,
    *,
    provenance: ProblemProvenance | None = None,
) -> DecomposedMVAUKernels:
    """Declare both Kernels and wire the folding from consumer to producer.

    ``compute_pool`` is the operation's one compute selection: the dot product
    is declared inside it, beside the fused Kernels it replaces, so choosing
    the decomposed implementation is the same kind of choice as choosing any
    other -- not a different operation and not a second axis.

    The dot product is declared first because it owns the folding; the replay
    Kernel is then told what that folding is.  Declaration order here is the
    supply waterfall made literal.
    """

    dot_product, dot_product_design = declare_kernel_design(
        DotProductKernel,
        kernel_namespace(compute_pool, DotProductKernel.id),
        DotProductInputs(
            repetitions=problem.repetitions,
            matrix_width=problem.matrix_width,
            matrix_height=problem.matrix_height,
            activation_element_type=problem.activation_element_type,
            weight_element_type=problem.weight_element_type,
            output_element_type=problem.output_element_type,
            accumulator_element_type=problem.accumulator_element_type,
            computation_profile=problem.computation_profile,
            # Both are optional problem fields, but a provider parameter that
            # needs the target cannot be derived without it.  Requiring them at
            # the use site makes the engine answer Unresolved with the missing
            # field in the trace, instead of an evaluator guessing.
            target_dsp_block=problem.target_dsp_block,
            target_clock_period_ns=problem.target_clock_period_ns,
            narrow_weights=MVAU_EFFECTIVE_NARROW_WEIGHTS,
        ),
        provenance=provenance,
    )
    pe = dot_product_design.handle("pe", int)
    simd = dot_product_design.handle("simd", int)
    pumping = dot_product_design.handle("compute_pumping", bool)
    replay, replay_design = declare_kernel_design(
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
    )
    return DecomposedMVAUKernels(
        dot_product,
        KernelSelection(
            REPLAY_POOL,
            (replay,),
            # Not optional.  When the compute is decomposed the replay node is
            # part of the semantics, not an option: without it the dot product
            # would expose its expanded activation sequence at the operation
            # boundary, which is a different contract from the source's.  The
            # applicability gate already withholds the decision entirely for
            # every other compute Kernel.
            applies_if=_replay_applies(compute_pool),
        ),
        pe,
        simd,
        pumping,
        DotProductPaths(
            pe.path,
            simd.path,
            pumping.path,
            dot_product_design.handle("dsp_version", int).path,
            dot_product_design.handle("signed_activations", bool).path,
            dot_product_design.handle("segment_length", int).path,
        ),
        ReplayPaths(
            replay_design.handle("buffer_length", int).path,
            replay_design.handle("buffer_repetitions", int).path,
            replay_design.handle("buffer_width", int).path,
        ),
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
    "ActivationReplayInputs",
    "ActivationReplayKernel",
    "DOT_PRODUCT_NODE",
    "DOT_PRODUCT_PROVIDER",
    "DecomposedMVAUKernels",
    "DotProductInputs",
    "DotProductKernel",
    "DotProductPaths",
    "ParameterOwnership",
    "ProviderParameter",
    "REPLAY_NODE",
    "REPLAY_POOL",
    "REPLAY_PROVIDER",
    "ReplayPaths",
    "WEIGHT_INTERFACE",
    "build_decomposed_mvau_kernels",
    "construct_decomposed_mvau_network",
]
