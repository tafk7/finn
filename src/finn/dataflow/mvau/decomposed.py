# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MVAU standard streamed form as two Regions and a Network.

``ActivationReplayKernel`` presents each activation row once per neuron fold;
``DotProductKernel`` does the arithmetic.  The monolithic Region did both, with
the replay hidden inside its schedule.

Both are **semantic** declarations.  Despite the class name -- which survives
until the migration renames it -- neither knows anything about hardware: no
source files, no RTL parameters, no target coverage, no provider.  What they
declare is a Region, the folding it is built from, what it requires supplied,
and what it is required to compute.  ``DotProductKernel`` owns ``pe`` and
``simd`` because it is the thing those numbers fold; ``ActivationReplayKernel``
owns nothing and derives its geometry from the folding it must feed.  That
direction is the supply waterfall: a producer presents what its consumer's
configuration requires, and never the reverse.

The hardware that realizes them -- ``DotpAxiKernel`` and ``ReplayBufferKernel``
-- is declared here too, because this module is what knows the wiring, but it
lives in :mod:`finn.dataflow.mvau.hardware` and imports these declarations
rather than being part of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring.kernel_design import (
    KernelDesign,
    declare_kernel_design,
    kernel_namespace,
)
from finn.dataflow.authoring.op_design import ProblemProvenance
from finn.dataflow.authoring.scope import Ref, divisors_of
from finn.dataflow.design import (
    AbsenceMode,
    Answer,
    Decided,
    DependencyRef,
    DependencyView,
    EvaluatorSpec,
    QualifiedPath,
)
from finn.dataflow.hardware import ComputationContract, hardware_namespace
from finn.dataflow.hardware.authoring import declare_hardware_kernel
from finn.dataflow.hardware.kernel import HardwareKernelDeclaration
from finn.dataflow.kernels import (
    KERNEL_ID_SEMANTICS,
    Kernel,
    KernelDeclaration,
    KernelSelection,
)
from finn.dataflow.mvau.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.mvau.compute_pool import (
    REGION_FORM_EXPORT,
    WEIGHT_INTERFACE,
    MVAUComputeKernelId,
)
from finn.dataflow.mvau.hardware.dotp_axi import DotpAxiKernel
from finn.dataflow.mvau.hardware.inputs import (
    ActivationReplayHardwareInputs,
    DotProductHardwareInputs,
)
from finn.dataflow.mvau.hardware.replay_buffer import ReplayBufferKernel
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.mvau.semantics import (
    ACTIVATION_EDGE,
    DOT_PRODUCT_NODE,
    HARDWARE_NUMERIC_TYPE_COVERAGE,
    REPLAY_NODE,
    accumulator_output_type_supported,
    construct_decomposed_mvau_network,
    dot_product_computation_supported,
    some_dot_product_hardware_covers_numeric_types,
)
from finn.dataflow.mvau_problem import (
    MVAU_EFFECTIVE_NARROW_WEIGHTS,
    MVAU_PROBLEM,
    MVAUComputationProfile,
    MVAUProblem,
)
from finn.dataflow.region import DataflowRegion, NumericElementType

#: Replay is its own pool because it is a second node, not a second compute
#: choice: it is present exactly when the decomposed compute member is selected.
#: The dot product has no pool of its own -- it is a member of the operation's
#: one compute pool, beside the fused Kernels it replaces.
REPLAY_POOL = "mvau.replay"

#: Where the two physical Kernels are placed.
HARDWARE_OWNER = "mvau.hardware"


@dataclass(frozen=True)
class DotProductInputs:
    """What the dot-product Region declaration is allowed to read.

    Target facts are absent on purpose.  A Region is the same Region on every
    board; what a board can build is the physical Kernel's question.
    """

    repetitions: Ref[int]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    computation_profile: Ref[MVAUComputationProfile]


@dataclass(frozen=True)
class ActivationReplayInputs:
    """What the replay Region declaration is allowed to read.

    ``pe`` and ``simd`` arrive from the dot-product declaration's choices, wired
    in by this module.  Replay does not decide them and does not name their
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

        # -- what the source must be for this Region to represent it at all --
        # Answerable from graph facts alone, so they gate inference.  Both are
        # statements about the *shape* of the computation, not about whether
        # any hardware exists for it.
        design.source_constraint(
            "computation_supported",
            dependencies={"profile": facts.computation_profile},
            evaluate=dot_product_computation_supported,
        )
        design.source_constraint(
            "accumulator_output_type_supported",
            dependencies={
                "accumulator": facts.accumulator_element_type,
                "output": facts.output_element_type,
            },
            evaluate=accumulator_output_type_supported,
        )
        # TRANSITIONAL, and not a property of this Region.
        #
        # A ``DataflowRegion`` admits floating-point element types, and
        # arithmetic behaviour is binding-owned, so "these operands are
        # integers" is a statement about the hardware that exists, not about
        # what this Region can represent.  What inference actually needs to ask
        # is existential: *does some physical Kernel covering this Region handle
        # these types?*  There is no generic bridge for that question yet --
        # ``admissible_kernels`` sees semantic declarations only -- so the
        # existential is spelled out here over a declared inventory.
        #
        # Adding a float Kernel therefore widens admission by extending
        # ``HARDWARE_NUMERIC_TYPE_COVERAGE``, not by editing this Region or the
        # source matcher.  Delete this once admission can ask coverage directly.
        design.source_constraint(
            "some_hardware_covers_the_numeric_types",
            dependencies={
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
                "accumulator": facts.accumulator_element_type,
                "output": facts.output_element_type,
            },
            evaluate=some_dot_product_hardware_covers_numeric_types,
        )

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
        # What this Region's traffic is required to mean.  A physical Kernel
        # declares what it implements and the two are compared, because equal
        # beats over equal schedules do not imply equal arithmetic.
        design.derived(
            "computation",
            ComputationContract,
            dependencies={},
            evaluate=lambda: DOT_PRODUCT_COMPUTATION,
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
        # operation reads it to decide the parameter topology and whether a
        # replay node belongs in the assembly.
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


class ActivationReplayKernel(Kernel):
    """Present each activation row once per neuron fold.

    Retained even at one neuron fold, where it is an identity: the Network shape
    should not depend on the matrix geometry, and eliding the physical buffer is
    the hardware's business, not the semantics'.
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
        design.derived(
            "computation",
            ComputationContract,
            dependencies={},
            evaluate=lambda: ACTIVATION_REPLAY_COMPUTATION,
        )


@dataclass(frozen=True)
class DecomposedMVAUKernels:
    """The decomposed slice: two Region declarations and the hardware for them.

    The dot product is a *declaration*, not a pool: it belongs to the operation's
    one compute pool alongside the Kernels it is replacing.  Replay is its own
    pool because it is a second node, present only when the compute choice is
    the decomposed one.

    The two physical Kernels are not pools either.  Exactly one covers each
    Region, so there is no choice to make and none is invented; their bindings
    are derived.
    """

    dot_product: KernelDeclaration
    activation_replay: KernelSelection
    dot_product_hardware: HardwareKernelDeclaration
    replay_hardware: HardwareKernelDeclaration
    pe: Ref[int]
    simd: Ref[int]
    compute_pumping: Ref[bool]
    #: The two Region declarations and what each is required to compute.
    #:
    #: Exposed because a Kernel covering *both* Regions has to name the same
    #: two declarations these physical Kernels name one each, and rebuilding
    #: their paths from the namespace convention would be a second statement of
    #: where they live -- the kind that stays right until it does not.  This is
    #: a pure addition: nothing new is declared, so the design space is
    #: unchanged.
    dot_product_region: Ref[DataflowRegion]
    dot_product_computation: Ref[ComputationContract]
    replay_region: Ref[DataflowRegion]
    replay_computation: Ref[ComputationContract]

    @property
    def hardware(self) -> tuple[HardwareKernelDeclaration, ...]:
        return (self.dot_product_hardware, self.replay_hardware)

    @property
    def coverage_constraints(self) -> tuple[QualifiedPath, ...]:
        """Every physical coverage condition, for the operation's feasibility set.

        Coverage is asked at feasibility rather than only at binding, so a point
        that no hardware can build is refused while it is still a design point
        -- not accepted, elaborated, and then rejected by a Kernel.
        """

        return tuple(
            dict.fromkeys(path for item in self.hardware for path in item.coverage_constraints)
        )


def _decomposed_selected(compute_pool: str) -> EvaluatorSpec[Answer[bool]]:
    """True exactly when the operation's compute choice is the decomposed one.

    Gates the replay pool and both physical Kernels.  Read from the compute
    *decision* rather than from a region form, because the question is which
    alternative was chosen and reading the choice says that without a derivation
    in between.
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
    """Declare both Regions, then the hardware that covers them.

    ``compute_pool`` is the operation's one compute selection: the dot product is
    declared inside it, beside the fused Kernels it replaces, so choosing the
    decomposed implementation is the same kind of choice as choosing any other.

    Declaration order is the supply waterfall made literal.  The dot product is
    declared first because it owns the folding; the replay Region is then told
    what that folding is; the physical Kernels are declared last because they
    import from both and contribute to neither.
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
        ),
        provenance=provenance,
    )
    pe = dot_product_design.handle("pe", int)
    simd = dot_product_design.handle("simd", int)
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

    dot_product_region = dot_product_design.handle("region", DataflowRegion)
    dot_product_computation = dot_product_design.handle("computation", ComputationContract)
    replay_region = replay_design.handle("region", DataflowRegion)
    replay_computation = replay_design.handle("computation", ComputationContract)

    applies = _decomposed_selected(compute_pool)
    dot_product_hardware, dotp_design = declare_hardware_kernel(
        DotpAxiKernel,
        hardware_namespace(HARDWARE_OWNER, DotpAxiKernel.id),
        DotProductHardwareInputs(
            region=dot_product_region,
            computation=dot_product_computation,
            pe=pe,
            simd=simd,
            activation_element_type=problem.activation_element_type,
            weight_element_type=problem.weight_element_type,
            output_element_type=problem.output_element_type,
            accumulator_element_type=problem.accumulator_element_type,
            narrow_weights=MVAU_EFFECTIVE_NARROW_WEIGHTS,
            # Both are optional problem fields, but a physical parameter that
            # needs the target cannot be derived without it.  Requiring them at
            # the use site makes the engine answer Unresolved with the missing
            # field in the trace, instead of an evaluator guessing.
            target_dsp_block=problem.target_dsp_block,
            target_clock_period_ns=problem.target_clock_period_ns,
        ),
        applies_if=applies,
    )
    replay_hardware, _ = declare_hardware_kernel(
        ReplayBufferKernel,
        hardware_namespace(HARDWARE_OWNER, ReplayBufferKernel.id),
        ActivationReplayHardwareInputs(
            region=replay_region,
            computation=replay_computation,
            matrix_width=problem.matrix_width,
            matrix_height=problem.matrix_height,
            pe=pe,
            simd=simd,
            activation_element_type=problem.activation_element_type,
        ),
        applies_if=applies,
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
            applies_if=applies,
        ),
        dot_product_hardware,
        replay_hardware,
        pe,
        simd,
        dotp_design.handle("compute_pumping", bool),
        dot_product_region,
        dot_product_computation,
        replay_region,
        replay_computation,
    )


__all__ = [
    "ACTIVATION_EDGE",
    "HARDWARE_NUMERIC_TYPE_COVERAGE",
    "ActivationReplayInputs",
    "ActivationReplayKernel",
    "DOT_PRODUCT_NODE",
    "DecomposedMVAUKernels",
    "DotProductInputs",
    "DotProductKernel",
    "HARDWARE_OWNER",
    "REPLAY_NODE",
    "REPLAY_POOL",
    "WEIGHT_INTERFACE",
    "build_decomposed_mvau_kernels",
    "construct_decomposed_mvau_network",
]
