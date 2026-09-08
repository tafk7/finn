# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What every MVAU Design agrees about before it disagrees about supply.

The folding and operand facts are authored once and reused by each concrete
Design. Compilation still gives every occurrence its own root-relative
coordinates: ``design.dot_product.pe`` and ``design.batch_interleaved.pe`` are
distinct persisted Decisions even though both come from the same Python
declaration object. The shared definition prevents duplicated authoring; it
does not merge choices across alternatives.
"""

from __future__ import annotations

from finn.dataflow.designs.design import DataflowDesign
from finn.dataflow.kernels.dotp_axi import DspBlock
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    constraint,
    divisors_of,
    reject,
)
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.ops.mvau.computation import MvauComputationProfile


class WeightedDotProductDesign(DataflowDesign):
    """The shared operand facts and the two folding choices MVAU owns."""

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Input(bool)
    target_dsp = Input(DspBlock)
    clock_period_ns = Input(float)
    computation_profile = Input(MvauComputationProfile)

    #: Owned here because each of them changes both Regions and their edge.
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

    @constraint(profile=computation_profile)
    def computes_a_bare_accumulator(*, profile: MvauComputationProfile) -> object:
        """These Designs build a dot product and stop; they fuse no activation.

        A Design limitation, argued from the Design's own structure rather than
        from the mathematics: every alternative below this class declares a
        Network with activation, weight and output boundaries and a compute
        Region that produces the accumulator directly.  There is no threshold
        boundary for the fourth operand to cross and no stage to apply it in,
        so a fused-threshold node has no *composition* here -- while remaining
        a perfectly valid problem that a later Design may build.

        Refusing it here rather than in the operation is what keeps that true:
        the node still binds, still projects its source facts, and reports an
        inapplicable Design instead of an unreadable node.
        """

        if not profile.fuses_activation:
            return True
        return reject(
            "mvau-design-fuses-no-activation",
            "this Design emits its accumulator directly and has no stage for a fused "
            f"threshold; this node computes {profile.name}",
            values={"computation_profile": profile.name},
        )

    dataflow_support = ConstraintGroup(computes_a_bare_accumulator)


#: Every Input the shared base consumes, for a caller assembling bindings.
SHARED_INPUTS = (
    "repetitions",
    "matrix_width",
    "matrix_height",
    "activation_type",
    "weight_type",
    "accumulator_type",
    "output_type",
    "narrow_weights",
    "target_dsp",
    "clock_period_ns",
    "computation_profile",
)

__all__ = ["SHARED_INPUTS", "WeightedDotProductDesign"]
