# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What every matrix-multiplication Kernel agrees about before supply differs.

The folding and operand facts are authored once and reused by each concrete
Kernel. Compilation still gives every occurrence its own root-relative
coordinates: ``kernel.dot_product.pe`` and ``kernel.batch_interleaved.pe`` are
distinct persisted Decisions even though both come from the same Python
declaration object. The shared definition prevents duplicated authoring; it
does not merge choices across alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.model.kernel import Kernel
from finn.dataflow.analysis.integer_dot import IntegerSupportReport
from finn.dataflow.model.logical.datatypes import QONNXDataType
from finn.dataflow.space.declarations import (
    ConstraintGroup,
    Decision,
    Input,
    constraint,
    divisors_of,
    reject,
)
from finn.dataflow.model.logical.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


class DspBlock(str, Enum):
    """DSP generation selected by the target platform."""

    DSP48E1 = "DSP48E1"
    DSP48E2 = "DSP48E2"
    DSP58 = "DSP58"


class AccumulationMode(str, Enum):
    """How matrix products are accumulated."""

    INTEGER = "integer"
    XNOR_POPCOUNT = "xnor_popcount"
    BIPOLAR_POPCOUNT = "bipolar_popcount"


class ActivationMode(str, Enum):
    """What happens to the accumulator afterwards."""

    NONE = "none"
    MULTITHRESHOLD = "multithreshold"


@dataclass(frozen=True, slots=True)
class MvauComputationProfile:
    """The independent accumulation and post-accumulation semantics."""

    accumulation: AccumulationMode
    activation: ActivationMode

    @property
    def fuses_activation(self) -> bool:
        return self.activation is ActivationMode.MULTITHRESHOLD

    @property
    def name(self) -> str:
        return f"{self.accumulation.value}+{self.activation.value}"


def computation_profile(
    *,
    no_activation: bool,
    binary_xnor: bool,
    activation_type: QONNXDataType | None = None,
    weight_type: QONNXDataType | None = None,
) -> MvauComputationProfile:
    """Derive the reusable mathematical profile from source-level facts."""

    from qonnx.core.datatype import DataType  # type: ignore[import-not-found]  # noqa: PLC0415

    bipolar = DataType["BIPOLAR"]
    if binary_xnor:
        accumulation = AccumulationMode.XNOR_POPCOUNT
    elif activation_type == bipolar and weight_type == bipolar:
        accumulation = AccumulationMode.BIPOLAR_POPCOUNT
    else:
        accumulation = AccumulationMode.INTEGER
    return MvauComputationProfile(
        accumulation,
        ActivationMode.NONE if no_activation else ActivationMode.MULTITHRESHOLD,
    )


class WeightedDotProductKernel(Kernel):
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
    numerical_support = Input(IntegerSupportReport, allow_absent=True)

    #: Owned here because each of them changes both Regions and their edge.
    pe = Decision(int, domain=divisors_of(matrix_height))
    simd = Decision(int, domain=divisors_of(matrix_width))

    @constraint(profile=computation_profile)
    def computes_a_bare_accumulator(*, profile: MvauComputationProfile) -> object:
        """These Kernels build a dot product and stop; they fuse no activation.

        A Kernel limitation, argued from the Kernel's own structure rather than
        from the mathematics: every alternative below this class declares a
        Network with activation, weight and output boundaries and a compute
        Region that produces the accumulator directly.  There is no threshold
        boundary for the fourth operand to cross and no stage to apply it in,
        so a fused-threshold node has no *composition* here -- while remaining
        a perfectly valid problem that a later Kernel may build.

        Refusing it here rather than in the operation is what keeps that true:
        the node still binds, still projects its source facts, and reports an
        inapplicable Kernel instead of an unreadable node.
        """

        if not profile.fuses_activation:
            return True
        return reject(
            "mvau-kernel-fuses-no-activation",
            "this Kernel emits its accumulator directly and has no stage for a fused "
            f"threshold; this node computes {profile.name}",
            values={"computation_profile": profile.name},
        )

    logical_support = ConstraintGroup(computes_a_bare_accumulator)


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
    "numerical_support",
)

__all__ = [
    "AccumulationMode",
    "ActivationMode",
    "DspBlock",
    "MvauComputationProfile",
    "SHARED_INPUTS",
    "WeightedDotProductKernel",
    "computation_profile",
]
