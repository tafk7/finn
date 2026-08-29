# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``DotpAxiKernel``: FinnLib's ``dotp_axi`` covering the dot-product Region.

Everything here is physical.  The folding it is built for -- ``PE`` and
``SIMD`` -- is imported from the Region declaration that already fixed it, and
the Kernel cannot choose a conflicting one because it never declares one.  What
it does own is the microarchitecture: whether the compute runs on a doubled
clock, which DSP generation the core instantiates, how long a cascade the
target clock can carry, and which operand widths the datapath covers.

The internal soft-vector versus packed-DSP58 dispatch is *inside* the core and
is not a Kernel identity.  It has no separate manifest, no separate build, and
no policy-visible choice: ``VERSION`` selects it mechanically from the target
family.  See the vocabulary note section 5.5.
"""

from __future__ import annotations

from typing import cast

from finn.dataflow.authoring.scope import Ref, finite, reject
from finn.dataflow.hardware import (
    HardwareDesign,
    HardwareKernel,
    KernelBinding,
    PhysicalComponent,
    scalar_parameters,
)
from finn.dataflow.mvau.computation import DOT_PRODUCT_COMPUTATION
from finn.dataflow.mvau.hardware.inputs import DotProductHardwareInputs
from finn.dataflow.mvau.rtl_parameters import (
    DSP_VERSION,
    dsp_version,
    segment_length,
    signed_activations,
)
from finn.dataflow.mvau_problem import MVAUDspBlock
from finn.dataflow.region import NumericElementType

#: FinnLib's half of the composition, relative to the FinnLib root, in compile
#: order -- ``dotp_axi`` instantiates ``dotp``, which instantiates the core.
FINNLIB_ROOT = "finnlib"
FINNLIB_SOURCES = (
    "rtl/arith/add_multi_pkg.sv",
    "rtl/arith/add_multi.sv",
    "rtl/linalg/dotp_8sx9_dsp58.sv",
    "rtl/linalg/dotp.sv",
    "rtl/linalg/dotp_axi.sv",
)

#: The physical module this Kernel instantiates.
DOTP_AXI_MODULE = "finnlib.rtl.dotp_axi"

#: Multiplier operand and accumulator widths each DSP generation offers.
_DSP_WIDTHS = {
    MVAUDspBlock.DSP48E1: (25, 18, 48),
    MVAUDspBlock.DSP48E2: (27, 18, 48),
    MVAUDspBlock.DSP58: (27, 24, 58),
}


def covers_operand_types(activation: NumericElementType, weight: NumericElementType) -> bool:
    """Whether this core's arithmetic exists for these operand types.

    Answerable from graph facts alone -- no target, no decision -- which is what
    lets the operation's transitional admission bridge ask it before anything is
    selected.  See :data:`finn.dataflow.mvau.decomposed.HARDWARE_OPERAND_TYPE_COVERAGE`.

    Integrality is *physical*, not semantic.  A ``DataflowRegion`` admits
    floating-point element types perfectly well, and arithmetic behaviour is
    binding-owned; a float dot product is a Region this Kernel cannot build, not
    a Region that does not exist.  Adding a float Kernel should widen what FINN
    admits without touching the Region declaration or the source matcher.
    """

    return activation.type_id in {"int", "uint"} and weight.type_id == "int"


def _operand_types_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    if not covers_operand_types(activation, weight):
        return reject(
            "dotp-axi-operands-not-integer",
            "this dot-product core multiplies integers",
            values={"activation": activation.type_id, "weight": weight.type_id},
        )
    return True


def _operand_widths_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    """This multiplier needs at least two bits of each operand.

    A one-bit dot product is a perfectly good Region, and something that
    implemented it -- an XNOR-popcount core, say -- would be a different Kernel
    over the same semantics, not a different Region.
    """

    if activation.bit_width < 2 or weight.bit_width < 2:
        return reject(
            "dotp-axi-operands-too-narrow",
            "the dot-product core needs at least two bits of each operand",
            values={"activation": activation.bit_width, "weight": weight.bit_width},
        )
    return True


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

    Coverage, not admission: the source is perfectly expressible, the board just
    cannot build it without the stronger contract on the weights.
    """

    return narrow if target is MVAUDspBlock.DSP48E1 else True


class DotpAxiKernel(HardwareKernel):
    """Folded multiply-accumulate over an already-expanded activation stream."""

    id = "dotp_axi"
    version = "1"

    @classmethod
    def define_design(cls, design: HardwareDesign[DotProductHardwareInputs]) -> None:
        facts = design.inputs
        design.covers_region(
            "compute",
            region=facts.region,
            computation=facts.computation,
            implements=DOT_PRODUCT_COMPUTATION,
            description="the folded dot product over an expanded activation stream",
        )
        design.source(FINNLIB_ROOT, *FINNLIB_SOURCES)

        # -- the one physical choice --------------------------------------
        # Pumping changes how fast the datapath runs, not what crosses the
        # boundary, so the Region is untouched by it.
        pumping = design.choice("compute_pumping", bool, domain=finite((False, True)))

        # -- derived physical parameters -----------------------------------
        version = design.derived(
            "dsp_version",
            int,
            dependencies={"target": facts.target_dsp_block},
            evaluate=dsp_version,
        )
        signed = design.derived(
            "signed_activations",
            bool,
            dependencies={"activation": facts.activation_element_type},
            evaluate=signed_activations,
        )
        segment = design.derived(
            "segment_length",
            int,
            dependencies={
                "clock_period_ns": facts.target_clock_period_ns,
                "pumping": pumping,
                "simd": facts.simd,
            },
            evaluate=segment_length,
        )
        # Widths reach the RTL as declared properties.  Projecting an element
        # type to its bit count on the way out would put a number in the
        # artifact that appears nowhere in the design point.
        activation_width = design.derived(
            "activation_width",
            int,
            dependencies={"element_type": facts.activation_element_type},
            evaluate=lambda element_type: element_type.bit_width,
        )
        weight_width = design.derived(
            "weight_width",
            int,
            dependencies={"element_type": facts.weight_element_type},
            evaluate=lambda element_type: element_type.bit_width,
        )
        accumulator_width = design.derived(
            "accumulator_width",
            int,
            dependencies={"element_type": facts.accumulator_element_type},
            evaluate=lambda element_type: element_type.bit_width,
        )

        # -- what this core can actually build ------------------------------
        design.coverage_constraint(
            "operand_types_supported",
            dependencies={
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
            },
            evaluate=_operand_types_supported,
        )
        design.coverage_constraint(
            "operand_widths_supported",
            dependencies={
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
            },
            evaluate=_operand_widths_supported,
        )
        design.coverage_constraint(
            "target_supported",
            dependencies={"target": facts.target_dsp_block},
            evaluate=lambda target: target in DSP_VERSION,
        )
        design.coverage_constraint(
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
        design.coverage_constraint(
            "narrow_weights_supported",
            dependencies={"target": facts.target_dsp_block, "narrow": facts.narrow_weights},
            evaluate=_narrow_weights_supported,
        )
        design.coverage_constraint(
            "pumping_supported",
            dependencies={"pumping": pumping, "simd": facts.simd},
            evaluate=lambda pumping, simd: simd >= 2 if pumping else True,
        )

        # -- the parameter table --------------------------------------------
        # No MW or MH: dotp_axi does not take them.  The fused wrapper did, only
        # to size the replay it contained -- which is now the replay Kernel's
        # LEN and REP.  That absence is the decomposition in the parameter list.
        design.parameter("PE", cast("Ref[object]", facts.pe))
        design.parameter("SIMD", cast("Ref[object]", facts.simd))
        design.parameter("PUMPED_COMPUTE", cast("Ref[object]", pumping))
        design.parameter("ACTIVATION_WIDTH", cast("Ref[object]", activation_width))
        design.parameter("WEIGHT_WIDTH", cast("Ref[object]", weight_width))
        design.parameter("ACCU_WIDTH", cast("Ref[object]", accumulator_width))
        design.parameter("VERSION", cast("Ref[object]", version))
        design.parameter("SIGNED_ACTIVATIONS", cast("Ref[object]", signed))
        design.parameter("SEGMENTLEN", cast("Ref[object]", segment))
        design.parameter("NARROW_WEIGHTS", cast("Ref[object]", facts.narrow_weights))
        design.constant(
            "ACTIVATION_BROADCASTING",
            1,
            why=(
                "this slice covers the MVU form only; the VVU form is a separate "
                "question that must not be decided from the Boolean alone"
            ),
        )
        design.constant(
            "FORCE_BEHAVIORAL",
            0,
            why="synthesis uses the inferred implementation; behavioural is a debug aid",
        )

    @classmethod
    def elaborate(cls, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        """One ``dotp_axi`` instance.  Composition is the assembly's business."""

        return (
            PhysicalComponent(
                "dot_product",
                DOTP_AXI_MODULE,
                scalar_parameters(dict(binding.parameters)),
            ),
        )


__all__ = [
    "DOTP_AXI_MODULE",
    "FINNLIB_ROOT",
    "FINNLIB_SOURCES",
    "DotpAxiKernel",
    "covers_operand_types",
]
