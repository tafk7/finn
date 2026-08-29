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

from dataclasses import dataclass
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
from finn.dataflow.mvau.numeric import MVAUNumericTypes
from finn.dataflow.mvau.rtl_parameters import (
    DSP_VERSION,
    dsp_version,
    segment_length,
    signed_activations,
)
from finn.dataflow.mvau_problem import MVAUDspBlock
from finn.dataflow.region import NumericElementType, element_width

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


#: The datatypes this core's datapath is a two's-complement multiplier for.
#:
#: Matched by **canonical identity**, not by family and width.  QONNX reports
#: ``is_integer()`` true for ``BINARY``, ``BIPOLAR``, and ``TERNARY`` as well,
#: and those are integer-*valued* domains rather than two's-complement ones:
#: ``TERNARY`` is two bits wide and spans -1..1, so a width-and-family test
#: admits it and the RTL then computes over a range it was never given.  That
#: is the ``TERNARY``-lowered-as-``INT2`` defect, and naming the families
#: explicitly is what closes it.
_MULTIPLIABLE_FAMILIES = ("INT", "UINT")


def _is_twos_complement_integer(datatype: NumericElementType) -> bool:
    """Whether ``datatype`` is a plain sized integer this datapath can multiply.

    ``INT<n>`` / ``UINT<n>`` and nothing else.  Special encodings are excluded
    by name even where they answer ``is_integer()``, and a datatype QONNX would
    canonicalize to a special name -- ``UINT1`` is ``BINARY`` -- is excluded
    with them, because the canonical name is the identity.
    """

    name = datatype.name
    return any(
        name.startswith(prefix) and name[len(prefix) :].isdigit()
        for prefix in _MULTIPLIABLE_FAMILIES
    )


#: The roles this core declares *signed* in its own RTL, and therefore cannot
#: accept an unsigned datatype for.
#:
#: ``dotp.sv`` and ``dotp_top.sv`` declare the result port as
#:
#:     output logic signed [PE-1:0][ACCU_WIDTH-1:0]  p
#:
#: so the accumulation and the value leaving the core are two's-complement
#: signed.  A ``UINT16`` accumulator would be reinterpreted at the boundary,
#: and every value above the signed maximum would come back negative.
#:
#: The weight operand is signed for the same kind of reason: the multiplier's
#: A port is fed as a signed operand.
#:
#: The activation is *not* in this set -- ``SIGNED_ACTIVATIONS`` exists
#: precisely so the core can be told which of the two it is getting.  So the
#: role contract is deliberately non-uniform:
#:
#:     activation   INT<n> or UINT<n>
#:     weight       INT<n>
#:     accumulator  INT<n>
#:     output       INT<n>, and equal to the accumulator
#:
#: Applying one rule to all four roles was the previous shape, and it admitted
#: an unsigned accumulator and output with no refusal at all.
_SIGNED_ROLES = frozenset({"weight", "accumulator", "output"})


@dataclass(frozen=True)
class _RoleVerdict:
    """One numeric role's answer, with the reason if it is a refusal."""

    role: str
    supported: bool
    detail: str = ""


def covers_numeric_types(types: MVAUNumericTypes) -> tuple[_RoleVerdict, ...]:
    """This core's one authoritative datatype predicate, over every role.

    The single implementation behind both the physical coverage constraint and
    the operation's transitional admission bridge.  They ask it through thin
    adapters rather than each restating the rule, so the two cannot answer
    differently -- which §10 of the adoption note requires while the bridge
    exists at all.

    Answerable from graph facts alone: no target, no decision.  That is what
    lets admission ask it before anything is selected, and it is also why the
    *width envelope* is not here -- the DSP datapath limits depend on the board,
    so they stay in :func:`_width_supported`, where a refusal removes a target
    from coverage rather than a graph from inference.

    Returns a verdict per role rather than one boolean, so a refusal names which
    operand was wrong; the callers reduce that as they need it.

    Integrality is *physical*, not semantic.  A ``DataflowRegion`` admits
    floating-point element types perfectly well, and arithmetic behaviour is
    binding-owned; a float dot product is a Region this Kernel cannot build, not
    a Region that does not exist.  Adding a float Kernel widens what FINN admits
    without touching the Region declaration or the source matcher.
    """

    roles = (
        ("activation", types.activation),
        ("weight", types.weight),
        ("accumulator", types.accumulator),
        ("output", types.output),
    )
    verdicts = []
    for role, datatype in roles:
        if not _is_twos_complement_integer(datatype):
            verdicts.append(
                _RoleVerdict(role, False, f"{datatype.name} is not a two's-complement integer")
            )
            continue
        if role in _SIGNED_ROLES and not datatype.signed():
            verdicts.append(
                _RoleVerdict(
                    role,
                    False,
                    f"{datatype.name} is unsigned; the core declares this role signed",
                )
            )
            continue
        verdicts.append(_RoleVerdict(role, True))
    return tuple(verdicts)


def covers_operand_types(types: MVAUNumericTypes) -> bool:
    """The boolean reduction, for callers that only need admission's answer."""

    return all(verdict.supported for verdict in covers_numeric_types(types))


def _operand_types_supported(
    activation: NumericElementType,
    weight: NumericElementType,
    accumulator: NumericElementType,
    output: NumericElementType,
) -> object:
    """The coverage adapter: the same predicate, reported as a finding."""

    refused = [
        verdict
        for verdict in covers_numeric_types(
            MVAUNumericTypes(activation, weight, accumulator, output)
        )
        if not verdict.supported
    ]
    if refused:
        return reject(
            "dotp-axi-numeric-types-unsupported",
            "this dot-product core multiplies two's-complement integers",
            values={verdict.role: verdict.detail for verdict in refused},
        )
    return True


def _operand_widths_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    """This multiplier needs at least two bits of each operand.

    A one-bit dot product is a perfectly good Region, and something that
    implemented it -- an XNOR-popcount core, say -- would be a different Kernel
    over the same semantics, not a different Region.
    """

    if element_width(activation) < 2 or element_width(weight) < 2:
        return reject(
            "dotp-axi-operands-too-narrow",
            "the dot-product core needs at least two bits of each operand",
            values={"activation": element_width(activation), "weight": element_width(weight)},
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
        2 <= element_width(weight) <= a_width
        and 2 <= element_width(activation) <= b_width
        and element_width(accumulator) <= p_width
        and element_width(output) <= p_width
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
            evaluate=element_width,
        )
        weight_width = design.derived(
            "weight_width",
            int,
            dependencies={"element_type": facts.weight_element_type},
            evaluate=element_width,
        )
        accumulator_width = design.derived(
            "accumulator_width",
            int,
            dependencies={"element_type": facts.accumulator_element_type},
            evaluate=element_width,
        )

        # -- what this core can actually build ------------------------------
        design.coverage_constraint(
            "operand_types_supported",
            # Every numeric role, not just the two that get multiplied.  An
            # integer dot product with a floating-point accumulator used to pass
            # here because nothing asked.
            dependencies={
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
                "accumulator": facts.accumulator_element_type,
                "output": facts.output_element_type,
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
