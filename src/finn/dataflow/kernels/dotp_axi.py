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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring import (
    Choice,
    Constant,
    Covers,
    Imported,
    KernelInput,
    Parameter,
    RegionClaim,
    Sources,
    constraint,
    derived,
)
from finn.dataflow.authoring.scope import Ref, reject
from finn.dataflow.computation import ComputationContract, DOT_PRODUCT_COMPUTATION
from finn.dataflow.design import DATAFLOW_REGION_SEMANTICS, QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.kernels import (
    Kernel,
    PhysicalComponent,
    scalar_parameters,
)
from finn.dataflow.kernels.dsp import DspBlock, pack_lanes
from finn.dataflow.kernels.numeric import DotProductNumericTypes, RoleVerdict
from finn.dataflow.kernels.rtl_parameters import (
    DSP_VERSION,
    dsp_version,
    segment_length,
    signed_activations,
)
from finn.dataflow.region import DataflowRegion, NumericElementType, element_width

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
    DspBlock.DSP48E1: (25, 18, 48),
    DspBlock.DSP48E2: (27, 18, 48),
    DspBlock.DSP58: (27, 24, 58),
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


@dataclass(frozen=True)
class DotProductKernelInputs:
    """Operation-neutral contracts and facts consumed by ``DotpAxiKernel``."""

    role: str
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]
    pe: Ref[int]
    simd: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    narrow_weights: Ref[bool]
    target_dsp_block: Ref[DspBlock]
    target_clock_period_ns: Ref[float]


@dataclass(frozen=True)
class DotpAxiHandles:
    """Typed handles exported by the Kernel authoring declaration."""

    compute_pumping: Ref[bool]


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


def covers_numeric_types(types: DotProductNumericTypes) -> tuple[RoleVerdict, ...]:
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
                RoleVerdict(role, False, f"{datatype.name} is not a two's-complement integer")
            )
            continue
        if role in _SIGNED_ROLES and not datatype.signed():
            verdicts.append(
                RoleVerdict(
                    role,
                    False,
                    f"{datatype.name} is unsigned; the core declares this role signed",
                )
            )
            continue
        verdicts.append(RoleVerdict(role, True))
    return _with_output_paired_to_accumulator(tuple(verdicts), types)


def _with_output_paired_to_accumulator(
    verdicts: tuple[RoleVerdict, ...], types: DotProductNumericTypes
) -> tuple[RoleVerdict, ...]:
    """The last clause of the role contract: the output *is* the accumulator.

    Stated in the table above and never asked.  The core drives ``PE *
    ACCU_WIDTH`` bits straight out of ``m_axis_output_tdata``, and the composed
    wrapper sizes its own ``out0_V_tdata`` from the *output* element type -- so
    an output that is not the accumulator produces a top whose port does not
    match what the core was connected to.  Nothing truncates, converts, or
    reports; the widths simply disagree.  ``INT32`` accumulator with ``INT24``
    output generated a wrapper declaring ``OSTREAM = 48`` around a core driving
    64 bits.

    The semantic Kernels carry the same rule as a source constraint, which
    gates *inference*.  A caller that commits a selection directly never passes
    through inference, so until this was a coverage question the disagreement
    reached a generated wrapper unremarked.

    Applied only when both roles are otherwise fine, so that an unsigned
    accumulator stays separately diagnosable instead of also reporting an
    output that never had a chance to match it.
    """

    by_role = {verdict.role: verdict for verdict in verdicts}
    if not (by_role["accumulator"].supported and by_role["output"].supported):
        return verdicts
    if types.output == types.accumulator:
        return verdicts
    return tuple(
        RoleVerdict(
            verdict.role,
            False,
            f"{types.output.name} is not the accumulator {types.accumulator.name}; "
            "the core drives the accumulator straight out",
        )
        if verdict.role == "output"
        else verdict
        for verdict in verdicts
    )


def covers_operand_types(types: DotProductNumericTypes) -> bool:
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
            DotProductNumericTypes(activation, weight, accumulator, output)
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
    target: DspBlock,
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


def _narrow_weights_supported(
    target: DspBlock,
    activation: NumericElementType,
    weight: NumericElementType,
    narrow: bool,
) -> object:
    """Whether these weights pack into the A port at all.

    Coverage, not admission: the source is perfectly expressible, the board just
    cannot build it without the stronger contract on the weights.

    Asked of ``dotp.sv``'s own lane calculation rather than of the target
    family.  This used to read ``narrow if target is DSP48E1 else True``, which
    was wrong in both directions -- it admitted 27-bit non-narrow weights on
    DSP48E2 and DSP58, which reach the generic ``dotp`` core and terminate in
    ``sliceLanes()``, and it refused 8-bit non-narrow weights on DSP48E1, which
    pack into three lanes with a bit to spare and which baseline FINN builds
    routinely.
    """

    a_width, _b_width, _p_width = _DSP_WIDTHS[target]
    weight_bits = element_width(weight)
    if weight_bits > a_width:
        # ``width_supported`` reports this; the packing is undefined past here
        # and must not be asked, because the RTL's subtraction is unsigned.
        return True
    packing = pack_lanes(
        a_width=a_width,
        weight_width=weight_bits,
        activation_width=element_width(activation),
        narrow_weights=narrow,
    )
    if not packing.fits:
        return reject(
            "dotp-axi-weights-do-not-pack",
            "these weights do not fit the DSP A datapath without the narrow-weight promise",
            values={
                "weight_width": weight_bits,
                "a_datapath_width": a_width,
                "narrow_weights": narrow,
                "bit_slack": packing.slack,
            },
        )
    return True


class DotpAxiKernel(Kernel):
    """Folded multiply-accumulate over an already-expanded activation stream."""

    id = "dotp_axi"
    version = "1"
    uses_class_authoring = True

    covered_region = Imported(DATAFLOW_REGION_SEMANTICS, stable_name="region")
    computation = Imported(ComputationContract)
    pe = Imported(int)
    simd = Imported(int)
    activation_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    weight_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    output_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    accumulator_element_type = Imported(QONNX_DATATYPE_VALUE_SEMANTICS)
    narrow_weights = Imported(bool)
    target_dsp_block = Imported(DspBlock)
    target_clock_period_ns = Imported(float)

    coverage = Covers(
        RegionClaim(
            KernelInput("role"),
            covered_region,
            computation,
            DOT_PRODUCT_COMPUTATION,
            "the folded dot product over an expanded activation stream",
        )
    )
    source_files = Sources(FINNLIB_ROOT, *FINNLIB_SOURCES)
    compute_pumping = Choice(bool, domain=(False, True))
    dsp_version_value = derived(target_dsp_block, value_type=int, stable_name="dsp_version")(
        dsp_version
    )
    signed_activations_value = derived(
        activation_element_type,
        value_type=bool,
        stable_name="signed_activations",
    )(signed_activations)
    segment_length_value = derived(
        target_clock_period_ns,
        compute_pumping,
        simd,
        value_type=int,
        stable_name="segment_length",
    )(segment_length)
    activation_width = derived(activation_element_type, value_type=int)(element_width)
    weight_width = derived(weight_element_type, value_type=int)(element_width)
    accumulator_width = derived(accumulator_element_type, value_type=int)(element_width)

    operand_types_supported = constraint(
        activation_element_type,
        weight_element_type,
        accumulator_element_type,
        output_element_type,
        sets=("coverage",),
    )(_operand_types_supported)
    operand_widths_supported = constraint(
        activation_element_type,
        weight_element_type,
        sets=("coverage",),
    )(_operand_widths_supported)
    target_supported = constraint(target_dsp_block, sets=("coverage",))(
        lambda target: target in DSP_VERSION
    )
    width_supported = constraint(
        target_dsp_block,
        activation_element_type,
        weight_element_type,
        accumulator_element_type,
        output_element_type,
        sets=("coverage",),
    )(_width_supported)
    narrow_weights_supported = constraint(
        target_dsp_block,
        activation_element_type,
        weight_element_type,
        narrow_weights,
        sets=("coverage",),
    )(_narrow_weights_supported)
    pumping_supported = constraint(compute_pumping, simd, sets=("coverage",))(
        lambda pumping, simd: simd >= 2 if pumping else True
    )

    pe_parameter = Parameter("PE", pe)
    simd_parameter = Parameter("SIMD", simd)
    pumping_parameter = Parameter("PUMPED_COMPUTE", compute_pumping)
    activation_width_parameter = Parameter("ACTIVATION_WIDTH", activation_width)
    weight_width_parameter = Parameter("WEIGHT_WIDTH", weight_width)
    accumulator_width_parameter = Parameter("ACCU_WIDTH", accumulator_width)
    version_parameter = Parameter("VERSION", dsp_version_value)
    signed_parameter = Parameter("SIGNED_ACTIVATIONS", signed_activations_value)
    segment_parameter = Parameter("SEGMENTLEN", segment_length_value)
    narrow_parameter = Parameter("NARROW_WEIGHTS", narrow_weights)
    activation_broadcasting = Constant(
        "ACTIVATION_BROADCASTING",
        1,
        "this implementation broadcasts one activation vector across its parallel output lanes",
    )
    force_behavioral = Constant(
        "FORCE_BEHAVIORAL",
        0,
        "synthesis uses the inferred implementation; behavioural is a debug aid",
    )

    @staticmethod
    def compiled_handles(members: Mapping[str, object]) -> DotpAxiHandles:
        return DotpAxiHandles(cast("Ref[bool]", members["compute_pumping"]))

    @classmethod
    def elaborate(cls, kernel: Kernel) -> tuple[PhysicalComponent, ...]:
        """One ``dotp_axi`` instance.  Composition is the assembly's business."""

        return (
            PhysicalComponent(
                "dotp_axi",
                DOTP_AXI_MODULE,
                scalar_parameters(dict(kernel.parameters)),
            ),
        )


__all__ = [
    "DOTP_AXI_MODULE",
    "DotpAxiHandles",
    "FINNLIB_ROOT",
    "FINNLIB_SOURCES",
    "DotpAxiKernel",
    "DotProductKernelInputs",
    "covers_operand_types",
]
