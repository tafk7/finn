# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Test-only ``MvuVvuAxiKernel`` covering both DotProduct Regions.

The forcing case for Phase 4 of the Region/Kernel binding migration, and the
one that proves physical coverage is not one-to-one with Regions.  Where the
decomposed path binds a ``ReplayBufferKernel`` to the replay node and a
``DotpAxiKernel`` to the dot-product node and wires them together in a
generated top, this Kernel covers both Regions *and the edge between them* and
elaborates to a single component.  Same selected semantics, same Network, one
physical reading of it instead of two.

The edge is the whole point.  ``mvu_vvu_axi`` instantiates ``replay_buffer``
internally (``mvu_vvu_axi.sv:141``), so the connection the Network declares is
realized as wiring inside the core rather than as a pair of exposed
interfaces.  Declaring it with ``absorbs_edge`` is what lets binding check the
selected Network really has that connection, running that way, between those
two nodes.

**Test-only, deliberately.**  This Kernel is a required forcing case and is
*not* production-selectable: it is absent from the operation's compute pool,
absent from production inference admission, and absent from
the production candidate inventory.  Making it selectable is a later decision
that needs measurements or a compatibility need behind it, and it is only
meaningful where both the fused and the decomposed Kernel are valid candidates
at the same semantic point.  See §16.3 of the migration plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.authoring.scope import Ref, finite, reject
from finn.dataflow.kernels import (
    KernelScope,
    Kernel,
    PhysicalComponent,
    scalar_parameters,
)
from finn.dataflow.computation import (
    ACTIVATION_REPLAY_COMPUTATION,
    ComputationContract,
    DOT_PRODUCT_COMPUTATION,
)
from finn.dataflow.kernels.dsp import (
    DspBlock,
    a_datapath_width,
    b_datapath_width,
    p_datapath_width,
    pack_lanes,
)
from finn.dataflow.kernels.numeric import DotProductNumericTypes, RoleVerdict
from finn.dataflow.kernels.rtl_parameters import (
    DSP_VERSION,
    dsp_version,
    segment_length,
    signed_activations,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.region import DataflowRegion, NumericElementType, element_width

#: FINN's fused core and everything under it, relative to the FINN root, in
#: compile order -- ``mvu_vvu_axi`` instantiates ``replay_buffer`` and one of
#: ``mvu`` / ``mvu_vvu_8sx9_dsp58``, and ``mvu`` instantiates ``add_multi``.
#:
#: This is the *existing* fused RTL, reused unchanged.  The migration does not
#: fork it: what changes is who declares it and what that declaration is
#: checked against, not the text being compiled.
FINN_ROOT = "finn"
FINN_SOURCES = (
    "finn-rtllib/mvu/mvu_pkg.sv",
    "finn-rtllib/mvu/replay_buffer.sv",
    "finn-rtllib/mvu/add_multi.sv",
    "finn-rtllib/mvu/mvu.sv",
    "finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv",
    "finn-rtllib/mvu/mvu_vvu_axi.sv",
)


@dataclass(frozen=True)
class FusedMatrixVectorHardwareInputs:
    replay_region: Ref[DataflowRegion]
    replay_computation: Ref[ComputationContract]
    compute_region: Ref[DataflowRegion]
    compute_computation: Ref[ComputationContract]
    network: Ref[DataflowNetwork]
    matrix_width: Ref[int]
    matrix_height: Ref[int]
    pe: Ref[int]
    simd: Ref[int]
    activation_element_type: Ref[NumericElementType]
    weight_element_type: Ref[NumericElementType]
    output_element_type: Ref[NumericElementType]
    accumulator_element_type: Ref[NumericElementType]
    narrow_weights: Ref[bool]
    target_dsp_block: Ref[DspBlock]
    target_clock_period_ns: Ref[float]


#: The physical module this Kernel instantiates.
MVU_VVU_AXI_MODULE = "finn-rtllib.mvu.mvu_vvu_axi"

#: The Region roles this Kernel covers, and the edge it absorbs.
REPLAY_ROLE = "replay"
COMPUTE_ROLE = "compute"
ACTIVATION_EDGE_ROLE = "activation"


# -- this core's datatype contract -------------------------------------------
#
# Written from ``mvu.sv`` rather than adapted from ``DotpAxiKernel``'s, and
# that separation is deliberate rather than incidental.  ``mvu_vvu_axi`` is a
# different physical core: it happens to agree with ``dotp_axi`` on every role
# today, and two cores agreeing today is a fact about today.  Delegating to the
# other Kernel's predicate would propagate one core's limits to the other
# silently, and the day they diverge nothing would say so.
#
# The agreement is instead *demonstrated* -- see
# ``test_the_two_cores_agree_today_and_the_agreement_is_checked_not_assumed``
# -- which is a test that fails loudly when it stops being true.


#: The datatypes this core's datapath is a two's-complement multiplier for.
#:
#: Matched by canonical identity, not by family and width.  ``BINARY``,
#: ``BIPOLAR`` and ``TERNARY`` all answer ``is_integer()`` true and none of
#: them is a two's-complement encoding: ``TERNARY`` is two bits wide and spans
#: -1..1, so a width-and-family test admits it and the RTL computes over a
#: range it was never given.
_MULTIPLIABLE_FAMILIES = ("INT", "UINT")

#: The roles ``mvu.sv`` declares signed in its own port list:
#:
#:     input  logic signed [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]  w   (line 55)
#:     input  logic        [SIMD-1:0][ACTIVATION_WIDTH-1:0]      a   (line 56)
#:     output logic signed [PE-1:0][ACCU_WIDTH-1:0]              p   (line 60)
#:
#: The activation is the one operand declared without a sign, because
#: ``SIGNED_ACTIVATIONS`` is how the core is told which of the two it is
#: getting.  Everything else is two's-complement signed, so an unsigned
#: accumulator would be reinterpreted at the output boundary and every value
#: above the signed maximum would come back negative.
#:
#: The output is included because this form writes the accumulator out
#: directly: ``p`` *is* the result port.
_SIGNED_ROLES = frozenset({"weight", "accumulator", "output"})


def _is_twos_complement_integer(datatype: NumericElementType) -> bool:
    """Whether ``datatype`` is a plain sized integer this datapath can multiply."""

    name = datatype.name
    return any(
        name.startswith(prefix) and name[len(prefix) :].isdigit()
        for prefix in _MULTIPLIABLE_FAMILIES
    )


def covers_numeric_types(types: DotProductNumericTypes) -> tuple[RoleVerdict, ...]:
    """This core's own authoritative datatype predicate, over every role.

    Complete by construction: it takes the whole :class:`DotProductNumericTypes`
    bundle, so a role cannot go unasked.  A predicate that inspected only the
    two operands being multiplied is exactly the shape that admitted an integer
    dot product with a floating-point accumulator.

    Answerable from graph facts alone -- no target, no decision -- which is what
    would let admission ask it if this Kernel were ever production-selectable.
    The width envelope is *not* here: the DSP datapath limits depend on the
    board, so they live in :func:`_width_supported`, where a refusal removes a
    target from coverage rather than a graph from inference.
    """

    verdicts = []
    for role, datatype in (
        ("activation", types.activation),
        ("weight", types.weight),
        ("accumulator", types.accumulator),
        ("output", types.output),
    ):
        if not _is_twos_complement_integer(datatype):
            verdicts.append(
                RoleVerdict(role, False, f"{datatype.name} is not a two's-complement integer")
            )
        elif role in _SIGNED_ROLES and not datatype.signed():
            verdicts.append(
                RoleVerdict(
                    role,
                    False,
                    f"{datatype.name} is unsigned; the core declares this role signed",
                )
            )
        else:
            verdicts.append(RoleVerdict(role, True))
    return tuple(verdicts)


def covers_operand_types(types: DotProductNumericTypes) -> bool:
    """The boolean reduction, for callers that only need a yes or no."""

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
            "mvu-vvu-axi-numeric-types-unsupported",
            "this fused core multiplies two's-complement integers",
            values={verdict.role: verdict.detail for verdict in refused},
        )
    return True


def _operand_widths_supported(activation: NumericElementType, weight: NumericElementType) -> object:
    """``mvu.sv`` documents both operands as at least two bits wide.

    Lines 55 and 56 say so in the port list itself: ``WEIGHT_WIDTH >= 2`` and
    ``ACTIVATION_WIDTH >= 2``.  A one-bit dot product is a perfectly good
    Region; something that implemented it would be a different Kernel over the
    same semantics.
    """

    if element_width(activation) < 2 or element_width(weight) < 2:
        return reject(
            "mvu-vvu-axi-operands-too-narrow",
            "the fused core needs at least two bits of each operand",
            values={"activation": element_width(activation), "weight": element_width(weight)},
        )
    return True


def _datapath_widths(target: DspBlock) -> tuple[int, int, int]:
    """The A, B and P datapath widths, as ``mvu.sv`` itself computes them.

    Derived from ``VERSION`` rather than tabulated, so a change to those
    localparams shows up here rather than in a constant that merely used to
    agree.
    """

    version = DSP_VERSION[target]
    return (
        a_datapath_width(version),
        b_datapath_width(version),
        p_datapath_width(version),
    )


def _width_supported(
    target: DspBlock,
    activation: NumericElementType,
    weight: NumericElementType,
    accumulator: NumericElementType,
    output: NumericElementType,
) -> object:
    """Whether the operands and accumulator fit the target's DSP datapath.

    The weight feeds the A port and the activation feeds B, which is why the
    two are not interchangeable here.
    """

    a_width, b_width, p_width = _datapath_widths(target)
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

    Asked of the core's own lane calculation rather than of the target family.
    The rule this replaced -- "DSP48E1 requires the promise" -- was wrong twice
    over: it admitted 27-bit non-narrow weights on DSP48E2 and DSP58, which
    reach the generic core and terminate at ``mvu.sv:113``, and it refused
    8-bit non-narrow weights on DSP48E1, which pack with a bit to spare.

    Coverage, not admission: the source is perfectly expressible either way, and
    the same weights on a wider A port -- or with the narrow promise -- build
    fine. What is refused is this board plus this promise.
    """

    a_width, _b_width, _p_width = _datapath_widths(target)
    weight_bits = element_width(weight)
    if weight_bits > a_width:
        # Reported by ``width_supported``; the packing is undefined past here
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
            "mvu-vvu-axi-weights-do-not-pack",
            "these weights do not fit the DSP A datapath without the narrow-weight promise",
            values={
                "weight_width": weight_bits,
                "a_datapath_width": a_width,
                "narrow_weights": narrow,
                "bit_slack": packing.slack,
            },
        )
    return True


class MvuVvuAxiKernel(Kernel):
    """Replay and folded multiply-accumulate as one core."""

    id = "mvu_vvu_axi"
    version = "1"

    @classmethod
    def define_design(cls, design: KernelScope[FusedMatrixVectorHardwareInputs]) -> None:
        facts = design.inputs
        # Two covered Regions and the edge between them.  Each names the
        # declaration it realizes and states what it computes, so the fused
        # claim is checked exactly as hard as two separate ones would be --
        # covering more semantics is not a licence to be vaguer about them.
        design.covers_region(
            REPLAY_ROLE,
            region=facts.replay_region,
            computation=facts.replay_computation,
            implements=ACTIVATION_REPLAY_COMPUTATION,
            description="the compact-to-expanded activation sequence, replayed internally",
        )
        design.covers_region(
            COMPUTE_ROLE,
            region=facts.compute_region,
            computation=facts.compute_computation,
            implements=DOT_PRODUCT_COMPUTATION,
            description="the folded dot product over the expanded activation stream",
        )
        design.absorbs_edge(
            ACTIVATION_EDGE_ROLE,
            network=facts.network,
            source_role=REPLAY_ROLE,
            sink_role=COMPUTE_ROLE,
            description="the replayed activation, realized as wiring inside the core",
        )
        design.source(FINN_ROOT, *FINN_SOURCES)

        # -- the one physical choice ---------------------------------------
        pumping = design.choice("compute_pumping", bool, domain=finite((False, True)))

        # -- derived physical parameters ------------------------------------
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
            dependencies={
                "target": facts.target_dsp_block,
                "activation": facts.activation_element_type,
                "weight": facts.weight_element_type,
                "narrow": facts.narrow_weights,
            },
            evaluate=_narrow_weights_supported,
        )
        design.coverage_constraint(
            "pumping_supported",
            dependencies={"pumping": pumping, "simd": facts.simd},
            evaluate=lambda pumping, simd: simd >= 2 if pumping else True,
        )

        # -- the parameter table --------------------------------------------
        # ``MW`` and ``MH`` are here and absent from ``dotp_axi``: this core
        # sizes its internal replay from them (``mvu_vvu_axi.sv:139-141``,
        # ``SF = MW/SIMD`` and ``NF = MH/PE``).  In the decomposed path that
        # same geometry is the replay Kernel's ``LEN`` and ``REP``.  The
        # difference between the two parameter tables *is* the fusion.
        design.parameter("MW", cast("Ref[object]", facts.matrix_width))
        design.parameter("MH", cast("Ref[object]", facts.matrix_height))
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
            "IS_MVU",
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
        design.constant(
            "M_REG_LUT",
            1,
            why=(
                "the core's own default, stated rather than left implicit: the "
                "equivalence fixture measured the core at this value, so leaving "
                "it out of the table would make the parameter set differ from the "
                "one the evidence covers"
            ),
        )

    @classmethod
    def elaborate(cls, kernel: Kernel) -> tuple[PhysicalComponent, ...]:
        """One ``mvu_vvu_axi`` instance -- replay and compute in one component.

        This is where the fusion is visible as a *count*.  The decomposed path
        elaborates the same two covered Regions into two components plus a
        generated top that wires them; this returns one, and there is nothing
        for a composition step to wire.
        """

        return (
            PhysicalComponent(
                "fused_compute",
                MVU_VVU_AXI_MODULE,
                scalar_parameters(dict(kernel.parameters)),
            ),
        )


__all__ = [
    "ACTIVATION_EDGE_ROLE",
    "COMPUTE_ROLE",
    "FINN_ROOT",
    "FINN_SOURCES",
    "FusedMatrixVectorHardwareInputs",
    "MVU_VVU_AXI_MODULE",
    "REPLAY_ROLE",
    "MvuVvuAxiKernel",
    "covers_numeric_types",
    "covers_operand_types",
]
