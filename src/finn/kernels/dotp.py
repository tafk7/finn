# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical FinnLib ``dotp_axi`` with caller-owned reduction bounds.

Each activation beat contains SIMD fields, broadcast to PE accumulators. Each
weight beat contains PE * SIMD fields, SIMD varying fastest. Fields are packed
low first; only the complete beat is padded to a byte boundary. The core pairs
activation and weight beats in order. Activation TLAST closes one nonempty
reduction frame and produces one PE-wide result beat; weights and results have
no TLAST. The caller supplies matching weight beats and terminates every frame.

``result_dtype`` specifies the signed accumulator encoding, not a proof that an
arbitrary frame fits it. The caller must bound each frame's accumulation to that
encoding (including intermediate sums); overflow is not exact arithmetic. All
declared activation and signed weight values are admitted, including the most
negative weight. NARROW_WEIGHTS is always zero.
"""

from finn.kernels.artifacts.abi import (
    Clock,
    ClockAlignment,
    Data,
    Derived as DerivedClock,
    Direction,
    Endpoint,
    Free,
    Reset,
    Signal,
)
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)
from finn.kernels.target import DspBlock, dsp_widths
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.space import (
    ConstraintGroup,
    Decision,
    Input,
    View,
    Readiness,
    constraint,
    derived,
    reject,
)

_DSP_VERSION = {DspBlock.DSP48E1: 1, DspBlock.DSP48E2: 2, DspBlock.DSP58: 3}


class DotpAxiKernel(Kernel):
    """Physical dot-product space and its assessed module-building view.

    SEGMENTLEN is literal: zero selects the RTL's full chain, and DSP48
    implementations ignore positive values. Pumped compute requires a
    phase-aligned 2x clock. The caller owns framing and accumulation bounds.
    """

    id = "exact_integer_dot_product_axi"
    version = "2"

    pe = Input(int)
    simd = Input(int)
    activation_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    result_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    target_dsp = Input(DspBlock)
    segment_length = Input(int)
    compute_pumping = Decision(bool, values=(False, True))

    @derived(AxiStream, dtype=activation_dtype, simd=simd)
    def activation(*, dtype: QONNXDataType, simd: int) -> object:
        try:
            return AxiStream("s_axis_input", dtype, simd, endpoint=Endpoint.TARGET, last=True)
        except ValueError as error:
            return reject("dotp-interface", str(error))

    @derived(AxiStream, dtype=weights_dtype, pe=pe, simd=simd)
    def weights(*, dtype: QONNXDataType, pe: int, simd: int) -> object:
        try:
            return AxiStream("s_axis_weights", dtype, pe * simd, endpoint=Endpoint.TARGET)
        except ValueError as error:
            return reject("dotp-interface", str(error))

    @derived(AxiStream, dtype=result_dtype, pe=pe)
    def result(*, dtype: QONNXDataType, pe: int) -> object:
        try:
            return AxiStream("m_axis_output", dtype, pe, endpoint=Endpoint.INITIATOR)
        except ValueError as error:
            return reject("dotp-interface", str(error))

    @constraint(target=target_dsp)
    def target_supported(*, target: DspBlock) -> object:
        if target not in _DSP_VERSION:
            return reject("dotp-target", "the RTL has no implementation for this DSP target")
        return True

    @constraint(pe=pe, simd=simd)
    def geometry_supported(*, pe: int, simd: int) -> object:
        if pe < 1 or simd < 1:
            return reject("dotp-geometry", "PE and SIMD must be positive integers")
        return True

    @constraint(target=target_dsp, activation=activation_dtype, weight=weights_dtype)
    def input_types_supported(
        *, target: DspBlock, activation: QONNXDataType, weight: QONNXDataType
    ) -> object:
        try:
            ordinary_integer_bounds(activation)
        except DatatypeError as error:
            return reject("dotp-activation-type", str(error))
        a_bits, b_bits, _ = dsp_widths(target)
        activation_bits = qonnx_datatype_width(activation)
        weight_bits = qonnx_datatype_width(weight)
        unsigned = not activation.signed()
        if activation_bits < 2 or activation_bits + unsigned > b_bits:
            return reject(
                "dotp-activation-width", "activation values must fit the signed DSP B input"
            )
        # dotp_axi selects its signed 9x8 INT8 path for this boundary case.
        if target is DspBlock.DSP58 and unsigned and activation_bits == 9 and weight_bits <= 8:
            return reject("dotp-activation-width", "the native INT8 path needs a ninth sign bit")
        if not weight.name.startswith("INT") or not 2 <= weight_bits < a_bits:
            return reject(
                "dotp-weight-type",
                "full-range signed weights need at least two bits and room for a DSP sign guard",
            )
        return True

    @constraint(result=result_dtype, target=target_dsp)
    def accumulator_width_supported(*, result: QONNXDataType, target: DspBlock) -> object:
        if not result.name.startswith("INT"):
            return reject(
                "dotp-result-type", "the accumulator requires an ordinary signed INT dtype"
            )
        bits, maximum = qonnx_datatype_width(result), dsp_widths(target)[2]
        if not 1 <= bits <= maximum:
            return reject(
                "dotp-accumulator-width",
                "the required result width exceeds the target accumulator capacity",
                values={"actual_bits": bits, "maximum_bits": maximum},
            )
        return True

    @constraint(pumping=compute_pumping, simd=simd)
    def pumping_supported(*, pumping: bool, simd: int) -> object:
        if pumping and simd < 2:
            return reject("dotp-pumping", "pumping must be boolean and requires SIMD >= 2")
        return True

    @constraint(target=target_dsp, segment=segment_length, simd=simd, pumping=compute_pumping)
    def segment_length_supported(
        *, target: DspBlock, segment: int, simd: int, pumping: bool
    ) -> object:
        products_per_stage = 6 if pumping else 3
        chain_length = (simd + products_per_stage - 1) // products_per_stage
        if segment < 0 or (target is DspBlock.DSP58 and segment > chain_length):
            return reject("dotp-segment", "SEGMENTLEN must be zero or fit the DSP58 compute chain")
        return True

    support = ConstraintGroup(
        target_supported,
        geometry_supported,
        input_types_supported,
        accumulator_width_supported,
        pumping_supported,
        segment_length_supported,
    )

    @derived(
        ModuleBuildRequirements,
        pe=pe,
        simd=simd,
        activation=activation,
        weights=weights,
        result=result,
        target_dsp=target_dsp,
        segment_length=segment_length,
        compute_pumping=compute_pumping,
    )
    def codegen(
        *,
        pe: int,
        simd: int,
        activation: AxiStream,
        weights: AxiStream,
        result: AxiStream,
        target_dsp: DspBlock,
        segment_length: int,
        compute_pumping: bool,
    ) -> ModuleBuildRequirements:
        parameters = tuple(
            sorted(
                {
                    "PE": pe,
                    "SIMD": simd,
                    "ACTIVATION_WIDTH": activation.element_bits,
                    "WEIGHT_WIDTH": weights.element_bits,
                    "ACCU_WIDTH": result.element_bits,
                    "SIGNED_ACTIVATIONS": int(activation.dtype.signed()),
                    "NARROW_WEIGHTS": 0,
                    "PUMPED_COMPUTE": int(compute_pumping),
                    "SEGMENTLEN": segment_length,
                    "VERSION": _DSP_VERSION[target_dsp],
                    "ACTIVATION_BROADCASTING": 1,
                    "FORCE_BEHAVIORAL": 0,
                }.items()
            )
        )
        abi = ModuleABIRequirements(
            FixedModuleName("dotp_axi"),
            (
                Signal("ap_clk", Direction.IN, 1, Clock(Free())),
                Signal(
                    "ap_clk2x",
                    Direction.IN,
                    1,
                    Clock(DerivedClock("ap_clk", 2)) if compute_pumping else Data(),
                ),
                Signal(
                    "ap_rst_n",
                    Direction.IN,
                    1,
                    Reset(
                        active_low=True,
                        synchronous=True,
                        synchronous_to=("ap_clk", "ap_clk2x") if compute_pumping else ("ap_clk",),
                    ),
                ),
                *(
                    stream.bus(clock="ap_clk", reset="ap_rst_n")
                    for stream in (activation, weights, result)
                ),
            ),
            tuple((name, str(value)) for name, value in parameters),
            (ClockAlignment("ap_clk", "ap_clk2x"),) if compute_pumping else (),
        )
        sources = tuple(
            CopiedSource(root, path, provides=(symbol,), requires=requires)
            for root, path, symbol, requires in (
                ("finnlib", "rtl/arith/add_multi_pkg.sv", "package:add_multi_pkg", ()),
                (
                    "finnlib",
                    "rtl/arith/add_multi.sv",
                    "module:add_multi",
                    ("package:add_multi_pkg",),
                ),
                ("finnlib", "rtl/linalg/dotp_8sx9_dsp58.sv", "module:dotp_8sx9_dsp58", ()),
                (
                    "finnlib",
                    "rtl/linalg/dotp.sv",
                    "module:dotp",
                    ("package:add_multi_pkg", "module:add_multi"),
                ),
                (
                    "kernels",
                    "dotp_axi.sv",
                    "module:dotp_axi",
                    ("module:dotp", "module:dotp_8sx9_dsp58"),
                ),
            )
        )
        return ModuleBuildRequirements(
            DotpAxiKernel.id, DotpAxiKernel.version, parameters, abi, sources
        )

    physical_ready = Readiness()
    physical = View(codegen, readiness=physical_ready, constraints=support)


__all__ = ["DotpAxiKernel"]
