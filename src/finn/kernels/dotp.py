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
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.kernels.datatypes.values import (
    DatatypeError,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.physical.axi_stream import AxiStream, axi_stream
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.streams import Port, StreamSpec
from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    View,
    constraint,
    derived,
    default_semantics,
    reject,
    view,
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

    pe = Param(int)
    simd = Param(int)
    target_dsp = Param(DspBlock)
    segment_length = Param(int)
    compute_pumping = Decision(bool, values=(False, True))

    activation_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    result_dtype = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    activation_type = integer_scalar(activation_dtype, Integer(min_bits=2))
    weights_type = integer_scalar(weights_dtype, SignedInteger(min_bits=2))
    result_type = integer_scalar(result_dtype, SignedInteger())
    activation_stream = Port(Endpoint.TARGET)
    weights_stream = Port(Endpoint.TARGET)
    result_stream = Port(Endpoint.INITIATOR)
    activation = axi_stream("s_axis_input", simd, Endpoint.TARGET, activation_type, last=True)
    weights = axi_stream("s_axis_weights", pe * simd, Endpoint.TARGET, weights_type)
    result = axi_stream("m_axis_output", pe, Endpoint.INITIATOR, result_type)

    @constraint
    def target_supported(self) -> bool | Rejected:
        target = self.target_dsp
        if target not in _DSP_VERSION:
            return reject("dotp-target", "the RTL has no implementation for this DSP target")
        return True

    @constraint
    def geometry_supported(self) -> bool | Rejected:
        pe = self.pe
        simd = self.simd
        if not 1 <= pe <= 0xFFFFFFFF or not 1 <= simd <= 0xFFFFFFFF:
            return reject("dotp-geometry", "PE and SIMD must be positive native unsigned integers")
        return True

    @constraint
    def stream_widths_supported(self) -> bool | Rejected:
        # The native payload and byte-aligned carrier localparams are uint32.
        widths = (self.activation.carrier_bits, self.weights.carrier_bits, self.result.carrier_bits)
        if any(width > 0xFFFFFFFF for width in widths):
            return reject("dotp-stream-width", "packed stream widths must fit native unsigned int")
        return True

    @constraint
    def input_types_supported(self) -> bool | Rejected:
        target = self.target_dsp
        activation = self.activation_dtype
        weight = self.weights_dtype
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

    @constraint
    def accumulator_width_supported(self) -> bool | Rejected:
        result = self.result_dtype
        target = self.target_dsp
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

    @constraint
    def pumping_supported(self) -> bool | Rejected:
        pumping = self.compute_pumping
        simd = self.simd
        if pumping and simd < 2:
            return reject("dotp-pumping", "pumping must be boolean and requires SIMD >= 2")
        return True

    @constraint
    def segment_length_supported(self) -> bool | Rejected:
        target = self.target_dsp
        segment = self.segment_length
        simd = self.simd
        pumping = self.compute_pumping
        products_per_stage = 6 if pumping else 3
        chain_length = (simd + products_per_stage - 1) // products_per_stage
        if segment < 0 or (target is DspBlock.DSP58 and segment > chain_length):
            return reject("dotp-segment", "SEGMENTLEN must be zero or fit the DSP58 compute chain")
        return True

    support = ConstraintGroup(
        target_supported,
        geometry_supported,
        stream_widths_supported,
        input_types_supported,
        accumulator_width_supported,
        pumping_supported,
        segment_length_supported,
    )

    @derived(semantics=default_semantics(ModuleBuildRequirements))
    def codegen(self) -> ModuleBuildRequirements | Rejected:
        pe = self.pe
        simd = self.simd
        if not 1 <= pe <= 0xFFFFFFFF or not 1 <= simd <= 0xFFFFFFFF:
            return reject("dotp-geometry", "PE and SIMD must be positive native unsigned integers")
        activation = self.activation.stream()
        weights = self.weights.stream()
        result = self.result.stream()
        if any(stream.carrier_bits > 0xFFFFFFFF for stream in (activation, weights, result)):
            return reject("dotp-stream-width", "packed stream widths must fit native unsigned int")
        target_dsp = self.target_dsp
        segment_length = self.segment_length
        compute_pumping = self.compute_pumping
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
                ("finnlib", "rtl/add_multi_pkg.sv", "package:add_multi_pkg", ()),
                (
                    "finnlib",
                    "rtl/add_multi.sv",
                    "module:add_multi",
                    ("package:add_multi_pkg",),
                ),
                ("finnlib", "rtl/dotp_8sx9_dsp58.sv", "module:dotp_8sx9_dsp58", ()),
                (
                    "finnlib",
                    "rtl/dotp.sv",
                    "module:dotp",
                    ("package:add_multi_pkg", "module:add_multi"),
                ),
                (
                    "finnlib",
                    "rtl/dotp_axi.sv",
                    "module:dotp_axi",
                    ("module:dotp", "module:dotp_8sx9_dsp58"),
                ),
            )
        )
        return ModuleBuildRequirements(
            DotpAxiKernel.id, DotpAxiKernel.version, parameters, abi, sources
        )

    build_requirements = View(codegen, constraints=(support,))

    @view(semantics=default_semantics(tuple))
    def interfaces(self) -> tuple[AxiStream, ...]:
        """Accepted (activation, weights, result) ports; framing is the caller's."""
        return (self.activation.stream(), self.weights.stream(), self.result.stream())

    def _port(self, index: int, spec: StreamSpec) -> StreamContract | Rejected:
        """A port over its bound stream: the stream's order, dotp's own encoding."""
        port = self.interfaces()[index]
        if spec.element.datatype_name != port.dtype.name:
            return reject(
                "dotp-stream-element",
                f"the stream carries {spec.element.datatype_name}, the port {port.dtype.name}",
            )
        transport = port.native(clock="ap_clk", reset="ap_rst_n")
        markers = {}
        if transport.markers:
            if len(spec.markers) != 1:
                return reject("dotp-framing", "the activation stream needs one frame marker rule")
            markers = {transport.markers[0].signal: spec.markers[0]}
        return StreamContract(transport, spec.element, spec.form, spec.repetition, markers)

    @view(semantics=STREAM_CONTRACT)
    def activation_port(self) -> StreamContract | Rejected:
        return self._port(0, self.activation_stream)

    @view(semantics=STREAM_CONTRACT)
    def weights_port(self) -> StreamContract | Rejected:
        return self._port(1, self.weights_stream)

    @view(semantics=STREAM_CONTRACT)
    def result_port(self) -> StreamContract | Rejected:
        return self._port(2, self.result_stream)


__all__ = ["DotpAxiKernel"]
