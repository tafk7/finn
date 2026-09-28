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

from __future__ import annotations

from math import ceil, floor, prod

from finn.kernels.artifacts.abi import (
    Clock,
    ClockAlignment,
    Data,
    Derived as DerivedRate,
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
from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.dataflow.datatypes import (
    DatatypeError,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.physical.axi_stream import AxiStream, axi_stream
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.physical.forms import (
    Loop,
    Step,
    Traversal,
    beat_walk,
    canonical_loops,
    split_walk,
    walk_axis,
)
from finn.kernels.streams import (
    MODULE,
    PORT,
    TIEOFFS,
    TIEOFFS_SEMANTICS,
    Stream,
    StreamSpec,
    Tieoffs,
)
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
# FINN's DSP58 chain timing model (rtl/matrixvectoractivation_rtl.py).
_FIRST_DSP_NS, _NEXT_DSP_NS = 0.741, 0.605


class DotpAxiKernel(Kernel):
    """Physical dot-product space and its assessed module-building view.

    ``target_period_ns`` is the clock period the module must meet. It sets
    SEGMENTLEN, the DSP58 chain length between pipeline registers, by FINN's
    timing model: about 0.741 ns through the first DSP and 0.605 ns through each
    further one, against half the period when compute is pumped. DSP48
    implementations ignore SEGMENTLEN. Pumped compute requires a phase-aligned
    2x clock. The caller owns framing and accumulation bounds.
    """

    id = "exact_integer_dot_product_axi"
    version = "2"

    pe: int = Param()
    simd: int = Param()
    target_dsp: DspBlock = Param()
    target_period_ns: float = Param()
    compute_pumping: bool = Decision(values=(False, True))

    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    result_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    activation_type = integer_scalar(activation_dtype, Integer(min_bits=2))
    weights_type = integer_scalar(weights_dtype, SignedInteger(min_bits=2))
    result_type = integer_scalar(result_dtype, SignedInteger())
    # The streams dotp sits on, when a parent places it between streams: reference
    # inputs, each a Stream node placed beside dotp.
    activation_stream: Stream = Param(required=False)
    weights_stream: Stream = Param(required=False)
    result_stream: Stream = Param(required=False)
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
        # The scalars admit the encodings (integer family, signedness, two bits);
        # these are the DSP's own bounds on them.
        target = self.target_dsp
        activation = self.activation_dtype
        weight = self.weights_dtype
        try:
            ordinary_integer_bounds(activation)
        except DatatypeError:
            return True  # not an integer: refused by the activation scalar
        a_bits, b_bits, _ = dsp_widths(target)
        activation_bits = qonnx_datatype_width(activation)
        weight_bits = qonnx_datatype_width(weight)
        unsigned = not activation.signed()
        if activation_bits + unsigned > b_bits:
            return reject(
                "dotp-activation-width", "activation values must fit the signed DSP B input"
            )
        # dotp_axi selects its signed 9x8 INT8 path for this boundary case.
        if target is DspBlock.DSP58 and unsigned and activation_bits == 9 and weight_bits <= 8:
            return reject("dotp-activation-width", "the native INT8 path needs a ninth sign bit")
        if weight_bits >= a_bits:
            return reject("dotp-weight-width", "signed weights need room for a DSP sign guard")
        return True

    @constraint
    def accumulator_width_supported(self) -> bool | Rejected:
        result = self.result_dtype
        target = self.target_dsp
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

    @derived(semantics=default_semantics(int))
    def segment_length(self) -> int | Rejected:
        """The longest DSP58 chain segment that meets the target period, at most the chain."""
        pumping = self.compute_pumping
        period = self.target_period_ns / 2 if pumping else self.target_period_ns
        if not period > _FIRST_DSP_NS:
            return reject(
                "dotp-clock-period",
                f"a {self.target_period_ns} ns target leaves no time for one DSP stage",
            )
        meets = floor((period - _FIRST_DSP_NS) / _NEXT_DSP_NS + 1)
        chain = ceil(self.simd / (6 if pumping else 3))
        return min(meets, chain)

    support = ConstraintGroup(
        target_supported,
        geometry_supported,
        stream_widths_supported,
        input_types_supported,
        accumulator_width_supported,
        pumping_supported,
    )

    @derived(semantics=default_semantics(ModuleBuildRequirements))
    def codegen(self) -> ModuleBuildRequirements:
        # Geometry and widths are the support group's; build_requirements requires it.
        pe = self.pe
        simd = self.simd
        activation = self.activation.stream
        weights = self.weights.stream
        result = self.result.stream
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
                    Clock(DerivedRate("ap_clk", 2)) if compute_pumping else Data(),
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
                    "finnlib",
                    "rtl/linalg/dotp_axi.sv",
                    "module:dotp_axi",
                    ("module:dotp", "module:dotp_8sx9_dsp58"),
                ),
            )
        )
        return ModuleBuildRequirements(
            DotpAxiKernel.id, DotpAxiKernel.version, parameters, abi, sources
        )

    build_requirements = View(codegen, requires=(support,))

    @view(semantics=default_semantics(tuple))
    def interfaces(self) -> tuple[AxiStream, ...]:
        """Accepted (activation, weights, result) ports; framing is the caller's."""
        return (self.activation.stream, self.weights.stream, self.result.stream)

    def _port(self, port: AxiStream, stream: Stream) -> StreamContract | Rejected:
        """A port over the stream it sits on: the stream's order, dotp's own encoding."""
        spec = stream.spec
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

    # What dotp reads. Activation beats carry SIMD consecutive columns of a
    # (rows, K) operand; weight beats carry PE rows of those same K columns, SIMD
    # fastest; result beats carry PE consecutive columns of a (rows, N) result.
    # Beat by beat the weight columns are the activation columns; each frame
    # (the activation marker period) stays within one activation row and one
    # group of weight rows, and produces one result beat whose row is the
    # activation row and whose columns are those weight rows.

    @view(semantics=STREAM_CONTRACT)
    def activation_port(self) -> StreamContract | Rejected:
        spec, simd = self.activation_stream.spec, self.simd
        refused = _fields(spec.form, (Loop(simd, 1),), "activation", f"SIMD={simd}")
        if refused is not None:
            return refused
        if len(spec.markers) == 1 and _frames(spec, spec.form) is None:
            return reject(
                "dotp-stream-form", "a reduction frame must stay within one activation row"
            )
        return self._port(self.activation.stream, self.activation_stream)

    @view(semantics=STREAM_CONTRACT)
    def weights_port(self) -> StreamContract | Rejected:
        weights, activation = self.weights_stream.spec, self.activation_stream.spec
        pe, simd = self.pe, self.simd
        width = activation.form.shape[-1]
        if len(weights.form.shape) != 2 or weights.form.shape[1] != width:
            return reject(
                "dotp-stream-form",
                f"weights must be a matrix over the activation's {width} columns",
            )
        required = (Loop(pe, width), Loop(simd, 1))
        refused = _fields(weights.form, required, "weights", f"PE={pe} rows of SIMD={simd}")
        if refused is not None:
            return refused
        mine, theirs = beat_walk(weights.form, width), beat_walk(activation.form, width)
        if mine is None or theirs is None or walk_axis(mine, 2) != walk_axis(theirs, 2):
            return reject("dotp-stream-form", "weight columns do not follow the activation columns")
        if len(activation.markers) == 1 and _frames(activation, weights.form) is None:
            return reject(
                "dotp-stream-form", "a reduction frame must read one group of weight rows"
            )
        return self._port(self.weights.stream, self.weights_stream)

    @view(semantics=STREAM_CONTRACT)
    def result_port(self) -> StreamContract | Rejected:
        result = self.result_stream.spec
        weights, activation = self.weights_stream.spec, self.activation_stream.spec
        pe = self.pe
        refused = _fields(result.form, (Loop(pe, 1),), "result", f"PE={pe}")
        if refused is not None:
            return refused
        rows = weights.form.shape[0]
        if result.form.shape[-1] != rows:
            return reject("dotp-stream-form", f"results must have the weights' {rows} columns")
        mine = beat_walk(result.form, rows)
        if len(activation.markers) == 1:
            # Frames the other ports refuse are theirs to report.
            activation_frames = _frames(activation, activation.form)
            weight_frames = _frames(activation, weights.form)
            if (
                activation_frames is not None
                and weight_frames is not None
                and (
                    mine is None
                    or walk_axis(activation_frames, 1) != walk_axis(mine, 1)
                    or walk_axis(weight_frames, 1) != walk_axis(mine, 2)
                )
            ):
                return reject(
                    "dotp-stream-form",
                    "each result beat must hold its frame's activation row and weight rows",
                )
        return self._port(self.result.stream, self.result_stream)

    @view(semantics=TIEOFFS_SEMANTICS)
    def tieoffs(self) -> Tieoffs:
        # Unpumped, the RTL ignores its 2x clock input: hold it low.
        return Tieoffs() if self.compute_pumping else Tieoffs((("ap_clk2x", 0),))

    exports = {
        MODULE: build_requirements,
        TIEOFFS: tieoffs,
        PORT: {
            activation_stream: activation_port,
            weights_stream: weights_port,
            result_stream: result_port,
        },
    }


def _frames(activation: StreamSpec, form: Traversal) -> tuple[Step, ...] | None:
    """The walk of ``form``'s frames, each frame one row: the activation marker period."""
    walk = beat_walk(form, form.shape[-1])
    framed = None if walk is None else split_walk(walk, activation.markers[0].period)
    if framed is None or any(row for _, row, _ in framed[1]):
        return None
    return framed[0]


def _fields(form: Traversal, required: tuple[Loop, ...], role: str, reads: str) -> Rejected | None:
    """Refuse a stream whose beats do not carry the fields dotp reads."""
    lanes = prod(loop.extent for loop in required)
    if form.lanes != lanes:
        return reject(
            "dotp-stream-lanes", f"the {role} stream carries {form.lanes} lanes; dotp reads {reads}"
        )
    if form.lane_loops != canonical_loops(required):
        return reject("dotp-stream-form", f"the {role} stream's lanes are not {reads}")
    return None


__all__ = ["DotpAxiKernel"]
