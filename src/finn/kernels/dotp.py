# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical FinnLib ``dotp_axi`` around one compute core, with caller-owned bounds.

Each compute core is its own kernel: ``PackedDotpKernel`` (FinnLib ``dotp``,
lanes packed into DSP48E1, DSP48E2 or DSP58 slices) and
``Int8Dsp58DotpKernel`` (``dotp_8sx9_dsp58``, the INT8 mode of DSP58). A parent
chooses between them with a Decision over nodes; each refuses what its core
cannot build. ``DotpAxiKernel`` is their shared declaration and places no core.

The ``form`` (``finn.dataflow.gemm``) says how activations meet the PE lanes.
``DENSE`` (``Y[m, n] = sum_k X[m, k] W[k, n]``): each activation beat carries
SIMD fields, broadcast to the PE accumulators. ``DEPTHWISE`` (``Y[m, n] =
sum_k X[m, k, n] W[k, n]``): each activation beat carries SIMD window positions
of PE channels, channel fastest, and lane p accumulates channel p alone; only
the INT8 core reads it. Each weight beat carries PE * SIMD fields, SIMD varying
fastest.
Fields are packed low first; only the complete beat is padded to a byte
boundary. The core pairs activation and weight beats in order. Activation TLAST
closes one nonempty reduction frame and produces one PE-wide result beat;
weights and results have no TLAST. The caller supplies matching weight beats
and terminates every frame.

Placed between streams, dotp takes its ``schedule`` (``finn.dataflow.schedule``):
the extents of ``m``, ``n`` and ``k``, their folds and the beat order. Every
port's beat sequence derives from it and the form (``dotp_sequences``) under
dotp_axi's field conventions: weights ``(n, k)``, results ``(n)``, and
activations ``(k)`` when broadcast or ``(k, n)`` (field ``s * PE + p``)
depthwise. The frame marker closes each reduction. One admission rule replaces
per-port checks: the schedule folds ``n`` by PE, ``k`` by SIMD and nothing
else, and ``k`` is the innermost beats.

``result_dtype`` specifies the signed accumulator encoding, not a proof that an
arbitrary frame fits it. The caller must bound each frame's accumulation to that
encoding (including intermediate sums); overflow is not exact arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import ClassVar

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
from finn.dataflow.gemm import Form, k, n
from finn.dataflow.schedule import SCHEDULE, Index, Refused, Schedule
from finn.dataflow.traversal import BeatSequence
from finn.kernels.streams import (
    MODULE,
    PORT,
    TIEOFFS,
    TIEOFFS_SEMANTICS,
    Stream,
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


@dataclass(frozen=True)
class DotpSequences:
    """What dotp's activation, weight and result ports present of their tensors."""

    activation: BeatSequence
    weights: BeatSequence
    result: BeatSequence


def dotp_sequences(
    schedule: Schedule,
    form: Form,
    *,
    pe: int,
    simd: int,
    activations: tuple[int, ...] = (),
) -> DotpSequences | Rejected:
    """Each port's beat sequence, derived from ``schedule``; refused unless dotp_axi admits it.

    ``activations`` is the activation tensor's shape when dotp reads it through
    a row-major view of the form's shape (a densely realized depthwise
    operation); empty, dotp reads the form's shape itself.
    """

    def refuse(message: str) -> Rejected:
        return reject("dotp-schedule", message)

    indices = schedule.beats
    missing = [index for index in (*form.x, *form.w) if index not in indices]
    if missing:
        return refuse(f"the schedule has no index {missing[0]!r}")
    if (schedule.fold(n), schedule.fold(k)) != (pe, simd):
        return refuse(
            f"the schedule folds n and k by {schedule.fold(n)} x {schedule.fold(k)}, "
            f"dotp's PE x SIMD {pe} x {simd}"
        )
    if any(schedule.fold(index) > 1 for index in indices if index not in (n, k)):
        return refuse("dotp's lanes fold only n (PE) and k (SIMD)")

    def shape(operand: tuple[Index, ...]) -> tuple[int, ...]:
        return tuple(schedule.extent(index) for index in operand)

    x = shape(form.x)
    try:
        marker = schedule.closing((k,))
        return DotpSequences(
            BeatSequence(
                schedule.present(
                    activations or x,
                    form.x,
                    lanes=(k, n) if form is Form.DEPTHWISE else (k,),
                    view=x if activations else None,
                ),
                markers=(marker,),
            ),
            BeatSequence(schedule.present(shape(form.w), form.w, lanes=(n, k))),
            BeatSequence(schedule.present(shape(form.y), form.y, lanes=(n,), reduces=(k,))),
        )
    except Refused as error:
        return refuse(str(error))


DOTP_SEQUENCES = default_semantics(DotpSequences)


class DotpAxiKernel(Kernel):
    """The ``dotp_axi`` space shared by its core kernels; it places no core itself.

    ``target_period_ns`` is the clock period the module must meet. It sets
    SEGMENTLEN, the DSP58 chain length between pipeline registers, by FINN's
    timing model: about 0.741 ns through the first DSP and 0.605 ns through each
    further one, against half the period when compute is pumped. Only the INT8
    core reads SEGMENTLEN. Pumped compute requires a phase-aligned 2x clock. The
    caller owns framing and accumulation bounds.
    """

    id = "finnlib.dotp_axi"
    version = "1"
    # FinnLib's CORE parameter: the name of the compute core module.
    core: ClassVar[str] = ""

    pe: int = Param()
    simd: int = Param()
    target_dsp: DspBlock = Param()
    target_period_ns: float = Param()
    form: Form = Param(default=Form.DENSE)
    compute_pumping: bool = Decision(values=(False, True))

    activation_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    weights_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    result_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    activation_type = integer_scalar(activation_dtype, Integer(min_bits=2))
    weights_type = integer_scalar(weights_dtype, SignedInteger(min_bits=2))
    result_type = integer_scalar(result_dtype, SignedInteger())
    # The streams dotp sits on, when a parent places it between streams: reference
    # inputs, each a Stream node placed beside dotp, and the schedule its ports
    # present, supplied by the parent.
    activation_stream: Stream = Param(required=False)
    weights_stream: Stream = Param(required=False)
    result_stream: Stream = Param(required=False)
    schedule: Schedule = Param(semantics=SCHEDULE, required=False)
    # The activation tensor's shape when dotp reads it through a view of the
    # form's shape (``dotp_sequences``); empty, it reads the form's shape.
    activation_shape: tuple[int, ...] = Param(default=(), semantics=default_semantics(tuple))

    @derived
    def activation_lanes(self) -> int:
        """SIMD fields, times PE when each lane reads its own channel."""
        return self.simd * (self.pe if self.form is Form.DEPTHWISE else 1)

    activation = axi_stream(
        "s_axis_input", activation_lanes, Endpoint.TARGET, activation_type, last=True
    )
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
    def core_supported(self) -> bool | Rejected:
        # The scalars admit the encodings (integer family, signedness, two bits);
        # these are the core's own bounds on them and on the form.
        try:
            ordinary_integer_bounds(self.activation_dtype)
        except DatatypeError:
            return True  # not an integer: refused by the activation scalar
        refused = self._core_refusal()
        return True if refused is None else refused

    def _core_refusal(self) -> Rejected | None:
        return reject("dotp-core", "dotp_axi is placed through one of its core kernels")

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
        core_supported,
        accumulator_width_supported,
        pumping_supported,
    )

    def _narrow_weights(self) -> bool:
        return False

    @derived(semantics=default_semantics(ModuleBuildRequirements))
    def codegen(self) -> ModuleBuildRequirements:
        # Geometry and widths are the support group's; build_requirements requires it.
        activation = self.activation.stream
        weights = self.weights.stream
        result = self.result.stream
        compute_pumping = self.compute_pumping
        settings: dict[str, int | str] = {
            "PE": self.pe,
            "SIMD": self.simd,
            "ACTIVATION_WIDTH": activation.element_bits,
            "WEIGHT_WIDTH": weights.element_bits,
            "ACCU_WIDTH": result.element_bits,
            "SIGNED_ACTIVATIONS": int(activation.dtype.signed()),
            "NARROW_WEIGHTS": int(self._narrow_weights()),
            "PUMPED_COMPUTE": int(compute_pumping),
            "SEGMENTLEN": self.segment_length,
            "VERSION": _DSP_VERSION[self.target_dsp],
            "ACTIVATION_BROADCASTING": int(self.form is Form.DENSE),
            "FORCE_BEHAVIORAL": 0,
            "CORE": f'"{self.core}"',
        }
        parameters = tuple(sorted(settings.items()))
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
        return ModuleBuildRequirements(
            type(self).id, type(self).version, parameters, abi, self._sources()
        )

    def _sources(self) -> tuple[CopiedSource, ...]:
        return ()

    build_requirements = View(codegen, requires=(support,))

    @view(semantics=default_semantics(tuple))
    def interfaces(self) -> tuple[AxiStream, ...]:
        """Accepted (activation, weights, result) ports; framing is the caller's."""
        return (self.activation.stream, self.weights.stream, self.result.stream)

    def _port(
        self, port: AxiStream, stream: Stream, sequence: BeatSequence
    ) -> StreamContract | Rejected:
        """A port over the stream it sits on: its beat sequence, dotp's own encoding."""
        element = stream.tensor.element
        if element.datatype_name != port.dtype.name:
            return reject(
                "dotp-stream-element",
                f"the stream carries {element.datatype_name}, the port {port.dtype.name}",
            )
        transport = port.native(clock="ap_clk", reset="ap_rst_n")
        markers = {}
        if transport.markers:
            markers = {transport.markers[0].signal: sequence.markers[0]}
        return StreamContract(transport, element, sequence.form, sequence.repetition, markers)

    @derived(semantics=DOTP_SEQUENCES)
    def sequences(self) -> DotpSequences | Rejected:
        return dotp_sequences(
            self.schedule,
            self.form,
            pe=self.pe,
            simd=self.simd,
            activations=self.activation_shape,
        )

    @view(semantics=STREAM_CONTRACT)
    def activation_port(self) -> StreamContract | Rejected:
        presented = self.sequences.activation
        return self._port(self.activation.stream, self.activation_stream, presented)

    @view(semantics=STREAM_CONTRACT)
    def weights_port(self) -> StreamContract | Rejected:
        presented = self.sequences.weights
        return self._port(self.weights.stream, self.weights_stream, presented)

    @view(semantics=STREAM_CONTRACT)
    def result_port(self) -> StreamContract | Rejected:
        presented = self.sequences.result
        return self._port(self.result.stream, self.result_stream, presented)

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


_ADD_MULTI = (
    CopiedSource("finnlib", "rtl/arith/add_multi_pkg.sv", provides=("package:add_multi_pkg",)),
    CopiedSource(
        "finnlib",
        "rtl/arith/add_multi.sv",
        provides=("module:add_multi",),
        requires=("package:add_multi_pkg",),
    ),
)


def _dotp_axi(core: str) -> CopiedSource:
    return CopiedSource(
        "finnlib", "rtl/linalg/dotp_axi.sv", provides=("module:dotp_axi",), requires=(core,)
    )


class PackedDotpKernel(DotpAxiKernel):
    """FinnLib ``dotp``: activation and weight lanes packed into DSP48E1, DSP48E2 or DSP58.

    Activations are broadcast (``DENSE`` only). ``narrow_weights`` promises that
    no weight is the most negative value of its type, which packs more lanes
    per DSP; the caller must keep that promise.
    """

    id = "finnlib.dotp_axi.dotp"
    version = "1"
    core = "dotp"
    narrow_weights: bool = Param(default=False)

    def _core_refusal(self) -> Rejected | None:
        if self.form is not Form.DENSE:
            return reject("dotp-form", "the packed core broadcasts activations to every PE lane")
        a_bits, b_bits, _ = dsp_widths(self.target_dsp)
        unsigned = not self.activation_dtype.signed()
        if qonnx_datatype_width(self.activation_dtype) + unsigned > b_bits:
            return reject(
                "dotp-activation-width", "activation values must fit the signed DSP B input"
            )
        if qonnx_datatype_width(self.weights_dtype) >= a_bits:
            return reject("dotp-weight-width", "signed weights need room for a DSP sign guard")
        return None

    def _narrow_weights(self) -> bool:
        return self.narrow_weights

    def _sources(self) -> tuple[CopiedSource, ...]:
        return (
            *_ADD_MULTI,
            CopiedSource(
                "finnlib",
                "rtl/linalg/dotp.sv",
                provides=("module:dotp",),
                requires=("package:add_multi_pkg", "module:add_multi"),
            ),
            _dotp_axi("module:dotp"),
        )


class Int8Dsp58DotpKernel(DotpAxiKernel):
    """FinnLib ``dotp_8sx9_dsp58``: three 9x8 signed products per DSP58 in INT8 mode.

    It takes signed weights of at most 8 bits and activations that fit 9 signed
    bits, broadcast (``DENSE``) or one channel per lane (``DEPTHWISE``).
    """

    id = "finnlib.dotp_axi.dotp_8sx9_dsp58"
    version = "1"
    core = "dotp_8sx9_dsp58"

    def _core_refusal(self) -> Rejected | None:
        if self.target_dsp is not DspBlock.DSP58:
            return reject("dotp-target", "the INT8 core is a DSP58 mode")
        unsigned = not self.activation_dtype.signed()
        if qonnx_datatype_width(self.activation_dtype) + unsigned > 9:
            return reject(
                "dotp-activation-width", "activation values must fit the 9-bit signed INT8 lanes"
            )
        if qonnx_datatype_width(self.weights_dtype) > 8:
            return reject("dotp-weight-width", "weights must fit the 8-bit INT8 lanes")
        return None

    def _sources(self) -> tuple[CopiedSource, ...]:
        return (
            CopiedSource(
                "finnlib", "rtl/linalg/dotp_8sx9_dsp58.sv", provides=("module:dotp_8sx9_dsp58",)
            ),
            _dotp_axi("module:dotp_8sx9_dsp58"),
        )


__all__ = [
    "DOTP_SEQUENCES",
    "DotpAxiKernel",
    "DotpSequences",
    "Int8Dsp58DotpKernel",
    "PackedDotpKernel",
    "dotp_sequences",
]
