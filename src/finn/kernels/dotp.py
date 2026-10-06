# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical FinnLib ``dotp_axi`` around one compute core, on three channels.

Each compute core is its own kernel: ``PackedDotpKernel`` (FinnLib ``dotp``,
lanes packed into DSP48E1, DSP48E2 or DSP58 slices) and
``Int8Dsp58DotpKernel`` (``dotp_8sx9_dsp58``, the INT8 mode of DSP58). A parent
chooses between them with a Decision over kernels; each refuses what its core
cannot build. ``DotpAxiKernel`` is their shared declaration.

dotp computes ``Y[m, n] = sum_k X * W`` in the ``form`` (``finn.dataflow.gemm``)
its parent gives. ``DENSE``: each activation beat carries SIMD lanes,
broadcast to the PE accumulators. ``DEPTHWISE``: each activation beat carries
SIMD window positions of PE channels, channel fastest, and lane p accumulates
channel p alone; only the INT8 core reads it. Each weight beat carries PE *
SIMD lanes, SIMD varying fastest; each result beat PE lanes. Lanes are
packed low first; only the complete beat is padded to a byte boundary.
Activation TLAST closes each reduction and produces one result beat.

Its three ports (``x``, ``w``, ``y``) sit on the channels its parent supplies
(``x_channel``, ``w_channel``, ``y_channel``). The extents are bound from the
tensors the ports read, which must agree (``kernel-extents``): x reads
``(m, k)`` (``(m, k, n)`` depthwise), w ``(k, n)`` (weights stored ``(k, n)``)
and y ``(m, n)``. PE and SIMD are dotp's own Decisions, the folding factors of ``n`` and ``k``,
and its ``schedule`` walks ``m``, then ``n``, then ``k`` innermost; every
port's beat sequence derives from it. ``reshape_activations`` reads (M, K, N)
activations as (M, K * N): a densely realized depthwise operation.

The results' range, ``result_range``, is its parent's statement (MatMul's);
each core derives from it its accumulator (``result_dtype``, ``ACCU_WIDTH``),
the encoding the ``y`` port presents over that range: both current cores
compute exactly at the range's own encoding, ``INT<b>`` or ``UINT<b>`` (FinnLib's
``p`` is the sum modulo ``2**ACCU_WIDTH``, read in either signedness). The range
is not a proof that an arbitrary frame fits it: the parent must bound each
frame's accumulation to it, every partial sum included.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, floor
from typing import ClassVar

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    divisors_of,
    reject,
    requires,
)
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)
from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.schedule import Index, Schedule
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import CLOCK2X, Clocking, Kernel, extent_of
from finn.kernels.channels import Channel
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import DspBlock, Platform, dsp_widths
from finn.kernels.values.domains import Integer, range_dtype
from finn.kernels.values.semantics import QONNX_DATATYPE_VALUE_SEMANTICS

_DSP_VERSION = {DspBlock.DSP48E1: 1, DspBlock.DSP48E2: 2, DspBlock.DSP58: 3}
# FINN's DSP58 chain timing model (rtl/matrixvectoractivation_rtl.py).
_FIRST_DSP_NS, _NEXT_DSP_NS = 0.741, 0.605


class DotpAxiKernel(Kernel):
    """The ``dotp_axi`` space shared by its core kernels; it places no core itself.

    The ``platform``'s ``period_ns`` is the clock period the module must meet. It sets
    SEGMENTLEN, the DSP58 chain length between pipeline registers, by FINN's
    timing model: about 0.741 ns through the first DSP and 0.605 ns through each
    further one, against half the period when compute is pumped. Only the INT8
    core reads SEGMENTLEN. Pumped compute requires a phase-aligned 2x clock: the
    ``platform``'s ``clk2x``. The DSP block is the ``platform``'s (``dsp``): a
    platform that states none is refused (``dotp-dsp``).
    """

    id = "finnlib.dotp_axi"
    version = 1
    rtl_module = "dotp_axi"
    # FinnLib's CORE parameter: the name of the compute core module.
    core: ClassVar[str] = ""

    form: Form = Param(default=Form.DENSE)
    reshape_activations: bool = Param(default=False)
    # The range of its results: its parent's statement (MatMul binds its result range),
    # so that it is known before the results channel exists.
    result_range: tuple[int, int] = Param()
    platform: Platform = Param()
    # The channels dotp sits on: reference inputs, each a Channel placed beside it.
    x_channel: Channel = Param(required=False)
    w_channel: Channel = Param(required=False)
    y_channel: Channel = Param(required=False)

    # Each extent bound from the tensors the ports read (``Kernel.extents``).
    rows = extent_of(m)  # M: the results' rows
    outputs = extent_of(n)  # N: the results' columns
    reduction = extent_of(k)  # K: the weights' rows, stored (k, n)

    pe: int = Decision(domain=divisors_of(outputs))
    simd: int = Decision(domain=divisors_of(reduction))

    @derived
    def pumpable(self) -> bool:
        """Whether pumping has SIMD lanes to split over the doubled clock: SIMD >= 2."""
        return self.simd >= 2

    # Pumping needs the platform's doubled clock and lanes to split (SIMD before it in
    # rank, so a configuration with SIMD 1 does not offer it).
    compute_pumping: bool = Decision(
        values=(False, True),
        requires=(
            requires(
                platform.clk2x, "clk2x-absent: the platform has no doubled clock", cases=(True,)
            ),
            requires(pumpable, "dotp-pumping: pumping requires SIMD >= 2", cases=(True,)),
        ),
    )

    @derived
    def schedule(self) -> Schedule | Rejected:
        """``n`` split by PE and ``k`` by SIMD; ``m``, then ``n``, then the reduction."""
        return self.bound_schedule(order=(m, n, k), factors={n: self.pe, k: self.simd})

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_dtype(self) -> QONNXDataType | Rejected:
        """The accumulator: the encoding the core computes and presents its results in,
        its own rule over the range it must hold (``_accumulator``)."""
        least, greatest = self.result_range
        return self._accumulator(least, greatest)

    def _accumulator(self, least: int, greatest: int) -> QONNXDataType | Rejected:
        return reject("dotp-core", "dotp_axi is placed through one of its core kernels")

    @derived
    def frame_cycles(self) -> int:
        """Its schedule's beats, one a cycle at best."""
        return self.schedule.beat_count

    @derived
    def x_index(self) -> tuple[Index, ...]:
        return self.form.x

    @derived
    def x_lanes(self) -> tuple[Index, ...]:
        """dotp_axi's activation lanes: SIMD alone, or ``s * PE + p`` depthwise."""
        return (k, n) if self.form is Form.DEPTHWISE else (k,)

    x = AxiStreamPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        channel=x_channel,
        admits=Integer(min_bits=2),
        schedule=schedule,
        index=x_index,
        lanes=x_lanes,
        closes=(k,),
        reshaped=reshape_activations,
    )
    w = AxiStreamPort(
        name="s_axis_weights",
        endpoint=Endpoint.TARGET,
        channel=w_channel,
        admits=Integer(min_bits=2, signed=True),
        schedule=schedule,
        index=(k, n),
        lanes=(n, k),
    )
    y = AxiStreamPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        channel=y_channel,
        admits=Integer(),
        schedule=schedule,
        index=(m, n),
        lanes=(n,),
        reduces=(k,),
        dtype=result_dtype,
        value_range=result_range,
    )

    @derived
    def dsp(self) -> DspBlock | Rejected:
        """The platform's DSP block, the slice the core is built in."""
        dsp = self.platform.dsp
        if dsp is None:
            return reject("dotp-dsp", "the platform states no DSP block")
        return dsp

    @constraint
    def core_supported(self) -> bool | Rejected:
        # The ports admit the encodings (integer family, signedness, two bits);
        # these are the core's own bounds on them and on the form.
        try:
            ordinary_integer_bounds(self.x.element.dtype)
        except DatatypeError:
            return True  # not an integer: refused by the activation port
        refused = self._core_refusal()
        return True if refused is None else refused

    def _core_refusal(self) -> Rejected | None:
        return reject("dotp-core", "dotp_axi is placed through one of its core kernels")

    @constraint
    def accumulator_width_supported(self) -> bool | Rejected:
        bits, maximum = self.y.element.bits, dsp_widths(self.dsp)[2]
        if not 1 <= bits <= maximum:
            return reject(
                "dotp-accumulator-width",
                "the required result width exceeds the target accumulator capacity",
                values={"actual_bits": bits, "maximum_bits": maximum},
            )
        return True

    @constraint
    def stream_widths_supported(self) -> bool | Rejected:
        # The native payload and byte-aligned carrier localparams are uint32.
        ports = (self.x.axis, self.w.axis, self.y.axis)
        if any(port.carrier_bits > 0xFFFFFFFF for port in ports):
            return reject("dotp-stream-width", "packed stream widths must fit native unsigned int")
        return True

    @derived
    def segment_length(self) -> int | Rejected:
        """The longest DSP58 chain segment that meets the target period, at most the chain."""
        pumping, target = self.compute_pumping, self.platform.period_ns
        period = target / 2 if pumping else target
        if not period > _FIRST_DSP_NS:
            return reject(
                "dotp-clock-period", f"a {target} ns target leaves no time for one DSP stage"
            )
        meets = floor((period - _FIRST_DSP_NS) / _NEXT_DSP_NS + 1)
        chain = ceil(self.simd / (6 if pumping else 3))
        return min(meets, chain)

    admission = ConstraintGroup(
        core_supported,
        accumulator_width_supported,
        stream_widths_supported,
    )

    @derived
    def clocking(self) -> Clocking:
        # Unpumped, the RTL ignores its 2x clock input: it is held low.
        return Clocking(doubled=CLOCK2X, doubling=self.compute_pumping)

    def _narrow_weights(self) -> bool:
        return False

    def _reducer(self) -> str:
        # dotp_axi declares REDUCER for every core and only its dotp core reads it;
        # another core binds FinnLib's default, as it binds NARROW_WEIGHTS.
        return "compressor"

    def parameters(self) -> Mapping[str, int | str]:
        x, w, y = self.x.element, self.w.element, self.y.element
        return {
            "PE": self.pe,
            "SIMD": self.simd,
            "ACTIVATION_WIDTH": x.bits,
            "WEIGHT_WIDTH": w.bits,
            "ACCU_WIDTH": y.bits,
            "SIGNED_ACTIVATIONS": int(x.signed),
            "NARROW_WEIGHTS": int(self._narrow_weights()),
            "PUMPED_COMPUTE": int(self.compute_pumping),
            "SEGMENTLEN": self.segment_length,
            "VERSION": _DSP_VERSION[self.dsp],
            "ACTIVATION_BROADCASTING": int(self.form is Form.DENSE),
            "FORCE_BEHAVIORAL": 0,
            "CORE": f'"{type(self).core}"',
            "REDUCER": f'"{self._reducer()}"',
        }


_COUNTERS = tuple(
    f"module:{name}"
    for name in (
        "adder_bin",
        "adder_quad",
        "comp_fa",
        "comp_five_two",
        "comp_six_three",
        "gate_lut2",
        "ten_six",
        "ternary_adder",
    )
)
# add_multi and its compressor: the compressor's schedule is built at
# elaboration by headers that add_multi.sv `includes; they are staged beside it and
# never compiled on their own (``finn.kernels.artifacts.sources.is_header``).
_ADD_MULTI = (
    CopiedSource("finnlib", "rtl/arith/add_multi_pkg.sv", provides=("package:add_multi_pkg",)),
    CopiedSource("finnlib", "rtl/arith/compressor_pkg.sv", provides=("package:compressor_pkg",)),
    CopiedSource("finnlib", "rtl/arith/compressor_counters.sv", provides=_COUNTERS),
    CopiedSource(
        "finnlib",
        "rtl/arith/compress_core.sv",
        provides=("module:compressor",),
        requires=("package:compressor_pkg", *_COUNTERS),
    ),
    CopiedSource("finnlib", "rtl/arith/schedule_core.svh", provides=("include:schedule_core.svh",)),
    CopiedSource("finnlib", "rtl/arith/sched_chunks.svh", provides=("include:sched_chunks.svh",)),
    CopiedSource(
        "finnlib",
        "rtl/arith/add_multi_sched.svh",
        provides=("include:add_multi_sched.svh",),
        requires=(
            "package:compressor_pkg",
            "include:schedule_core.svh",
            "include:sched_chunks.svh",
        ),
    ),
    CopiedSource(
        "finnlib",
        "rtl/arith/add_multi.sv",
        provides=("module:add_multi",),
        requires=("package:add_multi_pkg", "include:add_multi_sched.svh", "module:compressor"),
    ),
)


#: add_multi's two reductions of a lane's SIMD products (FinnLib dotp's ``REDUCER``): a
#: compressor tree of LUT counters with its pipeline registers at its output, or a binary
#: adder tree with them between its levels. Same latency and results; neither dominates
#: in area, timing or synthesis cost.
REDUCERS = ("compressor", "tree")


def _dotp_axi(core: str) -> CopiedSource:
    return CopiedSource(
        "finnlib", "rtl/linalg/dotp_axi.sv", provides=("module:dotp_axi",), requires=(core,)
    )


class PackedDotpKernel(DotpAxiKernel):
    """FinnLib ``dotp``: activation and weight lanes packed into DSP48E1, DSP48E2 or DSP58.

    Activations are broadcast (``DENSE`` only). ``narrow_weights`` derives from
    the weight channel's element: when its range excludes the type's most
    negative value (a value owner stated it), a weight needs no sign guard bit,
    which packs more lanes per DSP and admits weights as wide as the DSP's A
    input. FinnLib stops simulation on a weight that breaks it.

    Its accumulator is the range's own encoding: FinnLib's packed lanes resolve the sum
    modulo ``2**ACCU_WIDTH``, exact when the sum and every partial sum fit, which
    the range bounds (FinnLib ``229b35f``: ``ACCU_WIDTH >= 1``), so one product's width
    is no floor. It refuses an accumulator wider than the DSP's P path (48 bits, 58 on
    DSP58): FinnLib counts the top lane's wraps past P, but a single lane wider than
    P is not verified.
    """

    id = "finnlib.dotp_axi.dotp"
    version = 1
    core = "dotp"

    #: How add_multi reduces the SIMD products: a choice within this core. The order of
    #: its cases states no preference; a strategy ranks them (``finn.kernels.explore``).
    reducer: str = Decision(values=REDUCERS)

    @derived
    def narrow_weights(self) -> bool:
        """No weight its channel carries is its type's minimum."""
        element = self.w.element
        if element.value_range is None:
            return False  # not an integer encoding: the weight port refuses it
        return element.value_range[0] > ordinary_integer_bounds(element.dtype)[0]

    def _core_refusal(self) -> Rejected | None:
        if self.form is not Form.DENSE:
            return reject("dotp-form", "the packed core broadcasts activations to every PE lane")
        a_bits, b_bits, _ = dsp_widths(self.dsp)
        activation, weights = self.x.element, self.w.element
        if activation.bits + (not activation.signed) > b_bits:
            return reject(
                "dotp-activation-width", "activation values must fit the signed DSP B input"
            )
        if weights.bits + (not self.narrow_weights) > a_bits:
            return reject(
                "dotp-weight-width",
                "weights must fit the DSP A input, with a sign guard bit unless narrow",
            )
        return None

    def _accumulator(self, least: int, greatest: int) -> QONNXDataType | Rejected:
        return range_dtype(least, greatest)

    def _narrow_weights(self) -> bool:
        return self.narrow_weights

    def _reducer(self) -> str:
        return self.reducer

    def sources(self) -> tuple[CopiedSource, ...]:
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

    Its accumulator is the range's own encoding: it sums in the DSP58's 58 bits and
    presents the low ``ACCU_WIDTH``, exact when the sum fits, in either signedness:
    ``ACCU_WIDTH`` in ``[1, 58]`` (FinnLib refuses any other), the DSP58's P path.
    """

    id = "finnlib.dotp_axi.dotp_8sx9_dsp58"
    version = 1
    core = "dotp_8sx9_dsp58"

    @staticmethod
    def operand_refusal(
        dsp: DspBlock | None, activation: QONNXDataType, weights: QONNXDataType
    ) -> Rejected | None:
        """Why the core cannot take these operands on ``dsp``, or ``None``: its bounds read
        the datatypes alone, so a parent asks them before it places the core."""
        if dsp is not DspBlock.DSP58:
            return reject("dotp-target", "the INT8 core is a DSP58 mode")
        if qonnx_datatype_width(activation) + (not activation.signed()) > 9:
            return reject(
                "dotp-activation-width", "activation values must fit the 9-bit signed INT8 lanes"
            )
        if qonnx_datatype_width(weights) > 8:
            return reject("dotp-weight-width", "weights must fit the 8-bit INT8 lanes")
        return None

    def _core_refusal(self) -> Rejected | None:
        return self.operand_refusal(self.dsp, self.x.element.dtype, self.w.element.dtype)

    def _accumulator(self, least: int, greatest: int) -> QONNXDataType | Rejected:
        return range_dtype(least, greatest)

    def sources(self) -> tuple[CopiedSource, ...]:
        return (
            CopiedSource(
                "finnlib", "rtl/linalg/dotp_8sx9_dsp58.sv", provides=("module:dotp_8sx9_dsp58",)
            ),
            _dotp_axi("module:dotp_8sx9_dsp58"),
        )


__all__ = ["DotpAxiKernel", "Int8Dsp58DotpKernel", "PackedDotpKernel"]
