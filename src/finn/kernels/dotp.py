# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical FinnLib ``dotp_axi`` around one compute core, on three tensors.

Each compute core is its own kernel: ``PackedDotpKernel`` (FinnLib ``dotp``, lanes packed
into DSP48E1, DSP48E2 or DSP58 slices) and ``Int8Dsp58DotpKernel`` (``dotp_8sx9_dsp58``,
the INT8 mode of DSP58). A parent chooses between them with a selector over kernels (``Select``).
``DotpAxiKernel`` is their shared declaration, which cannot be placed: each core states
its ``sources``, ``resource_use``, ``accumulator``, the ``activations`` it takes,
``narrow_weights``, its ``reducer`` and its constraints (``core_supported``,
``accumulator_width_supported``), beside dotp_axi's own (``stream_widths_supported``).

dotp computes ``Y[m, n] = sum_k X * W`` in the ``form`` (``finn.dataflow.gemm``) its
parent gives. ``DENSE``: each activation beat carries SIMD lanes, broadcast to the PE
accumulators. ``DEPTHWISE`` (the INT8 core only): each activation beat carries SIMD
window positions of PE channels, and lane p accumulates channel p alone. Each weight beat
carries PE * SIMD lanes, SIMD fastest; each result beat PE lanes; activation TLAST closes
each reduction.

Its ports ``x`` and ``w`` read the tensors its parent gives (``x_tensor``, ``w_tensor``),
and ``y`` writes its results, ``y_tensor``; each presents its stream (``x_stream``,
``w_stream``, ``y_stream``) and states its contract (``x_contract``, ``w_contract``,
``y_contract``). The extents are bound from the input tensors: x reads ``(m, k)`` (``(m, k,
n)`` depthwise), w ``(k, n)``, and the results are ``(m, n)``. Dense activations at the
graph's rank, ``(..., K)``, are read through the view ``(m, k)`` (``rows_viewed``), the
results stated at that rank; ``reshape_activations`` reads ``(M, K, N)`` as ``(M, K *
N)``, a densely realized depthwise operation. PE and SIMD fold ``n`` and ``k``, and the
``schedule`` walks ``m``, then ``n``, then ``k`` innermost. A dense form read through a
``window`` (``finn.dataflow.gemm.Window``, a lowered convolution) reads the image's
pixels and window positions, ``(oh, ow, n, kh, kw, c)``; the core sees the beats of a
dense form of ``K = KH * KW * C``, the window being the activation channel's.

PE and SIMD are given (``pe``, ``simd``): the role that places a core holds them, so a
switch of cores keeps them (MatMul declares them over N and the widest SIMD extent). A
core states what it requires of them as rules over the given values, never as a domain
(a core's domain cannot read its parent's choice): SIMD's fit to the K of the datapath it
takes (``simd_fits``), and pumping's need of lanes (``pumping_has_lanes``). The parent
also states N (``columns``) and the widest K any datapath of the operation reads
(``widest_reduction``), from its own facts; their fit to the tensors is a rule
(``extents_stated``: N is the weights' last axis, and the datapath's K divides the widest
K).

The results' range, ``result_range``, is its parent's statement (MatMul's); each core
derives its accumulator (``ACCU_WIDTH``) and ``y_tensor``'s encoding from it. The range
is not a proof that a frame fits it: the parent must bound each frame's accumulation to
it, every partial sum included.
"""

from __future__ import annotations

from math import ceil, floor, prod
from typing import ClassVar, cast

from qonnx.core.ranges import TensorRange

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    required,
    view,
)
from finn.core.space.declarations import Constraint
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    qonnx_datatype_width,
)
from finn.dataflow.gemm import Form, Window, c, k, kh, kw, m, n, oh, ow
from finn.dataflow.schedule import Access, Affine, Index, Schedule
from finn.dataflow.tensor import Tensor
from finn.kernels.artifacts.abi import Clocking, Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel, extent_of
from finn.kernels.laws import Axis, doubled
from finn.kernels.library import ModuleSpec
from finn.kernels.port import axis
from finn.kernels.port_model import StreamPort
from finn.kernels.utilization import Fit
from finn.kernels.values.domains import Integer, admit_element, integer_bounds, range_dtype
from finn.platform.capabilities import DspBlock, Platform, Resources, dsp_widths

_DSP_VERSION = {DspBlock.DSP48E1: 1, DspBlock.DSP48E2: 2, DspBlock.DSP58: 3}
# FINN's DSP58 chain timing model (rtl/matrixvectoractivation_rtl.py).
_FIRST_DSP_NS, _NEXT_DSP_NS = 0.741, 0.605


def core_simd(simd: int, pumped_compute: bool) -> int:
    """The SIMD ``dotp_axi`` builds its core with (``DSP_SIMD``): pumped, the core takes
    half the lanes a cycle of the doubled clock, an odd SIMD padded by one."""
    return (simd + simd % 2) // 2 if pumped_compute else simd


def packed_dotp_resources(
    *,
    pe: int,
    simd: int,
    weight_width: int,
    activation_width: int,
    narrow_weights: bool,
    dsp: DspBlock,
) -> Resources:
    """FinnLib ``dotp`` with the ``tree`` reducer, inside ``dotp_axi``; ``simd`` is the
    core's (``core_simd``).

    DSP slices are exact, the RTL's: ``NUM_LANES`` PE lanes packed into a slice's A
    input, one slice a SIMD element for each of ``PIPE_COUNT = ceil(PE / NUM_LANES)``
    pipes. LUTs and FFs are a ``Fit`` over the cross-SIMD adder trees (one a PE lane,
    ``W + A - 1`` bits wide plus their growth) and the two-bit spill trees of the
    lanes below a pipe's top one; characterised unpumped, a pumped core's wrapper
    logic is not in it.
    """
    lane = weight_width + activation_width - 1
    a_bits = dsp_widths(dsp)[0]
    lanes = (
        1 if a_bits <= weight_width else 1 + (a_bits - (not narrow_weights) - weight_width) // lane
    )
    pipes = ceil(pe / lanes)
    growth = max(simd - 1, 0).bit_length()
    trees = pe * (simd - 1) * (lane + growth)
    spills = (pe - pipes) * (simd - 1) * (2 + growth)
    return Resources(
        lut=_DOTP_LUT.at(trees, spills), ff=_DOTP_FF.at(trees, spills), dsp=pipes * simd
    )


def int8_dsp58_dotp_resources(*, pe: int, simd: int) -> Resources:
    """FinnLib ``dotp_8sx9_dsp58``: a chain of ``CHAINLEN = ceil(SIMD / 3)`` DSP58s a PE
    lane, three products each; ``simd`` is the core's (``core_simd``). Its DSP slices
    only: its LUTs and FFs (the operands' skew registers along each chain) are not
    characterised, no DSP58 part being licensed where it was, and are stated as none,
    an under-count."""
    return Resources(dsp=pe * ceil(simd / 3))


# Features: the adder trees' bits, the spill trees' bits.
_DOTP_LUT = Fit(55.4, (1.032, 0.999))
_DOTP_FF = Fit(80.9, (1.026, 1.625))


class DotpAxiKernel(Kernel):
    """The ``dotp_axi`` space shared by its core kernels, which state what is theirs.

    The ``platform``'s ``period_ns`` is the clock period the module must meet. It sets
    SEGMENTLEN, the DSP58 chain length between pipeline registers, by FINN's
    timing model: about 0.741 ns through the first DSP and 0.605 ns through each
    further one, against half the period when compute is pumped. Only the INT8
    core reads SEGMENTLEN. Pumped compute takes a phase-aligned 2x clock
    (``ap_clk2x``), which its module states; whether the shell supplies one is the
    shell's answer (``clock-unavailable``), not a capability of the platform.
    The DSP block is the ``platform``'s (``dsp``): a platform that states none is
    refused (``dotp-dsp``).

    It binds no module itself: each core binds ``dotp_axi`` with its own sources
    (``dotp_axi``), reading the parameters stated here. Its ``required()`` members make it
    a base of kernels, not placed.
    """

    id = "finnlib.dotp_axi"
    # FinnLib's CORE parameter: the name of the compute core module.
    core: ClassVar[str] = ""
    module_spec: ClassVar[ModuleSpec]

    # What each core states (the module docstring).
    accumulator = required(QONNXDataType)  # type: ignore[type-abstract]
    activations = required(Integer)
    narrow_weights = required(bool)
    reducer = required(str)
    core_supported = required(Constraint)
    accumulator_width_supported = required(Constraint)
    # Each core's constraints: these two, beside ``stream_widths_supported``.

    form: Form = Param(default=Form.DENSE)
    reshape_activations: bool = Param(default=False)
    # A dense form read through a window (the module docstring): a lowered convolution.
    window: Window | None = Param(default=None)
    # The range of its results: its parent's statement (MatMul binds its result range),
    # so that it is known before the results channel exists.
    result_range: tuple[int, int] = Param()
    platform: Platform = Param()
    # The tensors its input ports read: facts its parent gives (MatMul's activations, and
    # the weights' tensor it carries, ``carried_weights``).
    x_tensor: Tensor = Param()
    w_tensor: Tensor = Param()
    # The extents PE and SIMD may divide, which its parent states from the operation's
    # own facts (``extents_stated``). N is the same in every datapath; the widest
    # reduction is the widest datapath's K: K * N for a depthwise operation, which a
    # dense realization reads, else K.
    columns: int = Param()
    widest_reduction: int = Param()
    # Its folding factors, given: the role's choices (MatMul's), which a switch of cores
    # keeps. What a core requires of them is a rule over the given values.
    pe: int = Param()
    simd: int = Param()

    @derived
    def windowed(self) -> bool:
        """Whether it reads its activations through a window."""
        return self.window is not None

    @derived
    def x_rank(self) -> int:
        """The activations' rank."""
        return len(self.x_tensor.shape)

    @derived
    def rows_viewed(self) -> bool:
        """Whether dense activations of another rank than two are read through the view
        ``(m, k)``, and the results written through ``(m, n)``: their leading axes are the
        rows."""
        return (
            self.form is Form.DENSE
            and not self.windowed
            and not self.reshape_activations
            and self.x_rank != 2
        )

    @derived
    def x_viewed(self) -> bool:
        """Whether x reads its tensor through a view."""
        return self.reshape_activations or self.rows_viewed

    @derived
    def extents(self) -> dict[Index, int] | Rejected:
        """Each index's extent, bound from the tensors its inputs read; a window's own
        (``Window.extents``) and its outputs, the weights' last axis, which they read
        through a view; and the rows of activations read through a view: densely realized
        depthwise ones', their leading axis, and dense ones' of another rank, every
        leading axis together."""
        given: dict[Index, int] = {}
        if self.windowed:
            # A window's outputs are the weights' last axis.
            window = cast(Window, self.window)
            given |= {**window.extents, n: self.w_tensor.shape[-1]}
        if self.reshape_activations:
            given[m] = self.x_tensor.shape[0]
        if self.rows_viewed:
            given[m] = prod(self.x_tensor.shape[:-1])
        return self._bound(given or None)

    rows = extent_of(m)  # M: the results' rows
    outputs = extent_of(n)  # N: the results' columns
    reduction = extent_of(k)  # K: the weights' rows, stored (k, n)
    channels = extent_of(c)  # C: a window's channels

    @derived
    def lanes_of(self) -> Index:
        """The index SIMD splits: the reduction, or a window's channels."""
        return c if self.windowed else k

    @derived
    def simd_extent(self) -> int:
        """The datapath's extent SIMD divides: K, or a window's channels."""
        return self.channels if self.windowed else self.reduction

    # SIMD's fit to the datapath it takes is a rule over the given SIMD.
    @constraint
    def simd_fits(self) -> bool | Rejected:
        if self.simd_extent % self.simd:
            return reject(
                "dotp-simd",
                f"SIMD {self.simd} does not divide the datapath's extent {self.simd_extent}",
            )
        return True

    @constraint
    def extents_stated(self) -> bool | Rejected:
        # The extents the parent states, which its folding's domains read: N is the
        # weights' last axis, and the datapath's K (a window's taps times its channels)
        # divides the widest K (a K stated too small would drop SIMD values that fit the
        # datapath).
        extents = self.extents
        reduction = prod(extents[index] for index in self.reduced)
        if self.columns != extents[n] or self.widest_reduction % reduction:
            return reject(
                "dotp-extents",
                f"the stated extents (N {self.columns}, widest K {self.widest_reduction}) "
                f"disagree with the tensors (N {extents[n]}, K {reduction})",
            )
        return True

    @derived
    def pumpable(self) -> bool:
        """Whether pumping has SIMD lanes to split over the doubled clock: SIMD >= 2."""
        return self.simd >= 2

    # Pumping needs lanes to split (``pumping_has_lanes``, so SIMD 1 does not offer it),
    # and a doubled clock, which the shell supplies.
    compute_pumping: bool = Decision(domain=(False, True))

    @constraint
    def pumping_has_lanes(self) -> bool | Rejected:
        if self.compute_pumping and not self.pumpable:
            return reject("dotp-pumping", "pumping requires SIMD >= 2")
        return True

    @derived
    def reduced(self) -> tuple[Index, ...]:
        """The reduction's indices: ``k``, or a window's ``(kh, kw, c)``."""
        return (kh, kw, c) if self.windowed else (k,)

    @derived
    def schedule(self) -> Schedule | Rejected:
        """``n`` split by PE and the reduction by SIMD; the rows (a window's pixels), then
        ``n``, then the reduction (a window's taps, then its channels)."""
        rows = (oh, ow) if self.windowed else (m,)
        return self.bound_schedule(
            order=(*rows, n, *self.reduced), factors={n: self.pe, self.lanes_of: self.simd}
        )

    @derived
    def signature(self) -> tuple[tuple[Index | Affine, ...], ...]:
        """The indices x, w and y read: the form's, or its window's over the image's rank."""
        found = cast(Window, self.window).over(self.x_rank) if self.windowed else self.form.value
        return found.x, found.w, found.y

    @derived
    def x_index(self) -> tuple[Index | Affine, ...]:
        return self.signature[0]

    @derived
    def w_index(self) -> tuple[Index | Affine, ...]:
        return self.signature[1]

    @derived
    def y_index(self) -> tuple[Index | Affine, ...]:
        return self.signature[2]

    @derived
    def y_tensor(self) -> Tensor | Rejected:
        """The results: ``(m, n)``, a window's ``(OH * OW, n)`` (``(1, OH, OW, n)`` over an
        image at the graph's rank), rows read through a view ``(..., n)`` at the
        activations' rank; in the core's accumulator over ``result_range``."""
        element = admit_element(self.accumulator, TensorRange(*self.result_range))
        if isinstance(element, Rejected):
            return element
        shape = self.spanned(self.y_index)
        if isinstance(shape, Rejected):
            return shape
        if self.rows_viewed:
            shape = (*self.x_tensor.shape[:-1], shape[-1])
        return Tensor(shape, element)

    @derived
    def x_lanes(self) -> tuple[Index, ...]:
        """dotp_axi's activation lanes: SIMD alone, or ``s * PE + p`` depthwise."""
        return (k, n) if self.form is Form.DEPTHWISE else (self.lanes_of,)

    @derived
    def w_lanes(self) -> tuple[Index, ...]:
        """Each weight beat's PE * SIMD lanes, SIMD fastest."""
        return (n, self.lanes_of)

    @derived
    def activation_policy(self) -> Integer:
        """The activation encodings the core takes (``activations``)."""
        return self.activations

    def _accesses(self) -> list[Access]:
        found: list[Access] = []
        if self.x.binds:
            found.append(self.x.access)
        if self.w.binds:
            found.append(self.w.access)
        return found

    x = StreamPort(
        transport=Axis(bus="s_axis_input"),
        endpoint=Endpoint.TARGET,
        tensor=x_tensor,
        admits=activation_policy,
        schedule=schedule,
        index=x_index,
        lanes=x_lanes,
        closes=reduced,
        reshaped=x_viewed,
    )
    w = StreamPort(
        transport=Axis(bus="s_axis_weights"),
        endpoint=Endpoint.TARGET,
        tensor=w_tensor,
        admits=Integer(min_bits=2, signed=True),
        schedule=schedule,
        index=w_index,
        lanes=w_lanes,
        reshaped=windowed,
    )
    y = StreamPort(
        transport=Axis(bus="m_axis_output"),
        endpoint=Endpoint.INITIATOR,
        tensor=y_tensor,
        admits=Integer(),
        schedule=schedule,
        index=y_index,
        lanes=(n,),
        reduces=reduced,
        reshaped=rows_viewed,
    )

    @derived
    def dsp(self) -> DspBlock | Rejected:
        """The platform's DSP block, the slice the core is built in."""
        dsp = self.platform.dsp
        if dsp is None:
            return reject("dotp-dsp", "the platform states no DSP block")
        return dsp

    def integer_activations(self) -> bool:
        """Whether the activations are an ordinary integer encoding: a core's bounds read
        them; another is the activation port's to refuse."""
        try:
            ordinary_integer_bounds(self.x.element.dtype)
        except DatatypeError:
            return False
        return True

    @constraint
    def stream_widths_supported(self) -> bool | Rejected:
        # The native payload and byte-aligned carrier localparams are uint32.
        x = axis(self.x.physical_spec(idle=False), self.x.stream)
        if isinstance(x, Rejected):
            return x
        w = axis(self.w.physical_spec(idle=False), self.w.stream)
        if isinstance(w, Rejected):
            return w
        y = axis(self.y.physical_spec(idle=False), self.y.stream)
        if isinstance(y, Rejected):
            return y
        if any(port.carrier_bits > 0xFFFFFFFF for port in (x, w, y)):
            return reject("dotp-stream-width", "packed stream widths must fit native unsigned int")
        return True

    @view
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

    @derived
    def clocking(self) -> Clocking:
        # Unpumped, the RTL ignores its 2x clock input: it is held low.
        return doubled(type(self).clock_domain, self.compute_pumping)

    @view
    def dsp_simd(self) -> int:
        """The SIMD the core is built with (``core_simd``: DSP_SIMD)."""
        return core_simd(self.simd, self.compute_pumping)

    @view
    def parameters(self) -> dict[str, int | str]:
        x, w, y = self.x.element, self.w.element, self.y.element
        return {
            "PE": self.pe,
            "SIMD": self.simd,
            "ACTIVATION_WIDTH": x.bits,
            "WEIGHT_WIDTH": w.bits,
            "ACCU_WIDTH": y.bits,
            "SIGNED_ACTIVATIONS": int(x.signed),
            "NARROW_WEIGHTS": int(self.narrow_weights),
            "PUMPED_COMPUTE": int(self.compute_pumping),
            "SEGMENTLEN": self.segment_length,
            "VERSION": _DSP_VERSION[self.dsp],
            "ACTIVATION_BROADCASTING": int(self.form is Form.DENSE),
            "FORCE_BEHAVIORAL": 0,
            "CORE": f'"{type(self).core}"',
            "REDUCER": f'"{self.reducer}"',
        }

    @view
    def cycles(self) -> int | Rejected:
        beats: list[int] = []
        if self.x.placed:
            beats.append(self.x.stream.sequence.form.beats)
        if self.w.placed:
            beats.append(self.w.stream.sequence.form.beats)
        if self.y.placed:
            beats.append(self.y.stream.sequence.form.beats)
        return (
            max(beats)
            if beats
            else reject(
                "kernel-cycles", f"no port of {type(self).__qualname__} carries a stream's beats"
            )
        )


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
#: binary adder tree with its pipeline registers between its levels, or a compressor tree
#: of LUT counters with them at its output. Same latency and results. The adder tree is
#: listed first, so first valid takes it: it meets timing with the larger margin, and one synthesis
#: of a packed dotp takes about 3 GB of memory with it against about 37 GB with the
#: compressor. The compressor saves LUTs and registers at wide SIMD.
REDUCERS = ("tree", "compressor")


def dotp_axi(core: str, *sources: CopiedSource) -> ModuleSpec:
    """``dotp_axi`` on the core module ``core`` (``module:dotp``), which ``sources``
    provide, providers first: a core kernel's library module, its parameters
    ``DotpAxiKernel``'s."""
    return ModuleSpec(
        "dotp_axi",
        constant_sources=(
            *sources,
            CopiedSource(
                "finnlib", "rtl/linalg/dotp_axi.sv", provides=("module:dotp_axi",), requires=(core,)
            ),
        ),
    )


class PackedDotpKernel(DotpAxiKernel):
    """FinnLib ``dotp``: activation and weight lanes packed into DSP48E1, DSP48E2 or DSP58.

    Activations are broadcast (``DENSE`` only). ``narrow_weights`` derives from
    the weight tensor's element: when its range excludes the type's most
    negative value (a value owner stated it), a weight needs no sign guard bit,
    which packs more lanes per DSP and admits weights as wide as the DSP's A
    input. FinnLib stops simulation on a weight that breaks it.

    Activations take at least two bits (``dotp.sv`` refuses ``ACTIVATION_WIDTH < 2``).

    Its accumulator is the range's own encoding: FinnLib's packed lanes resolve the sum
    modulo ``2**ACCU_WIDTH``, exact when the sum and every partial sum fit, which
    the range bounds (FinnLib's ``ACCU_WIDTH >= 1``), so one product's width
    is no floor, and the DSP's P path (48 bits, 58 on DSP58) no ceiling: the top lane
    takes its low bits from P and counts P's wraps in a high sideband sized from
    ``ACCU_WIDTH`` (``dotp.sv``'s ``sliceLanes`` and ``hi_width``), so an accumulator
    wider than P is exact, a single lane's included (the ``wide_*`` cases of
    ``tests/kernels/sweeps/pure_dot_product_numeric.py``).
    """

    id = "finnlib.dotp_axi.dotp"
    core = "dotp"

    #: How add_multi reduces the SIMD products: a choice within this core, the adder tree
    #: first (``REDUCERS``: why).
    reducer: str = Decision(domain=REDUCERS)

    @derived
    def narrow_weights(self) -> bool:
        """No weight its tensor holds is its type's minimum."""
        element = self.w.element
        try:
            least = ordinary_integer_bounds(element.dtype)[0]
        except DatatypeError:
            return False  # not an ordinary integer encoding: the weight port refuses it
        return integer_bounds(element)[0] > least

    @constraint
    def core_supported(self) -> bool | Rejected:
        if not self.integer_activations():
            return True
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
        return True

    @constraint
    def accumulator_width_supported(self) -> bool | Rejected:
        """Any width of a bit or more (FinnLib's ``ACCU_WIDTH >= 1``): the packed
        lanes resolve the sum modulo ``2**ACCU_WIDTH``."""
        bits = self.y.element.bits
        if bits < 1:
            return reject(
                "dotp-accumulator-width",
                "the packed core's ACCU_WIDTH is at least 1 (FinnLib dotp.sv)",
                values={"actual_bits": bits, "minimum_bits": 1},
            )
        return True

    @derived
    def activations(self) -> Integer:
        return Integer(min_bits=2)

    @derived
    def accumulator(self) -> QONNXDataType:
        least, greatest = self.result_range
        return range_dtype(least, greatest)

    @view
    def resource_use(self) -> Resources | Rejected:
        """``packed_dotp_resources`` at the core's SIMD; the tree reducer's only."""
        if self.reducer != "tree":
            return reject(
                "dotp-resources", "the compressor reducer's resources are not characterised"
            )
        return packed_dotp_resources(
            pe=self.pe,
            simd=self.dsp_simd,
            weight_width=self.w.element.bits,
            activation_width=self.x.element.bits,
            narrow_weights=self.narrow_weights,
            dsp=self.dsp,
        )

    module_spec: ClassVar[ModuleSpec] = dotp_axi(
        "module:dotp",
        *_ADD_MULTI,
        CopiedSource(
            "finnlib",
            "rtl/linalg/dotp.sv",
            provides=("module:dotp",),
            requires=("package:add_multi_pkg", "module:add_multi"),
        ),
    )


class Int8Dsp58DotpKernel(DotpAxiKernel):
    """FinnLib ``dotp_8sx9_dsp58``: three 9x8 signed products per DSP58 in INT8 mode.

    It takes signed weights of at most 8 bits and activations that fit 9 signed
    bits, of any width down to one (BINARY): it pads each to the 9-bit lane,
    broadcast (``DENSE``) or one channel per lane (``DEPTHWISE``).

    Its accumulator is the range's own encoding: it sums in the DSP58's 58 bits and
    presents the low ``ACCU_WIDTH``, exact when the sum fits, in either signedness:
    ``ACCU_WIDTH`` in ``[1, 58]`` (FinnLib refuses any other), the DSP58's P path.
    """

    id = "finnlib.dotp_axi.dotp_8sx9_dsp58"
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

    @constraint
    def core_supported(self) -> bool | Rejected:
        if not self.integer_activations():
            return True
        refused = self.operand_refusal(self.dsp, self.x.element.dtype, self.w.element.dtype)
        return True if refused is None else refused

    @constraint
    def accumulator_width_supported(self) -> bool | Rejected:
        bits = self.y.element.bits
        if bits > 58:
            return reject(
                "dotp-accumulator-width",
                "the INT8 core presents the low ACCU_WIDTH bits of its 58-bit DSP58 "
                "accumulation: at most 58 (FinnLib dotp_8sx9_dsp58.sv)",
                values={"actual_bits": bits, "maximum_bits": 58},
            )
        return True

    @derived
    def activations(self) -> Integer:
        # No floor on ACTIVATION_WIDTH: it pads each activation to its 9-bit lane.
        return Integer()

    @derived
    def accumulator(self) -> QONNXDataType:
        least, greatest = self.result_range
        return range_dtype(least, greatest)

    @derived
    def narrow_weights(self) -> bool:
        """FinnLib's default: the INT8 core reads no NARROW_WEIGHTS."""
        return False

    @derived
    def reducer(self) -> str:
        """FinnLib's default: dotp_axi declares REDUCER for every core, and only the packed
        one reads it."""
        return "compressor"

    @view
    def resource_use(self) -> Resources:
        return int8_dsp58_dotp_resources(pe=self.pe, simd=self.dsp_simd)

    module_spec: ClassVar[ModuleSpec] = dotp_axi(
        "module:dotp_8sx9_dsp58",
        CopiedSource(
            "finnlib", "rtl/linalg/dotp_8sx9_dsp58.sv", provides=("module:dotp_8sx9_dsp58",)
        ),
    )


__all__ = [
    "DotpAxiKernel",
    "dotp_axi",
    "Int8Dsp58DotpKernel",
    "PackedDotpKernel",
    "int8_dsp58_dotp_resources",
    "packed_dotp_resources",
]
