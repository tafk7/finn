# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer profile of FinnLib thresholding_axi, including resident parameter sets.

thresholds[set][channel][threshold] is an immutable, sorted table of numerical
integers. Its shape owns SETS, C and N. The input is sign/zero extended or
saturated to the threshold dtype before comparison, as the native RTL specifies.
Output is the threshold count plus bias. Runtime writes must preserve sorted
rows. With multiple sets, each input beat requires a matching set-selector beat.

All native pins remain present when AXI-Lite or set selection is disabled;
disabled outputs may be unspecified. Placed in a composite, the kernel sits on
an input, an output and (with several sets) a set-selector stream; its AXI-Lite
bus is exported through a ``ControlBus`` when thresholds are runtime-writable,
and otherwise held idle by its tie-offs, as is the set selector of a single
set. Multi-set AXI-Lite access is refused: the pinned wrapper's configuration
address width omits set bits. Static multi-set
selection remains supported. Floating-point threshold comparison is outside
this first profile; Eltwise and IntToFp32 exercise float authoring separately.
Biases below -N-1 are refused: the native unsigned width expression creates a
33-bit output, but the result addition zero-extends the negative 32-bit bias.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    constraint,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.schedule import Index, Refused, Schedule
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence, vector_major
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Signal, StandardProtocol
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.base import Kernel, Tieoffs
from finn.kernels.control import CONTROL, CONTROL_SEMANTICS, Control, ControlBus, held_bus
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.datatypes.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
    ThresholdTable,
)
from finn.kernels.port import GivenPort
from finn.kernels.streams import Stream


class ThresholdingAxiKernel(Kernel):
    id = "finnlib.thresholding_axi.integer"
    version = "1"
    module = "thresholding_axi"

    input_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    threshold_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    input_encoding = integer_scalar(input_dtype, Integer())
    threshold_encoding = integer_scalar(threshold_dtype, Integer())
    thresholds: ThresholdTable = Param(semantics=THRESHOLD_TABLE)
    bias: int = Param()

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_dtype(self) -> QONNXDataType | Rejected:
        table = self.thresholds
        bias = self.bias
        if not table or not table[0] or not table[0][0]:
            return reject("threshold-shape", "a nonempty threshold table is required")
        count = len(table[0][0])
        if bias >= 0:
            bits = max(1, (count + bias).bit_length())
            return resolve_qonnx_datatype_name(f"UINT{bits}")
        # N is unsigned in the native expression. Preserve its 32-bit arithmetic,
        # including the extra output bits when the whole result range is negative.
        candidate = max((-bias) & 0xFFFFFFFF, (count + bias + 1) & 0xFFFFFFFF)
        bits = 1 + (candidate - 1).bit_length()
        return resolve_qonnx_datatype_name(f"INT{bits}")

    pe: int = Param()
    # Where a parent places it: its streams, and the control
    # bus that exports its AXI-Lite interface when thresholds are runtime-writable.
    input_stream: Stream = Param(required=False)
    output_stream: Stream = Param(required=False)
    set_stream: Stream = Param(required=False)
    control: ControlBus = Param(required=False)
    use_axilite: bool = Decision(values=(False, True))
    deep_pipeline: bool = Decision(values=(False, True))
    depth_trigger_bram: int = Param()
    depth_trigger_uram: int = Param()

    @constraint
    def types_supported(self) -> bool | Rejected:
        a, t = self.input_dtype, self.threshold_dtype
        for dtype in (a, t):
            admitted = Integer().check(dtype)
            if isinstance(admitted, Rejected):
                return admitted
        if a.signed() != t.signed():
            return reject(
                "threshold-type", "input and threshold encodings must have the same signedness"
            )
        return True

    @constraint
    def table_supported(self) -> bool | Rejected:
        table = self.thresholds
        if not table or not table[0] or not table[0][0]:
            return reject("threshold-shape", "nonempty sets/channels/thresholds are required")
        channels, count = len(table[0]), len(table[0][0])
        if any(
            len(group) != channels or any(len(row) != count for row in group) for group in table
        ):
            return reject(
                "threshold-shape",
                "thresholds must be a rectangular (sets, channels, thresholds) table",
            )
        try:
            minimum, maximum = ordinary_integer_bounds(self.threshold_dtype)
        except DatatypeError as error:
            return reject("threshold-type", str(error))
        if any(not minimum <= item <= maximum for group in table for row in group for item in row):
            return reject("threshold-value", "every threshold must fit threshold_dtype")
        if any(
            any(left > right for left, right in zip(row, row[1:]))
            for group in table
            for row in group
        ):
            return reject("threshold-order", "threshold rows must be sorted in nondecreasing order")
        return True

    @constraint
    def folding_supported(self) -> bool | Rejected:
        table, pe = self.thresholds, self.pe
        if not table or not table[0] or not 1 <= pe <= 0xFFFFFFFF:
            return reject(
                "threshold-shape", "nonempty channels and positive native PE are required"
            )
        channels = len(table[0])
        if channels % pe and pe % channels:
            return reject("threshold-folding", "channels must divide PE or PE must divide channels")
        return True

    @constraint
    def memory_supported(self) -> bool | Rejected:
        if not all(
            0 <= value <= 0xFFFFFFFF for value in (self.depth_trigger_bram, self.depth_trigger_uram)
        ):
            return reject("threshold-memory", "memory triggers must fit native unsigned int")
        return True

    @constraint
    def bias_supported(self) -> bool | Rejected:
        bias, table = self.bias, self.thresholds
        if not table or not table[0] or not table[0][0]:
            return reject("threshold-shape", "a nonempty threshold table is required")
        if not -(1 << 31) <= bias < (1 << 31):
            return reject("threshold-bias", "BIAS must fit native signed int")
        if bias < -len(table[0][0]) - 1:
            return reject(
                "threshold-negative-range",
                "native RTL does not sign-extend BIAS correctly below -N-1",
            )
        return True

    @constraint
    def configuration_supported(self) -> bool | Rejected:
        if self.use_axilite and len(self.thresholds) > 1:
            return reject(
                "threshold-config-sets",
                "the native AXI wrapper does not address multiple configuration sets",
            )
        return True

    @constraint
    def carried(self) -> bool | Rejected:
        """Each placed stream carries the element its port takes."""
        placed: list[tuple[str, ScalarEncoding, QONNXDataType]] = []
        if self.present(ThresholdingAxiKernel.input_stream):
            placed.append(("input", self.input_stream.tensor.element, self.input_dtype))
        if self.present(ThresholdingAxiKernel.output_stream):
            placed.append(("output", self.output_stream.tensor.element, self.result_dtype))
        if self.present(ThresholdingAxiKernel.set_stream):
            placed.append(("set", self.set_stream.tensor.element, self.selector_dtype))
        for name, element, dtype in placed:
            if element.datatype_name != dtype.name:
                return reject(
                    "threshold-stream-element",
                    f"the {name} stream carries {element.datatype_name}, the port {dtype.name}",
                )
        return True

    admission = ConstraintGroup(
        types_supported,
        table_supported,
        folding_supported,
        memory_supported,
        bias_supported,
        configuration_supported,
        carried,
    )

    @derived(semantics=default_semantics(Bus))
    def config_bus(self) -> Bus | Rejected:
        """The AXI-Lite configuration bus, present in every configuration."""
        table, pe, bits = self.thresholds, self.pe, self.threshold_dtype.bitwidth()
        if not table or not table[0] or not table[0][0] or pe < 1:
            return reject("threshold-shape", "a nonempty table and a positive PE are required")
        channels, count = len(table[0]), len(table[0][0])
        cf, cpe = max(1, channels // pe), min(channels, pe)
        address_bits = (
            sum((value - 1).bit_length() for value in (cf, cpe, count, (bits + 31) // 32)) + 2
        )
        return Bus(
            "s_axilite",
            StandardProtocol.AXILITE,
            tuple(
                Member(name.lower(), "s_axilite_" + name, width)
                for name, width in (
                    ("AWVALID", 1),
                    ("AWREADY", 1),
                    ("AWADDR", address_bits),
                    ("WVALID", 1),
                    ("WREADY", 1),
                    ("WDATA", 32),
                    ("WSTRB", 4),
                    ("BVALID", 1),
                    ("BREADY", 1),
                    ("BRESP", 2),
                    ("ARVALID", 1),
                    ("ARREADY", 1),
                    ("ARADDR", address_bits),
                    ("RVALID", 1),
                    ("RREADY", 1),
                    ("RDATA", 32),
                    ("RRESP", 2),
                )
            ),
            associated_clock="ap_clk",
            associated_reset="ap_rst_n",
        )

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def selector_dtype(self) -> QONNXDataType:
        sets = len(self.thresholds)
        return resolve_qonnx_datatype_name(f"UINT{(sets - 1).bit_length() if sets > 2 else 1}")

    @derived(semantics=BEAT_SEQUENCE)
    def input_sequence(self) -> BeatSequence | Rejected:
        """PE consecutive channels a beat, channels the innermost axis of the tensor.

        The schedule walks every outer axis, then the channels folded by PE:
        T[..., c]. PE above the channel count would fold
        rows into the lanes as well, which needs rows divisible by PE / C; that
        is not modelled yet.
        """
        shape, pe, channels = self.input_stream.tensor.shape, self.pe, len(self.thresholds[0])
        if shape[-1] != channels or channels % pe:
            return reject(
                "threshold-stream-form",
                f"the input must walk its {channels} channels innermost, PE={pe} per beat",
            )
        outer = tuple(Index(f"a{axis}") for axis in range(len(shape) - 1))
        c = Index("c")
        schedule = Schedule(dict(zip((*outer, c), shape)), folds={c: pe})
        try:
            return BeatSequence(schedule.present(shape, (*outer, c), lanes=(c,)))
        except Refused as error:
            return reject("threshold-stream-form", str(error))

    @derived(semantics=BEAT_SEQUENCE)
    def output_sequence(self) -> BeatSequence | Rejected:
        """The input's order, over a tensor of the input's shape."""
        sequence = self.input_sequence
        if self.output_stream.tensor.shape != sequence.form.shape:
            return reject("threshold-stream-form", "the output keeps the input's shape")
        return sequence

    @derived(semantics=BEAT_SEQUENCE)
    def set_sequence(self) -> BeatSequence | Rejected:
        """One set index for each input beat."""
        shape = self.set_stream.tensor.shape
        if len(self.thresholds) < 2:
            return reject("threshold-set-stream", "a single threshold set takes no set stream")
        if shape != (self.input_sequence.form.beats,):
            return reject("threshold-set-stream", "each input beat needs one set index")
        return BeatSequence(vector_major(shape, 1))

    input = GivenPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        stream=input_stream,
        sequence=input_sequence,
        idle_dtype=input_dtype,
        idle_lanes=pe,
    )
    output = GivenPort(
        name="m_axis",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        sequence=output_sequence,
        idle_dtype=result_dtype,
        idle_lanes=pe,
    )
    set = GivenPort(
        name="s_axis_set",
        endpoint=Endpoint.TARGET,
        stream=set_stream,
        sequence=set_sequence,
        idle_dtype=selector_dtype,
    )

    def parameters(self) -> Mapping[str, int | str] | Rejected:
        table = self.thresholds
        if not table or not table[0] or not table[0][0]:
            return reject("threshold-shape", "a nonempty threshold table is required")
        a = self.input_encoding.encoding.dtype
        bits = self.threshold_encoding.encoding.dtype.bitwidth()
        mask = (1 << bits) - 1
        image = (
            "'{"
            + ", ".join(
                "'{"
                + ", ".join(
                    "'{" + ", ".join(f"{bits}'h{item & mask:x}" for item in row) + "}"
                    for row in group
                )
                + "}"
                for group in table
            )
            + "}"
        )
        return {
            "WI": a.bitwidth(),
            "WT": bits,
            "N": len(table[0][0]),
            "C": len(table[0]),
            "PE": self.pe,
            "SIGNED": int(a.signed()),
            "FPARG": 0,
            "BIAS": self.bias,
            "SETS": len(table),
            "THRESHOLDS": image,
            "THRESHOLDS_FILE": '""',
            "USE_AXILITE": int(self.use_axilite),
            "DEPTH_TRIGGER_BRAM": self.depth_trigger_bram,
            "DEPTH_TRIGGER_URAM": self.depth_trigger_uram,
            "DEEP_PIPELINE": int(self.deep_pipeline),
        }

    def sources(self) -> tuple[CopiedSource, ...]:
        return (
            CopiedSource("finnlib", "rtl/infra/axilite.sv", provides=("module:axilite",)),
            CopiedSource(
                "finnlib", "rtl/nonlin/thresholding.sv", provides=("module:thresholding",)
            ),
            CopiedSource(
                "finnlib",
                "rtl/nonlin/thresholding_axi.sv",
                provides=("module:thresholding_axi",),
                requires=("module:axilite", "module:thresholding"),
            ),
        )

    def other_pins(self) -> tuple[Signal | Bus, ...]:
        return (self.config_bus,)

    def held(self) -> Tieoffs | Rejected:
        """AXI-Lite, without runtime writes."""
        if not self.use_axilite:
            return held_bus(self.config_bus)
        if not self.present(ThresholdingAxiKernel.control):
            return reject("threshold-control", "runtime-writable thresholds need a control bus")
        return Tieoffs()

    @view(semantics=CONTROL_SEMANTICS)
    def control_bus(self) -> Control:
        return Control(self.config_bus if self.use_axilite else None)

    exports = {**Kernel.exports, CONTROL: {control: control_bus}}


__all__ = ["ThresholdingAxiKernel"]
