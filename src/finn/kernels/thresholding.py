# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer profile of FinnLib thresholding_axi, including resident parameter sets.

thresholds[set][channel][threshold] is an immutable, sorted table of numerical
integers. Its shape owns SETS, C and N. The input is sign/zero extended or
saturated to the threshold dtype before comparison, as the native RTL specifies.
Output is the threshold count plus bias. Runtime writes must preserve sorted
rows. With multiple sets, each input beat requires a matching set-selector beat.

PE, the channels a beat, is a Decision over the divisors of the table's C,
known without a stream, so a flat build commits it as a choice. Placed, the
input and output walk one schedule row-major, ``c`` split by PE innermost;
their tensors bind the extents, and the channel count must agree with the
table's (``kernel-extents``). The set port indexes beats, which no index of a
tensor expresses, so it presents a given sequence.

The threshold memories are the kernel's choice of resource, by pipeline stage.
The RTL compares through M = clog2(N + 1) stages; stage s keeps one memory per
PE lane, of depth ``base * 2**s`` (``base`` the channel folds; with several sets,
the sets times the folds rounded up to a power of two), and assigns each a
resource monotone in depth from its two depth triggers. So the expressible assignments are: the
deepest ``ultra_stages`` in UltraRAM, the ``block_stages`` above them in block
RAM, and the rest ``ram_style``: ``distributed``, or Vivado's choice (``auto``,
with no stage in block RAM). Each assignment has one spelling: with every stage
in UltraRAM none is left, and ``ram_style`` does not apply. Counted in stages,
not depths, the choices do not move with PE; ``parameters`` maps them to the
triggers (the depth of the first stage in each resource, 0 for none). An UltraRAM stage requires the
``platform``'s UltraRAM that takes initial contents (the table is the
memories' initial contents), and runtime-writable thresholds its control port,
each a named refusal of the case.

All native pins remain present when AXI-Lite or set selection is disabled;
disabled outputs may be unspecified. Placed in a kernel with children, it sits
on an input, an output and (with several sets) a set-selector stream; its
AXI-Lite bus is presented through a ``ControlBus`` when thresholds are
runtime-writable (``controlled``), and otherwise held idle by its module, as
is the set selector of a single set. Multi-set AXI-Lite access is refused: the
pinned wrapper's configuration address width omits set bits. Static multi-set
selection remains supported. Floating-point threshold comparison is outside
this first profile.
Biases below -N-1 are refused: the native unsigned width expression creates a
33-bit output, but the result addition zero-extends the negative 32-bit bias.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Domain,
    Param,
    Rejected,
    constraint,
    default_semantics,
    derived,
    divisors_of,
    domain,
    reject,
    requires,
    requiring,
    view,
)
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Signal, StandardProtocol
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Held
from finn.kernels.base import Kernel
from finn.kernels.control import CONTROL, Control, ControlBus, held_bus
from finn.kernels.datatypes.domains import Integer, set_index_dtype
from finn.kernels.datatypes.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
    ThresholdTable,
)
from finn.kernels.port import AxiStreamPort
from finn.kernels.streams import Stream
from finn.kernels.target import Platform

c = Index("c")


def _staged(count: object) -> bool:
    """The cases that place a stage: a count above 0."""
    return isinstance(count, int) and count > 0


def stage_counts(stages: object) -> Domain[int]:
    """How many of the ``stages`` pipeline stages: 0 to all of them."""

    def accepts(*, candidate: int, stages: int) -> bool:
        return type(candidate) is int and 0 <= candidate <= stages

    def candidates(*, stages: int) -> range:
        return range(stages + 1)

    return domain(
        accepts=accepts,
        candidates=candidates,
        semantics=default_semantics(int),
        stages=stages,
    )


class ThresholdingAxiKernel(Kernel):
    id = "finnlib.thresholding_axi.integer"
    version = 1
    rtl_module = "thresholding_axi"

    input_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    threshold_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    thresholds: ThresholdTable = Param(semantics=THRESHOLD_TABLE)
    bias: int = Param()

    @derived
    def shape(self) -> tuple[int, int, int] | Rejected:
        """SETS, C and N, as the table's first row states them (``table_supported``
        refuses a table that is not rectangular)."""
        table = self.thresholds
        if not table or not table[0] or not table[0][0]:
            return reject("threshold-shape", "nonempty sets/channels/thresholds are required")
        return len(table), len(table[0]), len(table[0][0])

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def result_dtype(self) -> QONNXDataType | Rejected:
        bias = self.bias
        _, _, count = self.shape
        if bias >= 0:
            bits = max(1, (count + bias).bit_length())
            return resolve_qonnx_datatype_name(f"UINT{bits}")
        # N is unsigned in the native expression. Preserve its 32-bit arithmetic,
        # including the extra output bits when the whole result range is negative.
        candidate = max((-bias) & 0xFFFFFFFF, (count + bias + 1) & 0xFFFFFFFF)
        bits = 1 + (candidate - 1).bit_length()
        return resolve_qonnx_datatype_name(f"INT{bits}")

    @derived
    def channels(self) -> int:
        """C, the table's: known without a stream, so a flat build has its factor domain."""
        return self.shape[1]

    # PE channels a beat; PE above C would carry rows in the lanes, not modelled yet.
    pe: int = Decision(domain=divisors_of(channels))
    # Where a parent places it: its streams, and the control
    # bus that exports its AXI-Lite interface when thresholds are runtime-writable.
    input_stream: Stream = Param(required=False)
    output_stream: Stream = Param(required=False)
    set_stream: Stream = Param(required=False)
    control: ControlBus = Param(required=False)
    platform: Platform = Param()
    use_axilite: bool = Decision(
        values=(False, True),
        requires=(
            requires(
                platform.control_ports,
                "control-absent: the platform has no control port for runtime-writable thresholds",
                cases=(True,),
            ),
        ),
    )
    deep_pipeline: bool = Decision(values=(False, True))

    @derived
    def stages(self) -> int:
        """M, the RTL's pipeline stages: clog2(N + 1)."""
        return self.shape[2].bit_length()

    # The threshold memories (module docstring): the deepest ``ultra_stages`` in
    # UltraRAM; when any stage is left above them, the ``block_stages`` in block RAM
    # and the rest ``ram_style``.
    ultra_stages: int = Decision(
        domain=requiring(
            stage_counts(stages),
            requires(platform.uram, "uram-absent: the platform has no UltraRAM", cases=_staged),
            requires(
                platform.uram_init,
                "uram-init: the platform's UltraRAM takes no initial contents",
                cases=_staged,
            ),
        )
    )

    @derived
    def left(self) -> bool:
        """Whether any stage is left above the UltraRAM ones."""
        return self.ultra_stages < self.stages

    ram_style: str = Decision(values=("auto", "distributed"), when=left)

    @derived
    def distributed(self) -> bool:
        return self.left and self.ram_style == "distributed"

    block_stages: int = Decision(domain=stage_counts(stages), when=distributed)

    def stage_depth(self, stage: int) -> int:
        """The depth of a stage's memory, as the RTL computes it from SETS, C and PE."""
        (sets, channels, _), pe = self.shape, self.pe
        folds = 1 if pe >= channels else channels // pe
        base = sets * (1 << (folds - 1).bit_length()) if sets > 1 else folds
        return base << stage

    @derived
    def depth_triggers(self) -> tuple[int, int] | Rejected:
        """DEPTH_TRIGGER_BRAM and DEPTH_TRIGGER_URAM: the depth of the first stage in
        block RAM (past the deepest when none is) or UltraRAM; 0 leaves it unset."""
        stages, ultra = self.stages, self.ultra_stages
        block = self.block_stages if self.distributed else 0
        if block + ultra > stages:
            return reject(
                "threshold-memory",
                f"{block} block RAM and {ultra} UltraRAM stages exceed the {stages} stages",
            )
        uram = self.stage_depth(stages - ultra) if ultra else 0
        if not self.distributed:
            return 0, uram
        return self.stage_depth(stages - ultra - block), uram

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
        table, (_, channels, count) = self.thresholds, self.shape
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
    def memory_supported(self) -> bool | Rejected:
        if not all(0 <= value <= 0xFFFFFFFF for value in self.depth_triggers):
            return reject("threshold-memory", "memory triggers must fit native unsigned int")
        return True

    @constraint
    def bias_supported(self) -> bool | Rejected:
        bias, (_, _, count) = self.bias, self.shape
        if not -(1 << 31) <= bias < (1 << 31):
            return reject("threshold-bias", "BIAS must fit native signed int")
        if bias < -count - 1:
            return reject(
                "threshold-negative-range",
                "native RTL does not sign-extend BIAS correctly below -N-1",
            )
        return True

    @constraint
    def configuration_supported(self) -> bool | Rejected:
        if self.use_axilite and self.shape[0] > 1:
            return reject(
                "threshold-config-sets",
                "the native AXI wrapper does not address multiple configuration sets",
            )
        return True

    admission = ConstraintGroup(
        types_supported,
        table_supported,
        memory_supported,
        bias_supported,
        configuration_supported,
    )

    @derived
    def config_bus(self) -> Bus | Rejected:
        """The AXI-Lite configuration bus, present in every configuration."""
        (_, channels, count), pe = self.shape, self.pe
        bits = self.threshold_dtype.bitwidth()
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
        return set_index_dtype(self.shape[0])

    @derived
    def indices(self) -> tuple[Index, ...]:
        """The input's axes: any leading ones, then the channels ``c``, innermost."""
        rank = len(self.input_stream.tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 1)), c)

    @derived
    def factors(self) -> dict[Index, int]:
        return {c: self.pe}

    @derived
    def schedule(self) -> Schedule | Rejected:
        """Row-major over the input's axes, ``c`` (the table's C) split by PE innermost."""
        indices = self.indices
        return self.bound_schedule(indices, self.factors, extents={c: self.channels})

    @derived
    def set_sequence(self) -> BeatSequence | Rejected:
        """One set index for each input beat."""
        shape = self.set_stream.tensor.shape
        if self.shape[0] < 2:
            return reject("threshold-set-stream", "a single threshold set takes no set stream")
        if shape != (self.input.presented.form.beats,):
            return reject("threshold-set-stream", "each input beat needs one set index")
        return BeatSequence(vector_major(shape, 1))

    input = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        stream=input_stream,
        schedule=schedule,
        factors=factors,
        index=indices,
        lanes=(c,),
        dtype=input_dtype,
    )
    output = AxiStreamPort(
        name="m_axis",
        endpoint=Endpoint.INITIATOR,
        stream=output_stream,
        schedule=schedule,
        factors=factors,
        index=indices,
        lanes=(c,),
        dtype=result_dtype,
    )
    # The set port indexes beats (``c``'s folds), which no index of a tensor expresses.
    set = AxiStreamPort(
        name="s_axis_set",
        endpoint=Endpoint.TARGET,
        stream=set_stream,
        sequence=set_sequence,
        dtype=selector_dtype,
    )

    def parameters(self) -> Mapping[str, int | str]:
        table, (sets, channels, count) = self.thresholds, self.shape
        a, bits = self.input_dtype, self.threshold_dtype.bitwidth()
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
            "N": count,
            "C": channels,
            "PE": self.pe,
            "SIGNED": int(a.signed()),
            "FPARG": 0,
            "BIAS": self.bias,
            "SETS": sets,
            "THRESHOLDS": image,
            "THRESHOLDS_FILE": '""',
            "USE_AXILITE": int(self.use_axilite),
            "DEPTH_TRIGGER_BRAM": self.depth_triggers[0],
            "DEPTH_TRIGGER_URAM": self.depth_triggers[1],
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

    def held(self) -> Held | Rejected:
        """AXI-Lite, without runtime writes."""
        if not self.use_axilite:
            return held_bus(self.config_bus)
        if not self.present(ThresholdingAxiKernel.control):
            return reject("threshold-control", "runtime-writable thresholds need a control bus")
        return Held()

    def controlled(self) -> tuple[Bus, ...]:
        """AXI-Lite, with runtime writes."""
        return (self.config_bus,) if self.use_axilite else ()

    @view
    def control_bus(self) -> Control:
        return Control(self.config_bus if self.controlled() else None)

    exports = {**Kernel.exports, CONTROL: {control: control_bus}}


__all__ = ["ThresholdingAxiKernel"]
