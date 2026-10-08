# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer profile of FinnLib thresholding_axi, including resident parameter sets.

thresholds[set][row][threshold] is an immutable, sorted table of numerical
integers. Its shape owns SETS, C and N: C is the RTL's number of threshold rows,
which it applies round-robin over the flat input stream. A table has one row for
each channel, or one row shared by every channel (C = 1: each PE lane keeps one
copy of it, and a runtime write reaches every lane). The input is sign/zero
extended or saturated to the threshold dtype before comparison, as the native
RTL specifies. Output is the threshold count plus bias. Runtime writes must
preserve sorted rows. With multiple sets, each input beat requires a matching
set-selector beat.

PE, the channels a beat, is a Decision over the divisors of the input's
channels; flat, with no channel to bind them, it is any PE the RTL takes with
the table's C (``lane_counts``), committed as a choice. Placed, the input and
output walk one schedule row-major, ``c`` split by PE innermost; their tensors
bind the extents, and the table has one row or one row a channel
(``threshold-rows``). The set port indexes beats, which no index of a tensor
expresses, so it presents a given sequence.

The threshold memories are the kernel's choice of resource, by pipeline stage.
The RTL compares through M = clog2(N + 1) stages; stage s keeps one memory per
PE lane, of depth ``base * 2**s`` (``base`` the row folds, C / PE or 1; with several sets,
the sets times the folds rounded up to a power of two), and assigns each a
resource monotone in depth from its two depth triggers. So the expressible assignments are: the
deepest ``ultra_stages`` in UltraRAM, the ``block_stages`` above them in block
RAM, and the rest ``ram_style``: ``distributed``, or Vivado's choice (``auto``,
with no stage in block RAM). Each assignment has one spelling: with every stage
in UltraRAM none is left, and ``ram_style`` does not apply. Counted in stages,
not depths, the choices do not move with PE; ``parameters`` maps them to the
triggers (the depth of the first stage in each resource, 0 for none). An UltraRAM stage requires the
``platform``'s UltraRAM that takes initial contents (the table is the
memories' initial contents), and runtime-writable thresholds its control port
and a ``control`` bus to be placed on, each a named refusal of the case.

All native pins remain present when AXI-Lite or set selection is disabled;
disabled outputs may be unspecified. Placed in a kernel with children, it sits
on an input, an output and (with several sets) a set-selector channel; its
AXI-Lite bus is presented through a ``ControlBus`` when thresholds are
runtime-writable (``controlled``), and otherwise held idle by its module, as
is the set selector of a single set. Presented, the bus carries the writes that
put the kernel's table into the memories (``register_map``), at the addresses
thresholding_axi decodes. Multi-set AXI-Lite access is refused: the
pinned wrapper's configuration address width omits set bits. Static multi-set
selection is supported. Floating-point threshold comparison is outside this
profile.
Biases below -N-1 are refused: the native unsigned width expression creates a
33-bit output, but the result addition zero-extends the negative 32-bit bias.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

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
from finn.kernels.artifacts.abi import Bus, Endpoint, Member, Pin, StandardProtocol
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Held, RegisterMap
from finn.kernels.base import CLOCK, RESET, Kernel, extent_of
from finn.kernels.channels import Channel
from finn.kernels.control import CONTROL, Control, ControlBus, held_bus
from finn.kernels.port import AxiStreamPort
from finn.kernels.target import Platform
from finn.kernels.values.domains import Integer, set_index_dtype
from finn.kernels.values.semantics import (
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
    ThresholdTable,
)

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
        ordered=True,
        stages=stages,
    )


def lane_counts(rows: object) -> Domain[int]:
    """PE's domain: the divisors of the input's channels, once a placed port binds them;
    flat, any PE the RTL takes with the table's ``rows`` (C a multiple of PE, or PE of C)."""
    divisors = divisors_of(1).candidates
    assert divisors is not None

    def accepts(*, candidate: int, rows: int, extents: Mapping[Index, int]) -> bool:
        if type(candidate) is not int or candidate < 1:
            return False
        if c in extents:
            return extents[c] % candidate == 0
        return candidate < 1 << 32 and (rows % candidate == 0 or candidate % rows == 0)

    def candidates(*, rows: int, extents: Mapping[Index, int]) -> tuple[int, ...] | Rejected:
        if c not in extents:
            return reject("kernel-extents", f"{c!r} is bound by no placed port")
        return tuple(cast("tuple[int, ...]", divisors(extent=extents[c])))

    return domain(
        accepts=accepts,
        candidates=candidates,
        semantics=default_semantics(int),
        ordered=True,
        rows=rows,
        extents=Kernel.extents,
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
            return reject("threshold-shape", "nonempty sets/rows/thresholds are required")
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
    def rows(self) -> int:
        """C, the table's rows: one a channel, or one shared by every channel."""
        return self.shape[1]

    channels = extent_of(c)  # the input's channels, bound by placement
    pe: int = Decision(domain=lane_counts(rows))  # channels a beat
    # Where a parent places it: the channels it sits on, and the control
    # bus that exports its AXI-Lite interface when thresholds are runtime-writable.
    input_channel: Channel = Param(required=False)
    output_channel: Channel = Param(required=False)
    set_channel: Channel = Param(required=False)
    control: ControlBus = Param(required=False)
    platform: Platform = Param()

    @derived
    def controllable(self) -> bool:
        """Whether a parent placed it on a control bus, which runtime writes reach it by."""
        return self.present(ThresholdingAxiKernel.control)

    # Runtime-writable thresholds need the platform's control port and a control bus
    # to export their AXI-Lite interface through.
    use_axilite: bool = Decision(
        values=(False, True),
        requires=(
            requires(
                platform.control_ports,
                "control-absent: the platform has no control port for runtime-writable thresholds",
                cases=(True,),
            ),
            requires(
                controllable,
                "threshold-control: runtime-writable thresholds need a control bus",
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
        (sets, rows, _), pe = self.shape, self.pe
        folds = 1 if pe >= rows else rows // pe
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
        table, (_, rows, count) = self.thresholds, self.shape
        if any(len(group) != rows or any(len(row) != count for row in group) for group in table):
            return reject(
                "threshold-shape",
                "thresholds must be a rectangular (sets, rows, thresholds) table",
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
        (_, rows, count), pe = self.shape, self.pe
        bits = self.threshold_dtype.bitwidth()
        cf, cpe = max(1, rows // pe), min(rows, pe)
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
            associated_clock=CLOCK,
            associated_reset=RESET,
        )

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def selector_dtype(self) -> QONNXDataType:
        return set_index_dtype(self.shape[0])

    @derived
    def indices(self) -> tuple[Index, ...]:
        """The input's axes: any leading ones, then the channels ``c``, innermost."""
        rank = len(self.input_channel.tensor.shape)
        return (*(Index(f"a{axis}") for axis in range(rank - 1)), c)

    @derived
    def factors(self) -> dict[Index, int]:
        return {c: self.pe}

    @derived
    def schedule(self) -> Schedule | Rejected:
        """Row-major over the input's axes, ``c`` (the input's channels) split by PE
        innermost; the table has one row, or one row a channel."""
        rows, channels = self.rows, self.channels
        if rows not in (1, channels):
            return reject(
                "threshold-rows",
                f"{rows} threshold rows are neither one nor the {channels} channels",
            )
        return self.bound_schedule(self.indices, self.factors)

    @derived
    def frame_cycles(self) -> int:
        """Its schedule's beats, one a cycle at best."""
        return self.schedule.beat_count

    @derived
    def set_sequence(self) -> BeatSequence | Rejected:
        """One set index for each input beat."""
        shape = self.set_channel.tensor.shape
        if self.shape[0] < 2:
            return reject("threshold-set-channel", "a single threshold set takes no set channel")
        if shape != (self.input.presented.form.beats,):
            return reject("threshold-set-channel", "each input beat needs one set index")
        return BeatSequence(vector_major(shape, 1))

    input = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        channel=input_channel,
        schedule=schedule,
        factors=factors,
        index=indices,
        lanes=(c,),
        dtype=input_dtype,
    )
    output = AxiStreamPort(
        name="m_axis",
        endpoint=Endpoint.INITIATOR,
        channel=output_channel,
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
        channel=set_channel,
        sequence=set_sequence,
        dtype=selector_dtype,
    )

    def parameters(self) -> Mapping[str, int | str]:
        table, (sets, rows, count) = self.thresholds, self.shape
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
            "C": rows,
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

    def other_pins(self) -> tuple[Pin, ...]:
        return (self.config_bus,)

    def held(self) -> Held:
        """AXI-Lite, without runtime writes (with them, ``use_axilite`` requires a control
        bus, which presents it)."""
        return Held() if self.use_axilite else held_bus(self.config_bus)

    def controlled(self) -> tuple[Bus, ...]:
        """AXI-Lite, with runtime writes."""
        return (self.config_bus,) if self.use_axilite else ()

    @derived
    def register_map(self) -> RegisterMap:
        """The AXI-Lite writes of its table: each threshold of a row at the wrapper's word
        address (the row's channel fold, its lane, the threshold), its bits in 32-bit words
        low first, the last of which commits it (FinnLib's ``axilite``). Rows are
        channels, ``c = fold * lanes + lane``, ``lanes`` the effective PE (min(C, PE)); a
        shared row (C = 1) is written once and reaches every lane."""
        (_, rows, count), pe = self.shape, self.pe
        bits = self.threshold_dtype.bitwidth()
        lanes, words = min(rows, pe), (bits + 31) // 32
        index_bits, lane_bits = (count - 1).bit_length(), (lanes - 1).bit_length()
        word_bits = (words - 1).bit_length()
        mask = (1 << bits) - 1
        writes = []
        for channel, row in enumerate(self.thresholds[0]):
            fold, lane = divmod(channel, lanes)
            for index, value in enumerate(row):
                address = (fold << (lane_bits + index_bits)) | (lane << index_bits) | index
                for word in range(words):
                    writes.append(
                        (
                            ((address << word_bits) | word) << 2,
                            ((value & mask) >> (32 * word)) & 0xFFFFFFFF,
                        )
                    )
        return RegisterMap(tuple(writes))

    @view
    def control_bus(self) -> Control:
        if not self.controlled():
            return Control(None)
        return Control(self.config_bus, self.register_map)

    exports = {**Kernel.exports, CONTROL: {control: control_bus}}


__all__ = ["ThresholdingAxiKernel"]
