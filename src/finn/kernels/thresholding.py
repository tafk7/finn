# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integer profile of FinnLib thresholding_axi, including resident parameter sets.

thresholds[set][channel][threshold] is an immutable, sorted table of numerical
integers. Its shape owns SETS, C and N. The input is sign/zero extended or
saturated to the threshold dtype before comparison, as the native RTL specifies.
Output is the threshold count plus bias. Runtime writes must preserve sorted
rows. With multiple sets, each input beat requires a matching set-selector beat.

All native pins remain present when AXI-Lite or set selection is disabled;
disabled outputs may be unspecified. Multi-set AXI-Lite access is refused: the
pinned wrapper's configuration address width omits set bits. Static multi-set
selection remains supported. Floating-point threshold comparison is outside
this first profile; Eltwise and IntToFp32 exercise float authoring separately.
Biases below -N-1 are refused: the native unsigned width expression creates a
33-bit output, but the result addition zero-extends the negative 32-bit bias.
"""

from __future__ import annotations

from finn.kernels.base import Kernel
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Direction,
    Endpoint,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.derivation import Scalar as BuildScalar
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    ScalarTable,
)
from finn.kernels.datatypes.semantics import (
    ThresholdTable,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    THRESHOLD_TABLE,
)
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.scalar import integer_scalar
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.physical.axi_stream import AxiStream
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


class ThresholdingAxiKernel(Kernel):
    id = "finnlib.thresholding_axi.integer"
    version = "1"

    input_dtype: Param[QONNXDataType] = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    threshold_dtype: Param[QONNXDataType] = Param(QONNX_DATATYPE_VALUE_SEMANTICS)
    input_encoding = integer_scalar(input_dtype, Integer())
    threshold_encoding = integer_scalar(threshold_dtype, Integer())
    thresholds: Param[ThresholdTable] = Param(THRESHOLD_TABLE)
    bias: Param[int] = Param(int)

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

    pe: Param[int] = Param(int)
    use_axilite = Decision(bool, values=(False, True))
    deep_pipeline = Decision(bool, values=(False, True))
    depth_trigger_bram: Param[int] = Param(int)
    depth_trigger_uram: Param[int] = Param(int)

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

    implementation_supported = ConstraintGroup(
        types_supported,
        table_supported,
        folding_supported,
        memory_supported,
        bias_supported,
        configuration_supported,
    )

    @view(
        semantics=default_semantics(ModuleBuildRequirements),
        requires=(implementation_supported,),
    )
    def build_requirements(self) -> ModuleBuildRequirements | Rejected:
        table = self.thresholds
        pe = self.pe
        a = self.input_encoding.encoding().dtype
        t = self.threshold_encoding.encoding().dtype
        result = self.result_dtype
        bias = self.bias
        axilite = self.use_axilite
        deep = self.deep_pipeline
        bram = self.depth_trigger_bram
        uram = self.depth_trigger_uram
        if pe < 1:
            return reject("threshold-interface", "PE must be positive")
        sets, channels, count = len(table), len(table[0]), len(table[0][0])
        bits = t.bitwidth()
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
        parameter_values: dict[str, BuildScalar] = {
            "WI": a.bitwidth(),
            "WT": bits,
            "N": count,
            "C": channels,
            "PE": pe,
            "SIGNED": int(a.signed()),
            "FPARG": 0,
            "BIAS": bias,
            "SETS": sets,
            "THRESHOLDS": image,
            "THRESHOLDS_FILE": '""',
            "USE_AXILITE": int(axilite),
            "DEPTH_TRIGGER_BRAM": bram,
            "DEPTH_TRIGGER_URAM": uram,
            "DEEP_PIPELINE": int(deep),
        }
        parameters: ScalarTable = tuple(sorted(parameter_values.items()))
        cf, cpe = max(1, channels // pe), min(channels, pe)
        address_bits = (
            sum((value - 1).bit_length() for value in (cf, cpe, count, (bits + 31) // 32)) + 2
        )
        selector_bits = (sets - 1).bit_length() if sets > 2 else 1
        config = Bus(
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
        streams = (
            AxiStream("s_axis", a, pe, endpoint=Endpoint.TARGET),
            AxiStream("m_axis", result, pe, endpoint=Endpoint.INITIATOR),
            AxiStream(
                "s_axis_set",
                resolve_qonnx_datatype_name(f"UINT{selector_bits}"),
                1,
                endpoint=Endpoint.TARGET,
            ),
        )
        abi = ModuleABIRequirements(
            FixedModuleName("thresholding_axi"),
            (
                Signal("ap_clk", Direction.IN, 1, Clock()),
                Signal(
                    "ap_rst_n",
                    Direction.IN,
                    1,
                    Reset(active_low=True, synchronous=True, synchronous_to=("ap_clk",)),
                ),
                config,
                *(stream.bus(clock="ap_clk", reset="ap_rst_n") for stream in streams),
            ),
            tuple((key, str(value)) for key, value in parameters),
        )
        sources = (
            CopiedSource("finnlib", "rtl/axilite.sv", provides=("module:axilite",)),
            CopiedSource("finnlib", "rtl/thresholding.sv", provides=("module:thresholding",)),
            CopiedSource(
                "finnlib",
                "rtl/thresholding_axi.sv",
                provides=("module:thresholding_axi",),
                requires=("module:axilite", "module:thresholding"),
            ),
        )
        return ModuleBuildRequirements(
            ThresholdingAxiKernel.id, ThresholdingAxiKernel.version, parameters, abi, sources
        )


__all__ = ["ThresholdingAxiKernel"]
