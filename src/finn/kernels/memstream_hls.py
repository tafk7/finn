# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib memstream with an explicit HLS memory/configuration top.

The top exposes resident memory through AXI-Lite and emits one scalar per AXI
stream beat. Its ap_ctrl_hs registers share the control bundle; the caller must
start/auto-restart the kernel for continuous delivery. Memory contents are
runtime configuration, not C++ constants or implicitly initialized zeros.

The physical View accepts HlsSourceRequirements, not a predicted RTL pin ABI.
The first datatype profile covers ordinary integers and IEEE FLOAT32.
"""

from finn.kernels.artifacts.contribution_types import CopiedSource, RenderedSource
from finn.kernels.artifacts.hls import HlsInterface, HlsSourceRequirements
from finn.kernels.artifacts.sources import CompileOptions, Language, Role
from finn.kernels.base import Kernel
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import DatatypeError, QONNXDataType, ordinary_integer_bounds
from finn.kernels.space import ConstraintGroup, Input, Readiness, View, constraint, derived, reject


class MemStreamHlsKernel(Kernel):
    id = "finnlib.memstream.hls"
    version = "1"

    element_dtype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    depth = Input(int)

    @derived(str, dtype=element_dtype)
    def cpp_type(*, dtype: QONNXDataType) -> object:
        if dtype.name == "FLOAT32":
            return "float"
        try:
            ordinary_integer_bounds(dtype)
        except DatatypeError as error:
            return reject("memstream-hls-type", str(error))
        if not 1 <= dtype.bitwidth() <= 1024:
            return reject(
                "memstream-hls-width", "the default HLS ap_int profile supports 1 through 1024 bits"
            )
        return f"{'ap_int' if dtype.signed() else 'ap_uint'}<{dtype.bitwidth()}>"

    @constraint(depth=depth)
    def depth_supported(*, depth: int) -> object:
        # memstream.hpp uses ap_uint<clog2(N)> for its pointer: N=1 is zero bits.
        if not 2 <= depth <= 0xFFFFFFFF:
            return reject(
                "memstream-hls-depth", "native memstream requires depth >= 2 fitting unsigned int"
            )
        return True

    @derived(HlsSourceRequirements, cpp=cpp_type, depth=depth)
    def codegen(*, cpp: str, depth: int) -> HlsSourceRequirements:
        includes = ("hls/util", "hls/infra")
        return HlsSourceRequirements(
            MemStreamHlsKernel.id,
            MemStreamHlsKernel.version,
            "memstream_hls",
            (
                HlsInterface("mem", cpp, (depth,), "s_axilite", "control"),
                HlsInterface("dst", cpp, (), "axis"),
            ),
            "ap_ctrl_hs in s_axilite control; enable auto-restart for continuous streaming",
            (
                CopiedSource(
                    "finnlib",
                    "hls/util/util.hpp",
                    language=Language.CPP,
                    role=Role.HEADER,
                    provides=("header:util.hpp",),
                ),
                CopiedSource(
                    "finnlib",
                    "hls/infra/memstream.hpp",
                    language=Language.CPP,
                    role=Role.HEADER,
                    provides=("header:memstream.hpp",),
                    requires=("header:util.hpp",),
                ),
                RenderedSource(
                    "memstream_hls.cpp",
                    "memstream_hls.cpp.j2",
                    language=Language.CPP,
                    standard="c++17",
                    options=CompileOptions(includes=includes),
                    provides=("function:memstream_hls",),
                    requires=("header:memstream.hpp",),
                ),
            ),
            (("CPP_TYPE", cpp), ("DEPTH", depth)),
            includes,
        )

    support = ConstraintGroup(depth_supported)
    physical_ready = Readiness()
    physical = View(codegen, readiness=physical_ready, constraints=support)


__all__ = ["MemStreamHlsKernel"]
