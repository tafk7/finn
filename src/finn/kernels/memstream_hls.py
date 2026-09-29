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

from __future__ import annotations


from finn.kernels.artifacts.contribution_types import CopiedSource, RenderedSource
from finn.kernels.artifacts.hls import HlsInterface, HlsSourceRequirements
from finn.kernels.artifacts.sources import CompileOptions, Language, Role
from finn.kernels.base import Kernel
from finn.dataflow.datatypes import QONNXDataType
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.domains import Integer
from finn.core.space import (
    Param,
    Rejected,
    constraint,
    derived,
    reject,
    view,
)


class MemStreamHlsKernel(Kernel):
    id = "finnlib.memstream.hls"
    version = "1"
    # Off the Kernel protocol until K2: it exports nothing of its own.
    exports = {}

    element_dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)

    @derived
    def cpp_type(self) -> str | Rejected:
        dtype = self.element_dtype
        if dtype.name == "FLOAT32":
            return "float"
        admitted = Integer(1, 1024).check(dtype)
        if isinstance(admitted, Rejected):
            return admitted
        return f"{'ap_int' if dtype.signed() else 'ap_uint'}<{dtype.bitwidth()}>"

    depth: int = Param()

    @constraint
    def depth_supported(self) -> bool | Rejected:
        # memstream.hpp uses ap_uint<clog2(N)> for its pointer: N=1 is zero bits.
        depth = self.depth
        if not 2 <= depth <= 0xFFFFFFFF:
            return reject(
                "memstream-hls-depth", "native memstream requires depth >= 2 fitting unsigned int"
            )
        return True

    @view(requires=(depth_supported,))
    def sources(self) -> HlsSourceRequirements:  # type: ignore[override]
        cpp = self.cpp_type
        depth = self.depth
        includes = ("hls/infra", "hls/util")
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


__all__ = ["MemStreamHlsKernel"]
