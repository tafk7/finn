############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# The wrapper template is finn-rtllib/memstream/hdl/memstream_wrapper_template.v
# verbatim; the .dat packing reproduces make_weight_file(...,"decoupled_verilog_dat")
# (matrixvectoractivation.py:715-811) as a pure function of (point, context).
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Hermetic RTL codegen for the ``decoupled`` (memstream) storage topology.

Emits the memstream wrapper ``.v`` (parameterized by the composed geometry —
``parameters.depth``/``width``/``sets``/``init_file`` + ``ram_style``/``pumpedMemory``)
and the ``memblock.dat`` weight file the on-chip RAM is initialized from. Both are
pure functions of the resolved point + the frozen context (which carries the weight
VALUES) — no graph, no ``FINN_ROOT``.

The geometry derived are contributed at compose time by the composing op's
cross-coordinate couplings (``ops/mvau/op.py`` §5 — they read the compute fold), so this
emit just reads them off the point. The ``.dat`` packing is the decoupled-verilog-dat path:
transpose ``(1,PE,WMEM,SIMD)`` → PE-flip → hex-pack each ``PE*SIMD`` group to a
4-bit-padded hex word; ``pumpedMemory`` splits each word into two half-width entries.
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.model.artifacts import (
    Artifacts,
    Bool,
    DataFile,
    Dim,
    GeneratedFile,
    Raw,
    RtlModule,
    Template,
    bind,
)
from finn.kernels.emit.manifest import ArtifactManifest, SourceFile
from finn.kernels.model.fold_depth import weight_fold_depth
from finn.kernels.model.ports import Direction, Kind, Port, Role

from .serialize import DAT_HEX, layout, weight_constraint

from finn.kernels.model.param_names import (
    depth_key,
    init_file_key,
    pumped_memory_key,
    ram_style_key,
    runtime_writeable_key,
    sets_key,
    sources_key,
    width_key,
)

from .names import WEIGHTS

# The single source-of-truth for the memstream cell's static finn-rtllib sources (F9). Each
# carries its own real subdir (base:1170,1182-1184) — axilite.sv under axi/hdl, the memstream
# cores under memstream/hdl — centralizing the per-file path resolution that was an inline
# override dict. The decoupled bundle's `sources` reads ``.filenames``; the emit resolves each
# resolved source name against this manifest's path table.
MEMSTREAM_MANIFEST = ArtifactManifest(
    sources=(
        SourceFile("memstream_axi.sv", root="finn-rtllib/memstream/hdl"),
        SourceFile("memstream.sv", root="finn-rtllib/memstream/hdl"),
        SourceFile("axilite.sv", root="finn-rtllib/axi/hdl"),
    ),
)
# filename -> resolved StaticFile, for the point-driven source list the emit reads.
_MANIFEST_BY_NAME = {s.filename: s for s in MEMSTREAM_MANIFEST.sources}


# The parameter tensor's fold-shape helper needs PE/SIMD (off the point) and WMEM (the
# topology-independent fold-depth query). We reuse the compute-side _hw_weight_tensor
# shape (transpose+interleave+reshape), then apply the decoupled PE-flip. Kept local to
# avoid a cross-package emit import.


# finn-rtllib/memstream/hdl/memstream_wrapper_template.v — 7 $UPPER$ slots. Verilog
# $clog2(...) expressions are NOT slots (lowercase) and pass through untouched.
_MEMSTREAM_WRAPPER = Template(
    """\
module $MODULE_NAME$_memstream_wrapper #(
	parameter  SETS = $SETS$,
	parameter  DEPTH = $DEPTH$,
	parameter  WIDTH = $WIDTH$,

	parameter  INIT_FILE = "$INIT_FILE$",
	parameter  RAM_STYLE = "$RAM_STYLE$",
	parameter  PUMPED_MEMORY = $PUMPED_MEMORY$,

	parameter  AXILITE_ADDR_WIDTH = $clog2(SETS * DEPTH * (2**$clog2((WIDTH+31)/32))) + 2,
	parameter  SET_BITS = SETS > 2? $clog2(SETS) : 1
)(
	// Global Control
	(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axilite:s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n" *)
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
	input	ap_clk,
	(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk2x CLK" *)
	input	ap_clk2x,
	(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
	input	ap_rst_n,

	// AXI-lite Write
	output	s_axilite_AWREADY,
	input	s_axilite_AWVALID,
	input	[2:0]  s_axilite_AWPROT,
	input	[AXILITE_ADDR_WIDTH-1:0]  s_axilite_AWADDR,

	output	s_axilite_WREADY,
	input	s_axilite_WVALID,
	input	[31:0]  s_axilite_WDATA,
	input	[ 3:0]  s_axilite_WSTRB,

	input	s_axilite_BREADY,
	output	s_axilite_BVALID,
	output	[1:0]  s_axilite_BRESP,

	// AXI-lite Read
	output	s_axilite_ARREADY,
	input	s_axilite_ARVALID,
	input	[2:0]  s_axilite_ARPROT,
	input	[AXILITE_ADDR_WIDTH-1:0]  s_axilite_ARADDR,

	input	s_axilite_RREADY,
	output	s_axilite_RVALID,
	output	[ 1:0]  s_axilite_RRESP,
	output	[31:0]  s_axilite_RDATA,

	// Set selector stream (ignored for SETS = 1)
	output	s_axis_0_tready,
	input	s_axis_0_tvalid,
	input	[SET_BITS-1:0]  s_axis_0_tdata,

	// Continuous output stream
	input	m_axis_0_tready,
	output	m_axis_0_tvalid,
	output	[((WIDTH+7)/8)*8-1:0]  m_axis_0_tdata
);

	memstream_axi #(
		.SETS(SETS),
		.DEPTH(DEPTH), .WIDTH(WIDTH),
		.INIT_FILE(INIT_FILE),
		.RAM_STYLE(RAM_STYLE),
		.PUMPED_MEMORY(PUMPED_MEMORY)
	) core (
		.clk(ap_clk), .clk2x(ap_clk2x), .rst(!ap_rst_n),

		// AXI-lite Write
		.awready(s_axilite_AWREADY),
		.awvalid(s_axilite_AWVALID),
		.awprot(s_axilite_AWPROT),
		.awaddr(s_axilite_AWADDR),
		.wready(s_axilite_WREADY),
		.wvalid(s_axilite_WVALID),
		.wdata(s_axilite_WDATA),
		.wstrb(s_axilite_WSTRB),
		.bready(s_axilite_BREADY),
		.bvalid(s_axilite_BVALID),
		.bresp(s_axilite_BRESP),

		// AXI-lite Read
		.arready(s_axilite_ARREADY),
		.arvalid(s_axilite_ARVALID),
		.arprot(s_axilite_ARPROT),
		.araddr(s_axilite_ARADDR),
		.rready(s_axilite_RREADY),
		.rvalid(s_axilite_RVALID),
		.rresp(s_axilite_RRESP),
		.rdata(s_axilite_RDATA),

		// Set selector stream (ignored for SETS = 1)
		.s_axis_0_tready(s_axis_0_tready),
		.s_axis_0_tvalid(s_axis_0_tvalid),
		.s_axis_0_tdata(s_axis_0_tdata),

		// Continuous output stream
		.m_axis_0_tready(m_axis_0_tready),
		.m_axis_0_tvalid(m_axis_0_tvalid),
		.m_axis_0_tdata(m_axis_0_tdata)
	);

endmodule // $MODULE_NAME$_memstream_wrapper
"""
)

# The TYPE signature of _MEMSTREAM_WRAPPER (F4) — one type per $SLOT$. SETS/DEPTH/WIDTH
# are geometry Dims; INIT_FILE/RAM_STYLE are free-form Verilog string literals (Raw);
# PUMPED_MEMORY is a Bool flag. bind() type-checks before render — bytes unchanged.
_MEMSTREAM_WRAPPER_SCHEMA = RtlModule(
    "memstream_wrapper",
    {
        "MODULE_NAME": Raw,
        "SETS": Dim,
        "DEPTH": Dim,
        "WIDTH": Dim,
        "INIT_FILE": Raw,
        "RAM_STYLE": Raw,
        "PUMPED_MEMORY": Bool,
    },
)


def emit_memstream(point, context, module_name: str = "mvau_top", iface: str = WEIGHTS) -> Artifacts:
    """Produce the decoupled (memstream) parameter-delivery artifacts from a resolved
    point: the memstream wrapper ``.v`` + the ``memblock.dat`` weight file + the static
    memstream HDL (read off ``parameters.<iface>.sources``). ``iface`` is the parameter
    interface this cell delivers (``weights`` this increment — the only stream-mode
    interface); it selects the interface-namespaced geometry keys."""
    init_file = point[init_file_key(iface)]

    bindings = bind(
        _MEMSTREAM_WRAPPER_SCHEMA,
        {
            "MODULE_NAME": Raw(module_name),
            "SETS": Dim(point[sets_key(iface)]),
            "DEPTH": Dim(point[depth_key(iface)]),
            "WIDTH": Dim(point[width_key(iface)]),
            "INIT_FILE": Raw(init_file),
            "RAM_STYLE": Raw(point[ram_style_key(iface)]),
            "PUMPED_MEMORY": Bool(point.get(pumped_memory_key(iface), 0)),
        },
    )
    wrapper = GeneratedFile(f"{module_name}_memstream_wrapper.v", _MEMSTREAM_WRAPPER, bindings)

    # The weight .dat is emitted only when the RAM is initialized from a file (i.e.
    # not the URAM-non-Versal case, where init_file is blank and weights load via
    # AXI-lite). base:330-332.
    data_files = ()
    if init_file:
        data_files = (DataFile("memblock.dat", _memblock_dat(point, context, iface)),)

    # Static memstream HDL — the selected topology's resolved source names, each mapped to
    # its real finn-rtllib subdir via the ONE manifest (axilite.sv under axi/hdl, the
    # memstream cores under memstream/hdl; base:1182-1184). One source-of-truth, so the
    # bundle's `sources` and this build copy cannot drift.
    static = tuple(
        _MANIFEST_BY_NAME[s].as_static() for s in point.get(sources_key(iface), ())
    )

    # The delivery cell publishes a WEIGHT_SOURCE (m_axis_0) — the resolver binds it to
    # the compute cell's WEIGHT_SINK (in1_V) by role, matching width. Its width is the
    # same padded PE*SIMD*wbits geometry (parameters.width). CONFIG (s_axilite) is
    # present only when weights are runtime-writable; it exports up as a boundary
    # register surface. The set-selector stream (s_axis_0, INDEX_SINK) exists in RTL but
    # is inert for SETS=1 (single-cardinality) — declared documented, not bound here.
    width = point[width_key(iface)]
    ports = [
        Port(Direction.OUT, Kind.AXIS, Role.WEIGHT_SOURCE, "m_axis_0", index=0,
             width=width),
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk"),
        # ap_clk2x: non-pumped memory ties it to ap_clk (fans from the region ap_clk in
        # the broadcast); a pumped-memory design binds it to a distinct 2x-clock port
        # (documented extension — the built path is non-pumped).
        Port(Direction.IN, Kind.CLOCK, Role.CLOCK, "ap_clk2x"),
        Port(Direction.IN, Kind.RESET, Role.RESET, "ap_rst_n"),
    ]
    if point.get(runtime_writeable_key(iface), 0):
        ports.append(
            Port(Direction.IN, Kind.AXILITE, Role.CONFIG, "s_axilite", index=0,
                 boundary=True)
        )

    # No per-emit IPICommands: emit_composed builds the authoritative IPI from stitch
    # over the declared ports and SUBSTITUTES it (compose_emit.py) — the per-emit
    # instantiation line was dead output. stitch is the sole IPI source (F5).
    return Artifacts(
        generated=(wrapper,),
        data_files=data_files,
        static_files=static,
        ports=tuple(ports),
    )


# ------------------------------------------------------------- pure helpers


def _memblock_dat(point, context, iface: str = WEIGHTS) -> str:
    """make_weight_file "decoupled_verilog_dat" (matrixvectoractivation.py:715-811) via the
    shared ``layout`` (DAT_HEX form + decoupled PE-flip + optional pumped split). The Part-1
    reshape is the SAME one the embedded ``params.h`` path calls — one serializer, so the two
    delivery forms cannot diverge. Pure over (point, context)."""
    weights = np.asarray(context.initializer(WEIGHTS))
    wdt = context.tensor_datatype(WEIGHTS)
    export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
    # WMEM from the topology-independent fold-depth query (F1) — the same geometry the
    # wrapper's DEPTH slot traces to via depth_key, no longer the op's point.WMEM alias.
    wmem = weight_fold_depth(point, context, iface)

    constraint = weight_constraint(
        point.PE,
        point.SIMD,
        wmem,
        wdt,
        export_wdt,
        form=DAT_HEX,
        decoupled_pe_flip=True,
        pumped_split=bool(point.get(pumped_memory_key(iface), 0)),
    )
    return layout(weights, constraint).text
