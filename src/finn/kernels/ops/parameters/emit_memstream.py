############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# The wrapper template is finn-rtllib/memstream/hdl/memstream_wrapper_template.v
# verbatim; the .dat packing reproduces make_weight_file(...,"decoupled_verilog_dat")
# (matrixvectoractivation.py:715-811) as a pure function of (point, context).
#
# SPDX-License-Identifier: MIT
############################################################################

"""Hermetic RTL codegen for the ``decoupled`` (memstream) storage topology.

Emits the memstream wrapper ``.v`` (parameterized by the composed geometry —
``parameters.depth``/``width``/``sets``/``init_file`` + ``ram_style``/``pumpedMemory``)
and the ``memblock.dat`` weight file the on-chip RAM is initialized from. Both are
pure functions of the resolved point + the frozen context (which carries the weight
VALUES) — no graph, no ``FINN_ROOT``.

The geometry derived are contributed at compose time by
``fixtures/mvau/parameters_coupling.py`` (they read the compute fold), so this emit
just reads them off the point. The ``.dat`` packing is the decoupled-verilog-dat path:
transpose ``(1,PE,WMEM,SIMD)`` → PE-flip → hex-pack each ``PE*SIMD`` group to a
4-bit-padded hex word; ``pumpedMemory`` splits each word into two half-width entries.
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.kernels.space import (
    Artifacts,
    DataFile,
    Direction,
    GeneratedFile,
    IPICommands,
    Kind,
    Port,
    Role,
    StaticFile,
    Template,
)
from finn.util.data_packing import pack_innermost_dim_as_hex_string

from .names import (
    PARAM_DEPTH,
    PARAM_INIT_FILE,
    PARAM_SETS,
    PARAM_WIDTH,
    PUMPED_MEMORY,
    RAM_STYLE,
    RUNTIME_WRITEABLE,
    WEIGHTS,
)

# Real finn-rtllib subdir per static source (base:1170,1182-1184). Anything not listed
# defaults to memstream/hdl (the memstream cores' home).
_SOURCE_DIRS = {"axilite.sv": "finn-rtllib/axi/hdl"}


# The parameter tensor's fold-shape helper needs PE/SIMD/WMEM — read off the point.
# We reuse the compute-side _hw_weight_tensor shape (transpose+interleave+reshape),
# then apply the decoupled PE-flip. Kept local to avoid a cross-package emit import.


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


def emit_memstream(point, context, module_name: str = "mvau_top") -> Artifacts:
    """Produce the decoupled (memstream) weight-delivery artifacts from a resolved
    point: the memstream wrapper ``.v`` + the ``memblock.dat`` weight file + the
    static memstream HDL (read off ``parameters.sources``)."""
    init_file = point[PARAM_INIT_FILE]

    bindings = {
        "MODULE_NAME": module_name,
        "SETS": point[PARAM_SETS],
        "DEPTH": point[PARAM_DEPTH],
        "WIDTH": point[PARAM_WIDTH],
        "INIT_FILE": init_file,
        "RAM_STYLE": point[RAM_STYLE],
        "PUMPED_MEMORY": int(point.get(PUMPED_MEMORY, 0)),
    }
    wrapper = GeneratedFile(f"{module_name}_memstream_wrapper.v", _MEMSTREAM_WRAPPER, bindings)

    # The weight .dat is emitted only when the RAM is initialized from a file (i.e.
    # not the URAM-non-Versal case, where init_file is blank and weights load via
    # AXI-lite). base:330-332.
    data_files = ()
    if init_file:
        data_files = (DataFile("memblock.dat", _memblock_dat(point, context)),)

    # Static memstream HDL — read straight off the selected topology's sources, each
    # resolved to its real finn-rtllib subdir. axilite.sv lives under axi/hdl/, the
    # memstream cores under memstream/hdl/ (matches FINN base:1182-1184); a bare
    # single-dir prefix would mis-locate axilite.sv.
    static = tuple(
        StaticFile("finn.data", f"{_SOURCE_DIRS.get(s, 'finn-rtllib/memstream/hdl')}/{s}")
        for s in point.get("parameters.sources", ())
    )

    # The delivery cell publishes a WEIGHT_SOURCE (m_axis_0) — the resolver binds it to
    # the compute cell's WEIGHT_SINK (in1_V) by role, matching width. Its width is the
    # same padded PE*SIMD*wbits geometry (parameters.width). CONFIG (s_axilite) is
    # present only when weights are runtime-writable; it exports up as a boundary
    # register surface. The set-selector stream (s_axis_0, INDEX_SINK) exists in RTL but
    # is inert for SETS=1 (single-cardinality) — declared documented, not bound here.
    width = point[PARAM_WIDTH]
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
    if point.get(RUNTIME_WRITEABLE, 0):
        ports.append(
            Port(Direction.IN, Kind.AXILITE, Role.CONFIG, "s_axilite", index=0,
                 boundary=True)
        )

    return Artifacts(
        generated=(wrapper,),
        data_files=data_files,
        static_files=static,
        ports=tuple(ports),
        ipi=IPICommands(
            (
                f"create_bd_cell -type hier -reference {module_name}_memstream_wrapper "
                f"{module_name}/{module_name}_wstrm",
            )
        ),
    )


# ------------------------------------------------------------- pure helpers


def _memblock_dat(point, context) -> str:
    """make_weight_file "decoupled_verilog_dat" (matrixvectoractivation.py:715-811):
    transpose (1,PE,WMEM,SIMD) → PE-flip → reshape (1,-1,PE*SIMD) → hex-pack each group
    to a 4-bit-padded hex word; pumpedMemory splits each word into two half-width
    entries. Pure over (point, context)."""
    from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

    weights = np.asarray(context.initializer(WEIGHTS))
    wdt = context.tensor_datatype(WEIGHTS)
    export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
    pe, simd, wmem = point.PE, point.SIMD, point.WMEM

    # get_hw_compatible_weight_tensor → (1, PE, WMEM, SIMD).
    ret = weights.T
    if wdt == DataType["BIPOLAR"]:
        ret = (ret + 1) / 2
    ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
    weight_tensor = ret.reshape(1, pe, wmem, simd)
    weight_tensor = np.flip(weight_tensor, axis=-1)  # SIMD flip (hw layout)

    # decoupled: transpose to (1, WMEM, PE, SIMD), then PE-flip, reshape (1,-1,PE*SIMD).
    unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
    pe_flipped = np.flip(unflipped, axis=-2)
    pe_flipped = pe_flipped.reshape(1, -1, pe * simd).copy()
    # TH=1: the TH-untile flip is a no-op, elided.

    # hex-pack each PE*SIMD group at 4-bit-padded width (base:783-790).
    weight_width = pe * simd * export_wdt.bitwidth()
    weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
    packed = pack_innermost_dim_as_hex_string(
        pe_flipped, export_wdt, weight_width_padded, prefix=""
    )
    weight_stream = packed.flatten().copy()

    if point.get(PUMPED_MEMORY, 0):
        # split each hex word into two half-width entries (low half first). base:801-808.
        split = []
        for w in weight_stream:
            split.append(w[len(w) // 2:])
            split.append(w[: len(w) // 2])
        weight_stream = split

    return "".join(str(v) + "\n" for v in weight_stream)
