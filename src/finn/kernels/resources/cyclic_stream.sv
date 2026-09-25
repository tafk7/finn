// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

// Read-only cyclic word source. INIT_DATA[W-1:0] is the first word.
// Reset is synchronous, active-high, and restarts delivery at word zero.
// Output is registered and held while valid && !ready. No framing is emitted.
// ROM_STYLE is "auto" (tool inference), "distributed" or "block"; the latter
// two are passed to synthesis as the rom_style attribute of the image.
`default_nettype none
module cyclic_stream #(
    parameter int unsigned W,
    parameter int unsigned DEPTH,
    parameter logic [W*DEPTH-1:0] INIT_DATA,
    parameter ROM_STYLE
)(
    input  wire clk,
    input  wire rst,
    output logic [W-1:0] odat,
    output logic ovld,
    input  wire ordy
);
    localparam int unsigned AW = DEPTH < 2 ? 1 : $clog2(DEPTH);
    logic [AW-1:0] ReadPtr;

    initial begin
        if (ROM_STYLE != "auto" && ROM_STYLE != "distributed" && ROM_STYLE != "block")
            $error("%m: unsupported ROM_STYLE %s", ROM_STYLE);
    end

    // A synchronous ROM read with an enabled output register. Keeping reset
    // out of this data path allows synthesis to infer memory resources.
    if (ROM_STYLE == "auto") begin : genRom
        logic [W-1:0] Memory [0:DEPTH-1];
        initial begin
            for (int i = 0; i < DEPTH; i++)
                Memory[i] = INIT_DATA[i*W +: W];
        end
        always_ff @(posedge clk) begin
            if (!ovld || ordy)
                odat <= Memory[ReadPtr];
        end
    end : genRom
    else begin : genRom
        (* rom_style = ROM_STYLE *)
        logic [W-1:0] Memory [0:DEPTH-1];
        initial begin
            for (int i = 0; i < DEPTH; i++)
                Memory[i] = INIT_DATA[i*W +: W];
        end
        always_ff @(posedge clk) begin
            if (!ovld || ordy)
                odat <= Memory[ReadPtr];
        end
    end : genRom

    always_ff @(posedge clk) begin
        if (rst) begin
            ReadPtr <= '0;
            ovld <= 1'b0;
        end else if (!ovld || ordy) begin
            ReadPtr <= ReadPtr == AW'(DEPTH-1) ? '0 : ReadPtr + 1'b1;
            ovld <= 1'b1;
        end
    end
endmodule
`default_nettype wire
