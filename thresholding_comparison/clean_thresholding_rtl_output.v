// Clean RTL backend: CG_Thresholding_rtl
// Template-based code generation

// Template values keys: ['module_name', 'n', 'wi', 'wt', 'c', 'pe', 'signed', 'fparg', 'bias', 'thresholds_path', 'use_axilite', 'depth_trigger_uram', 'depth_trigger_bram', 'deep_pipeline', 'o_bits', 'input_stream_width', 'output_stream_width', 'axilite_addr_width']

// Template values debug:
// module_name: '_wrapper'
// n: 2
// wi: 8
// wt: 8
// c: 4
// pe: 2
// signed: 1
// fparg: 0
// bias: 0
// thresholds_path: '"./_"'
// use_axilite: 0
// depth_trigger_uram: 0
// depth_trigger_bram: 0
// deep_pipeline: 1
// o_bits: 2
// input_stream_width: 16
// output_stream_width: 8
// axilite_addr_width: 0

// === Rendered wrapper.v ===
// Thresholding RTL Template for FINN
// Simple Verilog wrapper for thresholding operations
// Direct parameter substitution - no complex macros

module _wrapper #(
    // Core parameters
    parameter  N   = 2,          // output precision  
    parameter  WI  = 8,         // input precision
    parameter  WT  = 8,         // threshold precision
    parameter  C   = 4,          // channels
    parameter  PE  = 2,         // processing parallelism
    
    // Operation parameters
    parameter  SIGNED = 1,  // signed inputs
    parameter  FPARG  = 0,   // floating-point inputs
    parameter  BIAS   = 0,    // output offset
    
    // Memory configuration
    parameter  THRESHOLDS_PATH = ""./_"",
    parameter  USE_AXILITE = 0,
    parameter  DEPTH_TRIGGER_URAM = 0,
    parameter  DEPTH_TRIGGER_BRAM = 0,
    parameter  DEEP_PIPELINE = 1,
    
    // Derived parameters
    parameter  O_BITS = 2
)(
    // Global Control
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF in0_V:out0_V, ASSOCIATED_RESET ap_rst_n" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
    input  ap_clk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  ap_rst_n,
    
    
    // AXI Stream - Input
    output  in0_V_TREADY,
    input   in0_V_TVALID,
    input   [15:0] in0_V_TDATA,
    
    // AXI Stream - Output  
    input   out0_V_TREADY,
    output  out0_V_TVALID,
    output  [7:0] out0_V_TDATA
);

    // Instantiate the thresholding core
    thresholding_axi #(
        .N(N), .WI(WI), .WT(WT), .C(C), .PE(PE),
        .SIGNED(SIGNED),
        .FPARG(FPARG),
        .BIAS(BIAS),
        .THRESHOLDS_PATH(THRESHOLDS_PATH),
        .USE_AXILITE(USE_AXILITE),
        .DEPTH_TRIGGER_URAM(DEPTH_TRIGGER_URAM),
        .DEPTH_TRIGGER_BRAM(DEPTH_TRIGGER_BRAM),
        .DEEP_PIPELINE(DEEP_PIPELINE)
    ) core (
        .ap_clk(ap_clk), 
        .ap_rst_n(ap_rst_n),
        
        
        // AXI Stream connections
        .s_axis_tready(in0_V_TREADY), 
        .s_axis_tvalid(in0_V_TVALID), 
        .s_axis_tdata(in0_V_TDATA),
        .m_axis_tready(out0_V_TREADY), 
        .m_axis_tvalid(out0_V_TVALID), 
        .m_axis_tdata(out0_V_TDATA)
    );

endmodule