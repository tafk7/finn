/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Dot Product Unit (DOTP) AXI-lite interface wrapper.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @author	Nikolaos Kavvadias <nikolaos.kavvadias@amd.com>
 * @details
 *  The following compute cores are supported:
 *	 - 4-bit DOTP on DSP48 achieving 4 MACs/DSP,
 *	 - (4,8]-bit DOTP on DSP48 achieving 2 MACs/DSP,
 *	 - [4,9]-bit DOTP on DSP58 achieving 3 MACs/DSP,
 *  Folding:
 *	 - PE scaling determines the number of outputs worked on in parallel.
 *	 - SIMD scaling determines the number of input products ingested in parallel.
 ***************************************************************************/

// Local copy of FinnLib rtl/linalg/dotp_axi.sv at
// dfeafac81cd2a6da27e647ee03915ade5532186e.
// Upstream SHA-256: 3b574c5c9c59f1dceb82e98596fd96b9bdcc9e5b33313ae189e5d7053024ac7d.
// Local correction: bound output buffering for both compute backends and reserve
// the extra frame accepted while registered backpressure takes effect. No core,
// arithmetic, port or parameter behavior is changed.

module dotp_axi #(
	int unsigned  VERSION = 1,	// Allowed versions - 1: DSP48E1, 2: DSP48E2, 3: DSP58
	bit ACTIVATION_BROADCASTING,

	int unsigned PE,
	int unsigned SIMD,
	int unsigned SEGMENTLEN = 0,

	int unsigned ACTIVATION_WIDTH,
	int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH,
	bit NARROW_WEIGHTS     = 0,
	bit SIGNED_ACTIVATIONS = 0,

	bit PUMPED_COMPUTE = 0,
	bit FORCE_BEHAVIORAL = 0,

	// Safely deducible parameters
	localparam int unsigned  WEIGHT_STREAM_WIDTH    = PE * SIMD * WEIGHT_WIDTH,
	localparam int unsigned  WEIGHT_STREAM_WIDTH_BA = (WEIGHT_STREAM_WIDTH + 7)/8 * 8,
	localparam int unsigned  INPUT_STREAM_WIDTH     = (ACTIVATION_BROADCASTING ? 1 : PE) * SIMD * ACTIVATION_WIDTH,
	localparam int unsigned  INPUT_STREAM_WIDTH_BA  = (INPUT_STREAM_WIDTH  + 7)/8 * 8,
	localparam int unsigned  OUTPUT_STREAM_WIDTH    = PE*ACCU_WIDTH,
	localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUTPUT_STREAM_WIDTH + 7)/8 * 8,
	localparam bit  		 SIMD_UNEVEN = SIMD % 2
)(
	// Global Control
	input	logic  ap_clk,
	input	logic  ap_clk2x,	// synchronous, double-speed clock; only used for PUMPED_COMPUTE
	input	logic  ap_rst_n,

	// Weight Stream
	input	logic [WEIGHT_STREAM_WIDTH_BA-1:0]  s_axis_weights_tdata,
	input	logic  s_axis_weights_tvalid,
	output	logic  s_axis_weights_tready,

	// Input Stream
	input	logic [INPUT_STREAM_WIDTH_BA-1:0]  s_axis_input_tdata,
	input	logic  s_axis_input_tvalid,
	input	logic  s_axis_input_tlast,
	output	logic  s_axis_input_tready,

	// Output Stream
	output	logic [OUTPUT_STREAM_WIDTH_BA-1:0]  m_axis_output_tdata,
	output	logic  m_axis_output_tvalid,
	input	logic  m_axis_output_tready
);
`default_nettype none

//-------------------- Parameter sanity checks --------------------\\
	initial begin
		if(VERSION == 3) begin
			if(SEGMENTLEN == 0) begin
				$warning("Segment length of %0d defaults to chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
			end
			if(SEGMENTLEN > (SIMD+2)/3) begin
				$error("Segment length of %0d exceeds chain length of %0d", SEGMENTLEN, (SIMD+2)/3);
				$finish;
			end
		end
		if(!ACTIVATION_BROADCASTING) begin
			if(VERSION != 3) begin
				$error("VVU only supported on DSP58-based implementation");
				$finish;
			end
		end
	end

	uwire  rst = !ap_rst_n;

	//- Replay to Accommodate Neuron Fold -----------------------------------
	typedef logic [(ACTIVATION_BROADCASTING? 1:PE)*SIMD-1:0][ACTIVATION_WIDTH-1:0]  dotp_flatin_t;
	uwire dotp_flatin_t amvau = dotp_flatin_t'(s_axis_input_tdata);
	uwire alast = s_axis_input_tlast;
	uwire avld = s_axis_input_tvalid;
	uwire ardy;

	assign 	s_axis_input_tready = ardy;

	//- Unflatten inputs into structured matrices ---------------------------
	localparam int unsigned  ACT_PE = ACTIVATION_BROADCASTING? 1 : PE;
	typedef logic [PE    -1:0][SIMD-1:0][WEIGHT_WIDTH    -1:0]  dotp_w_t;
	typedef logic [ACT_PE-1:0][SIMD-1:0][ACTIVATION_WIDTH-1:0]  dotp_a_t;

	uwire  dotp_w_t  dotp_w = s_axis_weights_tdata;

	//- Conditional Activations Layout Adjustment for VVU
	uwire dotp_a_t  amvau_i;
	if (ACTIVATION_BROADCASTING || (PE == 1)) begin : genDOTPInput
		assign  amvau_i = amvau;
	end : genDOTPInput
	else begin : genVVUInput
		// The input stream will have the channels interleaved for VVU when PE>1
		// Hence, we need to 'untangle' the input stream, i.e. [..][SIMD*PE][..] --> [..][PE][SIMD][..]
		// Note that for each 'SIMD' (S) and 'PE' (P) element, we have something like:
		// (S_0, P_0), ..., (S_0, P_i), (S_1, P_0), ..., (S_1, P_i), ..., (S_i, P_i) which we need to 'untangle' to
		// (S_0, P_0), ..., (S_i, P_0), (S_0, P_1), ..., (S_i, P_1), ..., (S_i, P_i)
		for(genvar  pe = 0; pe < ACT_PE; pe++) begin
			for(genvar  simd = 0; simd < SIMD; simd++) begin
				assign	amvau_i[pe][simd] = amvau[simd*ACT_PE+pe];
			end
		end
	end : genVVUInput

	//- Flow Control Bracket around Compute Core ----------------------------
	uwire  idle;
	assign	ardy = !idle && s_axis_weights_tvalid;
	assign	s_axis_weights_tready = !idle && avld;

	//- Conditionally Pumped DSP Compute ------------------------------------
	typedef logic [PE-1:0][ACCU_WIDTH-1:0]  dsp_p_t;
	uwire  ovld;
	uwire dsp_p_t  odat;
	if(1) begin : blkDsp
		localparam int unsigned  EFFECTIVE_SIMD = SIMD_UNEVEN && PUMPED_COMPUTE ? SIMD+1 : SIMD;
		localparam int unsigned  DSP_SIMD = EFFECTIVE_SIMD/(PUMPED_COMPUTE+1);
		typedef logic [PE    -1:0][DSP_SIMD-1:0][WEIGHT_WIDTH    -1:0]  dsp_w_t;
		typedef logic [ACT_PE-1:0][DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  dsp_a_t;

		uwire  dsp_last;
		uwire  dsp_zero;
		uwire dsp_w_t  dsp_w;
		uwire dsp_a_t  dsp_a;

		uwire  dsp_vld;
		uwire dsp_p_t  dsp_p;

		if(!PUMPED_COMPUTE) begin : genUnpumpedCompute
			assign	dsp_zero = idle || !s_axis_weights_tvalid || !avld;
			assign	dsp_last = alast && !dsp_zero;
			assign	dsp_w = dotp_w;
			assign	dsp_a = amvau_i;

			assign	ovld = dsp_vld;
			assign	odat = dsp_p;
		end : genUnpumpedCompute
		else begin : genPumpedCompute

			// The input for a slow cycle is split across two fast cycles along the SIMD dimension.
			//	- Both fast cycles are controlled by the same enable state.
			//	- A zero cycle is duplicated across both fast cycles.
			//	- The last flag must be restricted to the second fast cycle.
			//                ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
			//  clk         ──┘     └─────┘     └─────┘     └─────┘     └─────┘     └
			//              ──┐
			//  rst           └──────────────────────────────────────────────────────
			//                ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌
			//  ap_clk2x    ──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘
			//                      ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌
			//  Active      ────────┘     └─────┘     └─────┘     └─────┘     └─────┘
			//
			//              ─────────────\ /─────────\ /─────────\ /─────────\ /─────
			//  Data (slow)    XXX        X {Hi0,Lo0} X {Hi1,Lo1} X {Hi2,Lo2} X {Hi3,
			//              ─────────────/ \─────────/ \─────────/ \─────────/ \─────
			//                                        ┌───────────┐
			//  Last (slow) ──────────────────────────┘           └──────────────────
			//              ───────────────────\ /───\ /───\ /───\ /───\ /───\ /───\
			//  Data (fast)    XXX              X Lo0 X Hi0 X Lo1 X Hi1 X Lo2 X Hi2 X
			//              ───────────────────/ \───/ \───/ \───/ \───/ \───/ \───/
			//                                                    ┌─────┐
			//  Last (fast) ──────────────────────────────────────┘     └────────────

			// Identify second fast cycle just before active slow clock edge
			logic  Active = 0;
			always_ff @(posedge ap_clk2x) begin
				if(rst)  Active <= 0;
				else     Active <= !Active;
			end

			dsp_w_t  W = 'x;
			for(genvar  pe = 0; pe < PE; pe++) begin : genPERegW

				uwire [2*DSP_SIMD-1:0][WEIGHT_WIDTH-1:0]  w;
				for(genvar  i =    0; i <       SIMD; i++)  assign  w[i] = dotp_w[pe][i];
				for(genvar  i = SIMD; i < 2*DSP_SIMD; i++)  assign  w[i] = 0;

				always_ff @(posedge ap_clk2x) begin
					if(rst)  W[pe] <= 'x;
					else     W[pe] <= w[(Active? DSP_SIMD : 0) +: DSP_SIMD];
				end

			end : genPERegW

			dsp_a_t  A = 'x;
			for(genvar  pe = 0; pe < ACT_PE; pe++) begin : genPERegA

				uwire [2*DSP_SIMD-1:0][ACTIVATION_WIDTH-1:0]  a;
				for(genvar  i =    0; i <       SIMD; i++)  assign  a[i] = amvau_i[pe][i];
				for(genvar  i = SIMD; i < 2*DSP_SIMD; i++)  assign  a[i] = 0;

				always_ff @(posedge ap_clk2x) begin
					if(rst)  A[pe] <= 'x;
					else     A[pe] <= a[(Active? DSP_SIMD : 0) +: DSP_SIMD];
				end

			end : genPERegA

			logic  Zero = 1;
			logic  Last = 0;
			always_ff @(posedge ap_clk2x) begin
				if(rst) begin
					Zero <= 1;
					Last <= 0;
				end
				else begin
					automatic logic  zero = idle || !s_axis_weights_tvalid || !avld;
					Zero <= zero;
					Last <= alast && !zero && Active;
				end
			end

			assign	dsp_last = Last;
			assign	dsp_zero = Zero;
			assign	dsp_w = W;
			assign	dsp_a = A;

			// Since no two consecutive last cycles will ever be asserted on the input,
			// valid outputs will also always be spaced by, at least, one other cycle.
			// We can always hold a captured output for two cycles to allow the slow
			// clock to pick it up.
			logic    Vld = 0;
			dsp_p_t  P = 'x;
			always_ff @(posedge ap_clk2x) begin
				if(rst) begin
					Vld <= 0;
					P   <= 'x;
				end
				else begin
					if(dsp_vld)  P <= dsp_p;
					Vld <= dsp_vld || (Vld && !Active);
				end
			end
			assign	ovld = Vld;
			assign	odat = P;

		end : genPumpedCompute

		// @todo	Push core selection decision entirely into dotp.sv.
		// This currently duplicates the lane count computation from dotp.sv
		// to intercept configurations that should map to the INT8 mode of DSP58.
		localparam int unsigned  MIN_LANE_WIDTH = WEIGHT_WIDTH + ACTIVATION_WIDTH - 1;
		// number of lanes: for only 1 lane, NARROW_WEIGHTS makes no difference
		localparam int unsigned  A_WIDTH = 25 + 2*(VERSION > 1);     // Width of A datapath
		localparam int unsigned  NUM_LANES = A_WIDTH == WEIGHT_WIDTH? 1 : 1 + (A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH) / MIN_LANE_WIDTH;

		if(!ACTIVATION_BROADCASTING || ((VERSION > 2) && (NUM_LANES <= 3) && (WEIGHT_WIDTH <= 8) && (ACTIVATION_WIDTH <= 9))) begin : genINT8
			dotp_8sx9_dsp58 #(
				.ACTIVATION_BROADCASTING(ACTIVATION_BROADCASTING),
				.PE(PE), .SIMD(DSP_SIMD),
				.WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
				.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
				.SEGMENTLEN(SEGMENTLEN),
				.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
			) core (
				.clk(PUMPED_COMPUTE? ap_clk2x : ap_clk), .rst, .en('1),
				.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
				.vld(dsp_vld), .p(dsp_p)
			);
		end : genINT8
		else begin : genSoftVec
			dotp #(
				.VERSION(VERSION),
				.PE(PE), .SIMD(DSP_SIMD),
				.WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
				.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS), .NARROW_WEIGHTS(NARROW_WEIGHTS),
				.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
			) core (
				.clk(PUMPED_COMPUTE? ap_clk2x : ap_clk), .rst, .en('1),
				.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
				.vld(dsp_vld), .p(dsp_p)
			);
		end : genSoftVec

	end : blkDsp

	if(1) begin : blkOutput
		// VERSION alone does not select the core: DSP58 can use genSoftVec.
		// Full SIMD conservatively bounds either core, including the input/output
		// register stages and fast-to-slow capture when compute is pumped.
		localparam int unsigned  INT8_PIPELINE_DEPTH =
			3 + (SEGMENTLEN == 0? 0 : ((SIMD+2)/3 -1)/SEGMENTLEN);
		localparam int unsigned  SOFTVEC_PIPELINE_DEPTH =
			3 + $clog2(SIMD+1) + (SIMD == 1);
		localparam int unsigned  CORE_PIPELINE_DEPTH =
			VERSION == 3 && INT8_PIPELINE_DEPTH > SOFTVEC_PIPELINE_DEPTH?
			INT8_PIPELINE_DEPTH : SOFTVEC_PIPELINE_DEPTH;

		// OLock is registered: its assertion edge can queue a second result and
		// accept a final frame. Reserve those two entries plus the pipeline tail.
		// OBuf has MAX_IN_FLIGHT+1 entries; OReg is separate.
		localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1;
		typedef logic [PE-1:0][ACCU_WIDTH-1:0]  output_t;

		logic signed [$clog2(MAX_IN_FLIGHT+1):0]  OPtr = '1;	// -1 | 0, 1, ..., MAX_IN_FLIGHT
		(* SHREG_EXTRACT = "YES" *)
		output_t  OBuf[0:MAX_IN_FLIGHT];
		logic     OVld  =  0;
		output_t  OReg  = 'x;
		logic     OLock =  0;	// Lock upon backpressure (second entry into queue)

		// Catch every output into (SRL) Output Queue
		always_ff @(posedge ap_clk) begin
			if(ovld)  OBuf <= { odat, OBuf[0:MAX_IN_FLIGHT-1] };
		end

		always_ff @(posedge ap_clk) begin
			if(rst) begin
				OPtr  <= '1;
				OVld  <=  0;
				OReg  <= 'x;
				OLock <=  0;
			end
			else begin
				automatic logic  push = ovld;
				automatic logic  pop  = (m_axis_output_tready || !OVld) && !OPtr[$left(OPtr)];
				assert(pop || !push || (OPtr < $signed(MAX_IN_FLIGHT))) else begin
					$error("%m: Overflowing output queue.");
					$stop;
				end
				OPtr <= OPtr + $signed(push == pop? 0 : push? 1 : -1);

				if(OPtr[$left(OPtr)])                   OLock <= 0;
				else if(OVld && !m_axis_output_tready)  OLock <= 1;

				if(m_axis_output_tready || !OVld) begin
					OVld <= !OPtr[$left(OPtr)];
					OReg <= OBuf[OPtr[$left(OPtr)-1:0]];
				end
			end
		end
		assign	idle = OLock;

		assign	m_axis_output_tvalid = OVld;
		// Why would we need a sign extension here potentially creating a higher signal load into the next FIFO?
		// These extra bits should never be used. Why not 'x them out?
		assign	m_axis_output_tdata = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){OReg[PE-1][ACCU_WIDTH-1]}}, OReg };

	end : blkOutput

`default_nettype wire
endmodule : dotp_axi
