/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Dot Product Unit (DOTP) core compute kernel utilizing DSP58.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @author	Mirza Mrahorovic <mirza.mrahorovic@amd.com>
 * @author	Nikolaos Kavvadias <nikolaos.kavvadias@amd.com>
 ***************************************************************************/

module dotp_8sx9_dsp58 #(
	bit  ACTIVATION_BROADCASTING,
	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  ACTIVATION_WIDTH,
	int unsigned  WEIGHT_WIDTH,
	int unsigned  ACCU_WIDTH,
	bit  SIGNED_ACTIVATIONS = 0,
	int unsigned  SEGMENTLEN = 0,
	bit  FORCE_BEHAVIORAL = 0,

	localparam int unsigned  ACTIVATION_ELEMENTS = (ACTIVATION_BROADCASTING? 1 : PE) * SIMD,
	localparam int unsigned  WEIGHT_ELEMENTS = PE * SIMD
)(
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	input	logic  last,
	input	logic  zero,
	input	logic [WEIGHT_ELEMENTS-1:0][WEIGHT_WIDTH-1:0]  w,
	input	logic [ACTIVATION_ELEMENTS-1:0][ACTIVATION_WIDTH-1:0]  a,

	output	logic  vld,
	output	logic [PE-1:0][ACCU_WIDTH-1:0]  p
);
`default_nettype none

	//=== Startup Recovery Watchdog ==========================================
	// The DSP slice needs 100ns of recovery time after initial startup before
	// being able to ingest input properly. This watchdog discovers violating
	// stimuli during simulation and produces a corresponding warning.
	if(1) begin : blkRecoveryWatch
		logic  Dirty = 1;
		initial begin
			#100ns;
			Dirty <= 0;
		end

		always_ff @(posedge clk) begin
			assert(!Dirty || rst || !en || zero) else begin
				$warning("%m: Feeding input during DSP startup recovery. Expect functional errors.");
			end
		end
	end : blkRecoveryWatch

	//=== Global Signals ====================================================
	localparam int unsigned  CHAINLEN = (SIMD + 2) / 3;
	localparam int unsigned  SEGLEN = SEGMENTLEN == 0? CHAINLEN : SEGMENTLEN;
	localparam int unsigned  PE_ACTIVATION = ACTIVATION_BROADCASTING? 1 : PE;

	(* KEEP = "TRUE" *)
	uwire [26:0]  a_in_i[PE_ACTIVATION * CHAINLEN];
	(* KEEP = "TRUE" *)
	uwire [23:0]  b_in_i[PE][CHAINLEN];
	uwire [PE-1:0][CHAINLEN-1:0][57:0]  pcout;

	//=== Shift Register for Opmode Select ==================================
	localparam int unsigned  MAX_PIPELINE_STAGES = (CHAINLEN + SEGLEN - 1) / SEGLEN;
	// After MAX_PIPELINE_STAGES of input data, we have 3 additional cycles
	// latency (A/B reg, Mreg, Preg). We add +2 since OPMODE is buffered
	// by 1 cycle in the DSP fabric.
	logic  L[0:1+MAX_PIPELINE_STAGES] = '{default: 0};

	always_ff @(posedge clk) begin
		if(rst)  L <= '{default: 0};
		else if(en) begin
			L[1+MAX_PIPELINE_STAGES] <= last;
			L[0:MAX_PIPELINE_STAGES] <= L[1:1+MAX_PIPELINE_STAGES];
		end
	end
	assign	vld = L[0];

	//=== Shift Register for ZERO Flag ======================================
	// We need MAX_PIPELINE_STAGES-1 delay stages (INMODE is registered
	// once more inside DSP).
	uwire [MAX_PIPELINE_STAGES-1:0]  inmode_zero;
	assign	inmode_zero[0] = zero;
	if(MAX_PIPELINE_STAGES > 1) begin : genZReg
		logic [MAX_PIPELINE_STAGES-1:1]  Z = '1;
		always_ff @(posedge clk) begin
			if(rst)      Z <= '1;
			else if(en)  Z <= inmode_zero[MAX_PIPELINE_STAGES-2:0];
		end
		assign	inmode_zero[MAX_PIPELINE_STAGES-1:1] = Z;
	end : genZReg

	//=== Buffer for Input Activations ======================================
	localparam int unsigned  PAD_BITS_ACT = 9 - ACTIVATION_WIDTH;
	for(genvar k = 0; k < PE_ACTIVATION; k++) begin : genActPE
		for(genvar i = 0; i < CHAINLEN; i++) begin : genActSIMD
			localparam int unsigned  TOTAL_PREGS = i / SEGLEN;
			localparam int unsigned  EXTERNAL_PREGS = TOTAL_PREGS > 1? TOTAL_PREGS - 1 : 0;
			localparam int unsigned  LANES_OCCUPIED = i == CHAINLEN - 1? SIMD - 3*i : 3;

			if(EXTERNAL_PREGS > 0) begin : genExternalPregAct
				(* EXTRACT_SHREG = "true" *)
				logic [0:EXTERNAL_PREGS-1][LANES_OCCUPIED-1:0][ACTIVATION_WIDTH-1:0]  A = '{default: 'x};
				always_ff @(posedge clk) begin
					if(en) begin
						A[EXTERNAL_PREGS-1] <=
// synthesis translate_off
							zero? '1 :
// synthesis translate_on
							a[SIMD*k + 3*i +: LANES_OCCUPIED];
						if(EXTERNAL_PREGS > 1)  A[0:EXTERNAL_PREGS-2] <= A[1:EXTERNAL_PREGS-1];
					end
				end
				for(genvar j = 0; j < LANES_OCCUPIED; j++) begin : genAin
					assign	a_in_i[CHAINLEN*k+i][9*j +: 9] =
						SIGNED_ACTIVATIONS?
							PAD_BITS_ACT == 0? A[0][j] : { {PAD_BITS_ACT{A[0][j][ACTIVATION_WIDTH-1]}}, A[0][j] } :
							PAD_BITS_ACT == 0? A[0][j] : { {PAD_BITS_ACT{1'b0}}, A[0][j] };
				end : genAin
				for(genvar j = LANES_OCCUPIED; j < 3; j++) begin : genAinZero
					assign	a_in_i[CHAINLEN*k+i][9*j +: 9] = 9'b0;
				end : genAinZero
			end : genExternalPregAct
			else begin : genInpDSPAct
				for(genvar j = 0; j < LANES_OCCUPIED; j++) begin : genAin
					assign	a_in_i[CHAINLEN*k+i][9*j +: 9] =
// synthesis translate_off
						zero? '1 :
// synthesis translate_on
						SIGNED_ACTIVATIONS?
							PAD_BITS_ACT == 0? a[SIMD*k+3*i+j] : { {PAD_BITS_ACT{a[SIMD*k+3*i+j][ACTIVATION_WIDTH-1]}}, a[SIMD*k+3*i+j] } :
							PAD_BITS_ACT == 0? a[SIMD*k+3*i+j] : { {PAD_BITS_ACT{1'b0}}, a[SIMD*k+3*i+j] };
				end : genAin
				for(genvar j = LANES_OCCUPIED; j < 3; j++) begin : genAinZero
					assign	a_in_i[CHAINLEN*k+i][9*j +: 9] = 9'b0;
				end : genAinZero
			end : genInpDSPAct
		end : genActSIMD
	end : genActPE

	//=== Buffer for Weights ================================================
	localparam int unsigned  PAD_BITS_WEIGHT = 8 - WEIGHT_WIDTH;

	for(genvar i = 0; i < PE; i++) begin : genWeightPE
		for(genvar j = 0; j < CHAINLEN; j++) begin : genWeightSIMD
			localparam int unsigned  TOTAL_PREGS = j / SEGLEN;
			localparam int unsigned  EXTERNAL_PREGS = TOTAL_PREGS > 1? TOTAL_PREGS - 1 : 0;
			localparam int unsigned  LANES_OCCUPIED = j == CHAINLEN - 1? SIMD - 3*j : 3;

			if(EXTERNAL_PREGS > 0) begin : genExternalPregWeight
				(* EXTRACT_SHREG = "true" *)
				logic [0:EXTERNAL_PREGS-1][LANES_OCCUPIED-1:0][WEIGHT_WIDTH-1:0]  B = '{default: 'x};
				always_ff @(posedge clk) begin
					if(en) begin
						B[EXTERNAL_PREGS-1] <=
// synthesis translate_off
							zero? '1 :
// synthesis translate_on
							w[SIMD*i + 3*j +: LANES_OCCUPIED];
						if(EXTERNAL_PREGS > 1)  B[0:EXTERNAL_PREGS-2] <= B[1:EXTERNAL_PREGS-1];
					end
				end
				for(genvar k = 0; k < LANES_OCCUPIED; k++) begin : genBin
					assign	b_in_i[i][j][8*k +: 8] =
						PAD_BITS_WEIGHT == 0? B[0][k] : { {PAD_BITS_WEIGHT{B[0][k][WEIGHT_WIDTH-1]}}, B[0][k] };
				end : genBin
				for(genvar k = LANES_OCCUPIED; k < 3; k++) begin : genBinZero
					assign	b_in_i[i][j][8*k +: 8] = 8'b0;
				end : genBinZero
			end : genExternalPregWeight
			else begin : genInpDSPWeight
				for(genvar k = 0; k < LANES_OCCUPIED; k++) begin : genBin
					assign	b_in_i[i][j][8*k +: 8] =
// synthesis translate_off
						zero? '1 :
// synthesis translate_on
						PAD_BITS_WEIGHT == 0? w[SIMD*i+3*j+k] : { {PAD_BITS_WEIGHT{w[SIMD*i+3*j+k][WEIGHT_WIDTH-1]}}, w[SIMD*i+3*j+k] };
				end : genBin
				for(genvar k = LANES_OCCUPIED; k < 3; k++) begin : genBinZero
					assign	b_in_i[i][j][8*k +: 8] = 8'b0;
				end : genBinZero
			end : genInpDSPWeight
		end : genWeightSIMD
	end : genWeightPE

	//=== DSP Chain: PE x CHAINLEN ==========================================
	for(genvar i = 0; i < PE; i++) begin : genDSPPE
		for(genvar j = 0; j < CHAINLEN; j++) begin : genDSPChain
			localparam int unsigned  TOTAL_PREGS = j / SEGLEN;
			localparam int unsigned  INTERNAL_PREGS = TOTAL_PREGS > 0? 2 : 1;
			localparam bit  PREG = (j+1) % SEGLEN == 0 || j == CHAINLEN - 1;
			localparam bit  FIRST = j == 0;
			localparam bit  LAST = j == CHAINLEN - 1;
			localparam bit  SEG_FIRST = j % SEGLEN == 0;
			localparam bit  SEG_LAST = PREG && !LAST && (j+1) < CHAINLEN && (j+1) % SEGLEN == 0;
			uwire [57:0]  pp;
			uwire [57:0]  pc;

			if(LAST) begin : genPOUT
				assign	p[i] = pp[ACCU_WIDTH-1:0];
			end

			// Note: Since the product B * AD is computed,
			//       rst can be only applied to AD and zero only to B
			//       with the same effect as zeroing both.
			if(FORCE_BEHAVIORAL) begin : genBehav
				// Stage #1: Input A/B
				logic signed [33:0]  Areg[INTERNAL_PREGS];
				always_ff @(posedge clk) begin
					if(rst)  Areg <= '{default: 0};
					else if(en) begin
						Areg[0] <= { 7'bx, a_in_i[(ACTIVATION_BROADCASTING? 0 : CHAINLEN*i) + j] };
						if(INTERNAL_PREGS == 2)  Areg[1] <= Areg[0];
					end
				end
				logic signed [23:0]  Breg[INTERNAL_PREGS];
				always_ff @(posedge clk) begin
					if(rst)  Breg <= '{default: 0};
					else if(en) begin
						Breg[0] <= b_in_i[i][j];
						if(INTERNAL_PREGS == 2)  Breg[1] <= Breg[0];
					end
				end

				// Stage #2: Multiply-Accumulate
				logic signed [57:0]  Mreg;
				logic  InmodeZero = 0;
				always_ff @(posedge clk) begin
					if(rst)      InmodeZero <= 0;
					else if(en)  InmodeZero <= inmode_zero[TOTAL_PREGS];
				end
				always_ff @(posedge clk) begin
					if(rst)  Mreg <= 0;
					else if(en) begin
						automatic logic signed [57:0]  m = 0;
						for(int k = 0; k < 3; k++)
							m = m + (InmodeZero? 0 : $signed(Areg[INTERNAL_PREGS-1][9*k +: 9]) * $signed(Breg[INTERNAL_PREGS-1][8*k +: 8]));
						Mreg <= m;
					end
				end

				// Stage #3: Accumulate
				logic signed [57:0]  Preg;
				logic  Opmode = 0;
				if(FIRST && !LAST) begin : genFirst
					if(PREG) begin : genPregBehav
						always_ff @(posedge clk) begin
							if(rst)      Preg <= 0;
							else if(en)  Preg <= Mreg;
						end
					end
					else  assign  Preg = Mreg;
				end
				else if(FIRST && LAST) begin : genSingle
					always_ff @(posedge clk) begin
						if(rst)      Opmode <= 0;
						else if(en)  Opmode <= L[1];
					end
					always_ff @(posedge clk) begin
						if(rst)      Preg <= 0;
						else if(en)  Preg <= (Opmode? 0 : Preg) + Mreg;
					end
				end
				else if(!FIRST && LAST) begin : genLast
					always_ff @(posedge clk) begin
						if(rst)      Opmode <= 0;
						else if(en)  Opmode <= L[1];
					end
					always_ff @(posedge clk) begin
						if(rst)      Preg <= 0;
						else if(en)  Preg <= (Opmode? 0 : Preg) + Mreg + pcout[i][j-1];
					end
				end
				else begin : genMid
					if(PREG) begin : genPregBehav
						always_ff @(posedge clk) begin
							if(rst)      Preg <= 0;
							else if(en)  Preg <= Mreg + pcout[i][j-1];
						end
					end
					else  assign  Preg = Mreg + pcout[i][j-1];
				end
				assign	pp = Preg;
				assign	pc = Preg;
			end : genBehav
			else begin : genDSP
				DSP58 #(
					// Feature Control Attributes: Data Path Selection
					.AMULTSEL("A"),
					.A_INPUT("DIRECT"),
					.BMULTSEL("B"),
					.B_INPUT("DIRECT"),
					.DSP_MODE("INT8"),
					.PREADDINSEL("A"),
					.RND(58'h000000000000000),
					.USE_MULT("MULTIPLY"),
					.USE_SIMD("ONE58"),
					.USE_WIDEXOR("FALSE"),
					.XORSIMD("XOR24_34_58_116"),
					// Pattern Detector Attributes
					.AUTORESET_PATDET("NO_RESET"),
					.AUTORESET_PRIORITY("RESET"),
					.MASK(58'h0ffffffffffffff),
					.PATTERN(58'h000000000000000),
					.SEL_MASK("MASK"),
					.SEL_PATTERN("PATTERN"),
					.USE_PATTERN_DETECT("NO_PATDET"),
					// Programmable Inversion Attributes
					.IS_ALUMODE_INVERTED(4'b0000),
					.IS_CARRYIN_INVERTED(1'b0),
					.IS_CLK_INVERTED(1'b0),
					.IS_INMODE_INVERTED(5'b00000),
					.IS_NEGATE_INVERTED(3'b000),
					.IS_OPMODE_INVERTED({
						LAST?  2'b01 : 2'b00,
						FIRST? 3'b000 : SEG_FIRST? 3'b011 : 3'b001,
						2'b01,
						2'b01
					}),
					.IS_RSTALLCARRYIN_INVERTED(1'b0),
					.IS_RSTALUMODE_INVERTED(1'b0),
					.IS_RSTA_INVERTED(1'b0),
					.IS_RSTB_INVERTED(1'b0),
					.IS_RSTCTRL_INVERTED(1'b0),
					.IS_RSTC_INVERTED(1'b0),
					.IS_RSTD_INVERTED(1'b0),
					.IS_RSTINMODE_INVERTED(1'b0),
					.IS_RSTM_INVERTED(1'b0),
					.IS_RSTP_INVERTED(1'b0),
					// Register Control Attributes
					.ACASCREG(INTERNAL_PREGS),
					.ADREG(0),
					.ALUMODEREG(0),
					.AREG(INTERNAL_PREGS),
					.BCASCREG(INTERNAL_PREGS),
					.BREG(INTERNAL_PREGS),
					.CARRYINREG(0),
					.CARRYINSELREG(0),
					.CREG(0),
					.DREG(0),
					.INMODEREG(1),
					.MREG(1),
					.OPMODEREG(1),
					.PREG(PREG),
					.RESET_MODE("SYNC")
				) DSP58_inst (
					// Cascade outputs
					.ACOUT(),
					.BCOUT(),
					.CARRYCASCOUT(),
					.MULTSIGNOUT(),
					.PCOUT(pc),
					// Control outputs
					.OVERFLOW(),
					.PATTERNBDETECT(),
					.PATTERNDETECT(),
					.UNDERFLOW(),
					// Data outputs
					.CARRYOUT(),
					.P(pp),
					.XOROUT(),
					// Cascade inputs
					.ACIN('x),
					.BCIN('x),
					.CARRYCASCIN('x),
					.MULTSIGNIN('x),
					.PCIN(SEG_FIRST? 'x : pcout[i][j-1]),
					// Control inputs
					.ALUMODE(4'h0),
					.CARRYINSEL('0),
					.CLK(clk),
					.INMODE({
						INTERNAL_PREGS == 2? 1'b0 : 1'b1,
						2'b00,
						inmode_zero[TOTAL_PREGS],
						INTERNAL_PREGS == 2? 1'b0 : 1'b1
					}),
					.NEGATE('0),
					.OPMODE({
						LAST? {1'b0, L[1]} : 2'b00,
						7'b000_0000
					}),
					// Data inputs
					.A({ 7'bx, a_in_i[(ACTIVATION_BROADCASTING? 0 : CHAINLEN*i) + j] }),
					.B(b_in_i[i][j]),
					.C(SEG_FIRST && !FIRST? pcout[i][j-1] : 'x),
					.CARRYIN('0),
					.D('x),
					// Reset/Clock Enable inputs
					.ASYNC_RST('0),
					.CEA1(en),
					.CEA2(INTERNAL_PREGS == 2? en : '0),
					.CEAD('0),
					.CEALUMODE('0),
					.CEB1(en),
					.CEB2(INTERNAL_PREGS == 2? en : '0),
					.CEC('0),
					.CECARRYIN('0),
					.CECTRL(en),
					.CED('0),
					.CEINMODE(en),
					.CEM(en),
					.CEP(PREG && en),
					.RSTA(rst),
					.RSTALLCARRYIN('0),
					.RSTALUMODE('0),
					.RSTB(rst),
					.RSTC('0),
					.RSTCTRL(rst),
					.RSTD('0),
					.RSTINMODE(rst),
					.RSTM(rst),
					.RSTP(PREG && rst)
				);
			end : genDSP
			// Route pcout: use P (fabric) at segment boundaries, PCOUT (cascade) within segments
			assign	pcout[i][j] = SEG_LAST? pp : pc;
		end : genDSPChain
	end : genDSPPE

`default_nettype wire
endmodule : dotp_8sx9_dsp58
