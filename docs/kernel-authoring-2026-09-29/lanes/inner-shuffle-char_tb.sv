// Characterization bench for FinnLib inner_shuffle: configs x input/output timing
// (record: inner-shuffle.md). Each instance streams ROUNDS matrices with distinct
// values per round and prints a RESULT line: its first mismatch, which lanes were
// x or wrong, and how many patterns entered rd_pattern_skid without a bank read.
//   F=<finnlib>/rtl
//   xvlog --sv $F/infra/fifo.sv $F/infra/elasticmem.sv $F/shape/inner_shuffle.sv inner-shuffle-char_tb.sv
//   xelab char_tb --snapshot char --timescale 1ns/1ps && xsim char --runall | grep RESULT
// (The unfixed inner_shuffle.sv needs --relax on xvlog and xelab.)
`timescale 1ns/1ps

module char_inst #(
	int unsigned  SIMD, I, J,
	int unsigned  MODE,
	int unsigned  ROUNDS = 4,
	int unsigned  BITS = 10
)(
	input  logic  clk, rst,
	input  int    cycle,
	output logic  done
);
	localparam int unsigned  BEATS = I*J/SIMD;
	typedef logic [SIMD-1:0][BITS-1:0]  data_t;

	function automatic data_t in_word(int unsigned r, int unsigned k);
		// input beat k of round r: row-major, SIMD elements of a row
		data_t  w;
		for(int unsigned  l = 0; l < SIMD; l++) begin
			automatic int unsigned  e = k*SIMD + l;	// flat row-major index
			w[l] = r*I*J + e;
		end
		return  w;
	endfunction
	function automatic data_t out_word(int unsigned r, int unsigned k);
		// output beat k: column-major, SIMD elements of a column
		data_t  w;
		for(int unsigned  l = 0; l < SIMD; l++) begin
			automatic int unsigned  f = k*SIMD + l;	// flat index in (J, I)
			automatic int unsigned  j = f / I, i = f % I;
			w[l] = r*I*J + i*J + j;
		end
		return  w;
	endfunction

	uwire  irdy, ovld;
	logic  ivld, ordy;
	data_t  idat;
	uwire data_t  odat;
	inner_shuffle #(.BITS(BITS), .I(I), .J(J), .SIMD(SIMD)) dut (
		.clk, .rst, .irdy, .ivld, .idat, .ordy, .ovld, .odat
	);

	int unsigned  Sent = 0, Recv = 0, Held = 0;
	int  first = -1, nfail = 0;
	logic [SIMD-1:0]  xlanes = 0, badlanes = 0;
	int unsigned  total = ROUNDS*BEATS;

	// Timing modes (valid / ready):
	//  0 free / free
	//  1 cycle%3!=0 / cycle%4!=1   (FINN conformance "stalled")
	//  2 cycle%3!=0 / free
	//  3 free / cycle%4!=1
	//  4 cycle%8<3 (bursts) / free
	//  5 free, but the last beat of every matrix held back 200 cycles / free
	//  6 random 1/37 gaps / random 6/7   (FinnLib tb)
	//  7 cycle%8<3 / cycle%4!=1
	function automatic logic want_valid(int c);
		case(MODE)
		1, 2: return  c%3 != 0;
		4, 7: return  c%8 < 3;
		5:    return  (Sent % BEATS != BEATS-1) || Held >= 200;
		6:    return  $urandom()%37 != 0;
		default: return  1;
		endcase
	endfunction
	function automatic logic want_ready(int c);
		case(MODE)
		1, 3, 7: return  c%4 != 1;
		6:       return  $urandom()%7 != 0;
		default: return  1;
		endcase
	endfunction

	always @(negedge clk) begin
		ivld = !rst && Sent < total && want_valid(cycle);
		idat = ivld? in_word(Sent / BEATS, Sent % BEATS) : 'x;
		ordy = !rst && want_ready(cycle);
	end

	always_ff @(posedge clk) begin
		if(!rst) begin
			if(ivld && irdy) begin
				Sent <= Sent + 1;
				Held <= 0;
			end
			else if(Sent % BEATS == BEATS-1)  Held <= Held + 1;
			if(ovld && ordy && Recv < total) begin
				automatic data_t  exp = out_word(Recv / BEATS, Recv % BEATS);
				if(odat !== exp) begin
					if(first < 0) begin
						first = Recv;
						$display("MISMATCH SIMD=%0d I=%0d J=%0d MODE=%0d word %0d (round %0d beat %0d): got %h exp %h",
							SIMD, I, J, MODE, Recv, Recv/BEATS, Recv%BEATS, odat, exp);
					end
					nfail++;
					for(int unsigned  l = 0; l < SIMD; l++) begin
						if($isunknown(odat[l]))  xlanes[l] = 1;
						else if(odat[l] !== exp[l])  badlanes[l] = 1;
					end
				end
				Recv <= Recv + 1;
			end
		end
	end
	assign	done = Recv == total;

	// A pattern pushed into rd_pattern_skid without a bank read issued with it
	int  orphan = 0;
	always @(posedge clk)  if(!rst && dut.rd_req_en && !dut.rd_req_rdy_all)  orphan++;

	final begin
		$display("RESULT SIMD=%0d I=%0d J=%0d MODE=%0d %s recv=%0d/%0d first=%0d nfail=%0d xlanes=%b badlanes=%b orphan=%0d",
			SIMD, I, J, MODE, (Recv == total && nfail == 0)? "PASS" : "FAIL",
			Recv, total, first, nfail, xlanes, badlanes, orphan);
	end
endmodule : char_inst

module char_tb;
	typedef struct { int unsigned simd, i, j; } cfg_t;
	localparam int unsigned  NC = 14;
	localparam cfg_t  CFGS[NC] = '{
		'{1,6,6}, '{2,6,6}, '{3,6,6}, '{6,6,6},
		'{4,4,4}, '{4,8,8}, '{4,4,8}, '{4,8,4},
		'{2,4,6}, '{3,6,4}, '{3,6,7}, '{4,8,10}, '{2,2,3}, '{5,10,4}
	};
	localparam int unsigned  NM = 8;

	logic  clk = 0, rst = 1;
	always #5 clk = !clk;
	int  cycle = 0;
	always @(posedge clk)  cycle <= cycle + 1;

	logic [NC*NM-1:0]  done;
	for(genvar c = 0; c < NC; c++) begin : gC
		for(genvar m = 0; m < NM; m++) begin : gM
			char_inst #(.SIMD(CFGS[c].simd), .I(CFGS[c].i), .J(CFGS[c].j), .MODE(m)) u (
				.clk, .rst, .cycle, .done(done[c*NM+m])
			);
		end
	end

	initial begin
		repeat(4) @(posedge clk);
		rst <= 0;
		fork
			begin
				forever begin
					@(posedge clk);
					if(&done) break;
				end
				repeat(8) @(posedge clk);
			end
			begin
				repeat(20000) @(posedge clk);
				$display("TIMEOUT");
			end
		join_any
		$finish;
	end
endmodule : char_tb
