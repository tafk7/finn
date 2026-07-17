# Census: requant

*A three-layer family (agnostic Requant base + HLS and RTL backends, both cleanly inheriting the base) implementing clip(round(x*scale+bias)) as a streaming alternative to thresholding. It is mostly well-structured, but its embedded (non-streamed) scale/bias inputs force execute_node/stream-width/folding overrides across all layers, and the HLS and RTL backends implement divergent numeric algorithms with duplicated, differently-laid-out parameter code.*

**Files:** `src/finn/custom_op/fpgadataflow/requant.py`, `src/finn/custom_op/fpgadataflow/hls/requant_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/requant_rtl.py`, `finn-rtllib/requant/hdl/requant_wrapper_template.sv`, `finn-rtllib/requant/hdl/requant_wrapper_template.v`, `finn-rtllib/requant/hdl/requant.sv`, `finn-rtllib/requant/hdl/requant_axi.sv`, `finn-rtllib/requant/hdl/queue.sv`, `src/finn/custom_op/fpgadataflow/rtlbackend.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `Requant` | `requant.py` | `(HWCustomOp)` | yes |
| `Requant_hls` | `requant_hls.py` | `(Requant, HLSBackend)` | yes |
| `Requant_rtl` | `requant_rtl.py` | `(Requant, RTLBackend)` | yes |

## Redesign pressure

The dominant stress is that requant is a multi-input node whose non-data inputs (scale, bias) are embedded constants rather than streams, which violates the base assumption that every node.input is a stream: it forces get_instream_width to return 0 for ind!=0, forces get_folded_input_shape to special-case ind, and forces a full execute_node re-implementation in BOTH backends (and the base) purely to avoid the RTLBackend/HLSBackend loops that would try to stream scale/bias. A clean abstraction would model 'embedded parameter inputs' as a first-class concept instead of per-op override surgery. Secondly, the two backends do not share one quantization definition — HLS does float math while RTL derives fixed-point params in SystemVerilog from 6-decimal-truncated literals — so the HLS/RTL axis here hides genuinely different numeric algorithms (and divergent [CF][PE] vs [PE][CF] param layouts with duplicated broadcast code), not two codegens of one algorithm. Finally the base class's knowledge of an RTL-only restriction (narrow=0/unsigned, requant.py:48) documented but unenforced shows backend constraints leaking upward into the agnostic layer.

## Overrides (30)

| method | class | line | reason |
|---|---|---|---|
| `get_input_datatype` | Requant | 95 | Multi-input op: ind==0 is quantized data, but ind 1/2 (scale/bias) are hard-forced to FLOAT32 since they are embedded params, not FINN-quantized streams. |
| `get_output_datatype` | Requant | 103 | Standard single output datatype from nodeattr. |
| `get_normal_input_shape` | Requant | 107 | Shape derived from numInputVectors + NumChannels (channel-last). |
| `get_normal_output_shape` | Requant | 113 | Output shape identical to input shape (elementwise requant). |
| `get_folded_input_shape` | Requant | 117 | Folds channels into [fold, PE]; for ind!=0 returns the UNFOLDED normal shape because scale/bias are embedded not streamed. |
| `get_folded_output_shape` | Requant | 128 | Same folded shape as data input. |
| `get_instream_width` | Requant | 169 | Returns 0 for ind!=0 because scale/bias (inputs 1,2) are embedded as constants, not streamed. |
| `get_outstream_width` | Requant | 179 | PE * output bitwidth. |
| `get_exp_cycles` | Requant | 132 | One cycle per output value (II=1 streaming). |
| `verify_node` | Requant | 92 | No-op stub; no verification implemented. |
| `execute_node` | Requant | 136 | Python functional model: clip(round(x*scale+bias)) with round-half-up. |
| `get_nodeattr_types` | Requant | 32 | Adds PE, NumChannels, inputDataType, outputDataType, numInputVectors, narrow. |
| `get_exp_cycles` | Requant_hls | 39 | Adds constant hls_overhead=10 to account for HLS pipeline init latency. |
| `generate_params` | Requant_hls | 47 | Writes scale/bias to params.h as [CF][PE] float arrays; also sets scale-is-one/bias-is-zero optimization flags. |
| `global_includes` | Requant_hls | 83 | HLS abstract: pulls hls_math, flatten.hpp, params.h. |
| `defines` | Requant_hls | 92 | HLS abstract: emits PE/CF/TOTAL_FOLD/MIN_VAL/MAX_VAL, type aliases, and an inline clip() template. |
| `docompute` | Requant_hls | 140 | HLS abstract: main compute loop; expression varies by scale-is-one/bias-is-zero flags. |
| `blackboxfunction` | Requant_hls | 196 | HLS abstract: top function signature with only in0_V/out0_V (no scale/bias ports). |
| `read_npy_data` | Requant_hls | 206 | Reads only input_0.npy; half vs float npy type selection. |
| `strm_decl` | Requant_hls | 219 | Declares only in0_V/out0_V streams. |
| `dataoutstrm` | Requant_hls | 226 | Writes output_0.npy with half/float selection. |
| `save_as_npy` | Requant_hls | 240 | Emptied because saving is handled inside dataoutstrm. |
| `pragmas` | Requant_hls | 244 | AXIS interface pragmas + ap_ctrl_none. |
| `execute_node` | Requant_hls | 253 | cppsim/rtlsim that only writes input_0.npy since scale/bias are embedded; includes BIPOLAR handling. |
| `get_nodeattr_types` | Requant_hls | 33 | Merges Requant + HLSBackend attrs. |
| `generate_hdl` | Requant_rtl | 38 | RTL abstract: string-fills SV/V templates with K/N/C/PE/VERSION/SCALES/BIASES/widths. |
| `get_rtl_file_list` | Requant_rtl | 134 | Lists queue.sv/requant.sv/requant_axi.sv + generated _impl.sv and .v wrappers. |
| `code_generation_ipi` | Requant_rtl | 157 | TCL add_files + create_bd_cell of gen_top_module. |
| `execute_node` | Requant_rtl | 169 | rtlsim only writes input_0.npy (embedded params); else delegates to Requant.execute_node Python model. |
| `get_nodeattr_types` | Requant_rtl | 21 | Merges Requant + RTLBackend attrs. |

## Hacks (17 — 1 blocker, 5 major)

- **[blocker/magic-number]** `requant_rtl.py:78` — format_sv_array emits scale/bias with fixed '{:.6f}' — only 6 decimal places for a 32-bit shortreal. Scale/bias are truncated to 6 decimals in the generated SV, which can perturb the quantization result vs the Python golden model (which uses full float32).
- **[major/brittle-assumption]** `requant.py:48` — narrow attr comment 'RTL backend only supports narrow=0 and unsigned output' — the agnostic base documents an RTL-backend-specific restriction, but nothing enforces it; an RTL node with narrow=1 or signed output will silently mis-quantize (requant.sv stage-4 clamps neg->0, i.e. unsigned-only).
- **[major/brittle-assumption]** `requant.py:142` — execute_node branches on hasattr(graph,'model') to decide between model initializers and context.get(...) fallback for scale/bias — brittle duck-typing of the graph argument with silent default arrays [1.0]/[0.0].
- **[major/brittle-assumption]** `requant_hls.py:59` — _scale_is_one/_bias_is_zero set as instance attributes in generate_params and read via getattr(...,False) in docompute (lines 142-143). Order-dependent: if docompute runs before generate_params the optimization silently defaults off, producing full multiply-add even when scale=1/bias=0.
- **[major/duplicated-logic]** `requant_hls.py:253` — execute_node re-implements HLSBackend.execute_node (npy save, cppsim exec, rtlsim pack/unpack, BIPOLAR conversion at 286-288/304-307) with the sole change of writing only input_0.npy. ~85 lines duplicated to drop param inputs.
- **[major/duplicated-logic]** `requant_hls.py:69` — Scale/bias broadcast+reshape to [CF][PE] here vs Requant_rtl reshapes to [PE][CF] (transposed) at rtl:69-70. The two backends carry divergent, independently-maintained param-layout code for the same tensors.
- **[minor/magic-number]** `requant.py:164` — np.floor(x*scale+bias + 0.5) hard-codes round-half-up; comment explicitly rejects np.round banker's rounding. Golden model must match RTL/HLS rounding exactly or verification breaks.
- **[minor/magic-number]** `requant_hls.py:44` — hls_overhead = 10 constant added to expected cycles with no derivation.
- **[minor/brittle-assumption]** `requant_hls.py:110` — Comment 'Use explicit width constants instead of TI::width which doesn't work for float' — TI_WIDTH/TO_WIDTH hand-injected because the HLS float type lacks ::width; works around HLS type system rather than using a uniform packing abstraction.
- **[minor/template-surgery]** `requant_hls.py:129` — A full C++ clip<>() function template is embedded as a raw string literal inside the $DEFINES$ list rather than living in an .hpp include.
- **[minor/magic-number]** `requant_rtl.py:30` — _resolve_dsp_version maps DSP58->3, DSP48E2->2, else->1; hard-coded version integers that requant.sv derive_MUL_WIDTHS switches on (aw/bw 25/18, 27/18, 27/24).
- **[minor/duplicated-logic]** `requant_rtl.py:86` — Byte-aligned stream width ((PE*K+7)//8)*8 computed in Python AND independently recomputed in requant_axi.sv (lines 22-23). Two sources of truth for stream width.
- **[minor/template-surgery]** `requant_rtl.py:97` — SV impl generated by 10 sequential str.replace of $PLACEHOLDER$ tokens in requant_wrapper_template.sv; no structured templating.
- **[minor/template-surgery]** `requant_rtl.py:118` — Second raw str.replace pass fills the Verilog stub requant_wrapper_template.v (needed because Vivado IP packaging rejects SV top).
- **[minor/other]** `requant_top.sv:1` — requant_top.sv exists in the rtllib hdl dir but is never referenced by generate_hdl or get_rtl_file_list — dead/unused source relative to the Python codegen path.
- **[minor/other]** `requant_hls.py:1` — Copyright 2024 on the HLS file vs 2026 on requant.py and requant_rtl.py — inconsistent provenance suggesting the HLS backend was templated/copied earlier from another op.
- **[minor/brittle-assumption]** `requant_hls.py:26` — Docstring says HLS backend is 'primarily for FLOAT32 inputs' and to prefer RTL for integer inputs, yet defines()/docompute() cast via TI integer types — the intended input-datatype regime is ambiguous and only documented in prose.

## Hermeticity violations (4)

- **[env-var]** `requant_rtl.py:90` — generate_hdl reads os.environ['FINN_ROOT'] to locate finn-rtllib/requant/hdl templates.
- **[env-var]** `requant_rtl.py:136` — get_rtl_file_list reads os.environ['FINN_ROOT'] again to locate rtllib sources (second independent lookup).
- **[module-mutable-state]** `requant_hls.py:59` — generate_params writes self._scale_is_one/_bias_is_zero which docompute later reads (getattr default False) — creates a required call-ordering dependency between two contract methods.
- **[filesystem-path]** `requant_rtl.py:90` — Hard-coded path suffix '/finn-rtllib/requant/hdl/' appended to FINN_ROOT in both generate_hdl and get_rtl_file_list.

## finn-rtllib coupling (6)

- `requant/hdl/requant_wrapper_template.sv` via **string-replace** — Read at rtl/requant_rtl.py:94, then 10 str.replace calls fill $TOP_MODULE_NAME$/$VERSION$/$K$/$N$/$C$/$PE$/$SCALES$/$BIASES$/$IN_STREAM_WIDTH$/$OUT_STREAM_WIDTH$ (lines 98-107); written to <top>_impl.sv (line 109). SCALES/BIASES are Python-formatted SV array literals injected at lines 104-105.
- `requant/hdl/requant_wrapper_template.v` via **string-replace** — Read at line 114; str.replace fills $TOP_MODULE_NAME$/$IN_STREAM_WIDTH$/$OUT_STREAM_WIDTH$ (lines 119-121); written to <top>.v (line 123) — Verilog stub required because Vivado IP packaging rejects a SV top module.
- `requant/hdl/requant.sv` via **file-copy** — Added verbatim to file list at rtl/requant_rtl.py:141; parameterized indirectly via SV module params (VERSION/K/N/C/PE/SCALES/BIASES) flowing from the _impl wrapper through requant_axi into requant. Contains the actual float->fixed-point derive_PARAMS conversion (requant.sv:87-147).
- `requant/hdl/requant_axi.sv` via **file-copy** — Added verbatim at rtl/requant_rtl.py:142; AXIS wrapper, credit-based admission, recomputes stream widths independently of Python (requant_axi.sv:22-23).
- `requant/hdl/queue.sv` via **file-copy** — Added verbatim at rtl/requant_rtl.py:140; generic output elasticity queue instantiated by requant_axi with ELASTICITY=CREDIT.
- `requant/hdl/*` via **tcl-instantiate** — code_generation_ipi (rtl/requant_rtl.py:157-167) emits 'add_files -norecurse <f>' for every file and 'create_bd_cell -type module -reference <gen_top_module>' to instantiate the block in the BD.

## Seams (5)

- **clean-seam** — The agnostic Requant base cleanly owns shapes, datatypes, stream widths, and the Python golden model; both backends inherit it and add only codegen. Backend selection (Requant_hls vs Requant_rtl) is a clean substitution point.
- **clean-seam** — Param emission is cleanly per-backend: HLS generate_params writes params.h (hls:47), RTL generate_hdl fills SCALES/BIASES into the SV template (rtl:104-105). Neither leaks into the other file.
- **fused-no-seam** — execute_node is overridden in ALL THREE layers (Requant:136 Python model, Requant_hls:253 cpp/rtlsim, Requant_rtl:169 rtlsim). Both backend overrides exist ONLY to strip the embedded scale/bias inputs from the base HLSBackend/RTLBackend execute_node loops. The embedded-param design is smeared across every layer and cannot be separated from execution.
- **fused-no-seam** — HLS docompute optimization is fused to generate_params via _scale_is_one/_bias_is_zero instance state (hls:59,142) — the codegen of the compute loop cannot be generated without first having run param generation.
- **fused-no-seam** — The numeric algorithm differs between backends, not just its codegen: HLS does float multiply-add + hls::lrint (hls:181-184) while RTL converts scale/bias to fixed-point in SV (requant.sv derive_PARAMS) with 6-decimal-truncated inputs. There is no single shared quantization definition — the 'same op' has two independent implementations that can disagree.
