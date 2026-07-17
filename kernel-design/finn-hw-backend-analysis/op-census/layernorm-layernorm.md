# Census: layernorm (LayerNorm)

*A functionally-clean, weightless LayerNorm HW op with a fully backend-agnostic base (SIMD last-dim folding + torch golden model) and two well-separated HLS/RTL backends; the RTL variant is a thin template-fill + verbatim-copy of 5 finn-rtllib .sv sources. Its main abuse is that the input/output datatype attributes are decorative — both backends hard-code fp32/32-bit widths that the width methods nonetheless derive from those attrs.*

**Files:** `src/finn/custom_op/fpgadataflow/layernorm.py`, `src/finn/custom_op/fpgadataflow/hls/layernorm_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/layernorm_rtl.py`, `finn-rtllib/layernorm/layernorm_wrapper_template.v`, `src/finn/custom_op/fpgadataflow/rtlbackend.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `LayerNorm` | `layernorm.py` | `(HWCustomOp)` | yes |
| `LayerNorm_hls` | `layernorm_hls.py` | `(LayerNorm, HLSBackend)` | yes |
| `LayerNorm_rtl` | `layernorm_rtl.py` | `(LayerNorm, RTLBackend)` | yes |

## Redesign pressure

The family is structurally clean under the 2-axis model — a genuinely backend-agnostic base with two well-separated, non-contaminating backend classes and no reliance on the HWCustomOp generate_hdl_{memstream,fetch_weights,dynload} leak. The real pressure is the datatype axis, not the HLS/RTL axis: the op advertises inputDataType/outputDataType attrs and width methods derived from them, yet both backends silently pin data to 32-bit float (RTL wrapper hard-codes [31:0], HLS hard-codes TO=float), so the datatype contract is decorative and unenforced — a redesign needs a single authoritative width/datatype source that the RTL template and HLS defines both consume. Secondary pressure comes from the RTL variant hand-duplicating its 5-file source list three times and embedding microarchitecture-specific latency magic numbers in get_exp_cycles that are invisibly coupled to the finn-rtllib SV pipeline. The 'cppsim' exec_mode also overloads to mean 'run the torch golden model' for the RTL backend, blurring the sim-mode taxonomy.

## Overrides (25)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | LayerNorm | 28 | Adds op params SIMD, ifm_dim, epsilon, inputDataType, outputDataType on top of base attrs. |
| `get_normal_input_shape` | LayerNorm | 55 | Returns the ifm_dim attribute verbatim as the input shape. |
| `get_normal_output_shape` | LayerNorm | 58 | LayerNorm is shape-preserving so output == input. |
| `get_folded_input_shape` | LayerNorm | 61 | Folds last dim into (fold, SIMD); asserts SIMD divides last dim. |
| `get_folded_output_shape` | LayerNorm | 69 | Shape-preserving; output folding equals input folding. |
| `get_input_datatype` | LayerNorm | 72 | Reads inputDataType attr; only ind==0 valid. |
| `get_output_datatype` | LayerNorm | 79 | Reads outputDataType attr; ignores ind. |
| `infer_node_datatype` | LayerNorm | 83 | Warns if graph input dtype differs from attr, sets input attr and propagates output dtype. |
| `get_instream_width` | LayerNorm | 98 | in_bits * SIMD. |
| `get_outstream_width` | LayerNorm | 103 | out_bits * SIMD. |
| `execute_node` | LayerNorm | 42 | Functional reference via torch F.layer_norm over last dim with epsilon; weight/bias removed. |
| `get_nodeattr_types` | LayerNorm_hls | 19 | Merges LayerNorm + HLSBackend attrs and adds cpp_interface=hls_vector, hls_style=freerunning. |
| `global_includes` | LayerNorm_hls | 31 | HLS abstract: pulls hls_vector.h and layernorm.hpp. |
| `defines` | LayerNorm_hls | 37 | Emits SIMD, N, TI (from input dtype), TO=float constants. |
| `docompute` | LayerNorm_hls | 50 | Calls layernorm<N>(in0_V, out0_V). |
| `blackboxfunction` | LayerNorm_hls | 53 | Declares vector-stream top function. |
| `pragmas` | LayerNorm_hls | 63 | AXIS + aggregate compact=bit + ap_ctrl_none + dataflow disable_start_propagation (freerunning kernel). |
| `execute_node` | LayerNorm_hls | 76 | Delegates to HLSBackend.execute_node. |
| `timeout_value` | LayerNorm_hls | 79 | Sets rtlsim timeout to max(prod(input_shape),100). |
| `get_nodeattr_types` | LayerNorm_rtl | 30 | Merges RTLBackend + LayerNorm attrs. |
| `generate_hdl` | LayerNorm_rtl | 36 | RTL abstract: string-replace template fill + copies 5 .sv files, sets gen_top_module/ipgen_path/ip_path. |
| `get_rtl_file_list` | LayerNorm_rtl | 77 | Lists 5 rtllib .sv sources plus generated wrapper .v. |
| `code_generation_ipi` | LayerNorm_rtl | 95 | add_files for the 5 .sv + wrapper, then create_bd_cell. |
| `execute_node` | LayerNorm_rtl | 119 | cppsim->LayerNorm.execute_node (torch model); rtlsim->RTLBackend.execute_node. |
| `get_exp_cycles` | LayerNorm_rtl | 126 | Analytic latency: two queue depths (N//SIMD + ceil(log2(SIMD))*2 + {7,24}) plus prod(idim)//SIMD + 5. |

## Hacks (12 — 2 blocker, 5 major)

- **[blocker/magic-number]** `layernorm_wrapper_template.v:20` — Input TDATA width is hard-coded [$SIMD$-1:0][31:0] — every element is fixed at 32 bits regardless of inputDataType. get_instream_width() (layernorm.py:98) computes i_bits*SIMD from the datatype, so any input dtype whose bitwidth != 32 silently mismatches the RTL port. Effectively locks input to a 32-bit (float) type.
- **[blocker/magic-number]** `layernorm_wrapper_template.v:25` — Output TDATA hard-coded [$SIMD$-1:0][31:0] (32-bit float) while get_output_datatype/get_outstream_width read the outputDataType attr. The attr is effectively ignored by the RTL; output is always fp32.
- **[major/hard-coded-param]** `layernorm_hls.py:46` — defines() hard-codes 'using TO = float;' ignoring the outputDataType nodeattr entirely. get_outstream_width() still derives width from outputDataType.bitwidth(), so a non-fp32 outputDataType would desync the HLS output stream from the computed width.
- **[major/duplicated-logic]** `layernorm_rtl.py:69` — The .sv source list [layernorm.sv, queue.sv, accuf.sv, binopf.sv, rsqrtf.sv] is hand-duplicated in three places: generate_hdl (line 69), get_rtl_file_list (lines 85-91) and code_generation_ipi (lines 98-104). Adding/removing a source requires editing all three.
- **[major/magic-number]** `layernorm_rtl.py:134` — get_exp_cycles hard-codes pipeline-latency constants (+7 and +24 for the two queue lengths, +5 overall). These are undocumented magic numbers tied to the specific RTL microarchitecture in rsqrtf.sv/accuf.sv; any RTL pipeline change silently invalidates the estimate.
- **[major/inheritance-irregularity]** `layernorm_rtl.py:122` — For exec_mode 'cppsim' the RTL backend runs LayerNorm.execute_node (torch F.layer_norm golden model), not a compiled C++ sim. 'cppsim' here is a functional-python alias, differing in meaning from HLS cppsim.
- **[major/inheritance-irregularity]** `layernorm_hls.py:76` — execute_node must explicitly re-dispatch to HLSBackend.execute_node because the MRO (LayerNorm, HLSBackend) would otherwise resolve to LayerNorm.execute_node (the torch model). Fragile reliance on left-parent-wins ordering; a base rename breaks it silently.
- **[minor/template-surgery]** `layernorm_rtl.py:60` — generate_hdl does naive str.replace of $N$, $SIMD$, $TOP_MODULE_NAME$ over the whole wrapper template with no escaping/uniqueness guard. Works only because keys are distinct dollar-delimited tokens.
- **[minor/inheritance-irregularity]** `layernorm_rtl.py:30` — get_nodeattr_types merges RTLBackend first then LayerNorm, whereas the HLS variant (hls line 19) merges LayerNorm first then HLSBackend. Inconsistent merge ordering across the two backends of the same family; last-writer-wins on any colliding key differs between them.
- **[minor/brittle-assumption]** `layernorm.py:52` — execute_node normalizes only over the last dim [ishape[-1]] and drops weight & bias (affine params assumed folded away / absorbed). Output forced to float32. Assumes upstream transforms guaranteed weightless LayerNorm.
- **[minor/hard-coded-param]** `layernorm_hls.py:81` — timeout_value floor of 100 is a magic constant with no derivation from the actual pipeline depth.
- **[minor/brittle-assumption]** `layernorm.py:22` — LayerNorm base defines no make_shape_compatible_op, get_number_output_values, verify_node, or resource-estimation overrides; it relies entirely on HWCustomOp defaults. Resource/BRAM/DSP estimates for a fp32 rsqrt/accumulate pipeline are therefore whatever the generic default returns (likely 0/inaccurate).

## Hermeticity violations (4)

- **[env-var]** `layernorm_rtl.py:37` — generate_hdl reads os.environ['FINN_ROOT'] to locate finn-rtllib/layernorm/. Hard dependency on ambient env var + fixed filesystem layout.
- **[env-var]** `layernorm_rtl.py:80` — get_rtl_file_list also reads os.environ['FINN_ROOT'] (second, independent use of the same ambient path).
- **[filesystem-path]** `layernorm_rtl.py:37` — Hard-coded relative subpath 'finn-rtllib/layernorm/' assumed to exist under FINN_ROOT; template + 5 .sv filenames are string-literal paths.
- **[filesystem-path]** `layernorm_rtl.py:71` — generate_hdl performs shutil.copy of 5 .sv files into code_gen_dir_ipgen and mutates nodeattrs (gen_top_module, ipgen_path, ip_path). Side-effecting filesystem writes + node mutable state during HDL gen.

## finn-rtllib coupling (6)

- `layernorm/layernorm_wrapper_template.v` via **verilog-template-fill** — generate_hdl reads the wrapper template (rtl line 38/58), then str.replace injects $N$, $SIMD$, $TOP_MODULE_NAME$ (rtl lines 46-61) and writes <gen_top_module>.v. Parameters flow into the Verilog `layernorm #(.N,.SIMD)` instantiation at template lines 33-34.
- `layernorm/layernorm.sv` via **file-copy** — Copied verbatim via shutil.copy (rtl line 71); NOT parameterized directly — its N/SIMD come only through the wrapper's module instantiation. FORCE_BEHAVIORAL is toggled by the template's `ifdef FINN_SIMULATION guard (template lines 30-32).
- `layernorm/queue.sv` via **file-copy** — Copied verbatim (rtl line 71); no parameterization from Python.
- `layernorm/accuf.sv` via **file-copy** — Copied verbatim (rtl line 71); no parameterization from Python.
- `layernorm/binopf.sv` via **file-copy** — Copied verbatim (rtl line 71); no parameterization from Python.
- `layernorm/rsqrtf.sv` via **file-copy** — Copied verbatim (rtl line 71); no parameterization from Python. get_exp_cycles magic constants are implicitly tied to this pipeline's latency.

## Seams (4)

- **clean-seam** — The base LayerNorm class is fully backend-agnostic: shapes (SIMD folding of last dim), datatypes, and the torch golden model in execute_node contain no HLS/RTL concepts. Both LayerNorm_hls and LayerNorm_rtl attach cleanly on top.
- **clean-seam** — HLS variant fills only the four HLSBackend abstracts (global_includes/defines/docompute/blackboxfunction) plus pragmas/timeout_value; RTL variant fills only the three RTLBackend abstracts plus get_exp_cycles. Neither leaks logic into the other's file — no cross-backend contamination.
- **fused-no-seam** — The fp32/32-bit data width is fused implicitly across three layers with no single source of truth: HLS defines TO=float (hls:46), the RTL wrapper hard-codes [31:0] for both in/out TDATA (template:20,25), while get_instream_width/get_outstream_width derive width from the datatype attrs (layernorm.py:98,103). These cannot be varied independently — the datatype attributes are effectively decorative and cannot be cleanly swapped without editing the RTL template and HLS defines together.
- **fused-no-seam** — get_exp_cycles magic constants (rtl:134-136) are fused to the concrete rtllib pipeline depths (accuf.sv/rsqrtf.sv/queue.sv). The Python estimate and the SV implementation must change in lockstep; there is no seam decoupling them.
