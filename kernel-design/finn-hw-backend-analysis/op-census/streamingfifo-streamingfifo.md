# Census: streamingfifo (StreamingFIFO)

*StreamingFIFO is a passthrough FIFO op with an RTL-only backend that, unusually, multiplexes two real backends (FINN Q_srl SRL RTL vs Xilinx axis_data_fifo Vivado IP) via the runtime impl_style attr. Its 'agnostic' base is not agnostic -- it reaches into subclass-only impl_style/get_adjusted_depth via try/except, and several resource-estimation guards are dead code from bool-vs-string comparisons.*

**Files:** `src/finn/custom_op/fpgadataflow/streamingfifo.py`, `src/finn/custom_op/fpgadataflow/rtl/streamingfifo_rtl.py`, `finn-rtllib/fifo/hdl/fifo_template.v`, `src/finn/custom_op/fpgadataflow/rtlbackend.py`, `src/finn/custom_op/fpgadataflow/hwcustomop.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `StreamingFIFO` | `streamingfifo.py` | `(HWCustomOp)` | yes |
| `StreamingFIFO_rtl` | `streamingfifo_rtl.py` | `(StreamingFIFO, RTLBackend)` | yes |

## Redesign pressure

The strongest redesign pressure is that StreamingFIFO does not fit the 2-axis HLS/RTL model at all: there is no HLS variant, and the single StreamingFIFO_rtl class secretly hosts a THIRD backend ('vivado', the Xilinx axis_data_fifo infrastructure IP) selected by the impl_style runtime attribute rather than by class hierarchy (streamingfifo_rtl.py:120-199, 53-67). Simultaneously the supposedly backend-agnostic base is not agnostic -- its shape/resource methods depend on impl_style and get_adjusted_depth, which exist only on the subclass, papered over with try/except AttributeError (streamingfifo.py:88-259), so base and backend are inseparable. Compounding this, several resource-estimation guards compare a bool to backend strings and are effectively dead code (streamingfifo.py:112, 165, 200, 254), signaling the impl_style branching has already rotted. A redesign should make each backend (FINN-RTL, Vivado-IP) a first-class variant and give the shared op a genuinely backend-free interface instead of exception-guarded subclass reach-through.

## Overrides (15)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | StreamingFIFO | 39 | Adds FIFO-specific attrs: depth, folded_shape, normal_shape, dataType, ram_style, depth_monitor, in/outFIFODepths, debug_log_path. Calls super(). |
| `get_normal_input_shape` | StreamingFIFO | 102 | Returns stored normal_shape attr; also emits an SRL-depth efficiency warning. |
| `get_normal_output_shape` | StreamingFIFO | 116 | FIFO is a passthrough: output shape == input shape. |
| `get_folded_input_shape` | StreamingFIFO | 119 | Returns stored folded_shape attr. |
| `get_folded_output_shape` | StreamingFIFO | 122 | Passthrough: same folded_shape as input. |
| `get_instream_width` | StreamingFIFO | 125 | width = folded_shape[-1] * dtype.bitwidth(). |
| `get_outstream_width` | StreamingFIFO | 131 | Passthrough: identical to instream width (body duplicated verbatim from get_instream_width). |
| `get_input_datatype` | StreamingFIFO | 137 | Returns single dataType attr. |
| `get_output_datatype` | StreamingFIFO | 140 | Passthrough: same dataType attr. |
| `execute_node` | StreamingFIFO | 143 | Functional/cppsim model of a FIFO is a pure copy input->output. |
| `get_verilog_top_module_intf_names` | StreamingFIFO | 87 | Adds 'maxcount' ap_none port when impl_style==rtl and depth_monitor==1. |
| `get_verilog_top_module_intf_names` | StreamingFIFO_rtl | 69 | Near-identical duplicate of the base override (maxcount port), but without the try/except guard. |
| `bram_estimation` | StreamingFIFO | 147 | Analytic BRAM count from depth/width buckets for ram_style==block. |
| `uram_estimation` | StreamingFIFO | 182 | Analytic URAM count for ram_style==ultra. |
| `lut_estimation` | StreamingFIFO | 235 | Address LUTs + LUTRAM for distributed/rtl impl. |

## Hacks (14 — 3 blocker, 4 major)

- **[blocker/base-class-leak]** `streamingfifo.py:104` — The 'agnostic' base StreamingFIFO calls self.get_adjusted_depth() wrapped in try/except AttributeError, falling back to raw depth attr. get_adjusted_depth() is defined ONLY on the RTL subclass (streamingfifo_rtl.py:53). Base op depends on a subclass-only method via exception control-flow. Repeated at lines 104, 160, 195, 208, 224, 247.
- **[blocker/base-class-leak]** `streamingfifo.py:90` — Base reads self.get_nodeattr('impl_style') inside try/except AttributeError, but impl_style is declared ONLY on the RTL subclass (streamingfifo_rtl.py:46). AttributeError branch raises 'still in hw abstraction format, run SpecializeLayers'. The base class is thus not backend-agnostic at all. Repeated at 90, 109, 150, 186, 238.
- **[blocker/cross-backend-leak]** `streamingfifo_rtl.py:141` — code_generation_ipi contains a large 'vivado' branch (141-195) that instantiates Xilinx axis_data_fifo:2.0 infrastructure IP via TCL -- a THIRD backend hidden inside the class named *_rtl. impl_style multiplexes {rtl, vivado} at runtime instead of via the class hierarchy; the RTL class carries an entire non-FINN-RTL codegen path.
- **[major/brittle-assumption]** `streamingfifo.py:165` — Dead-code logic bug: line 150 sets impl = (get_nodeattr('impl_style') == 'rtl'), so impl is a BOOL. Line 165 then tests `if impl == 'rtl' or (impl == 'vivado' and ram_type != 'block')` comparing a bool to strings -- always False. The non-BRAM early-return never fires via impl. Same pattern in uram_estimation (186/200) and lut_estimation (238/254).
- **[major/inheritance-irregularity]** `streamingfifo_rtl.py:41` — get_nodeattr_types does NOT call super(). It hand-builds a dict with impl_style, then explicitly merges StreamingFIFO.get_nodeattr_types(self) and RTLBackend.get_nodeattr_types(self). Deliberately bypasses MRO to control merge order and inject impl_style; fragile if a third base is added.
- **[major/template-surgery]** `streamingfifo_rtl.py:104` — generate_hdl reads fifo_template.v and does naive string .replace() of $KEY$ placeholders ($TOP_MODULE_NAME$, $COUNT_WIDTH$, $COUNT_RANGE$, $IN_RANGE$, $OUT_RANGE$, $WIDTH$, $DEPTH$, $DATA_LOGFILE$) in a loop (104-106), then writes the result. No structured templating.
- **[major/hard-coded-param]** `streamingfifo_rtl.py:78` — Hard-coded rtllib path built from os.environ['FINN_ROOT'] + '/finn-rtllib/fifo/hdl'. Same env-var path repeated in get_rtl_file_list (line 204).
- **[minor/brittle-assumption]** `streamingfifo.py:112` — get_normal_input_shape: line 109 assigns impl_style = (get_nodeattr('impl_style')=='rtl') (a bool), line 111 sets it to '' on AttributeError, then line 112 tests `if depth>256 and impl_style=='rtl'` -- bool/'' never equals 'rtl', so the SRL depth-efficiency warning is unreachable.
- **[minor/duplicated-logic]** `streamingfifo_rtl.py:69` — get_verilog_top_module_intf_names (69-75) is a near-verbatim copy of the parent StreamingFIFO version (streamingfifo.py:87-100), differing only in the try/except guard. Two copies of the maxcount-port logic drift-prone.
- **[minor/hard-coded-param]** `streamingfifo_rtl.py:113` — Source file names 'fifo_gauge.sv' and 'Q_srl.v' hard-coded in three places: shutil.copy in generate_hdl (113-114), sourcefiles list in code_generation_ipi (126-127), and get_rtl_file_list (210-211).
- **[minor/magic-number]** `streamingfifo_rtl.py:60` — get_adjusted_depth rounds depth up to nearest power-of-2 for impl_style=='vivado' (line 60) with redundant inline `if impl=='vivado' else depth` inside an already-vivado-guarded block. Magic power-of-2 rounding for the Vivado IP quirk.
- **[minor/todo-marker]** `streamingfifo.py:220` — TODO in uram_efficiency_estimation: Versal URAM flexible bit-width (9/18/36/72) not modeled vs UltraScale+ 72-bit assumption.
- **[minor/magic-number]** `streamingfifo.py:170` — bram_estimation width/depth bucket constants (16384, 8192, 4096, 2048, 1024, 512, 36, 18, 9, 4) and uram/bram capacity constants (72*4096, 36*512) hard-code device BRAM/URAM geometry inline with no named constants.
- **[minor/other]** `streamingfifo_rtl.py:223` — execute_node manually dispatches: cppsim -> StreamingFIFO.execute_node (copy), rtlsim -> RTLBackend.execute_node. Explicit class-qualified calls rather than cooperative super(); brittle to MRO changes.

## Hermeticity violations (5)

- **[env-var]** `streamingfifo_rtl.py:78` — os.environ['FINN_ROOT'] read in generate_hdl to locate rtllib sources.
- **[env-var]** `streamingfifo_rtl.py:204` — os.environ['FINN_ROOT'] read again in get_rtl_file_list.
- **[filesystem-path]** `streamingfifo_rtl.py:113` — shutil.copy of fifo_gauge.sv and Q_srl.v from rtllib into code_gen_dir; also writes generated top .v (107-111). Side-effecting filesystem I/O inside codegen.
- **[hidden-coupling]** `streamingfifo_rtl.py:99` — debug_log_path nodeattr is baked into $DATA_LOGFILE$ of the generated Verilog (fifo_gauge simulation logging), embedding a host filesystem path into HW source.
- **[module-mutable-state]** `streamingfifo_rtl.py:84` — generate_hdl mutates node attrs as side effects: gen_top_module (84), ipgen_path/ip_path (117-118). Later methods (code_generation_ipi, get_rtl_file_list) depend on gen_top_module being set first -- order-dependence.

## finn-rtllib coupling (3)

- `fifo/hdl/fifo_template.v` via **string-replace** — Read at streamingfifo_rtl.py:102, placeholders $TOP_MODULE_NAME$/$COUNT_WIDTH$/$COUNT_RANGE$/$IN_RANGE$/$OUT_RANGE$/$WIDTH$/$DEPTH$/$DATA_LOGFILE$ filled via .replace() loop at 104-106 (values assembled 87-99), written to <top>.v at 107-111. Template instantiates the actual FIFO cores.
- `fifo/hdl/Q_srl.v` via **verilog-template-fill** — The synthesizable SRL-based FIFO. Instantiated inside fifo_template.v line 62 as `Q_srl #(.depth($DEPTH$), .width($WIDTH$))` (non-FINN_SIMULATION branch). File copied verbatim at streamingfifo_rtl.py:114, listed in get_rtl_file_list at 210 and code_generation_ipi sourcefiles at 128.
- `fifo/hdl/fifo_gauge.sv` via **verilog-template-fill** — Simulation-only depth gauge. Instantiated in fifo_template.v line 55 under ifdef FINN_SIMULATION as fifo_gauge #(.WIDTH($WIDTH$), .COUNT_WIDTH($COUNT_WIDTH$), .DATA_LOGFILE("$DATA_LOGFILE$")). File copied at streamingfifo_rtl.py:113, listed at 211 and 127.

## Seams (3)

- **clean-seam** — The pure RTL codegen surface (generate_hdl, get_rtl_file_list, code_generation_ipi rtl-branch, prepare_rtlsim) lives cleanly on StreamingFIFO_rtl and pulls only from finn-rtllib/fifo/hdl via template-fill -- a well-isolated FINN-RTL backend.
- **fused-no-seam** — The 'agnostic' StreamingFIFO base is fused to its RTL subclass: bram/uram/lut estimation and get_normal_input_shape reach into impl_style and get_adjusted_depth (subclass-only) via try/except AttributeError (streamingfifo.py:88-114, 149-259). Base and backend cannot be separated -- the base is unusable standalone.
- **fused-no-seam** — impl_style multiplexes two disjoint backends (FINN Q_srl RTL vs Xilinx axis_data_fifo Vivado IP) inside a single StreamingFIFO_rtl class. code_generation_ipi (streamingfifo_rtl.py:120-199) and get_adjusted_depth branch on impl_style at runtime; the 'vivado' backend has no class of its own, so it cannot be swapped out without editing the RTL class.
