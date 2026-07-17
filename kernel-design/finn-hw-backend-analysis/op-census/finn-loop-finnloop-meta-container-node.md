# Census: finn_loop (FINNLoop meta/container node)

*FINNLoop is a backend-only RTL meta-node that wraps a whole FINN-ONNX subgraph and executes it in a loop, assembling child IPs into a Vivado block design with a template-filled loop-control shell and per-layer stream taps. It is an outlier that delegates all core-contract accessors into its child graph and hard-codes deep knowledge of sibling ops (MVAU/Thresholding/Elementwise), file naming, and Vivado Tcl.*

**Files:** `src/finn/custom_op/fpgadataflow/rtl/finn_loop.py`, `src/finn/custom_op/fpgadataflow/rtlbackend.py`, `src/finn/custom_op/fpgadataflow/templates.py`, `finn-rtllib/mlo/loop_control_wrapper.v`, `finn-rtllib/stream_tap/hdl/stream_tap_wrapper_template.v`, `finn-rtllib/skid/skid.sv`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `FINNLoop` | `finn_loop.py` | `(HWCustomOp, RTLBackend)` | NO (backend-only) |

## Redesign pressure

FINNLoop is not a leaf kernel at all — it is a meta/container node that wraps an entire FINN-ONNX subgraph (stored in a graph-typed 'body' nodeattr) and stitches its children into a Vivado block design with a hand-written loop-control shell, per-layer stream taps, skid buffers, and an HBM/aximm memory bus. This shatters the current HLS/RTL 2-axis abstraction in several ways: (1) there is no HW-agnostic base and no HLS counterpart, so 'RTL variant of an op' is a category error — the RTLBackend abstract methods (get_rtl_file_list) are stubbed to None and its rtlsim/ipgen seams are overridden wholesale; (2) all eight core shape/width/datatype contract methods are delegated into child ops rather than owned, with a pervasive hard-coded 'param is input index 1' and op_type-string dispatch (MVAU/Thresholding/Elementwise) that leaks sibling-op internals into this op; (3) the real work (~540 lines of ipgen_singlenode_code plus generate_params file-surgery) is a Vivado-Tcl/subprocess orchestrator that depends on prior transformations (CreateStitchedIP metadata, per-child generate_params output filenames) and external tool state, none of which the abstraction models. A redesign needs a distinct 'container/hierarchical op' concept — subgraph-owning nodes with their own composition contract — rather than forcing this into the leaf HLS/RTL kernel taxonomy.

## Overrides (18)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | FINNLoop | 83 | Adds a 'body' attribute of ONNX type 'g' (a whole subgraph) plus iteration count and a cppsim-only iteration_context_path; the op wraps an entire FINN-ONNX model, not a leaf kernel. |
| `get_normal_input_shape` | FINNLoop | 146 | Delegates to the first node of the loop body (ind==0) or to the param consumer node (ind>0) rather than owning its own shape; a container has no intrinsic shape. |
| `get_normal_output_shape` | FINNLoop | 169 | Delegates to the last node of the loop body. |
| `get_folded_input_shape` | FINNLoop | 181 | Delegates folded shape to first body node / param consumer node (input 1). |
| `get_folded_output_shape` | FINNLoop | 197 | Delegates folded output shape to last body node. |
| `get_input_datatype` | FINNLoop | 208 | ind==0 reads own inputDataType nodeattr; ind>0 fetches from the param consumer node inside the body. |
| `get_output_datatype` | FINNLoop | 224 | Reads own outputDataType nodeattr. |
| `get_instream_width` | FINNLoop | 228 | Delegates to first body node (ind 0) or param consumer input 1. |
| `get_outstream_width` | FINNLoop | 261 | Delegates to last body node. |
| `get_exp_cycles` | FINNLoop | 244 | Runs AnnotateCycles + dataflow_performance analysis on the whole loop body, multiplies critical_path_cycles by iteration count, adds fixed per-iteration overhead. |
| `get_number_output_values` | FINNLoop | 269 | Delegates to last body node. |
| `generate_params` | FINNLoop | 403 | Regenerates per-iteration weight/threshold .dat files for each body param node across all iterations, concatenates them, and string-replaces .dat paths inside already-generated child _memstream_wrapper.v / .v files. |
| `generate_hdl` | FINNLoop | 365 | Fills the loop_control_wrapper.v template with iteration counts, stream widths, input/output byte sizes and a power-of-2 layer offset; also triggers stream-tap HDL and param generation. |
| `get_rtl_file_list` | FINNLoop | 1175 | Returns nothing (pass) — the loop's RTL is assembled via a Vivado block-design (ipgen_singlenode_code), not a flat file list. |
| `code_generation_ipi` | FINNLoop | 1156 | Collects child IP dirs, updates catalog, instantiates the packaged block-design IP by VLNV. |
| `get_verilog_top_module_intf_names` | FINNLoop | 1119 | Hand-builds interface dict: in0_V/out0_V AXIS, an aximm 'm_axi_hbm' bus, a 'done_if' control in ap_none, plus all aximm interfaces harvested from the body's stitched-IP metadata. |
| `execute_node` | FINNLoop | 300 | rtlsim mode packs/unpacks I/O and drives xsi with an MLO pre-hook; cppsim mode literally re-runs execute_onnx on the body subgraph 'iteration' times, feeding per-iteration param slices. |
| `verify_node` | FINNLoop | 0 | NOT overridden — inherits HWCustomOp default despite being a radically non-standard node. |

## Hacks (22 — 3 blocker, 13 major)

- **[blocker/inheritance-irregularity]** `finn_loop.py:1175` — get_rtl_file_list is a required RTLBackend @abstractmethod but is implemented as a bare 'pass' returning None. Any base-class code path that calls it (RTLBackend.prepare_rtlsim at rtlbackend.py:57) would break; FINNLoop dodges this by overriding prepare_rtlsim (line 277) to read all_verilog_srcs.txt instead. The abstract contract is satisfied only nominally.
- **[blocker/cross-backend-leak]** `finn_loop.py:421` — generate_params dispatches on child op_type string prefixes ('MVAU', 'Elementwise', 'Thresholding') to decide how to rename/concatenate .dat files, with 'else: raise Exception' (line 450). This hard-codes knowledge of three sibling ops' internal param-file naming conventions inside the loop op.
- **[blocker/brittle-assumption]** `finn_loop.py:712` — adjacency_list is built with a predicate hard-coding exactly three op_type families (Thresholding_rtl, MVAU_rtl, Elementwise*) each gated on a 'mlo_max_iter' attribute > 0. The block-design stream-tap wiring only works for these ops.
- **[major/inheritance-irregularity]** `finn_loop.py:99` — get_nodeattr and set_nodeattr are fully overridden to special-case dtype=='g' (a subgraph). For 'g' it wraps the AttributeProto graph into a ModelWrapper on read and CopyFrom on write. This duplicates and forks the base nodeattr container logic just to smuggle an entire model into a node attribute.
- **[major/magic-number]** `finn_loop.py:259` — overhead_per_iter = 40 is a hard-coded cycle overhead added per loop iteration in get_exp_cycles with no derivation or source.
- **[major/hard-coded-param]** `finn_loop.py:385` — LAYER_OFFS_INT is computed as 2**ceil(log2(input_bytes)); comment on line 388 literally says 'need to get correct value' — an admitted placeholder for the DRAM per-layer offset.
- **[major/template-surgery]** `finn_loop.py:468` — For Elementwise param nodes, walks the child's ipgen dir, opens any *_memstream_wrapper.v, and string-replaces the hard-coded 'memblock.dat' path with the concatenated per-loop file path. Comment (line 467) admits it is 'Adapted from transformations.fpgadataflow.replace_verilog_relpaths' — duplicated logic copied from a transformation.
- **[major/template-surgery]** `finn_loop.py:521` — For Thresholding param nodes, walks the child ipgen dir and string-replaces './<node.name>' with '<path>/Thresholding_id_<i+1>' in every .v file. Second copy of the replace_verilog_relpaths logic (comment line 522).
- **[major/magic-number]** `finn_loop.py:512` — Threshold .dat blocks are padded up to the next power of 2 (cnt & (cnt-1)) with pad_val = 2**o_bitwidth - 1, encoding an implicit hardware assumption about the stream-tap replication requiring power-of-2 depth.
- **[major/brittle-assumption]** `finn_loop.py:161` — Repeated assumption that 'the second input is the parameter input' (comments at 161, 191, 215, 238) — param streams are always assumed to be consumer input index 1. Any op that takes its parameter on a different input index breaks shape/datatype/width queries.
- **[major/brittle-assumption]** `finn_loop.py:285` — top_module_name = top_module_file_name.strip('.v') — str.strip strips the character SET {'.', 'v'} from both ends, not the '.v' suffix. A module name ending in 'v' or starting with '.' would be silently corrupted.
- **[major/brittle-assumption]** `finn_loop.py:558` — generate_hdl_stream_tap special-cases node.op_type == 'Thresholding_rtl' to compute TAP_REP = prod(folded_input_shape[:-1]); all other param nodes get tap_rep=1. Hard-coded op_type string dependency embedded in HDL generation.
- **[major/duplicated-logic]** `finn_loop.py:56` — collect_ip_dirs is a near-copy of the same-named helper in the stitched-IP transformation, including the 'MVAU'/'Thresholding_hls' + internal_decoupled memstreamer special-case (lines 68-74). Duplicated cross-op resource logic.
- **[major/brittle-assumption]** `finn_loop.py:757` — pruned_adj_list manipulation relies on stringly-typed sentinel node names '__INPUT0__', '__INPUT', '__OUTPUT0__' produced by adjacency_list, with fragile dict key/value inversion (762-763) and O(n^2) double-edge de-duplication (770-791). Highly implicit graph-shape assumptions.
- **[major/hard-coded-param]** `finn_loop.py:651` — ext_intf_signals = ['in0_V','out0_V','m_axi_hbm'] and ext_signals = ['done_if'] hard-code the exact top-level port names of the loop_control_wrapper shell.
- **[major/template-surgery]** `finn_loop.py:1024` — A run of set_property name commands (1024-1034) renames auto-generated '<sig>_0' external BD ports back to canonical names (in0_V, ap_clk, ap_rst_n, out0_V, m_axi_hbm, done_if, sim_finish). Brittle dependence on Vivado's '_0' suffixing convention.
- **[minor/magic-number]** `finn_loop.py:1121` — addr_bits = 64 hard-coded for the m_axi_hbm aximm interface width in get_verilog_top_module_intf_names.
- **[minor/todo-marker]** `finn_loop.py:547` — '# TODO check if this needs to be padded' next to stream-tap data_width computation.
- **[minor/todo-marker]** `finn_loop.py:1139` — '# TODO: rename because it might not be hbm?' — the m_axi_hbm interface name is a hard assumption about the backing memory being HBM.
- **[minor/hard-coded-param]** `finn_loop.py:1071` — Hotfix to ipx::remove_segment m_axi_gmem0:APERTURE_0 copied from IODMA packaging; hard-codes gmem0 segment name. Comment (1069-1070) flags it as a workaround for IP packager inferring bad aperture.
- **[minor/other]** `finn_loop.py:1036` — validate_bd_design is commented out (1036-1037), so the generated block design is never validated before packaging — errors surface only at synthesis.
- **[minor/brittle-assumption]** `finn_loop.py:205` — infer_node_datatype overridden to a no-op (pass), so datatype inference silently does nothing for this node; downstream relies on inputDataType/outputDataType being pre-set correctly.

## Hermeticity violations (13)

- **[env-var]** `finn_loop.py:390` — Reads os.environ['FINN_ROOT'] to locate the loop_control_wrapper.v template.
- **[env-var]** `finn_loop.py:541` — os.environ['FINN_ROOT'] again to locate stream_tap_wrapper_template.v.
- **[env-var]** `finn_loop.py:698` — os.environ['FINN_ROOT'] to locate skid.sv and stream_tap.sv in finn-rtllib.
- **[env-var]** `finn_loop.py:1082` — os.environ['FINN_ROOT'] to copytree the qnn-data/mdd-data example data dir into the stitch project.
- **[env-var]** `finn_loop.py:1103` — os.environ['PWD'] captured as working_dir to cd back after the Vivado batch run.
- **[filesystem-path]** `finn_loop.py:588` — Hard-coded Tcl path '$::env(FINN_ROOT)/finn-rtllib/memstream' injected as an ip_repo_path (also line 74 in collect_ip_dirs).
- **[filesystem-path]** `finn_loop.py:1111` — Spawns a subprocess ('bash make_loop_ip.sh') that launches Vivado in batch mode; heavyweight external-tool coupling inside a custom-op method (also resolve_xilinx_tool at 1104).
- **[hidden-coupling]** `finn_loop.py:955` — Consumes the loop body's stitched-IP metadata props (vivado_stitch_proj, vivado_stitch_vlnv, vivado_stitch_ifnames) via eval(); requires CreateStitchedIP to have already run on the child model — strong order-dependence on a prior transformation.
- **[hidden-coupling]** `finn_loop.py:984` — Wires a 'sim_finish' pin that only exists because CreateStitchedIP inserted a sim_ctrl into the body; comment (978-983) documents the dependency on the body's SystemVerilog final blocks flushing fifo_gauge logs during characterization.
- **[sibling-op-coupling]** `finn_loop.py:553` — Reads a 'mlo_max_iter' nodeattr off child op instances to decide which get a stream tap; this attribute is defined on the sibling MVAU/Thresholding/Elementwise ops, not on FINNLoop.
- **[filesystem-path]** `finn_loop.py:419` — Relies on child generate_params writing exactly '<path>/memblock.dat' (MVAU/Elementwise) and '<node.name>_threshs_<pe>_<stage>.dat' (Thresholding) — hard-coded knowledge of sibling ops' output filenames.
- **[filesystem-path]** `finn_loop.py:282` — prepare_rtlsim reads a side-channel file '<code_gen_dir_ipgen>/all_verilog_srcs.txt' that must have been produced earlier; order-dependent file coupling in place of get_rtl_file_list.
- **[hidden-coupling]** `finn_loop.py:296` — derive_characteristic_fxns injects an MLO-specific pre-hook (mlo_prehook_func_factory) into the base characterization flow; couples to finn.util.mlo_sim.

## finn-rtllib coupling (5)

- `mlo/loop_control_wrapper.v` via **verilog-template-fill** — generate_hdl (finn_loop.py:390-401) reads the template and str.replace-fills placeholders $LOOP_CONTROL_WRAPPER_NAME$, $N_MAX_LAYERS$, $N_LAYERS$, $ILEN_BITS$, $OLEN_BITS$, $INPUT_BYTES$, $OUTPUT_BYTES$, $LAYER_OFFS_INT$ (dict built 371-388) then writes <node>_wrapper.v into code_gen_dir_ipgen.
- `stream_tap/hdl/stream_tap_wrapper_template.v` via **verilog-template-fill** — generate_hdl_stream_tap (finn_loop.py:538-577) fills $MODULE_NAME$, $DATA_WIDTH$ (smallest-dtype-for-iteration rounded to multiple of 8), $TAP_REP$ per param node and writes IN_<idx>_stream_tap_wrapper.v; template placeholders confirmed at stream_tap_wrapper_template.v:34-36,70.
- `stream_tap/hdl/stream_tap.sv` via **file-copy** — ipgen_singlenode_code (finn_loop.py:699,708-710) locates stream_tap.sv and add_files -copy_to into the block-design source dir; instantiated as BD cells referencing the generated wrappers (735-751).
- `skid/skid.sv` via **file-copy** — skid_file built at finn_loop.py:698 and added via add_files -copy_to alongside stream_tap sources (708-710) to provide skid buffers for the stream-tap graph.
- `memstream` via **tcl-instantiate** — Both ipgen_singlenode_code (finn_loop.py:588) and collect_ip_dirs (line 74) inject '$::env(FINN_ROOT)/finn-rtllib/memstream' as an ip_repo_path so the child MVAU/Thresholding decoupled memstreamers resolve during catalog update.

## Seams (5)

- **fused-no-seam** — FINNLoop is a backend-only op (inherits HWCustomOp+RTLBackend directly, no HW-agnostic base). There is no HLS variant and the 2-axis abstraction has nothing to abstract: the entire 'implementation' is a ~540-line Vivado-Tcl block-design assembler (ipgen_singlenode_code) fused with child-subgraph introspection. Base contract methods and backend logic cannot be separated.
- **fused-no-seam** — Every core-contract shape/width/datatype accessor (146-275) is fused to reaching into the child loop-body subgraph via getCustomOp and assuming param-on-input-1. The op has no self-describing shape; you cannot swap a backend without also owning subgraph traversal.
- **fused-no-seam** — generate_params (403-536) fuses FINN-ONNX param regeneration, per-op .dat file naming knowledge (MVAU/Elementwise/Thresholding), power-of-2 padding, and child-Verilog string-surgery into one method. Backend RTL emission and model-level param handling are inseparable here.
- **clean-seam** — The three verilog template-fills (loop_control_wrapper.v, stream_tap_wrapper_template.v) via placeholder str.replace ARE a clean, isolated seam: swapping the RTL shell would only require new templates + the code_gen_dict keys, independent of the rest of the op.
- **fused-no-seam** — prepare_rtlsim (277-294) is overridden to bypass RTLBackend.prepare_rtlsim (which needs get_rtl_file_list) and instead read all_verilog_srcs.txt from the stitched block design. rtlsim path and the block-design assembly are fused; the RTLBackend rtlsim seam is deliberately broken (get_rtl_file_list returns None at 1175).
