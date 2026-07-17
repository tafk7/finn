# FINN HW Backend — Core Interface Contract (Phase A)

*As-is model of the four core abstraction files under
`src/finn/custom_op/fpgadataflow/`. This is the fixed vocabulary the Phase B
per-op census references: when an op "overrides a core method" or "violates a
core contract", it means one of the methods catalogued here.*

Source scan: 4 read-only agents (Opus, high effort), one per file.

---

## Summary: the true contract surface

| File | Class | Abstract (contract) | Concrete (defaults) | Hermeticity smells |
|------|-------|--------------------:|--------------------:|-------------------:|
| `hwcustomop.py` (603 LOC) | `HWCustomOp(CustomOp)` | **8** | 30 | 9 |
| `hlsbackend.py` (638 LOC) | `HLSBackend(ABC)` | **4** | 25 | 11 |
| `rtlbackend.py` (146 LOC) | `RTLBackend(ABC)` | **3** | 5 | 4 |
| `templates.py` (435 LOC) | module-level | 0 (6 template strings) | — | 7 |

**Central finding for the redesign:** "backend" is not a uniform concept. The
HLS contract (4 abstracts + 25 concrete helpers, 638 LOC) and the RTL contract
(3 abstracts + thin wrappers, 146 LOC) are wildly asymmetric in surface area and
responsibility. HLS backends generate C++ through a shared mutable
`code_gen_dict` side-channel; RTL backends emit/parameterize handwritten Verilog
from `finn-rtllib`. They share almost nothing beyond the `HWCustomOp` substrate.

---

## 1. `HWCustomOp` — the backend-agnostic substrate

Every op inherits this, HLS or RTL. Defines the node-attribute schema, resource
estimation hooks, rtlsim orchestration, HDL wrapper generation, and FIFO
characteristic-function derivation.

### The 8 abstract methods — THE contract every op variant must implement

| Method | Line | Signature | Returns |
|--------|-----:|-----------|---------|
| `get_input_datatype` | 261 | `(self, ind=0)` | FINN DataType of input stream |
| `get_output_datatype` | 265 | `(self, ind=0)` | FINN DataType of output stream |
| `get_normal_input_shape` | 269 | `(self, ind=0)` | unfolded input shape |
| `get_normal_output_shape` | 273 | `(self, ind=0)` | unfolded output shape |
| `get_folded_input_shape` | 277 | `(self, ind=0)` | folded shape (packed last dim) |
| `get_folded_output_shape` | 281 | `(self, ind=0)` | folded shape (packed last dim) |
| `get_instream_width` | 285 | `(self, ind=0)` | input stream bit width |
| `get_outstream_width` | 289 | `(self, ind=0)` | output stream bit width |

These 8 are the genuine downstream contract. `make_shape_compatible_op`,
`get_number_output_values`, `get_instream_width_padded/get_outstream_width_padded`,
and `node_res_estimation` all transitively depend on them.

### Concrete defaults (override seams)

- **Resource estimation** (all return trivial defaults, meant to be overridden):
  `bram_estimation`→0, `uram_estimation`→0, `lut_estimation`→0,
  `dsp_estimation`→0, `bram_efficiency_estimation`→1,
  `uram_efficiency_estimation`→1, `get_exp_cycles`→0, `get_op_and_param_counts`→{}.
  `node_res_estimation` (L165) aggregates these into a summary dict.
- **rtlsim lifecycle**: `get_rtlsim` (L144), `close_rtlsim` (L161),
  `reset_rtlsim` (L220), `rtlsim_multi_io` (L225) — all reach into the module
  global `finnxsi`.
- **Stream/shape helpers**: `get_verilog_top_module_name` (L108),
  `get_verilog_top_module_intf_names` (L116, always returns empty
  aximm/axilite/ap_none — pushes customization to subclasses),
  `get_instream_width_padded` (L292), `get_outstream_width_padded` (L301),
  `get_number_output_values` (L254), `make_shape_compatible_op` (L103).
- **No-op hooks**: `verify_node` (L242), `generate_params` (L248),
  `adapt_for_loop_body` (L585, called by LoopRolling).
- **Characteristic functions**: `derive_characteristic_fxns` (L440, 125-line
  method mixing rtlsim + accumulation + file offload), `get_io_chrc_in` (L565),
  `get_io_chrc_out` (L575).

### ⚠ The instantiation mess lives here (base→subclass hidden coupling)

Three HDL-wrapper generators on the *base class* hard-code knowledge of specific
subclasses — this is the abstraction leak the redesign targets:

- `generate_hdl_memstream` (L307) — branches on `op_type` ∈
  {MVAU_hls, VVAU, Thresholding_hls, Elementwise…}; calls `self.calc_wmem`,
  `self.calc_tmem`, `self.get_nodeattr('ram_style')` — **none defined on the base**.
- `generate_hdl_fetch_weights` (L355) — MVAU/Elementwise only; reads
  MW/MH/PE/SIMD/numInputVectors/rhs_shape. Contains TODO "use broadcast rhs
  shape here" (L375).
- `generate_hdl_dynload` (L407) — **no op_type guard, no docstring**;
  unconditionally assumes MVAU-style MW/MH/PE/SIMD attrs.

All three duplicate the same template-read/replace/write boilerplate and read
`os.environ["FINN_ROOT"]` to locate `finn-rtllib` templates.

### Hermeticity smells (9)

- **`finnxsi` singleton** (L39): resolved once at import; all rtlsim methods
  couple to it, not injectable.
- **`FINN_ROOT` env** (L313, L360, L409): the three `generate_hdl_*` helpers.
- **Filesystem layout**: `get_rtlsim` string-splits `rtlsim_so` on literal
  `'xsim.dir'` (L149); characteristic fxns write `io_chrc_*.npy` sidecars into
  `code_gen_dir_ipgen` and store abs paths as nodeattrs (L556).
- **global-config** (L230): `rtlsim_multi_io` reads process-wide
  `get_liveness_threshold_cycles()`.
- **hidden-coupling** (L311): the base-branches-on-subclass problem above.

### Doc bugs / notes
- L273 `get_normal_output_shape` docstring wrongly says "Returns folded output
  shape" (copy-paste from folded variant).

---

## 2. `HLSBackend` — the HLS (C++/Vitis) code-generation contract

Mixed into every `hls/` variant. Drives cppsim/rtlsim/ipgen by assembling C++
via a shared `self.code_gen_dict` of `$PLACEHOLDER$` → substitution lists, then
filling FINN templates.

### The 4 abstract methods — the per-op HLS contract

| Method | Line | Must populate |
|--------|-----:|---------------|
| `global_includes` | 418 | `code_gen_dict['$GLOBALS$']` (#include lines) |
| `defines` | 425 | `code_gen_dict['$DEFINES$']` (takes `var` = codegen mode) |
| `docompute` | 542 | `code_gen_dict['$DOCOMPUTE$']` (the HLS function call — the compute core) |
| `blackboxfunction` | 605 | `code_gen_dict['$BLACKBOXFUNCTION$']` (IP top signature) |

### Concrete defaults & override seams

- **Codegen orchestration**: `code_generation_ipgen` (L130),
  `code_generation_cppsim` (L213), `code_generation_ipi` (L246),
  `ipgen_singlenode_code` (L193), `compile_singlenode_code` (L252).
- **Override seams** (default behavior ops commonly customize):
  `read_npy_data` (L436, branches packed vs hls_vector), `strm_decl` (L482),
  `dataoutstrm` (L549), `pragmas` (L612, default axis in0_V/out0_V + ap_ctrl_none),
  `fold_input_for_npy` (L309), `ipgen_extra_directives` (L189, default empty),
  `ipgen_default_directives` (L177), `save_as_npy` (L601, empty).
- **Freerunning (`hls_style`) hooks**: `timeout_value` (L628, →'1000'),
  `timeout_condition` (L632), `timeout_read_stream` (L636).
- **Execution**: `execute_node` (L315, 100+ lines, mixes cppsim+rtlsim,
  duplicated invalid-exec_mode exception blocks at L324 and L411),
  `npy_to_dynamic_output` (L286), `exec_precompiled_singlenode_model` (L296).
- **Verilog discovery**: `get_all_verilog_paths` (L79),
  `get_all_verilog_filenames` (L102), `find_subcore_path` (L64) — all hard-code
  Vitis output directory layout.
- `get_ap_int_max_w` (L619, asserts ≤ 8191), `get_nodeattr_types` (L53, adds
  code_gen_dir_cppsim/executable_path/res_hls/**cpp_interface**/**hls_style**).

### Hermeticity smells (11)
- **`finnxsi` singleton** (L44). **Env vars**: `XILINX_VIVADO` (L256, regex, no
  None-guard), `HLS_PATH` (L261), `VITIS_PATH` (L263). **FINN_ROOT-relative
  include paths** hard-coded into compile flags (L267–274). **Vitis dir layout**
  hard-coded in glob/paths (L67, L87).
- **`code_gen_dict` mutable side-channel** (L136): steps communicate only
  through this ordering-dependent instance dict (populated, then cleared at
  L154/175/244) — non-composable, hard to test in isolation.

### Notes / latent bugs
- **`cpp_interface` / `hls_style`** described in-code (L58–61) as "temporary"
  node attributes — now load-bearing branching (packed vs hls_vector,
  ifm_aware vs freerunning) across read_npy_data/strm_decl/dataoutstrm.
- **Multi-input vector bug**: `read_npy_data` hls_vector branch (L470) calls
  `get_folded_input_shape()` with no index — always uses input 0.
- `get_ap_int_max_w` uses default ind=0 → only reflects stream-0 widths.

---

## 3. `RTLBackend` — the RTL (handwritten Verilog) contract

Much thinner. Mixed into `rtl/` variants; each corresponds to a `finn-rtllib`
module.

### The 3 abstract methods — the per-op RTL contract

| Method | Line | Responsibility |
|--------|-----:|----------------|
| `generate_hdl` | 53 | `(self, model, fpgapart, clk)` — generate the RTL source |
| `get_rtl_file_list` | 77 | `(self, abspath=False)` — list RTL source files |
| `code_generation_ipi` | 82 | `(self)` — emit TCL to instantiate the RTL module into IPI |

### Concrete
- `code_generation_ipgen` (L85) — thin wrapper delegating to `generate_hdl`.
- `prepare_rtlsim` (L56), `get_verilog_paths` (L70), `execute_node` (L88,
  rtlsim-only), `get_nodeattr_types` (L46, adds `gen_top_module`).

### Hermeticity smells (4)
- **`finnxsi` singleton** (L37); AttributeError if xsi unavailable (L64, no
  guard). `make_build_dir` side-effect (L61). npy scratch-file paths off
  `code_gen_dir_ipgen` (L104).

### Notes
- The 3 abstracts assume a rich host object (`onnx_node`, `get_nodeattr`,
  `get_normal_input_shape`, `get_instream_width`, `get_rtlsim`…) provided by the
  `HWCustomOp` base RTLBackend is mixed into but **does not declare or inherit**
  — an implicit, undeclared interface dependency (mirror of the HLS side).
- `execute_node` error message says exec_mode must be `("cppsim","rtlsim")` but
  RTLBackend has no cppsim path — misleading.

---

## 4. `templates.py` — shared code-gen string templates

Six module-level string constants (no functions/classes/abstracts):

| Template | Line | Purpose |
|----------|-----:|---------|
| `docompute_template` | 31 | C++ cppsim testbench main() |
| `docompute_template_timeout` | 64 | freerunning variant (⚠ omits `HLS_NO_XIL_FPO_LIB` that the base has) |
| `ipgen_template` | 111 | C++ top-level for HLS IP gen |
| `ipgentcl_template` | 128 | Vitis/Vivado HLS synth TCL driver |
| `ip_package_tcl` | 157 | Vivado ipx packaging (hard-coded device allowlist L186–208) |
| `ip_gen_loop_op` | 261 | MLO/CDMA loop-op flow (⚠ uses `@VAR@` not `$VAR$`) |

### Hermeticity smells (7)
- **`$::env(FINN_ROOT)` embedded in generated TCL** (L134, L136, L331 ×~27
  add_files lines). **Hard-coded finn-rtllib source paths** baked into
  `ip_gen_loop_op` (L331–357: cdma/dwc/skid/ram/mlo/fifo subtrees) — any RTL
  library reorg silently breaks IP gen. **Working-dir assumptions** (`ip/src`,
  `ip/component.xml`, L375–417).

### Notes
- **Two placeholder conventions** in one file: `$VAR$` (most) vs `@VAR@`
  (`ip_gen_loop_op`).
- `ip_gen_loop_op` hard-codes special-case string matches on `*FINNLoop*`
  (L423) and component.xml surgery (`.xci`, `spirit:fileSet`, `xilinx:subCoreRef`)
  — brittle string logic in a data template.

---

## Cross-cutting themes (seed for Phase C)

1. **Undeclared host-object interface.** Both HLSBackend and RTLBackend are
   mixins that call ~15 `HWCustomOp` methods without declaring the dependency.
   The real coupling is implicit multiple-inheritance, not an interface.
2. **`code_gen_dict` side-channel.** HLS codegen is a sequence of methods
   mutating one shared dict in a required order — the opposite of hermetic.
3. **Base-knows-subclasses leak.** `HWCustomOp.generate_hdl_{memstream,
   fetch_weights,dynload}` branch on concrete op types and call methods that
   don't exist on the base. This is the MVAU "instantiation mess" root.
4. **Ambient config everywhere.** `finnxsi` import-time singleton, `FINN_ROOT`,
   `XILINX_VIVADO`/`HLS_PATH`/`VITIS_PATH`, liveness-threshold global, and
   hard-coded Vitis/finn-rtllib filesystem layouts.
5. **HLS ≫ RTL surface asymmetry.** Any redesign must reconcile that "HLS
   backend" and "RTL backend" are not peers under a common contract today.
