# Census: fmpadding (FMPadding)

*A clean RTL-only padding op: FMPadding is a fully backend-agnostic base implementing all 8 contract methods plus a numpy np.pad reference, and FMPadding_rtl is a thin (FMPadding, RTLBackend) mixin that fills a Verilog template via string-replace and copies three static SV sources from finn-rtllib. Main smells are env-var/filesystem coupling, triple-duplicated hard-coded source lists, hard-coded AXI-lite register offsets, and a 'cppsim' mode that is actually numpy.*

**Files:** `src/finn/custom_op/fpgadataflow/fmpadding.py`, `src/finn/custom_op/fpgadataflow/rtl/fmpadding_rtl.py`, `finn-rtllib/fmpadding/hdl/fmpadding_template.v`, `src/finn/custom_op/fpgadataflow/rtlbackend.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `FMPadding` | `fmpadding.py` | `(HWCustomOp)` | yes |
| `FMPadding_rtl` | `fmpadding_rtl.py` | `(FMPadding, RTLBackend)` | yes |

## Redesign pressure

This family is one of the cleaner ones: a genuinely backend-agnostic base (FMPadding) implements all 8 contract methods plus a numpy reference, and the RTL variant is a thin mixin. The main friction with the current 2-axis HLS/RTL abstraction is that fmpadding is RTL-only with NO HLS sibling, yet still carries an exec_mode=='cppsim' path that is really numpy — the 'cppsim' axis is a misnomer here and reveals the HLS/RTL dichotomy doesn't map onto 'has-C++-codegen vs functional-sim'. The second pressure is that all RTL coupling is expressed as raw string.replace template surgery plus triply-duplicated hard-coded SV filename lists and hard-coded AXI-lite register byte offsets that mirror axi2we.sv — there is no structured contract binding the Python template values to the RTL module's parameter/register interface, so the seam between op and finn-rtllib is convention-only and brittle. A redesign would benefit from a declarative rtllib-source manifest and a typed template/parameter binding rather than env-var path lookup + str.replace.

## Overrides (17)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | FMPadding | 43 | Declares padding-specific attrs (ImgDim, Padding[Hb,Wb,He,We], NumChannels, SIMD, inputDataType, numInputVectors); merges super via update. |
| `get_nodeattr_types` | FMPadding_rtl | 46 | Adds dynamic_mode {0,1} for runtime-reprogrammable padding; explicitly merges FMPadding + RTLBackend attr dicts by hand. |
| `get_normal_input_shape` | FMPadding | 83 | NHWC image shape (1, H, W, C) from ImgDim/NumChannels. |
| `get_normal_output_shape` | FMPadding | 89 | Padded NHWC (1, odim_h, odim_w, C) using get_padded_odim. |
| `get_folded_input_shape` | FMPadding | 96 | Folds channel dim by SIMD -> (...,fold,simd); asserts SIMD divides channels. |
| `get_folded_output_shape` | FMPadding | 105 | SIMD fold of padded output shape. |
| `get_input_datatype` | FMPadding | 127 | Reads inputDataType attr; asserts DataType can represent 0 because hlslib pads with zeros. |
| `get_output_datatype` | FMPadding | 135 | Padding preserves datatype -> returns input datatype. |
| `get_instream_width` | FMPadding | 139 | ibits * SIMD. |
| `get_outstream_width` | FMPadding | 144 | obits * SIMD. |
| `get_exp_cycles` | FMPadding | 75 | (channels/simd)*batch*odim_h*odim_w cycle estimate over padded output. |
| `execute_node` | FMPadding | 149 | Pure-numpy functional sim via np.pad with constant (zero) padding. |
| `execute_node` | FMPadding_rtl | 183 | Dispatches by exec_mode: cppsim -> FMPadding.execute_node (numpy), rtlsim -> RTLBackend.execute_node. |
| `get_verilog_top_module_intf_names` | FMPadding_rtl | 56 | Adds s_axilite interface only when dynamic_mode enabled. |
| `generate_hdl` | FMPadding_rtl | 110 | RTLBackend abstract impl: fills fmpadding_template.v and copies static SV sources into code_gen dir. |
| `get_rtl_file_list` | FMPadding_rtl | 145 | Lists 3 static SV files + generated top .v; abspath toggles rtllib vs codegen dir prefixes. |
| `code_generation_ipi` | FMPadding_rtl | 161 | Emits Vivado TCL add_files + create_bd_cell -type module for the generated top. |

## Hacks (10 — 0 blocker, 6 major)

- **[major/hard-coded-param]** `fmpadding_rtl.py:111` — generate_hdl builds rtlsrc path from os.environ['FINN_ROOT'] + '/finn-rtllib/fmpadding/hdl' to locate the Verilog template and SV sources. Hard filesystem/env-var dependency.
- **[major/duplicated-logic]** `fmpadding_rtl.py:148` — get_rtl_file_list independently re-derives rtllib_dir from os.environ['FINN_ROOT'] via os.path.join; same env-var path coupling repeated in a second method.
- **[major/template-surgery]** `fmpadding_rtl.py:127` — generate_hdl does raw string.replace of '$KEY$' placeholders in fmpadding_template.v for each code_gen_dict entry (lines 127-129). No structured templating; relies on exact $NAME$ tokens present in the .v (e.g. $STREAM_BITS$, $XCOUNTER_BITS$, $INIT_XON$...).
- **[major/duplicated-logic]** `fmpadding_rtl.py:137` — The static SV source list ['fmpadding_axi.sv','fmpadding.sv','axi2we.sv'] is hard-coded THREE times: generate_hdl copy loop (137), get_rtl_file_list (153-156), code_generation_ipi (165-170). Adding/renaming an SV file requires editing all three.
- **[major/brittle-assumption]** `fmpadding_rtl.py:101` — get_dynamic_config hard-codes AXI-lite register byte offsets (0*4..5*4) for XON/XOFF/XEND/YON/YOFF/YEND. These must exactly match the axi2we.sv register map; any HW register reorder silently breaks runtime reconfig.
- **[major/brittle-assumption]** `fmpadding_rtl.py:121` — generate_hdl caches the top module name into gen_top_module nodeattr because later passes (GiveUniqueNodeNames prefix during MakeZynqProject) rename the node; get_rtl_file_list/code_generation_ipi then read gen_top_module. Order-dependent: file list is wrong if generate_hdl hasn't run first.
- **[minor/magic-number]** `fmpadding_rtl.py:70` — STREAM_BITS forced to roundup_to_integer_multiple(stream_bits, 8) — magic 8-bit AXI-stream byte alignment baked into template value; not derived from a named constant.
- **[minor/brittle-assumption]** `fmpadding.py:132` — get_input_datatype assertion message reads 'FMPadding_Batch DataType must support zero' — stale copy-paste of the old op class name FMPadding_Batch, which no longer matches this class (FMPadding). Indicates logic lifted from a predecessor op.
- **[minor/brittle-assumption]** `fmpadding.py:86` — get_normal_input_shape / get_normal_output_shape hard-code batch dim to 1 and ignore the numInputVectors attr, yet get_exp_cycles (line 79) multiplies by numInputVectors as batch_size. Inconsistent treatment of batch across shape vs cycle model.
- **[minor/other]** `fmpadding_rtl.py:183` — execute_node treats exec_mode=='cppsim' as an alias for the pure-numpy FMPadding.execute_node — there is no actual C++ codegen path for this op, so 'cppsim' is a misnomer that quietly runs the abstraction-level reference.

## Hermeticity violations (5)

- **[env-var]** `fmpadding_rtl.py:111` — os.environ['FINN_ROOT'] read in generate_hdl to locate finn-rtllib template/sources.
- **[env-var]** `fmpadding_rtl.py:148` — os.environ['FINN_ROOT'] read again in get_rtl_file_list.
- **[filesystem-path]** `fmpadding_rtl.py:111` — Hard-coded relative subpath '/finn-rtllib/fmpadding/hdl' and template name 'fmpadding_template.v'; reads and copies files from the source tree at codegen time.
- **[filesystem-path]** `fmpadding_rtl.py:137` — shutil.copy of SV sources into code_gen_dir_ipgen; writes generated top .v into same dir (line 131). Side-effecting filesystem mutation during generate_hdl.
- **[order-dependence]** `fmpadding_rtl.py:121` — Sets gen_top_module / ipgen_path / ip_path nodeattrs (121,142,143) as mutable state that get_rtl_file_list and code_generation_ipi later depend on; those methods break if generate_hdl hasn't executed.

## finn-rtllib coupling (4)

- `fmpadding/hdl/fmpadding_template.v` via **verilog-template-fill** — Read at rtl/fmpadding_rtl.py:112-129; $KEY$ tokens (TOP_MODULE_NAME, STREAM_BITS, XCOUNTER_BITS, YCOUNTER_BITS, NUM_CHANNELS, SIMD, ELEM_BITS, INIT_X/Y ON/OFF/END) replaced via str.replace loop (127-129) using code_gen_dict from get_template_values (63-86). Written as <top>.v (131-135).
- `fmpadding/hdl/fmpadding_axi.sv` via **file-copy** — shutil.copy from FINN_ROOT/finn-rtllib/fmpadding/hdl into code_gen_dir (rtl/fmpadding_rtl.py:137-139); instantiated as parameterized submodule 'fmpadding_axi #(...)' inside the filled template (fmpadding_template.v:77-90).
- `fmpadding/hdl/fmpadding.sv` via **file-copy** — Copied unmodified (rtl/fmpadding_rtl.py:137-139); core padding datapath used by fmpadding_axi.sv. Listed in get_rtl_file_list (155) and code_generation_ipi (167).
- `fmpadding/hdl/axi2we.sv` via **file-copy** — Copied unmodified (rtl/fmpadding_rtl.py:137-139); AXI-lite-to-write-enable register block whose register offsets are mirrored by get_dynamic_config offsets 0*4..5*4 (88-108).

## Seams (4)

- **clean-seam** — FMPadding (agnostic base) cleanly holds all 8 contract methods + numpy execute_node; FMPadding_rtl adds only RTL codegen (generate_hdl/get_rtl_file_list/code_generation_ipi) via RTLBackend mixin. A new backend could subclass (FMPadding, XBackend) without touching the base — the shape/datatype/reference-sim layer is fully backend-agnostic.
- **clean-seam** — execute_node dispatch (rtl/fmpadding_rtl.py:183-188) delegates cppsim to base numpy and rtlsim to RTLBackend, so the functional reference is reusable across backends unchanged.
- **fused-no-seam** — get_dynamic_config register offsets (88-108) and get_template_values INIT_X/Y* fields (63-86) are fused to the specific axi2we.sv / fmpadding_axi.sv register map and parameter names; the Python config layout cannot be separated from these exact RTL sources.
- **fused-no-seam** — The three hard-coded SV filename lists (137/153/165) fuse the Python op to the exact finn-rtllib file set; swapping the RTL implementation requires editing three separate Python methods, not one seam.
