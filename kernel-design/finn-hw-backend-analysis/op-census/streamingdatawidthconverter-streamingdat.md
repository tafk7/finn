# Census: streamingdatawidthconverter (StreamingDataWidthConverter / DWC)

*A datatype/shape-preserving bit-repacking op with a clean agnostic base but two functionally non-equivalent backends: the HLS variant supports arbitrary widths via an LCM two-stage path, while the RTL variant is restricted to integer width ratios and fills a Verilog template via raw $KEY$ string replacement. The key hazard is a base-class no-op capability hook (check_divisible_iowidths) that hides this backend-dependent constraint.*

**Files:** `src/finn/custom_op/fpgadataflow/streamingdatawidthconverter.py`, `src/finn/custom_op/fpgadataflow/hls/streamingdatawidthconverter_hls.py`, `src/finn/custom_op/fpgadataflow/rtl/streamingdatawidthconverter_rtl.py`, `finn-rtllib/dwc/hdl/dwc_template.v`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `StreamingDataWidthConverter` | `streamingdatawidthconverter.py` | `(HWCustomOp)` | yes |
| `StreamingDataWidthConverter_hls` | `streamingdatawidthconverter_hls.py` | `(StreamingDataWidthConverter, HLSBackend)` | yes |
| `StreamingDataWidthConverter_rtl` | `streamingdatawidthconverter_rtl.py` | `(StreamingDataWidthConverter, RTLBackend)` | yes |

## Redesign pressure

The DWC family is mostly clean at the shape/datatype layer (a genuinely backend-agnostic base), but it resists the 2-axis HLS/RTL abstraction because the two backends implement DIFFERENT converter algorithms with DIFFERENT capabilities: HLS supports arbitrary widths via an LCM two-stage stream (needs_lcm/get_iowidth_lcm woven into defines and docompute), while RTL only supports integer width ratios and enforces that by overriding the base's no-op check_divisible_iowidths hook. The base class silently under-specifies this — its empty check_divisible_iowidths pretends both backends accept the same configs, so backend choice can invalidate a node. Secondary pressure comes from the RTL variant's untyped $KEY$ string-replace template surgery plus the triplicated hard-coded SystemVerilog source list (dwc_axi.sv/dwc.sv across generate_hdl, get_rtl_file_list, code_generation_ipi) and the self-admitted 'copy-pasted' HLS codegen TODO. A redesign wants the per-backend capability constraint to be a first-class, declared contract rather than a base-class no-op hook.

## Overrides (17)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | StreamingDataWidthConverter | 43 | Adds DWC-specific attrs shape/inWidth/outWidth/dataType then merges super. |
| `get_input_datatype` | StreamingDataWidthConverter | 56 | Returns the single 'dataType' attr; DWC preserves datatype across conversion. |
| `get_output_datatype` | StreamingDataWidthConverter | 60 | Same single 'dataType' attr as input; DWC is datatype-preserving. |
| `get_normal_input_shape` | StreamingDataWidthConverter | 64 | Returns raw 'shape' attr verbatim. |
| `get_normal_output_shape` | StreamingDataWidthConverter | 68 | Returns raw 'shape' attr verbatim, identical to input. |
| `get_folded_input_shape` | StreamingDataWidthConverter | 87 | Folds last dim by inWidth//elem_bitwidth; used to size numInWords for HLS packing. |
| `get_folded_output_shape` | StreamingDataWidthConverter | 107 | Folds last dim by outWidth//elem_bitwidth. |
| `get_instream_width` | StreamingDataWidthConverter | 128 | Stream width IS the configured inWidth (the whole point of DWC). |
| `get_outstream_width` | StreamingDataWidthConverter | 132 | Stream width IS the configured outWidth. |
| `infer_node_datatype` | StreamingDataWidthConverter | 136 | Propagates input datatype to output unchanged, warns on change. |
| `verify_node` | StreamingDataWidthConverter | 150 | Checks backend attr and single data input. |
| `execute_node` | StreamingDataWidthConverter | 167 | Functional identity: copies input tensor to output (DWC only repacks bits, no math). |
| `lut_estimation` | StreamingDataWidthConverter | 178 | Models shift-based DWC LUT cost via lcm/gcd of in/out widths. |
| `get_nodeattr_types` | StreamingDataWidthConverter_hls | 44 | Merges base DWC attrs with HLSBackend attrs. |
| `execute_node` | StreamingDataWidthConverter_hls | 117 | cppsim path does identity copy; rtlsim delegates to HLSBackend. |
| `get_nodeattr_types` | StreamingDataWidthConverter_rtl | 42 | Merges base DWC attrs with RTLBackend attrs. |
| `execute_node` | StreamingDataWidthConverter_rtl | 66 | cppsim delegates to base identity copy; rtlsim delegates to RTLBackend. |

## Hacks (10 — 1 blocker, 5 major)

- **[blocker/brittle-assumption]** `streamingdatawidthconverter.py:84` — check_divisible_iowidths() is a no-op 'pass' in the agnostic base, silently permitting arbitrary in/out width ratios. The base thus admits configs (non-integer-ratio widths) that only the HLS backend can build via its LCM path; the RTL backend must re-add the constraint by overriding this hook (rtl line 48). Capability differs by backend but the base pretends both are equal.
- **[major/todo-marker]** `streamingdatawidthconverter_hls.py:81` — docompute() carries literal comment '# TODO continue with fxns below, they are copy-pasted' — self-admitted copy-paste of HLS codegen from another op, left unfinished.
- **[major/duplicated-logic]** `streamingdatawidthconverter_rtl.py:106` — The hard-coded SystemVerilog source list ['dwc_axi.sv','dwc.sv'] is repeated three times independently: generate_hdl (line 106), get_rtl_file_list (lines 122-124), and code_generation_ipi (lines 134-138). Adding/renaming a source requires editing all three.
- **[major/hard-coded-param]** `streamingdatawidthconverter_rtl.py:85` — rtllib source dir hard-coded as os.environ['FINN_ROOT'] + '/finn-rtllib/dwc/hdl'; template filename 'dwc_template.v' hard-coded at line 86. Path repeated with os.path.join form at line 117.
- **[major/template-surgery]** `streamingdatawidthconverter_rtl.py:96` — generate_hdl does raw string.replace of $KEY$ tokens over the verilog template text (loop at 96-98) to inject IBITS/OBITS/TOP_MODULE_NAME — untyped, no escaping, silent no-op if a token is missing/renamed.
- **[major/brittle-assumption]** `streamingdatawidthconverter_hls.py:66` — defines() asserts numInWords % (lcmWidth/inWidth)==0 with message 'Error in DWC LCM calculation'; the LCM two-stage codegen (docompute 84-90) is entirely HLS-only and has no RTL equivalent — the two backends implement genuinely different converter algorithms.
- **[minor/other]** `streamingdatawidthconverter.py:91` — get_folded_input_shape builds a real np.random.randn(*ishape) tensor and reshapes it just to read back dummy_t.shape — a throwaway allocation used purely to compute a tuple. Same trick at line 111 in get_folded_output_shape.
- **[minor/magic-number]** `streamingdatawidthconverter.py:175` — execute_node wraps output as np.asarray([output]) then reshapes to exp_shape — redundant listify+reshape identity dance; duplicated again in hls execute_node line 122.
- **[minor/inheritance-irregularity]** `streamingdatawidthconverter_hls.py:46` — get_nodeattr_types explicitly calls StreamingDataWidthConverter.get_nodeattr_types(self) and HLSBackend.get_nodeattr_types(self) by name rather than using super()/MRO — fragile to base reordering, and diamond-merges two parents manually. Same pattern in rtl line 44.
- **[minor/brittle-assumption]** `streamingdatawidthconverter_hls.py:119` — cppsim execute_node reshapes the input to exp_shape (normal INPUT shape) and calls it output — assumes in/out shapes identical and performs no actual width repacking, so cppsim never exercises the conversion logic (only rtlsim does).

## Hermeticity violations (5)

- **[env-var]** `streamingdatawidthconverter_rtl.py:85` — generate_hdl reads os.environ['FINN_ROOT'] to locate finn-rtllib/dwc/hdl — hard dependency on ambient env var.
- **[env-var]** `streamingdatawidthconverter_rtl.py:117` — get_rtl_file_list again reads os.environ['FINN_ROOT'] to build rtllib_dir.
- **[filesystem-path]** `streamingdatawidthconverter_rtl.py:108` — shutil.copy of dwc_axi.sv/dwc.sv from rtlsrc into code_gen_dir — side-effecting filesystem writes during generate_hdl.
- **[module-mutable-state]** `streamingdatawidthconverter_rtl.py:90` — Sets nodeattr 'gen_top_module' during generate_hdl and later reads it in get_rtl_file_list/code_generation_ipi — order-dependent: those methods break if generate_hdl hasn't run first.
- **[module-mutable-state]** `streamingdatawidthconverter_rtl.py:111` — Sets ipgen_path and ip_path nodeattrs purely to placate downstream HLS-Synth/stitch transforms (comment lines 109-110) — cross-transform coupling via node state.

## finn-rtllib coupling (3)

- `dwc/hdl/dwc_template.v` via **verilog-template-fill** — generate_hdl reads dwc_template.v (rtl line 86,94), then string-replaces $TOP_MODULE_NAME$, $IBITS$, $OBITS$ (values built in get_template_values lines 73-82; replace loop lines 96-98) and writes <top>.v. Template params drive dwc_axi #(.IBITS,.OBITS) instantiation (dwc_template.v line 57-59).
- `dwc/hdl/dwc_axi.sv` via **file-copy** — shutil.copy'd verbatim into code_gen_dir (rtl line 106-108); listed for synthesis in get_rtl_file_list (line 123) and code_generation_ipi (line 135). Not parameterized in Python — receives IBITS/OBITS through the generated top wrapper only.
- `dwc/hdl/dwc.sv` via **file-copy** — shutil.copy'd verbatim into code_gen_dir (rtl line 106-108); listed in get_rtl_file_list (line 124) and code_generation_ipi (line 136). Instantiated by dwc_axi.sv, no direct Python parameterization.

## Seams (4)

- **clean-seam** — The agnostic base (datatype/shape/stream-width/folding) is fully backend-independent; both _hls and _rtl inherit it unchanged. A new backend could subclass StreamingDataWidthConverter + <Backend> and only supply codegen.
- **clean-seam** — check_divisible_iowidths() is a deliberate hook: base no-op (line 84), RTL asserts integer-ratio widths (rtl line 48). This is the intended per-backend capability seam — though it silently under-constrains the base.
- **fused-no-seam** — The LCM/non-divisible-width conversion capability lives ONLY in the HLS backend (needs_lcm/get_iowidth_lcm consumed in defines line 64-69 and docompute line 83-90). The RTL backend cannot express it and forbids such configs. The two backends are NOT functionally interchangeable — same op, different supported width ratios — so they cannot be swapped transparently.
- **fused-no-seam** — execute_node is split three ways (base identity, hls cppsim/rtlsim branch, rtl cppsim/rtlsim branch) with each backend re-implementing the exec_mode dispatch; the functional (cppsim) semantics are duplicated identity copies rather than a shared seam.
