# Census: concat (StreamingConcat)

*A clean HLS-only multi-input concat op: a fully backend-agnostic base (StreamingConcat) plus a thin HLS codegen subclass, with well-separated seams. Its main blemishes are internal to the base -- an accumulator-style output-datatype inference copied from an add-style op and unchecked SIMD divisibility -- not cross-backend leakage.*

**Files:** `src/finn/custom_op/fpgadataflow/concat.py`, `src/finn/custom_op/fpgadataflow/hls/concat_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `StreamingConcat` | `concat.py` | `(HWCustomOp)` | yes |
| `StreamingConcat_hls` | `concat_hls.py` | `(StreamingConcat, HLSBackend)` | yes |

## Redesign pressure

This family is one of the cleaner examples of the current 2-axis abstraction: the agnostic base (StreamingConcat) contains zero HLS/RTL leakage and the HLS variant contains only codegen, so backend substitution is genuinely clean and there is no reliance on the HWCustomOp.generate_hdl_* op_type-string leak. The real friction is not the HLS/RTL split but two op-specific policy warts baked into the base: (1) an accumulator-derived output-datatype inference with leftover 'acc_min/acc_max' comments copied from an add-style op, which is semantically off for a pack-only concat and mishandles mixed signed/unsigned inputs; and (2) unchecked SIMD-divides-channels integer divisions in the folding methods. There is also no RTL variant, so nothing yet stresses whether the base truly generalizes beyond HLS. Net: little redesign pressure on the backend axis, moderate pressure on cleaning op-level datatype/folding policy.

## Overrides (13)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | StreamingConcat | 45 | adds SIMD, ChannelsPerStream (per-input elem counts), inputDataTypes, numInputVectors; concat is a multi-input op so it needs a list-of-streams attr model |
| `get_normal_input_shape` | StreamingConcat | 68 | shape is per-input: ChannelsPerStream[ind] as last dim, prefixed by numInputVectors |
| `get_folded_input_shape` | StreamingConcat | 75 | folds channel dim of input ind by SIMD -> (vecs, folds, SIMD) |
| `get_normal_output_shape` | StreamingConcat | 81 | output channel dim is sum of all ChannelsPerStream (total_elems) |
| `get_folded_output_shape` | StreamingConcat | 86 | folds total_elems by SIMD |
| `get_input_datatype` | StreamingConcat | 110 | per-input datatype looked up from inputDataTypes[ind] list |
| `get_output_datatype` | StreamingConcat | 114 | output dt inferred by scanning min/max across ALL input datatypes and deriving a common UINT/INT bitwidth via log2 |
| `get_instream_width` | StreamingConcat | 136 | input stream width = input_dt(ind).bitwidth * SIMD |
| `get_outstream_width` | StreamingConcat | 140 | output stream width = output_dt.bitwidth * SIMD |
| `get_exp_cycles` | StreamingConcat | 145 | cycle estimate = product of folded output shape minus last dim |
| `execute_node` | StreamingConcat | 148 | pure-numpy functional execution via np.concatenate on last axis |
| `get_nodeattr_types` | StreamingConcat_hls | 41 | merges StreamingConcat attrs with HLSBackend attrs |
| `execute_node` | StreamingConcat_hls | 47 | routes to HLSBackend.execute_node (cppsim/rtlsim) instead of base numpy path |

## Hacks (8 — 0 blocker, 2 major)

- **[major/duplicated-logic]** `concat.py:114` — get_output_datatype derives the output datatype with accumulator-style bitwidth math (min/max scan + log2). The comments literally reference 'acc_max' (line 124) and 'acc_min' (line 128) -- accumulator terminology copied from an add/accumulate op. For a concat, values are packed side-by-side and are not summed, so the log2 bitwidth reasoning is semantically inappropriate; it just happens to yield a wide-enough common type. Mixing signed+unsigned inputs (e.g. INT8 + UINT8) inflates to INT9 rather than a natural pack.
- **[major/brittle-assumption]** `concat.py:77` — get_folded_input_shape does ChannelsPerStream[ind] // SIMD with no check that SIMD divides each stream's channel count; a non-dividing SIMD silently truncates the fold count. Same unchecked integer division for the output at line 89 (total_elems // SIMD).
- **[minor/brittle-assumption]** `concat.py:51` — inputDataTypes default is [""] (single empty string). get_input_datatype does DataType[inputDataTypes[ind]] which would raise on the empty-string default; correctness relies on the attr being populated (via infer_node_datatype) before any datatype/width query.
- **[minor/hard-coded-param]** `concat.py:40` — Op only supports concatenation along the last (channel) axis (docstring line 40, repeated in hls docstring line 36). numInputVectors folds all non-concat axes into a flat prefix; no support for arbitrary concat axis.
- **[minor/other]** `concat_hls.py:83` — pragmas() aliases the local list `pragmas` with self.code_gen_dict["$PRAGMAS$"] at line 88, then interleaves appends to both the local var (lines 91,92) and the dict entry (lines 89,93) relying on them being the same object. Correct output but a fragile mutable-aliasing pattern that breaks if line 88 is ever changed to copy.
- **[minor/other]** `concat_hls.py:57` — docompute() sets $DOCOMPUTE$ = [] at line 57 then unconditionally reassigns it at line 66; the initial empty-list assignment is dead code.
- **[minor/template-surgery]** `concat_hls.py:65` — docompute builds the C++ call StreamingConcat<fold0,fold1,...>(out0_V, in0_V, in1_V, ...) by string-joining per-input fold values as template args and per-input stream names. blackboxfunction (lines 68-81) likewise string-assembles the full void fn signature with N input hls::stream<hls::vector<...,SIMD>> refs. Variadic-input signature generation is purely string surgery.
- **[minor/other]** `concat_hls.py:77` — out_stream format uses a 1-element tuple with trailing comma `% (... ,)` for a single %s -- works but is an easy-to-break stylistic footgun in the signature template.

## Hermeticity violations (0)


## Seams (3)

- **clean-seam** — The base StreamingConcat (concat.py) is fully backend-agnostic: all shape/fold/datatype/width/exec logic is HLS/RTL-free, and StreamingConcat_hls holds ONLY the HLS codegen (global_includes/defines/docompute/blackboxfunction/pragmas). A new backend could inherit (StreamingConcat, RTLBackend) without touching the base -- a textbook clean 2-axis split.
- **clean-seam** — execute_node is cleanly layered: base provides a pure-numpy functional exec (concat.py:148) and the HLS variant explicitly delegates to HLSBackend.execute_node (concat_hls.py:47) to resolve the diamond, so backend swap does not disturb the functional reference implementation.
- **fused-no-seam** — get_output_datatype's accumulator-style bitwidth inference (concat.py:114-134) lives in the agnostic base -- it is backend-independent so it is not a cross-backend fusion, but it fuses concat's datatype policy into hard-coded log2 math that a redesign cannot parameterize without editing the base.
