# Census: crop

*Crop is a dtype-preserving, SIMD-folded streaming crop op with a clean agnostic base (Crop over HWCustomOp) and a single HLS backend (Crop_hls over Crop+HLSBackend); no RTL variant exists. It is one of the tidier families, its only notable quirks being an MRO-driven execute_node override and sentinel-valued shape attributes.*

**Files:** `src/finn/custom_op/fpgadataflow/crop.py`, `src/finn/custom_op/fpgadataflow/hls/crop_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `Crop` | `crop.py` | `(HWCustomOp)` | yes |
| `Crop_hls` | `crop_hls.py` | `(Crop, HLSBackend)` | yes |

## Redesign pressure

This family is genuinely clean and simple: a dtype-preserving, HLS-only streaming op with a well-separated agnostic base (Crop) and a thin HLS backend (Crop_hls). It never touches the HWCustomOp.generate_hdl_* leaks, uses no env/filesystem/global state, and has no RTL variant, so the 2-axis HLS/RTL abstraction fits it comfortably. The only real friction points are (1) the execute_node MRO fusion — the base provides a numpy reference that must be manually shadowed by the backend, which any redesign should make an explicit 'reference-vs-backend-exec' contract rather than a base-ordering accident; and (2) sentinel encodings (ImgDim h==0 for 1D, numInputVectors==[0] for no-batch) that leak shape-semantics into magic values rather than typed shape descriptors.

## Overrides (18)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | Crop | 26 | Adds crop-specific attrs (ImgDim, NumChannels, Crop{North,South,West,East}, SIMD, numInputVectors) then super().update. |
| `get_normal_input_shape` | Crop | 41 | Builds NHWC/NWC shape from ImgDim+NumChannels, with h==0 meaning a 1D image and numInputVectors!=[0] gating batch dims. |
| `get_normal_output_shape` | Crop | 51 | Subtracts CropWest/East from width and CropNorth/South from height to size the cropped output. |
| `get_folded_input_shape` | Crop | 120 | Folds innermost channel dim by SIMD (asserts divisibility). |
| `get_folded_output_shape` | Crop | 112 | Folds cropped output channel dim by SIMD (asserts divisibility). |
| `get_input_datatype` | Crop | 86 | Returns DataType from the single 'DataType' attr. |
| `get_output_datatype` | Crop | 109 | Returns same DataType attr as input (crop is dtype-preserving). |
| `get_instream_width` | Crop | 99 | ibits * SIMD. |
| `get_outstream_width` | Crop | 104 | obits * SIMD (== instream since dtype preserved). |
| `get_exp_cycles` | Crop | 128 | Cycle estimate = numInputVectors*H*W*(ch//SIMD), with h==0 pretending height=1. |
| `execute_node` | Crop | 67 | numpy reference: slices input by crop offsets, dispatching on len(inp.shape) 2/3/4. |
| `execute_node` | Crop_hls | 88 | Forces HLSBackend.execute_node (cppsim/rtlsim) instead of Crop's numpy path via MRO. |
| `get_nodeattr_types` | Crop_hls | 21 | Merges Crop and HLSBackend attr dicts explicitly with '\|'. |
| `global_includes` | Crop_hls | 24 | Emits #include "crop.hpp". |
| `defines` | Crop_hls | 29 | Emits constexpr SIMD/H/W/CF/CROP_* and TV vector typedef for the HLS template. |
| `docompute` | Crop_hls | 51 | Instantiates hls::stream buffers and calls move()/crop<...>() templates. |
| `blackboxfunction` | Crop_hls | 65 | Top function signature with in0_V/out0_V AXIS streams. |
| `pragmas` | Crop_hls | 75 | AXIS interface + aggregate compact=bit + ap_ctrl_none + dataflow disable_start_propagation. |

## Hacks (9 — 0 blocker, 4 major)

- **[major/brittle-assumption]** `crop.py:44` — h==0 in ImgDim is a magic sentinel meaning '1D image' (no height axis). Repeated across get_normal_input_shape (44), get_normal_output_shape (61), get_exp_cycles (133), and crop_hls.defines (33) where it is silently remapped to height=1. Non-obvious contract encoded in a coordinate value.
- **[major/magic-number]** `crop.py:49` — numInputVectors uses [0] as a sentinel meaning 'no batch/vector dims'. Compared as `num_vec != [0]` at lines 49, 65, 139. A list-valued attr overloaded as a boolean flag.
- **[major/inheritance-irregularity]** `crop_hls.py:88` — Crop_hls.execute_node exists solely to override the MRO: Crop (first base) defines a numpy execute_node, so without this method a cppsim/rtlsim call would silently run the numpy reference. The 2-line override forces HLSBackend.execute_node. Fragile coupling to base-ordering.
- **[major/template-surgery]** `crop_hls.py:60` — docompute string-templates a call `crop< H, W, CF, CROP_N, CROP_E, CROP_S, CROP_W, TV>(src0, dst0)` and `move(...)` against crop.hpp. Template arg order and the move() helper are an implicit hardcoded contract with the hlslib header; a tab character is even embedded between W and CF.
- **[minor/inheritance-irregularity]** `crop.py:31` — CropNorth/CropSouth/CropWest/CropEast declared as type 'i' (int) but with default value [] (empty list) instead of 0 (lines 31-34). Type/default mismatch; only works because they are marked required=True so the default is never used.
- **[minor/duplicated-logic]** `crop.py:120` — get_folded_input_shape (120) and get_folded_output_shape (112) are byte-for-byte the same SIMD-fold logic differing only in normal-shape source; the standard boilerplate copied across FINN ops.
- **[minor/magic-number]** `crop_hls.py:56` — hls::stream depth hardcoded to 2 for both src0 and dst0 (lines 56-57), independent of image/tile size.
- **[minor/brittle-assumption]** `crop.py:75` — execute_node dispatches purely on len(inp.shape) (2/3/4) to choose the numpy slice; a 2D input (line 78) applies CropWest/CropEast on axis 0 (treating it as the width axis) with no channel/height crop, an implicit assumption tied to the h==0 1D-image encoding.
- **[minor/base-class-leak]** `crop_hls.py:22` — get_nodeattr_types uses explicit `Crop.get_nodeattr_types(self) | HLSBackend.get_nodeattr_types(self)` rather than cooperative super(); hardwires both parent names and dict-union semantics, so any third base is silently dropped.

## Hermeticity violations (0)


## Seams (3)

- **clean-seam** — Crop (crop.py) is a fully backend-agnostic base: shape/datatype/width/cycle logic and a numpy execute_node, with zero HLS or RTL references. A new backend (e.g. an RTL variant) could subclass Crop the same way Crop_hls does without touching the base.
- **clean-seam** — Crop_hls confines all HLS specifics (crop.hpp include, constexpr defines, docompute template call, blackbox/pragmas) to the four HLSBackend abstract methods plus pragmas; the HLS logic does not leak back into the base.
- **fused-no-seam** — execute_node is fused across the seam by MRO: the base Crop.execute_node (numpy golden) and HLSBackend.execute_node (sim) coexist, and Crop_hls must manually re-dispatch (crop_hls.py:88) to pick the backend path. Which execute_node runs is not cleanly selectable without knowledge of base ordering.
