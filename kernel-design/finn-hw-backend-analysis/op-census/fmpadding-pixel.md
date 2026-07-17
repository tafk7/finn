# Census: fmpadding_pixel

*FMPadding_Pixel is a datatype-preserving pixel/dilation (stride-insertion) padding op with a clean backend-agnostic base and a single thin HLS backend; there is no RTL variant and no finn-rtllib coupling. Its only notable irregularities are a manual execute_node stub to resolve the base/backend MRO diamond and inconsistent batch (numInputVectors vs hard-coded 1) handling.*

**Files:** `src/finn/custom_op/fpgadataflow/fmpadding_pixel.py`, `src/finn/custom_op/fpgadataflow/hls/fmpadding_pixel_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `FMPadding_Pixel` | `fmpadding_pixel.py` | `(HWCustomOp)` | yes |
| `FMPadding_Pixel_hls` | `fmpadding_pixel_hls.py` | `(FMPadding_Pixel, HLSBackend)` | yes |

## Redesign pressure

This family is mostly clean and simple: a datatype-preserving pixel/dilation padder with an agnostic base that is genuinely backend-free and a thin HLS backend that only fills the four HLSBackend template hooks. It is HLS-only (no RTL variant, no finn-rtllib coupling; the kernel comes from hlslib streamtools.h), so the 2-axis abstraction fits it well. The two real points of friction are (1) the manual execute_node re-dispatch stub needed to defeat the (agnostic-base, backend) MRO diamond so simulation resolves to the backend rather than the Python reference — a pattern that a redesign should formalize (e.g. explicit sim-vs-backend dispatch) rather than leave to per-op stubs, and (2) inconsistent batch handling (normal shapes hard-code batch=1 while cycle estimate and sim honor numInputVectors). The axis-swap magic in defines() and the hard-coded hlslib template name are minor template-surgery smells common to all HLS ops.

## Overrides (13)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | FMPadding_Pixel | 41 | Declares op-specific attrs (ImgDim, Stride, NumChannels, SIMD, inputDataType, numInputVectors) then merges super(). |
| `get_normal_input_shape` | FMPadding_Pixel | 75 | NHWC input geometry from ImgDim/NumChannels. |
| `get_normal_output_shape` | FMPadding_Pixel | 81 | Output spatial size computed via pixel-padding formula odim = idim + (idim-1)*(stride-1) using helper get_padded_odim. |
| `get_folded_input_shape` | FMPadding_Pixel | 87 | Splits channel dim into (fold, SIMD); asserts NumChannels % SIMD == 0. |
| `get_folded_output_shape` | FMPadding_Pixel | 96 | Same SIMD channel fold applied to padded output shape. |
| `get_input_datatype` | FMPadding_Pixel | 118 | Returns DataType from inputDataType attr, but additionally asserts the DataType can represent 0 because hlslib always pads with zeros. |
| `get_output_datatype` | FMPadding_Pixel | 126 | Padding is datatype-preserving; output dtype == input dtype. |
| `get_instream_width` | FMPadding_Pixel | 130 | ibits * SIMD packing width. |
| `get_outstream_width` | FMPadding_Pixel | 135 | obits * SIMD; identical structure to instream width. |
| `get_exp_cycles` | FMPadding_Pixel | 67 | Cycle estimate = (channels/SIMD) * batch * odim_h * odim_w over the padded output geometry. |
| `execute_node` | FMPadding_Pixel | 140 | Pure-Python functional reference: allocates zero tensor and scatters input pixels at stride*index positions (pixel/dilation padding). |
| `get_nodeattr_types` | FMPadding_Pixel_hls | 37 | Merges FMPadding_Pixel attrs with HLSBackend attrs explicitly by classname rather than via super(). |
| `execute_node` | FMPadding_Pixel_hls | 91 | Re-dispatches to HLSBackend.execute_node to run cppsim/rtlsim path instead of the Python sim inherited first in MRO. |

## Hacks (7 — 0 blocker, 4 major)

- **[major/inheritance-irregularity]** `fmpadding_pixel_hls.py:91` — FMPadding_Pixel_hls.execute_node exists solely to break the diamond: because bases are declared (FMPadding_Pixel, HLSBackend), MRO resolves execute_node to the agnostic Python-sim version first. This stub explicitly re-routes to HLSBackend.execute_node(self, ...) so cppsim/rtlsim actually run. A fragile, order-dependent workaround that must be kept in sync if base ordering changes.
- **[major/brittle-assumption]** `fmpadding_pixel.py:78` — get_normal_input_shape hard-codes batch/leading dim to 1 (ishape = (1, idim_h, idim_w, num_ch)) and get_normal_output_shape does the same at line 84, ignoring the numInputVectors attribute. Yet get_exp_cycles (line 71) multiplies by numInputVectors, and execute_node (line 148) reads the actual ishape[0]. Inconsistent treatment of batch across shape/estimate/sim.
- **[major/brittle-assumption]** `fmpadding_pixel.py:123` — get_input_datatype asserts ret.allowed(0) ('FMPadding_Pixel DataType must support zero') because the hlslib kernel always inserts zeros. A hidden datatype constraint baked into a getter; couples the op contract to a hlslib implementation detail and will hard-fail (assert) on otherwise-legal bipolar/custom datatypes rather than warning.
- **[major/magic-number]** `fmpadding_pixel_hls.py:51` — defines() emits a raw string template with an axis swap: OutputDim_x<-odim_w, OutputDim_y<-odim_h, Stride_x<-stride_w, Stride_y<-stride_h (args passed at lines 58-63 in width,height / x=w order). The x/y-to-w/h remapping is implicit and unchecked; any reordering silently transposes padding. Also embeds literal C #define names via .format positional substitution.
- **[minor/hard-coded-param]** `fmpadding_pixel_hls.py:71` — docompute hard-codes the hlslib template-function name hls_call = 'FMPadding_Pixel_Nonsquare' and the full template-argument order in a format string (lines 72-76). No abstraction over the streamtools.h API; renaming/parametrizing the kernel requires editing this string.
- **[minor/duplicated-logic]** `fmpadding_pixel.py:96` — get_folded_output_shape (96-103) is a near-verbatim copy of get_folded_input_shape (87-94) with only the normal-shape source swapped; same SIMD-divisibility assert duplicated. Likewise get_outstream_width (135) duplicates get_instream_width (130).
- **[minor/duplicated-logic]** `fmpadding_pixel.py:148` — execute_node re-derives the padded output geometry inline (ishape[1] + (ishape[1]-1)*(s_h-1), etc.) instead of reusing get_padded_odim(), duplicating the pixel-padding formula that already lives in get_padded_odim (lines 63-64). Two independent copies of the sizing math can drift.

## Hermeticity violations (0)


## Seams (2)

- **clean-seam** — The agnostic base FMPadding_Pixel (fmpadding_pixel.py) is genuinely backend-free: all shape/datatype/width/cycle/sim logic is expressible without HLS or RTL, so an alternate backend could subclass it cleanly. The only HLS-specific artifacts live entirely in fmpadding_pixel_hls.py (global_includes/defines/docompute/blackboxfunction).
- **fused-no-seam** — execute_node is fused across the diamond: the base defines the Python-sim version and the HLS subclass must add an explicit stub (line 91) to reach HLSBackend.execute_node. The base op cannot be composed with a backend without the subclass manually resolving which execute_node wins — the seam is not clean, it requires per-subclass MRO surgery.
