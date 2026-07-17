# Census: pool

*A small, largely clean depthwise pooling op (MaxPool/QuantAvgPool) with a backend-agnostic base (Pool) and a single HLS backend (Pool_hls); no RTL variant exists. The notable defects are an execute_node dual-meaning/MRO hazard requiring a re-dispatch shim and a Function-string dispatch duplicated across three methods.*

**Files:** `src/finn/custom_op/fpgadataflow/pool.py`, `src/finn/custom_op/fpgadataflow/hls/pool_hls.py`

## Variants

| class | file | bases | agnostic base? |
|---|---|---|---|
| `Pool` | `pool.py` | `(HWCustomOp)` | NO (backend-only) |
| `Pool_hls` | `pool_hls.py` | `(Pool, HLSBackend)` | yes |

## Redesign pressure

The pool family is mostly clean: the base Pool class is properly backend-agnostic with no HLS/RTL leakage, and the HLS specialization is well-contained, so it would be easy to add an RTL variant. The one real structural strain is the overloaded execute_node contract: the base uses it as a Python golden/behavioral model while backends use the same name for cppsim/rtlsim, and Python MRO (Pool listed before HLSBackend) silently resolves to the behavioral sim, forcing Pool_hls to add a no-op shim that re-dispatches to HLSBackend.execute_node. A redesign should split 'behavioral reference' from 'backend execute' into distinct contract methods. Secondary friction is the Function-string switch hand-duplicated across get_output_datatype, execute_node, and docompute, plus the opaque 'Size'/'AccumBits' attrs whose meaning only surfaces at C++ template instantiation.

## Overrides (19)

| method | class | line | reason |
|---|---|---|---|
| `get_nodeattr_types` | Pool | 54 | Adds pool-specific attrs (Channels, PE, KernelSize, Function enum {MaxPool,QuantAvgPool}, OutImgDims, AccumBits, Size, BatchSize) then merges super(). |
| `get_input_datatype` | Pool | 76 | Returns InputDataType nodeattr as DataType. |
| `get_output_datatype` | Pool | 80 | Function-dependent: MaxPool forces odt==idt; QuantAvgPool only enforces matching signedness. |
| `get_normal_input_shape` | Pool | 100 | Input packs kernel window: (batch, *OutImgDims, KernelSize_prod * Channels) to be im2col-compatible. |
| `get_folded_input_shape` | Pool | 109 | Folds last dim by PE (PE acts as SIMD from input side); asserts Channels % PE == 0. |
| `get_normal_output_shape` | Pool | 118 | Output collapses kernel window: (batch, *OutImgDims, Channels). |
| `get_folded_output_shape` | Pool | 125 | Folds Channels by PE; asserts Channels % PE == 0. |
| `get_exp_cycles` | Pool | 134 | ((Channels*K_prod)/PE) * prod(OutImgDims) * BatchSize latency model. |
| `get_instream_width` | Pool | 145 | input_dt_bits * PE. |
| `get_outstream_width` | Pool | 151 | output_dt_bits * PE. |
| `verify_node` | Pool | 163 | Checks backend attr, single input, supported Function. |
| `execute_node` | Pool | 186 | Pure-Python NumPy BEHAVIORAL reference sim: reshapes to (…,k2,ch), np.max for MaxPool, sum+right_shift quant for QuantAvgPool. |
| `get_nodeattr_types` | Pool_hls | 54 | Merges Pool + HLSBackend attr dicts explicitly. |
| `global_includes` | Pool_hls | 60 | Emits #include pool.hpp (HLS abstract). |
| `defines` | Pool_hls | 63 | Emits ISIZE = osz*cf*k and K constants for the C++ kernel (HLS abstract). |
| `docompute` | Pool_hls | 72 | Instantiates MaxPoolFunction<> or QuantAvgPoolFunction<> then Pool_batch<ISIZE,K> (HLS abstract). |
| `pragmas` | Pool_hls | 98 | super().pragmas() + dataflow disable_start_propagation + aggregate compact=bit on both streams. |
| `blackboxfunction` | Pool_hls | 104 | Emits top-level signature with hls::vector<idt,PE> in / hls::vector<odt,PE> out (HLS abstract). |
| `execute_node` | Pool_hls | 116 | Explicitly delegates to HLSBackend.execute_node to force cppsim/rtlsim path. |

## Hacks (9 — 1 blocker, 4 major)

- **[blocker/inheritance-irregularity]** `pool_hls.py:116` — Pool_hls.execute_node exists ONLY to call HLSBackend.execute_node(self,...). Because MRO is (Pool_hls, Pool, HLSBackend, HWCustomOp), an un-overridden execute_node would resolve to Pool.execute_node (the python behavioral golden model), NOT the HLS cppsim/rtlsim path. The base class overloads execute_node with two incompatible meanings and MRO order silently picks the wrong one without this shim.
- **[major/duplicated-logic]** `pool.py:199` — Function string dispatch ('MaxPool'/'QuantAvgPool') is hand-written in THREE places: get_output_datatype (pool.py:85-96), execute_node (pool.py:199-211), and docompute (pool_hls.py:80-94). Adding a Function means editing all three; they can drift.
- **[major/brittle-assumption]** `pool.py:202` — QuantAvgPool quantization arithmetic in the python sim (shift_bits = bit_length(max_value*k2) - obits, clamped >=0, then right_shift) is a hand-reimplementation that must stay bit-exact with finn-hlslib QuantAvgPoolFunction and the HLS 'Size'/'AccumBits' params. No shared source of truth; silent divergence risk between behavioral sim and hardware.
- **[major/magic-number]** `pool_hls.py:83` — The nodeattr 'Size' (declared in pool.py:69 default 1) is silently repurposed as the QuantAvgPool right-shift template parameter passed to QuantAvgPoolFunction<...,shift>. Opaque name whose meaning is only discoverable at the C++ instantiation site.
- **[major/template-surgery]** `pool_hls.py:77` — hls::vector<...> C++ type strings are assembled via % formatting in docompute (77,85) and blackboxfunction (108-109); the QuantAvgPool accumulator type 'hls::vector<ap_[u]int<AccumBits>,PE>' is built by string interpolation of signedness+AccumBits. Type correctness is unchecked until C++ compile.
- **[minor/brittle-assumption]** `pool.py:88` — get_output_datatype (a getter) contains assert statements (odt==idt for MaxPool; signed()==signed() for QuantAvgPool). A datatype accessor throws on inconsistent model state; callers cannot query the type without risking an AssertionError.
- **[minor/todo-marker]** `pool.py:62` — '# TODO add support for AvgPool and AccPool' — Function enum is knowingly incomplete.
- **[minor/magic-number]** `pool_hls.py:100` — Three hard-coded HLS pragmas appended unconditionally: 'dataflow disable_start_propagation', 'aggregate variable=in0_V compact=bit', 'aggregate variable=out0_V compact=bit'. No parameterization or rationale in code.
- **[minor/brittle-assumption]** `pool.py:96` — Exception/message text references legacy op name 'Pool_Batch' (pool.py:96,176 and pool_hls.py:94) while the classes are named Pool/Pool_hls — stale naming that misleads diagnostics.

## Hermeticity violations (1)

- **[hidden-coupling]** `pool.py:202` — execute_node's QuantAvgPool math is implicitly coupled to the external finn-hlslib QuantAvgPoolFunction semantics and to the 'Size'/'AccumBits' HLS params; the python golden model and the generated C++ must agree, but the coupling is not expressed anywhere in code.

## Seams (3)

- **clean-seam** — Pool (pool.py) is a genuinely backend-agnostic base: shapes, datatypes, folding, cycle model and attrs contain zero HLS/RTL leakage. An RTL variant could be added purely by writing pool_rtl.py(Pool, RTLBackend) with no edits to the base.
- **clean-seam** — All HLS-specific code (includes/defines/docompute/pragmas/blackboxfunction) is confined to pool_hls.py; the HLSBackend contract methods are cleanly isolated in the backend subclass.
- **fused-no-seam** — execute_node is fused across the abstraction: Pool.execute_node is a python behavioral reference model while HLSBackend.execute_node is the cppsim/rtlsim driver. The 'execute' contract carries two meanings and MRO ordering (Pool before HLSBackend) forces every backend to re-override execute_node just to re-dispatch (pool_hls.py:116). Base behavioral sim and backend execution cannot be cleanly separated under the current single-method contract.
