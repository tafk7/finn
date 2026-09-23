# Flat kernel authoring results

Implemented in `finn-kernels-extraction` on top of `37cc2aa6d`.
FinnLib source revision: `dfeafac81cd2a6da27e647ee03915ade5532186e`.

The seven-example set consists of the existing DotpAxiKernel plus six new flat
declarations: FifoKernel, InputGeneratorKernel, ThresholdingAxiKernel,
EltwiseKernel, IntToFp32Kernel and MemStreamHlsKernel. All are exported from
`finn.kernels`. The existing dotp declaration, RTL correction, MVAU assembly,
Space engine and graph/compiler implementation are unchanged.

The new HLS source handoff represents top-function arguments and interface
directives before synthesis determines physical pins. Its renderer uses the
existing contribution, source-closure and template components. No new AXI
authoring DSL, kernel composition system or compiler integration was added.

## Validation

- Complete `scripts/check-kernels.sh`: **958 tests passed**, no skips;
  formatting and lint passed; strict mypy passed for **66 production files**
  and **28 selected test files**.
- The complete gate includes native RTL elaboration, C++ execution with Vitis
  headers, six new Vivado numerical/transport simulations, and installed-wheel
  materialization tests.
- Following final threshold-bias profile tightening, **60 focused tests**
  covering admission, native interfaces, source generation, datatypes and HLS
  C++ were rerun. The six numerical configurations from the complete gate are
  unaffected by that refusal.
- Retained dataflow package boundaries and facade tests: **19 passed**.
  The boundary test now permits the physical library to grow while checking
  that every export is owned by `finn.kernels`.
- Representative MemStreamHLS synthesis: **passed with Vitis HLS 2025.2**,
  INT9 elements, depth 3, target `xc7z020clg400-1`, requested clock 10 ns.
  HLS generated the AXI-Lite `control` bundle for memory/control and an AXI
  stream output with 16-bit TDATA for the 9-bit scalar. The report is retained
  as `memstream_hls_csynth.rpt`. This is HLS synthesis, not place-and-route or
  measured board timing.

Commands from the repository root:

```bash
PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python bash scripts/check-kernels.sh

PYTHONPATH=src:tests:deps/qonnx/src /home/tkeller/prj-kernels/.kernel-venv/bin/python \
  -m pytest -q --confcutdir=tests/kernels tests/kernels/test_flat_kernels.py \
  -k 'not generated_rtl and not combinational_conversion'

PYTHONPATH=src:tests:deps/qonnx/src /home/tkeller/prj-kernels/.kernel-venv/bin/python \
  -m pytest -q --confcutdir=tests/dataflow \
  tests/dataflow/test_package_boundaries.py \
  tests/dataflow/test_space_model_boundary.py tests/dataflow/model/test_facade.py
```

For HLS synthesis, stage `render_hls_sources(...)` output without changing its
relative paths. Invoke `vitis-run --mode hls --tcl synth.tcl` with:

```tcl
open_project project
set_top memstream_hls
add_files memstream_hls.cpp -cflags {-Ihls/util -Ihls/infra -std=c++17}
open_solution -flow_target vivado solution
set_part xc7z020clg400-1
create_clock -period 10
csynth_design
exit
```

## Findings reflected in the declarations

1. **Native words are not uniformly byte-padded AXI beats.** FIFO,
   InputGenerator and Eltwise retain unpadded pins; IntToFp32 has no stream or
   clock. Dotp and thresholding retain their actual AXI layouts.
2. **Parameter arrays need an explicit immutable value boundary.** Traversal
   vectors and threshold tables use typed tuple recognition before entering
   Space; emission produces the native SV array literal at the artifact boundary.
3. **Threshold configuration has a native restriction.** Multi-set AXI-Lite
   configuration is refused because the pinned wrapper omits set address bits.
   Static multiple-set selection remains supported.
4. **Negative threshold bias needs a numerical check, not just ABI matching.**
   With N=3 and BIAS=-10 the native module derives a 33-bit output. A numeric
   probe with input zero returned `00fffffff8` on its 40-bit padded output,
   whereas signed -8 in the declared 33-bit encoding is `01fffffff8`.
   BIAS < -N-1 is therefore refused in this profile.
5. **The AXI-Lite source needed a declaration-order correction.** Vivado rejects
   `snk_re` before its declaration. The local `resources/axilite.sv` declares
   it before use and preserves the original continuous expression. Shared
   FinnLib sources remain untouched.
6. **HLS source interfaces and synthesized RTL ABIs are different products.**
   The MemStream example carries the former and verifies the latter through a
   real synthesis run, without hardcoding predicted pin names in the kernel.

No full FINN Op conversion/build run was requested or performed. The authoring
comparison and supported profiles are in the adjacent README.
