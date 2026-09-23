# Concrete kernel extraction validation

Worktree: `/home/tkeller/prj-kernels/finn-kernels-extraction`.
Baseline checkpoint: `459fc8cc8d16694c8afec14250cff82abb3c6eae`.
Date: 2026-09-23.

Runtime environment:

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=src:tests:deps/qonnx/src
FINN_ROOT=/home/tkeller/prj-kernels/finn-kernels-extraction
FINNLIB_ROOT=/home/tkeller/prj-kernels/finn-kernels-extraction/deps/finnlib
FINN_XELAB_MT=2
LD_LIBRARY_PATH=/home/tkeller/Xilinx/2025.2/Vivado/lib/lnx64.o
```

Python: `/home/tkeller/prj-kernels/.kernel-venv/bin/python`.

Executed from the worktree:

```text
python -m pytest -q --confcutdir=tests/kernels tests/kernels/test_dotp.py tests/kernels/test_mvau_assembly.py tests/kernels/test_streaming_components.py tests/kernels/test_axi_stream_declaration.py tests/kernels/rtlsim/test_observed_transport.py
python -m kernels.rtlsim.mvau_assembly_numeric --case packed --output /tmp/kernel-extraction-rtl-ICkDVH/mvau
python -m kernels.rtlsim.pure_dot_product_numeric --case one_beat_dsp58 --output /tmp/kernel-extraction-rtl-ICkDVH/dotp
```

- Focused suite: 148 passed in 32.86s, no skips. This includes the Vivado
  replay/cyclic component simulation for framing, stalls, wrap and reset.
- MVAU: packed INT3, DSP48E2, external/cyclic delivery, free/stalled outputs:
  four successful XSI simulations. Each emitted 8 result beats; input counts
  were 12 activation words and, for external delivery, 24 weight words.
- Dotp: one-beat INT3, DSP58, free output and 32-cycle output stalls:
  two successful XSI simulations. Each consumed 64 activation/weight words
  and emitted 64 result beats. Stalled execution took 5882 cycles.
- Every XSI run retains request.json, response.json, simulation.log and compile
  inputs in its case/mode directory. Each included a 4000-cycle drain window.
- Cold imports of both numeric harnesses and shared packing support passed
  with an import hook rejecting dataflow, finn.dataflow and
  finn.custom_op.dataflow; DspBlock.__module__ is finn.kernels.target.
- Ruff formatting and lint passed for the five owned kernel source modules,
  four concrete test modules, tests/kernels/rtlsim, and the adapted legacy
  dotp_axi_numeric.py harness.
- Strict mypy passed for the five source modules with PYTHONPATH unset and
  MYPYPATH=src:tests.

Resource SHA-256 values match the baseline bytes:

```text
466397881624b4d0a4884762700620d548271541dc97bb265ea4b80129a5cd90 dotp_axi.sv
0fca60b76592a308c354a84022722e52936261ec89e90f807156bea6be7f4173 cyclic_stream.sv
3115c7181620dac035d6714ac6698723dada0099d980fdb5cefac30341676096 decomposed_wrapper.sv.j2
```

No simulator binaries or compile outputs were added to the worktree.
