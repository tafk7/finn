# MatMulKernel: implementation record

The [spec](SPEC.md) and its [M0 analysis](M0.md) define the work; this record
states what each increment changed and the evidence for it. Evidence files are
under [`evidence/`](evidence/). XSim runs use the local Vivado 2025.2
libraries; each simulation runs in a fresh process.

## M1: `MVAU` becomes `MatMulKernel` (dense)

**Change.** `finn.kernels.mvau` is replaced by `finn.kernels.matmul`:

| Before | After |
|---|---|
| `MVAU`, `MVAUAssembly`, `mvau_assembly` | `MatMulKernel`, `MatMulAssembly`, `matmul_assembly` |
| facts `repetitions`, `matrix_width`, `matrix_height` | `rows`, `reduction`, `outputs` |
| Decision `implementation` (`implementation.cyclic.rom_style`) | `delivery` (`delivery.cyclic.rom_style`); `selected(delivery)` is `delivered` |
| instance `u_implementation_cyclic` | `u_delivery_cyclic` |
| module `finn_mvau_<delivery>`, producer `finn.mvau.<delivery>` | `finn_matmul_<delivery>`, `finn.matmul.<delivery>` |
| refusal codes `mvau-arithmetic`, `mvau-folding` | `matmul-arithmetic`, `matmul-folding` |

Tests, the numeric harness (`rtlsim/matmul_numeric.py`), the benchmark
workload (which also drops its stale `segment_length` argument) and the README
follow. MVAU had no thresholding references to remove.

**Evidence** (`evidence/m1/`).

- Structures: the six fingerprinted configurations
  ([`structure_dump.py`](structure_dump.py)) are identical to before once the
  module and cyclic instance names are normalized. All six fingerprints change,
  by those names alone.
- Keys: `delivery` and `delivery.cyclic.rom_style` replace `implementation` and
  `implementation.cyclic.rom_style`; the rest are unchanged.
- Numeric XSI: 28/28 (direct, and a weight FIFO for `packed` and
  `int8_pumped`; external and cyclic; free and stalled output).
- Gates: Space 427, kernels 787 passed; ruff, mypy clean; dataflow gate clean.
