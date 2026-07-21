# `finn.kernels` — the declarative design-space engine

A new FINN-native hardware-kernel backend built on one thesis: **the design space of a
kernel is declarative data.** A kernel declares its space as free choices, computed
quantities, and feasibility gates; a resolver turns (space + device context + choices)
into a legal build point — or an *explained* illegality — and a hermetic emit lowers that
point to typed artifacts. This module is the proven foundation for replacing FINN's
`HWCustomOp` + `HLSBackend`/`RTLBackend` subsystem.

For the *why* and the multi-stage plan, see
[`MOTIVATION.md`](../../../kernel-design/kernel-final-design/MOTIVATION.md) and
[`KERNEL_REFACTOR_PLAN.md`](../../../kernel-design/kernel-final-design/KERNEL_REFACTOR_PLAN.md).

## Layout

| Path | Role |
|---|---|
| `space/` | **The engine.** The four resolve primitives (Context, Axis, Derived, Predicate) → `resolve` → a `Point`; plus composition/codegen: `implementation` (pool + emit dispatch), `artifacts` (typed emit outputs), `ports` (role-tagged port taxonomy), `stitch` (op-agnostic block-design wiring). |
| `primitives/` | Reused domain value objects — ordered parameters, datatype range-builders, template resolution, interface shape/width. **Vendored from `microsoft/brainsmith` (MIT); see Provenance.** |
| `ops/` | The op library: `mvau`, `vvau`, `thresholding`, `parameters` (weight/threshold delivery). Each declares its design space as data over the engine. |
| `tests/unit/` | The venv-pure test suite (126 tests, no Vivado/Docker). |
| `tests/hardware/` | Docker/Vivado validation harnesses (elaborate, rtlsim, byte-diff vs FINN). See [`tests/hardware/README.md`](tests/hardware/README.md). |

Dependency direction is one-way: `ops → space → primitives`.

## Running the tests

The unit suite runs in the lightweight kernel venv against FINN's pinned qonnx
(`deps/qonnx`), no FINN Docker image required:

```bash
PYTHONPATH="deps/qonnx/src:src" .kernel-venv/bin/python -m pytest \
  src/finn/kernels/tests/unit/ -q          # expect 126 passed
```

A `conftest.py` guards that `qonnx` resolves to the finn-pinned `deps/qonnx` and fails
loudly on a drifted sibling checkout.

The hardware harnesses require the FINN Docker container (Vivado / `xsi.so`) and are run
explicitly — see [`tests/hardware/README.md`](tests/hardware/README.md).

## Provenance & licensing

- This module is **BSD-3-Clause** (FINN's license), except `primitives/`.
- The seven `primitives/` files are **MIT**, inlined from
  [`microsoft/brainsmith`](https://github.com/microsoft/brainsmith)
  (`brainsmith/dataflow/`) @ `38faaf9`. `ordered_parameter.py`, `template_resolution.py`,
  `schemas.py`, and `dse_models.py` are byte-identical; `spec_helpers.py` retargets three
  docstring import examples to the `finn.kernels.primitives` home; `types.py` adds a local
  `TilingSpec` alias; `interface.py` is a distilled variant. `schemas.py` (the
  `KernelSchema`/`InputSchema`/`OutputSchema` interface-list declaration) and `dse_models.py`
  (the TENSOR→BLOCK→STREAM shape resolution) back the KernelOp folding model — see
  [`kernel-design/kernel-final-design/kernelop-tensor-block-stream.md`](../../../kernel-design/kernel-final-design/kernelop-tensor-block-stream.md).
  Each file carries a provenance header. These are a **temporary vendored copy** —
  de-vendoring is Stage 3 (the source model) of the refactor plan.

## Design docs

Under [`kernel-design/kernel-final-design/`](../../../kernel-design/kernel-final-design/):
[`design-space-model.md`](../../../kernel-design/kernel-final-design/design-space-model.md)
(the engine primitives),
[`port-taxonomy.md`](../../../kernel-design/kernel-final-design/port-taxonomy.md) (kind vs
role, the stitch),
[`param-delivery-design-space.md`](../../../kernel-design/kernel-final-design/param-delivery-design-space.md)
(composition), and
[`generality-gaps.md`](../../../kernel-design/kernel-final-design/generality-gaps.md)
(op-zoo coverage).
