# `finn.kernels` — the declarative design-space engine

A new FINN-native hardware-kernel backend built on one thesis: **the design space of a
kernel is declarative data.** A kernel declares its space as free choices, computed
quantities, and feasibility gates; a resolver turns (space + device context + choices)
into a legal build point — or an *explained* illegality — and a hermetic emit lowers that
point to typed artifacts. This module is the proven foundation for replacing FINN's
`HWCustomOp` + `HLSBackend`/`RTLBackend` subsystem.

For the *why* and the multi-stage plan, see
[`MOTIVATION.md`](../../../../scratchpad/current/MOTIVATION.md) and
[`KERNEL_REFACTOR_PLAN.md`](../../../../scratchpad/archive/superseded/KERNEL_REFACTOR_PLAN.md).

## Layout

| Path | Role |
|---|---|
| `engine/` | **The pure resolve core.** The four resolve primitives (Context, Axis, Derived, Predicate) → `resolve` → a `Point`; plus the folded-in brainsmith value objects (`ordered_parameter`, `spec_helpers`). No dependency on any other kernel package. |
| `model/` | **The op-model framework.** `Kernel`/`Backend`/`Interface`, the tiling/fold-depth projections, `ports` (role-tagged port taxonomy), the parameter-source CONTRACT (`param_contract`, `parameter_source`), `source_backend`, and — folded in here — `registry` (the op-agnostic implementation-registry factory) and `artifacts` (the typed emit-output vocabulary). |
| `emit/` | **The emit PROCESS.** `manifest` (artifact manifest reader) + `stitch` (op-agnostic block-design wiring). Consumes a resolved Point → produces artifacts. |
| `ir/` | **The host-facing seam.** The `KernelOp` bridge to FINN's node model, the nodeattr registry, and `routing` (the single kernel-side host-routing seam). |
| `compute/` | Per-op bounded contexts — `mvau`, `thresholding` — each a folder owning its Kernel/DesignSpace definition, backends, and emit. |
| `dataflow/` | Infrastructure kernels. Today `parameters/` (weight/threshold source); `fifo`/`dwc`/`iodma` to come. |
| `tests/` | The mirror test tree: `tests/{engine,model,emit,ir,compute,dataflow,integration,hw}/`. |

Dependency spine (acyclic, no cycles): `engine` (leaf) ← `model` ← {`ir`, `emit`,
`compute`, `dataflow`}. The only cross-kind edge is `compute → dataflow` (one-way —
compute composes infra). `ir` is the host-facing seam.

## Running the tests

The suite runs inside the FINN Docker env against FINN's pinned qonnx (`deps/qonnx`), via
a committed runner:

```bash
./run-docker.sh bash scripts/run_kernel_tests.sh   # fast + integration + finn_codegen
```

This covers `tests/{engine,model,emit,ir,compute,dataflow,integration}/` (`-m "not
slow_hw"`). The `slow_hw` hardware tier (`tests/hw/`, requiring Vivado / `xsi.so`) is
gated and run separately via `scripts/run_kernel_hw.sh`.

## Provenance & licensing

- This module is **BSD-3-Clause** (FINN's license), except the vendored value objects
  below.
- `engine/ordered_parameter.py` and `engine/spec_helpers.py` are **MIT**, inlined from
  [`microsoft/brainsmith`](https://github.com/microsoft/brainsmith)
  (`brainsmith/dataflow/`) @ `38faaf9`. Each file carries a provenance header. These are a
  **temporary vendored copy** — de-vendoring is Stage 3 (the source model) of the refactor
  plan. Other kernel files reference brainsmith *concepts* (the 3-category port model, the
  TENSOR→BLOCK→STREAM folding model) but are original BSD-3-Clause code.

## Design docs

The private design corpus lives under [`../../../../scratchpad/`](../../../../scratchpad/)
(see its `README.md` for the map). It is tiered: `current/` is normative, `reference/`
describes systems outside our control (FINN's baseline, brainsmith, the prototype), and
`archive/` is historical — **do not read an archived doc as the spec.**

Current: [`MOTIVATION.md`](../../../../scratchpad/current/MOTIVATION.md) (why this exists)
and [`generality-gaps.md`](../../../../scratchpad/current/generality-gaps.md) (op-zoo
coverage).

The engine-facing specs — the design-space model, composition/parameter delivery, and the
port taxonomy — are **being rewritten** and their stale predecessors sit in
`archive/pending-harvest/`. Until they land, **the code is the spec**: start at
`engine/design_space.py` + `engine/resolve.py` for the primitives, `model/kernel.py` +
`model/backend.py` for the op model, and `model/parameter_source.py` for composition. The
module docstrings there are maintained.
