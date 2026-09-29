# Status: code quality pass over finn.kernels and finn.dataflow (2026-09-29)

Branch `feature/code-quality-2026-09-28`, worktree
`/home/tkeller/prj-kernels/finn-code-quality`, from `9118a3cb1`
(`feature/kernel-package-extraction`). Local commits only, nothing pushed.
The per-increment log, with line counts and gate output, is
[RECORD.md](RECORD.md).

**Goal.** The same behaviour with less code and fewer concepts: no new
features, registries, wrappers or families; stale code and weak tests
deleted; producers present data as it is, receivers adapt.

**Totals against `9118a3cb1`.** src −389 net (+620 −1009, including a
zero-sum move of ~340 lines), tests −273 net (+331 −604). No decision key
changed; module parameters, memory images and wrapper fingerprints are
unchanged (identity dump).

## Commits, in review order

| Commit | Change | src | tests |
|---|---|---|---|
| `b99691538` A | Dead code and test-only views deleted | −209 | −38 |
| `232c9f372` B | Wiring moves to `composite.py`; `Endpoints` folds into `Connection` | −47 | −10 |
| `bdd00a0e2` C | A `GivenPort` carries its kernel's element; the stream checks it | −53 | −2 |
| `20623b20d` D | The seven adapter classes become a table | −74 | −7 |
| `f08d7f093` E | Sequence derivations, the cyclic-source rule and helpers stated once | −24 | −7 |
| `2a0a809fb` F | One XSim test harness | 0 | −194 |

Each commit is self-contained and passes every gate on its own; they can be
reviewed one at a time.

## What changed, and why

**A. Dead code.** `AxiStreamPort`, `TypedStream` and `physical/ports.py`, the
`interfaces` views of FIFO, `input_gen` and eltwise, `configure.compatible`
and the adapters' `admitted` views were used only by tests; so were
`canonical_loops`, `is_repetition`, `once` and `gemm.FORM` in dataflow. The
scalar-admission tests that read through `AxiStreamPort` now read the
scalar's own `encoding` (`test_scalar_declaration.py`); the adapter test uses
the engine's `compatible_cases` with the kernels' `admission`.

**B. `streams.py` held two things.** The stream family stays (776 → 415
lines); the composite's wiring (`netlist`, `merge_parts`, `Parts`,
`Composed`, the clock and tie-off helpers) moves unchanged to `composite.py`,
its only consumer. `Endpoints` had `Connection`'s fields without the stages;
the ends are now a `Connection` and `link` is `replace(endpoints,
stages=...)`. Re-exports of other modules' names from `streams` and `port`
are removed: each name has one import path.

**C. Element checks (the main conceptual fix).** Every stream end's contract
took its element from the stream's own tensor, so `Stream.well_formed`'s
element check could never fail, while memstream, thresholding, eltwise and
transpose each re-checked their placed streams (four `carried`-style
constraints). Now a `GivenPort` presents the element its kernel gives it
(`dtype`) and the stream refuses a mismatch (`stream-tensor`); the kernel
checks are deleted. A `ScheduledPort` (dotp) still carries its stream's
element, admitted by its policy. *Review point:* the refusal now belongs to
the stream's `connection` rather than the kernel's `build_requirements` /
`admission`; a composite still refuses, through its `structure`.

**D. Adapter chains.** Seven classes differed only in their stage tuple.
`CHAINS` lists the tuples and `ADAPTERS` builds each candidate with the
engine's existing `composite()` (no new mechanism), keyed by its modules
joined, so every adapter key is unchanged.

**E. Stated once.**
- Thresholding's input order and the transpose's expected input were
  hand-built `vector_major` (verified equal by a probe before the change).
- `physical.contract._presented` restated `dataflow.plan.presented`.
- FinnLib's set-index width had three copies (`set_index_dtype`), the
  nested-integer walk two (`integers`), the undecided-keys query two
  (`configure.undecided`).

**F. Tests.** `tests/kernels/xsim.py` gains `materialize`, `simulate` and a
multi-port `stream_through` (other top inputs held at zero). The inline
testbenches of `test_interfaces`, the flat kernels and the FIFO capacity
bench, and the numeric harnesses' source builds, use them. A test that
hand-wired an eltwise-plus-constant `Composition` (re-implementing `netlist`
down to its wrapper) is a `Design`.

## Public names and codes that changed

- **Renamed.** `StreamPort.idle_dtype` → `dtype`.
- **Moved.** `finn.kernels.streams.{netlist, merge_parts, Parts, PARTS,
  PARTS_SEMANTICS, Composed, COMPOSED}` → `finn.kernels.composite`.
- **Removed.** `AxiStreamPort`, `axi_stream`, `TypedStream`,
  `physical.ports`, `STREAM_INTERFACES`, `*.interfaces` views,
  `configure.compatible`, `Stream.adapter_admitted`,
  `StreamAdapter.admitted`, the seven adapter classes (`InputGenAdapter`,
  ...), `Endpoints`, `ENDPOINTS`, `boundary_sequence`, `stored_element`,
  `MemStreamKernel.set_bits`, `base.merged`; dataflow `canonical_loops`,
  `is_repetition`, `once`, `split_beats` (now private), `gemm.FORM`.
- **New.** `adapters.CHAINS`, `adapters.ADAPTERS`,
  `datatypes.domains.set_index_dtype`, `datatypes.semantics.integers`,
  `configure.undecided`; in tests, `xsim.materialize` and `xsim.simulate`.
- **Refusal codes removed.** `memory-element`, `threshold-stream-element`,
  `eltwise-stream-element`, `transpose-element` (now the stream's
  `stream-tensor`), and `eltwise-interface` (its PE bound is already refused
  by `eltwise-operation`).
- **Keys.** None changed.

## Verification

- **Fast gates, every commit** (both gate scripts, Vivado off `PATH`): Space
  448, kernels 791 + 14 XSim skipped, graph 4 + 2 XSim skipped, dataflow 40;
  ruff and mypy clean. The base had kernels 792: one test that only exercised
  a deleted view was dropped (A).
- **Documentation examples:** 27 pass at every commit (scratchpad
  `check-examples.py --finn-root`, including the kernels README).
- **Identity dump,** every commit: `identity.py --api=k1` identical to
  `evidence/identity-norom.txt` (module parameters, images, top ports, wire
  and beat counts, wrapper fingerprints, keys).
- **XSim from the commits** (a `git archive` snapshot with its own `deps`
  and `xsi.so`; sweep counts compared with the post-ROM-removal baseline
  `/tmp/xsim/norom`):
  - A: XSim pytest kernels 805, graph 6, all pass.
  - C (covers B, C): pytest 805, graph 6; sweeps dense 26, FIFO 4 and 4,
    depthwise 22, memstream 14 and 12, pumped 14, writable 14, sets 14, dotp
    27, dotp-stress 17, adapters 26: all pass, every count equal to the
    baseline.
  - F (covers D, E, F, including the new harness): pytest 805, graph 6;
    every sweep passes with the baseline count (214 passes, no failures).
  - Before F's commit, every rewritten XSim test passed locally.

## Considered and not done

- **Transpose takes `simd` instead of `input_form`** (the receiver declares
  what it reads and the stream adapts): a behaviour change, so left for a
  decision.
- **A shared base for memstream's and thresholding's AXI-Lite handling**
  (`held`, `control_bus`, exports): it would be a new family.
- **Deriving a composite's `PORT` views from `boundaries`:** new machinery for
  one user (MatMul).
- **`matmul_assembly` onto `settle`:** `realization` is a scalar Decision,
  which `settle` never settles; the manual choice is needed.
- **Splitting `traversal.py` or merging `plan`/`schedule`/`stream`:** the
  layers are distinct; only the dead code (A) and the duplicate (E) went.
- **`StreamPort`'s AXIS contract and `boundary_contract`** share one small
  marker line; not worth a helper.
