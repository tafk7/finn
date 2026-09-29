# Status: kernel composition plan (2026-09-28)

Plan: [PLAN.md](PLAN.md). Record of what landed: [RECORD.md](RECORD.md).
Branch `feature/kernel-package-extraction`, local commits only, nothing pushed.

## Done (committed)

| Step | Commit | Fast gates | XSim |
|---|---|---|---|
| Plan | `9d7bac4f8` | — | — |
| E0 probe, report, review | `1051948f5` | probe 52 | — |
| E1 engine refinement | `d9f1764b9` | Space 446, kernels 819, dataflow 33 | — |
| V1 values, logical Stream | `0c1a5123f`, `95c814bd6` | Space 446, kernels 819, dataflow 40 | — |
| K1 protocol, ports, MatMul | `1e44982cb` | Space 447, kernels 810, dataflow 40 | all sweeps: 224 passes (`7dc0f99b1`) |
| K2 every kernel on the protocol | `8d68a7c4b` | Space 448, kernels 806 (XSim in pytest included), dataflow 40 | sweeps from the commit: see below |
| S4 composites and the Design | `c7764f7f9` | Space 448, kernels 800 + 15 XSim skipped, dataflow 40 | Design gate passed locally; full run from the commit: see below |
| G1 graph adapter (`finn.graph`) | `d6952761d` | kernels 800, graph 4 (+ XSim skipped), dataflow 40 | ONNX gate passed locally; pytest XSim from the commit: see below |
| ROM removed; constant-weights design note | `b865e8c38` | kernels 792, graph 4 (+ XSim skipped), dataflow 40 | sweeps 208 pass; 4 XSim tests hit a FinnLib defect |
| FinnLib pin `d03f2fc` (memstream padding) | `9118a3cb1` | same | everything passes: XSim tests 806, sweeps 208 |

Every step keeps ruff and mypy clean and the 27 documentation examples
passing; module parameters and memory images are unchanged over the identity
configurations (`evidence/identity-*.txt`).

## XSim from the commits (the working practice)

Each increment is committed once its fast gates pass (XSim tests skipped by
leaving Vivado off `PATH`); XSim then runs from a snapshot of the commit
(`git archive`, its own `deps` copy) while development continues. Runners:
`/tmp/xsim-commit.sh <commit> <tag>` (XSim pytest and every numeric sweep)
and `/tmp/xsim-pytest.sh <commit> <tag>`; results in `/tmp/xsim/<tag>/`.
Findings are fixed when found or batched after the next stage, and each
run's results are recorded in RECORD.md in a follow-up commit.

- K2 sweeps: all pass (224), recorded.
- S4: XSim pytest passes (815), recorded; its sweeps could not start (the
  snapshot lacked `finn_xsi/xsi.so`), covered by the next commit's run.
- G1: XSim tests pass (kernels 815, graph 6), recorded.
- ROM removal: every sweep passes (208, S4's included); 4 XSim tests failed
  on FinnLib's memstream padding, fixed and pinned (`9118a3cb1`, FinnLib
  `d03f2fc`), where everything passes: XSim tests 806, sweeps 208.

## Decisions recorded for the user

- G0.2: weights `(k, n)` kept; revisit weight-file generation in detail.
- Engine additions: `Users` through forwarded inputs; `settle` treats a
  pending admission as not refused (K1, accepted); a body projects its own
  derived values, `Annotated` names a projection's semantics, the mapping
  form of `Decision` is refused (K2).
- The artifact layer gained nested composed modules: a fixed name from the
  requirements fingerprint and a wrapper carrying its own render values (S4);
  every composed wrapper fingerprint changed once as a result.
- A fused composite with a control bus is refused (`composite-control`);
  unfused, its bus is exported as `<composite>_<port>`.
- `finn.graph` is a new layer above `finn.kernels` (which never reads a
  graph). FINN `mem_mode` mapping: none → `external`, memstream →
  `internal_decoupled`.
- The cyclic ROM is removed (user): it eliminated no constants. A
  constant-weight RTL dot product is proposed in
  `docs/constant-weights-2026-09-28/DESIGN.md`, with five questions.
- The flat-kernel tests stay flat: an unplaced kernel's idle ports take their
  pins from its own dtypes.

## Open, outside the plan

- The weight layout and weight-file generation revisit (G0.2).
- Exporting a fused composite's control bus through its parent.
- Depthwise MatMul to FINN (`VVAU`); other ONNX operators in `finn.graph`.
- Housekeeping: the temporary worktree `/home/tkeller/prj-kernels/finn-v1`
  (branch `wip/v1-values`, cherry-picked as `0c1a5123f`) can be removed.
