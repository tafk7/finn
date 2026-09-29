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

- K2 sweeps: `/tmp/k2/sweeps.log`.
- S4 (all XSim): `/tmp/xsim/s4/summary.log`.
- G1 (XSim pytest): `/tmp/xsim/g1/summary.log`.

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
  graph). FINN `mem_mode` mapping: none → `external`, ROM →
  `internal_embedded`, memstream → `internal_decoupled`.
- The flat-kernel tests stay flat: an unplaced kernel's idle ports take their
  pins from its own dtypes.

## Open, outside the plan

- The weight layout and weight-file generation revisit (G0.2).
- Exporting a fused composite's control bus through its parent.
- Depthwise MatMul to FINN (`VVAU`); other ONNX operators in `finn.graph`.
- Housekeeping: the temporary worktree `/home/tkeller/prj-kernels/finn-v1`
  (branch `wip/v1-values`, cherry-picked as `0c1a5123f`) can be removed.
