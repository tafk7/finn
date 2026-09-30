# Status: kernel authoring (2026-09-29)

Plan: [PLAN.md](PLAN.md). Record of what landed: [RECORD.md](RECORD.md). Probes:
[p0/REPORT.md](p0/REPORT.md). Branch `feature/kernel-authoring`, worktree
`finn-kernel-authoring`; local commits only, except FinnLib `99d75e8` (pushed
as `kernels/inner-shuffle-20260929`, and pinned).

## What a kernel author writes now

The hardware facts, once: the module and its sources, its generics
(`parameters()`), one `Index` per dimension, each fold as a `Decision` over
the divisors of a bound extent (`extent_of`), the RTL's loop nest
(`bound_schedule`), one `AxiStreamPort` per interface (the indices it reads,
lane order, what it reduces or holds, which reduction its TLAST closes, the
element a producer states), and the RTL's own limits (`admission`). The model
binds the extents from the ports' tensors (refusing ports that disagree),
derives every port's traversal, idle widths and pins, infers ordinary value
semantics, and the conformance harness checks the declared order against the
RTL in XSim. Guide: [`src/finn/kernels/AUTHORING.md`](../../src/finn/kernels/AUTHORING.md).

## Done (committed)

| Step | Commits | Fast gates | XSim from the commit |
|---|---|---|---|
| P0 probes | `6278ac53a` | probes 29 | — |
| A1 conformance harness | `59299e0a3`, `a0e881750`, `e4ab6646a`, follow-ups `78a574dd4`, `eabb488fb` | kernels 809 | all cases; loop- and lane-order planted errors fail in every sample |
| A2 `T \| Rejected` semantics; 88 removals | `68e68ee8a`, `a361697be` | Space 454, kernels 809 | all pass |
| A3 extents from the ports (lane A) | `451c4747b`, `22e12e9a3` | dataflow 61 | with wave 1 |
| A3b RTL checker names without values (lane B) | `ecaad8080`, `36036a62f`, `4e974e6dd`; merge `a12bd9ea1` | kernels 822; checker binds 34/34 | all pass |
| `inner_shuffle` fix adopted (lane C) | FinnLib `99d75e8`; FINN `b2cb336a3` | kernels 822 | transpose passes; adapters 32 |
| A4 `AxiStreamPort`, base extents, dotp | `2b6572282` | kernels 830 | all cases; sweeps at baseline |
| A5 every kernel migrated; folds are Decisions | `7dafec804` | kernels 832 | all cases; sweeps at baseline |
| A6 producers state their element | `dd1ba4568` | kernels 834 | all cases; graph 6; sweeps at baseline |
| A7 authoring guide; close | `6ecb83720`, then the XSim record | kernels 834; examples 33 | code identical to A6 |

Every step keeps ruff and mypy clean; the identity dump
(`docs/kernel-composition-2026-09-28/identity.py --api=k1`) is identical to
`evidence/identity-norom.txt` at every step. Gates run with Vivado off `PATH`
and `FORCE_COLOR` unset (a restarted shell sets it, and ANSI codes in mypy's
output fail five Space typing-fixture tests).

## Renamed after the plan

The code now uses the dataflow theory's terms
([`docs/dataflow-model/THEORY.md`](../dataflow-model/THEORY.md)); this record,
RECORD.md and PLAN.md keep the names they were written with:

| Written here | Now |
|---|---|
| `Schedule(folds=)`, `.folds`, `.fold(i)` (the lanes, `F_i`) | `factors=`, `.factors`, `.factor(i)`: the folding factor; the fold `E_i / F_i` is `.steps(i)` |
| `bound_schedule(beats, folds)`, `AxiStreamPort(folds=)`, `fold_domain`, kernels' `folds` | `bound_schedule(order, factors)`, `factors=`, `factor_domain`, `factors` |
| `Schedule(beats=)`, `Schedule.beats` (the beat order) | `order=`, `Schedule.order`; `Traversal.beats` (a count) is unchanged |
| a lane index "field": `Traversal.position(beat, field)`, memstream's `FIELD`, `FieldPlacement(field_index, …)`, `PackedBeatLayout.fields` | a lane: `position(beat, lane)`, `LANE`, `LanePlacement(lane, …)`, `PackedBeatLayout.lanes` |
| `conformance(..., folds=)`, `Sample.folds` | `factors=`, `Sample.factors` |

## Open

- The compiler-facing op wrapping one kernel: designed next, from the plan's
  "Forward interface" table (scratchpad `issues/kernel-op-design.md`); its two
  known gaps are a per-kernel derived output shape and resource estimates.
- The tiled MVU's tile extents: `bind_extents` does not solve an affine axis
  with one unbound index; not needed by any kernel here.
- Thresholding PE above C (rows folded into lanes): the RTL takes it, the
  model refuses it (G0.4a) until the stream model presents it.
- slang rejects `thresholding_axi`'s own default for `THRESHOLDS`; every
  binding supplies it, so nothing is affected.
- A design fed by a cyclic source repeats; the harness compares the first
  pass. Whether a composite should say so is open.
- The documentation examples are not a fast gate (A6 broke one unnoticed).
