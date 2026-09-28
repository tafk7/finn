# E0 review

Reviewer: the lead, 2026-09-28, against the plan's F3 and E0 text
([`../PLAN.md`](../PLAN.md)) and the subagent's [`REPORT.md`](REPORT.md).

## What was checked

- The probe's tests, rerun independently: `52 passed` (`PYTHONPATH=src
  .kernel-venv/bin/python -m pytest docs/kernel-composition-2026-09-28/e0`).
- `src/` is untouched: the worktree shows only `e0/` as new.
- The gates at the same revision, run separately in the V1 worktree before any
  V1 edit: Space 427, kernels 819, dataflow 33, ruff and mypy clean. They
  match the report's numbers.
- `refined.py` and `spike.py`, read in full. The refined Decision is a
  class-creation expansion into `node_choice`; the strict checks run before
  any binding is recorded; the spike changes only the two functions it names,
  inside a context manager. Nothing in either depends on kernel notions except
  the `admission` default, addressed below.

## Verdicts

| # | Item | Verdict |
|---|---|---|
| 1 | Entries (class or call), shared bindings as keywords, strict: every candidate declares each, no double assignment | **accepted** as built |
| 2 | Reserved shared names `values`, `domain`, `semantics`, `name` (Q1) | **accepted**; `CyclicDelivery`/`MemStreamKernel`'s `values` becomes `contents` in K1 |
| 3 | `optional=True` adds a None candidate keyed `"none"`, first (Finding 4) | **accepted**; the probe-only `optional="<key>"` does not ship. K1 renames `external` to `none` (D7) |
| 4 | Narrowing and pinning a Decision over nodes by key (Finding 1) | **accepted as an E1 engine change**, with the selections, collapse and inspection tests the report says are missing (Findings 9, 10 included) |
| 5 | Direct reads need a member every candidate declares, with one value type; qualified reads `compute["packed"]` | **accepted** |
| 6 | Typed qualified reads (Finding 2) | **accepted**: in strictly typed kernel code a qualified read is spelled with a class-body handle that is also the entry (`packed = PackedDotpKernel(narrow_weights=...)`, entry `"packed": packed`); the handle carries only that candidate's own bindings, the shared ones stay on the Decision. `compute["packed"]` stays for untyped code and enclosing bodies |
| 7 | `required()` as a field specifier with the placement check in `declare_node` (Finding 7) | **accepted**; no metaclass |
| 8 | Unsupplied required inputs stay the link-time error (Finding 3) | **accepted**; no change |
| 9 | `settle` over Decisions over nodes, repeated until nothing changes (Finding 5) | **accepted, with one change**: the engine is generic, so `finn.core.space.settle` takes the admission as an argument (none: a case is compatible when its commit is accepted). `finn.kernels.configure.settle` supplies the kernels' convention, the `admission` member (F9). The engine never names `admission` |
| 10 | Q2: migrate `values={...}` over nodes to entries | **accepted**: E1 keeps the mapping form working; K2's close migrates the remaining sites (kernels, Space tests, docs) and turns the mapping form into an error pointing at entries |
| 11 | Q3: each buffering adapter owns its `ram_style` Decision (option C) | **accepted for K2**; option A until then |
| 12 | Shared bindings unchecked by mypy (Finding 6); provenance names the Decision's line (Finding 8) | **accepted** as known limits |

## Not settled by E0, carried forward

- The `realization` Decision is scalar and stays outside `settle`; K1 keeps its
  explicit handling.
- G1's ONNX adapter itself: only run-time composite declaration was probed.

## Result

The engine design is fixed as the report's "Engine changes" list, with
verdict 9's split of `settle`. E1 proceeds.
