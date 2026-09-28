# Record: kernel choices, one interface per port, composable kernels

The execution record of [PLAN.md](PLAN.md). Each increment appends what landed,
the gate results as observed, key and name changes (D7), and deviations.

## G0 (2026-09-28)

| # | Answer |
|---|---|
| 1 | Weight memory is a discrete `memory` Decision in the composite (F10) |
| 2 | Open; V1 proceeds with `(k, n)` unless the user says otherwise |
| 3 | `required()` ships with E1 |
| 4 | Registration dropped; candidates are listed in the Decision |
| 5 | A port's contract is named after the nearest ancestor that owns a module |
| 6 | `replay_buffer` stays scrapped |

## E0: engine probe

- **By** a subagent; probe, tests and report in [`e0/`](e0/); the report is
  [`e0/REPORT.md`](e0/REPORT.md), the review [`e0/REVIEW.md`](e0/REVIEW.md).
- **Result.** The refined Decision builds today's modules and keys over the
  identity configurations for the dot-product core, weight memory, the stream
  adapter and transport; case 5 (disjoint choices) holds on stubs. Narrowing
  and pinning by key needed an engine change (spiked).
- **Gates.** Probe 52 passed (rerun by the reviewer); kernel gate Space 427,
  kernels 819; dataflow 33; unchanged by the probe.
- **Review verdicts** fixed the E1 design: the report's engine changes, with
  `settle` split into a generic engine service (admission passed in) and the
  kernels' convention (`finn.kernels.configure.settle`, the `admission`
  member).
- Commit `1051948f5`.

## E1: the engine refinement

- **Engine** (`finn.core.space`):
  - `Decision(entries, /, *, optional=False, when=None, **shared)`: entries are
    families or calls; shared bindings strict (every candidate declares each;
    no double assignment); the scalar form's argument names are refused as
    shared bindings; `optional=True` adds a `None` candidate keyed `"none"`,
    first. The `values={...}` mapping form is unchanged (migrated in K2).
  - Direct reads through an entry Decision need a member every candidate
    declares, with one value type (checked at class creation, from each
    candidate's collected semantics); `choice["key"]` reads one candidate's
    node (in a body, from class access, and for enclosing assignments).
  - An enclosing body pins a Decision over nodes with a key (the selector
    becomes a constant, the key is listed as pinned) or narrows it with a
    Decision over keys (the key stays, its domain shrinks); either keeps the
    declared candidates with their declared bindings. The earlier
    fresh-node narrowing still works.
  - `required(T)`, `Required`, `unmet_required`: a family with an unmet member
    cannot be placed (checked in `declare_node`) and cannot be an entry.
  - `settle(point, admission=None)` and `compatible_cases` in
    `finn.core.space.settling`; `settle` and `Settlement` exported.
  - `inspection.choices()` reports a pinned choice's selector as `None`;
    `codecs.decode` refuses a narrowed-out case.
- **Kernels.** `finn.kernels.configure.settle` and `admission` (the member
  named `admission`: a constraint group, a constraint or a view).
- **Docs.** Scratchpad `space/AUTHORING.md` ("Choices over nodes": entries,
  shared bindings, reads, `required`, `settle`; the overrides table; the house
  example in the entries form and the estate narrowing by key),
  `space/DESIGN.md` §4.3–4.4, and a migration note in `space/MIGRATION.md`.
  27 documentation examples pass.
- **Deviation from the plan.** `required()` is spelled `required(T)` (for
  example `schedule = required(Schedule)`), not `schedule: Schedule =
  required()`. As a dataclass field specifier (E0's Finding 7), mypy refuses a
  subclass that meets the member with a derived value ("Dataclass attribute
  may only be overridden by another attribute") and demands the member as a
  call keyword. Typed by its argument, like a derived value, it checks cleanly
  under `--strict`, and a Param, derived value, view or constant meets it. No
  settled decision changes.
- **Changed test.** `test_overrides.py`: a key now pins a Decision over nodes
  (it was refused); `test_inspection.py`: a choice's selector is typed
  `DecisionHandle[str] | None`.
- **Keys.** None changed.
- **Gates.** `scripts/check-kernels.sh`: Space 446 passed (427 plus 19 in
  `tests/core/space/test_candidate_entries.py`), kernels 819 passed, ruff and
  mypy clean, exit 0. `scripts/check-dataflow-design.sh`: 33 passed, exit 0.
  The kernel gate is unchanged, as E1 requires (nothing migrated).
