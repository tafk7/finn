# E0 report: the refined Decision, probed on today's engine

Written by the E0 subagent (2026-09-28); saved here by the lead, since the
subagent could not write files outside its code. The lead's review is
[`REVIEW.md`](REVIEW.md).

## Result

The refined `Decision` (F3) can be built on today's engine. At class creation
it is expanded into the Decision over nodes the engine already compiles
(`node_choice`).

For cases 1–4, over the real kernels, the refined form matches today's
declarations:

- It builds the same module for all 14 identity-dump configurations: equal
  `module_build_fingerprint` and equal `MatMulAssembly`.
- It declares the same decision keys over the 3 fact sets (20 keys each).
- Core and adapter compatibility agree. Refusals agree on code and text,
  except for the provenance clause.
- `settle` makes the same choices as `compatible` plus `commit_adapters`.

Five things did not hold as written. None of them changes a settled decision
(F1–F10).

1. **Narrowing and pinning a Decision over kernels is not possible today**
   once bindings are shared: an outer body cannot restate a stream binding. A
   spike shows the fix (Findings 1).
2. **`compute["packed"]` fails mypy `--strict`**, because the attribute is
   annotated as the union of candidate types. A class-body handle naming the
   entry is the typed spelling and reaches the same node.
3. **The unsupplied-input check cannot run at class creation**, because an
   enclosing body may still supply the input. The engine's existing link-time
   error already names the candidate.
4. **The plan does not say what the None candidate's key is.** The probe uses
   `"none"`; today's `delivery` uses `"external"`.
5. **`settle` over "every Decision" can only mean Decisions over kernels.**
   Scalar Decisions have no candidate admission. `realization` stays a special
   case.

## Files

| File | What it is |
|---|---|
| `refined.py` | The refined `Decision` (a function standing in for the engine class), `RefinedChoice` (the class-body face), `required()`, `unmet_required`, `RequiredMeta`, `settle`, `compatible_cases`, admission helpers |
| `spike.py` | `engine_spike()`: narrowing and pinning by key. It patches `_nodes._choice_supplier` and `_linker._Linker.choice` with `mock.patch` inside a `with` block only |
| `stubs.py` | Case 5 stubs: `ProtoKernel` (`schedule = required()`, a `width` fact), `PackedCore` (owns `pe`, `simd`), `StubCore` (owns `rows`), `WideCore`, `Unfinished`, `Narrow` |
| `real_cases.py` | `RefinedStream`, `RefinedBufferedStream`, `RefinedMatMul`, `OwnedRamStyleStream` (Q3 option C), and `refined_assembly` (`matmul_assembly` using `settle`) |
| `test_rules.py` | 20 tests: the rules, over the stubs |
| `test_real_cases.py` | 32 tests: cases 1–4 against today's declarations |
| `conftest.py` | Puts `e0/` and `tests/` on `sys.path`; checks the evaluation context is reset |

Run: `PYTHONPATH=src /home/tkeller/prj-kernels/.kernel-venv/bin/python -m pytest docs/kernel-composition-2026-09-28/e0 -q`.

Cases 1–4 use the real kernels, unchanged. The refined families subclass
today's and replace only the Decision under test, under the same name, so
every other member and key is today's.

## The API as built

### `Decision`

```python
Decision(entries: Mapping[str, type[Space] | Space] | None = None, /, *,
         optional: bool | str = False, when: ValueRef[bool] | None = None,
         **shared: object) -> Any
```

- **Without `entries`** it is today's engine `Decision`, unchanged.
- **Entries.** Each entry is a `Space` class or a call on one. A class becomes
  a fresh node `cls()`. Each entry is placed as a candidate through
  `node_choice`, exactly as `values={...}` does today.
- **Shared bindings.** Every keyword other than `optional` and `when` is merged
  into every entry, checked with the engine's `check_supplier` and recorded
  with the Decision's origin.
- **`optional=True`** adds a None candidate keyed `"none"`, placed first.
  `optional="<key>"` is a probe-only bridge keeping today's `"external"` key in
  case 2; it should not ship.
- **`when=`** guards the whole Decision, as today.
- The result is a `RefinedChoice` (a `NodeChoice` subclass). The persisted
  value and the keys are today's.

Errors, all `DefinitionError` at class creation, naming the author's line:

| Rule | Message |
|---|---|
| A shared binding some candidate lacks (or declares as behaviour) | `Decision (declared at f:l): shared binding 'width' is not declared by candidates ['narrow (Narrow)']. A shared binding goes to every candidate: write it on the entries that take it instead` (every lacking pair in one error) |
| A shared binding also written on an entry | `Decision (declared at f:l): candidate 'packed' already binds width (at f:l), and the shared bindings bind it again; a body sets a member once: remove it from the entry or from the shared bindings` |
| A shared name `values`, `domain`, `semantics` or `name` | `Decision (declared at f:l): shared bindings ['values'] are named like the Decision's own arguments; rename the Param (values -> contents), or write the binding on each entry that takes it` |
| An entry that is neither a class nor a call | `candidate 'x' must be a kernel class or a call on one` |
| An entry keyed like the None candidate | `'none' is the key of the None candidate` |
| A class entry with an unmet `required()` member | `Unfinished (candidate 'unfinished' of a Decision (declared at f:l)) leaves required members unmet: ProtoKernel.schedule; a class with an unmet required() member cannot be placed (define them in a subclass)` |

An **unsupplied required input** stays the engine's existing link-time
authoring error, raised by `design_space`, never a refusal, naming the
candidate's path:

`compute.packed.width is not supplied: ProtoKernel.width (declared at stubs.py:33) is required, and nothing supplies it for the node compute.packed (declared at f:l) before Open is prepared; supply it at the call or assign it (compute.packed.width = ...)`

### Reads (`RefinedChoice`)

- **Direct, in the class body (`compute.cycles`)** produces a
  `ChoiceMemberRef`. It is refused unless every non-None candidate declares
  the member (`... a direct read needs a member every candidate declares;
  candidates ['stub'] do not declare 'pe'. Read it qualified, as
  compute["packed"].pe`). Where value types are known before linking, they
  must match. A name no candidate declares raises `AttributeError`. The None
  candidate does not count as lacking; a read through it is inapplicable.
- **Direct, in a method (`self.compute.cycles`)**: unchanged; the engine cannot
  check method bodies, but mypy enforces the rule as `union-attr`.
- **Qualified (`compute["packed"]`)** is the candidate node, the same object a
  class-body handle is: inapplicable when another candidate is selected; works
  in the class body, from class access and for enclosing bodies' assignments
  (`mm.compute["packed"].pe = 2`, keyed `mm.compute.packed.pe`). In methods
  and on configurations it is today's `inspection.candidate(point,
  Family.compute, "packed")`.

### `required()`

Returns a `Required` marker, which collection ignores. A subclass meets it
with any attribute. Calling a class that leaves one unmet raises `... leaves
required members unmet: ProtoKernel.schedule; ...`. In the probe the check
lives in a `SpaceMeta` subclass; E1 must move it (Findings 7).

### `settle`

```python
settle(point, *, admission=None) -> Settlement(point, committed, open)
```

It looks at every applicable, undecided Decision over nodes. A case is
compatible when committing it alone is accepted and the candidate's admission
is Available (no admission: admitted; None case: admitted when accepted). It
commits a Decision with exactly one compatible case, then rescans. Several or
none stay in `open`. `admission` defaults to the member named `admission`
(F9); for today's kernels the probe maps `DotpAxiKernel.support` and
`StreamAdapter.admitted` explicitly.

### The spike: narrowing and pinning by key

Inside `engine_spike()`: `unit.compute = "stub"` pins (the selector becomes a
constant, the key is listed in `inspection.pinned`, `commit` refuses it as
stale); `unit.compute = Decision(values=("stub",))` narrows (the key stays,
its domain shrinks); an unknown key is refused (`an override narrows the
cases, it does not add one`). Every declared candidate keeps its node, with
its bindings read in the declaring body.

## The five cases

1. **The dot-product core.** Two handle calls with the same 13 bindings become
   one Decision with the 13 shared bindings and `narrow_weights` on the packed
   entry. Equal fingerprints, assemblies and keys; equal compatible-core sets
   over four fact sets; equal refusal codes and texts except provenance.
2. **Weight memory.** `values` is refused as a shared binding, so it is
   written on each entry (Q1). Cases `external, cyclic, memstream` in today's
   order. Same equalities.
3. **The stream adapter.** Seven entries, shared `tensor` and `plan`,
   `when=T.adapting`; `ram_style` written on each of the six buffering
   entries. Same equalities; `settle` commits `input_gen` wherever
   `commit_adapters` does. Option C of Q3 was probed as `OwnedRamStyleStream`.
4. **Transport.** `Decision({"direct": _Direct, "fifo": StreamFifo(...)})`;
   nothing is shared (sharing `tensor` is refused, naming `direct`). Keys
   unchanged.
5. **Disjoint choices** (stubs). Keys `compute`, `compute.packed.pe`,
   `compute.packed.simd`, `compute.stub.rows`, identical in both forms; each
   candidate's keys inapplicable while the other is selected.

## Coverage (52 tests, observed `52 passed`)

| Item | Result |
|---|---|
| Cases 1–4 | pass (14 identity configurations, 3 key sets, case tests) |
| Case 5 | pass |
| Shared binding a candidate lacks; double assignment; reserved names | pass |
| Unsupplied required input | pass, at link time |
| Unmet `required()` | pass |
| Direct and qualified reads, inapplicable under another candidate | pass |
| Namespaced keys; pinning a candidate's choice from outside | pass |
| Narrowing and pinning the Decision itself | **fails on today's engine**; passes with the spike |
| Switching candidates (stale-choice rule) | pass |
| Selections by key; unknown key or case refused | pass |
| Collapse on and off | pass |
| `optional` | pass |
| Composite declared at run time, stable keys | pass |
| `settle` | pass |

The identity dump's `replay-input-gen` configuration passes a keyword that
today's `matmul_assembly` no longer takes; dropped, it equals `external`.

## Answers to the three questions

**Q1. Shared bindings named like the Decision's own arguments.** `when` cannot
collide (the engine reserves it as a member name). `values` (`CyclicDelivery`,
`MemStreamKernel`) and `name` (`physical/ports.py`) collide; `domain` and
`semantics` could. **Recommendation:** keep shared bindings as keywords,
refuse the four names as shared bindings, and rename `values` to `contents` on
the two memory kernels in K1.

**Q2. Keep `values={...}` over nodes beside entries, or migrate?**
**Recommendation: migrate.** E1 keeps the old form working (its gate requires
the kernel gate unchanged); removal waits until nothing uses it. 48 sites: 4 in
`src/finn/kernels`, 2 in kernel tests, 42 in 16 Space test files, and the
Space documentation's "Choices over nodes" section.

**Q3. A binding six of seven candidates take (`ram_style`).** All seven
adapters declare `ram_style` (on `StreamAdapter`); six use it. **A:** written
on each entry (today's key). **B:** shared across all seven, rejected (passes
only because the base over-declares). **C:** each buffering adapter owns its
`ram_style` Decision, keys `<stream>.adapter.<case>.ram_style`.
**Recommendation:** C in K2; A until then.

## Findings

1. **Narrowing is unworkable today once bindings are shared.** Fix in E1:
   narrow and pin by key, keeping the declared candidates (the spike).
2. **`compute["packed"]` is not typable** under mypy `--strict`. In kernel
   code, spell a qualified read with a class-body handle naming the entry;
   keep `compute["packed"]` for untyped code and enclosing bodies.
3. **The unsupplied-input rule holds only at link time.** No E1 change.
4. **The None candidate's key** is fixed as `"none"`; `external` becomes `none`
   in K1 (D7).
5. **`settle` applies to Decisions over kernels only**, and needs F9's standard
   `admission` member.
6. **Shared bindings are not checked by mypy** (`**object`); the runtime check
   at class creation covers the same errors.
7. **`required()` must not use a metaclass** (a `SpaceMeta` subclass loses
   `dataclass_transform`). Put the check in `declare_node` and declare
   `required` as a field specifier with `init=False`.
8. **A shared binding's provenance names the Decision's line.**
9. **`inspection.choices()` raises on a pinned Decision over nodes.**
10. **Selections under narrowing**: `codecs.decode` checks declared cases, not
    the narrowed domain; the restore then refuses with `domain-membership`.

## Engine changes E1 must make (`src/finn/core/space/`)

1. `declarations.py`, `Decision.__new__`: positional-only `entries`,
   `optional`, `**shared`, with an overload ahead of the scalar overloads;
   refuse the reserved shared names; delegate to `_nodes.entry_choice`.
   `values=` over nodes stays until K2, then raises.
2. `declarations.py`: `required()`, `Required`, `unmet_required`, exported;
   `required` in `SpaceMeta`'s `dataclass_transform(field_specifiers=...)`.
3. `_nodes.py`: `entry_choice(entries, shared, *, optional, when)` with the
   checks above, then `node_choice`.
4. `_nodes.py`, `declare_node`: refuse families with unmet `required()`.
5. `_nodes.py`, `NodeChoice`: `__getattr__` (common members) and
   `__getitem__` (qualified).
6. `_nodes.py`, `_choice_supplier`: accept a key (pin) and a scalar Decision
   over keys (narrow).
7. `_linker.py`, `_Linker.choice`: key selections keep the declared
   candidates; a pin makes the selector a constant and records it as pinned; a
   narrowing restricts its domain.
8. `_linker.py`, `check_semantics`: name disagreeing candidates.
9. `inspection.py`, `choices()`: handle a pinned selector.
10. `settle` and `compatible_cases` as a public engine service, re-exported by
    `finn.kernels.configure`.
11. Optional: `codecs._check_case` checks the narrowed domain.

## Open risks

- The spike patches private functions; E1 must test selections and collapse
  with it.
- The qualified-read spelling in strictly typed kernel code (Findings 2).
- `settle` cost: 5.0 s today against 5.3 s refined over the 14
  configurations.
- Strictness is only as honest as the declarations (Q3).
- G1's ONNX adapter itself is not probed; only run-time composite declaration.

## Test counts (as observed)

- Probe: `52 passed`; ruff clean on the probe files; mypy not clean on the
  probe's tests and 5 probe lines (Findings 2 and 7).
- Gates before and after adding the probe, identical:
  `scripts/check-kernels.sh` Space 427, kernels 819, exit 0;
  `scripts/check-dataflow-design.sh` 33, exit 0.
