# Landing the declarative Space on `feature/kernel-package-extraction`

Date: 2026-09-26. Status: **landed locally, not pushed.** Design record:
[`DESIGN.md`](DESIGN.md). Evidence for this landing:
[`evidence/landing/`](evidence/landing/) (`.txt`, since `*.log` is ignored).

## 1. What landed, and from where

Source: `spike/space-declarative-4` at `1853a67d6`, the fourth iteration of a
chain rooted at merge base `39482b480` (`spike/space-declarative`,
`-2`, `-3`, `-4`). It was brought in with a three-way
`git merge --squash`, so the two docs-only commits the feature branch gained
after the merge base (`224b636d7`, design records; `599a31c67`, the
artifact-integration SPEC) are untouched. The merge had no conflicts; after it
the index differed from the spike's tree only by those two commits' 12 docs
files. The squashed change was split by path into curated commits. The spike
branches are kept as the record of how the design got here.

| Commit | What |
|---|---|
| `b244ec8be` | `src/finn/core/space`: the engine (declarations, `design_space`, Decisions over nodes, references and assignment edges, overrides with provenance, annotated value types, reference inputs and `Users`, collapse, views read as values) |
| `53e7992c5` | `tests/core/space`, including the strict-mypy typing fixtures |
| `959ed0de5` | `src/finn/kernels`, `tests/kernels`, `scripts/benchmark-space.py` |
| `ec24fa607` | this directory: the design record, the spike's evidence and probe scripts |
| `8c7ca08c6` | `docs/space-design-graph-2026-09-26/DESIGN.md` marked superseded by this record. `docs/space-graph-composition-2026-09-25/DESIGN.md` already said it was superseded (by the design-graph record) |
| `902efb29f` | `docs/robust-mvau-2026-09-26/SPEC.md`, the robust MVAU task spec, committed as written |
| `bfb31ad48` | R6: identical findings kept once where answers merge (section 4) |
| (this commit) | this record and `evidence/landing/` |

## 2. Validation

All runs on the landed tree, from
`/home/tkeller/prj-kernels/finn-kernels-extraction`, with the kernel venv
(`/home/tkeller/prj-kernels/.kernel-venv/bin/python`, Python 3.10), ruff and
mypy from `PATH`, and the normal `PATH` with Xilinx 2025.2 `xvlog`/`xelab`/`xsim`,
so the XSim-backed tests executed locally (no Docker). The gates and
fingerprints ran on the working tree that was then committed unchanged as
`bfb31ad48` (their transcript header reads "`902efb29f` + working-tree R6
fix"); the numeric sweep ran on the commit itself.

| Evidence | Result | Transcript (in `evidence/landing/`) |
|---|---|---|
| Gate `scripts/check-kernels.sh` | Space **423 passed** (418 on the spike + 5 R6 tests); kernels **763 passed, 0 skipped** in 294 s, so the 9 XSim-backed tests (they skip without a Vivado simulator; listed in `evidence/xsim-tests.txt`) executed; format, lint and strict mypy clean; exit 0 | `gate-check-kernels.txt` |
| Gate `scripts/check-dataflow-design.sh` | **16 passed**; format, lint and strict mypy clean; exit 0 | `gate-check-dataflow-design.txt` |
| MVAU fingerprints (`../space-graph-composition-2026-09-25/fingerprints.py`) | identical to the spike (`evidence/fingerprints.txt`); against `0d700b1ab`, the four non-cyclic configurations are identical and the two cyclic ones differ (section 3) | `fingerprints.txt` |
| Fingerprints with the cyclic instance renamed (`fingerprints_renamed.py`) | **bit-identical** to `../space-graph-composition-2026-09-25/evidence/fingerprints-baseline-0d700b1ab.txt` | `fingerprints-renamed.txt`, `fingerprints-compare.txt` |
| MVAU numeric XSI sweep (`tests/kernels/rtlsim/mvau_assembly_numeric.py`) | **28/28 PASS, 0 FAIL** on `bfb31ad48` with Vivado Simulator v2025.2 and FinnLib `b17eae6a0`: direct run **20 PASS** (5 cases x external/cyclic x free/stalled, 386 s); depth-2 weight FIFO **8 PASS** (`packed` and `int8_pumped` x external/cyclic x free/stalled, 83 s + 80 s); each run exit 0 | `mvau-numeric-direct.txt`, `mvau-numeric-fifo-packed.txt`, `mvau-numeric-fifo-int8-pumped.txt` |

**The numeric sweep** had not been run on any spike iteration. It builds each
MVAU through `mvau_assembly`, prepares and materializes the module sources from
the artifact store, wraps the top ABI in an observation wrapper and drives it in
XSI with random INT/UINT stimulus (seed 83, extremes forced), checking the
result words, the replayed activation stream and its `last` flags, and the
weight words the compute core consumed, once with free and once with stalled
outputs. The five cases are `single` (DSP48E1, 1x1), `packed` (DSP48E2, 6x6,
PE 3, SIMD 2), `one_beat_reductions` (DSP58), `padded_output` (UINT3
activations) and `int8_pumped` (DSP58, pumped compute), each with external and
cyclic weight delivery. Environment, as in
`../kernel-package-extraction/rtl/VALIDATION.md`:

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=src:tests:deps/qonnx/src
FINN_ROOT=/home/tkeller/prj-kernels/finn-kernels-extraction
FINNLIB_ROOT=/home/tkeller/prj-kernels/finn-kernels-extraction/deps/finnlib
FINN_XELAB_MT=2
LD_LIBRARY_PATH=/home/tkeller/Xilinx/2025.2/Vivado/lib/lnx64.o

python -m kernels.rtlsim.mvau_assembly_numeric --output /tmp/mvau-landing
python -m kernels.rtlsim.mvau_assembly_numeric --case packed --weight-fifo-depth 2 --output /tmp/mvau-landing-fifo
python -m kernels.rtlsim.mvau_assembly_numeric --case int8_pumped --weight-fifo-depth 2 --output /tmp/mvau-landing-fifo2
```

The harness's CLI matched the planned commands (`--case`, `--delivery`,
`--output`, `--rom-style`, `--weight-fifo-depth`). **One simulation per
process** (the `CLAUDE.md` XSI rule) was verified in the code, not assumed:
every `drive_observed` call writes a request and runs
`rtl_transport.py --simulate` in a fresh `sys.executable` subprocess, which
compiles, makes the single `load_sim_obj` and runs; the parent never loads a
simulation. The 28 runs left 28 `simulation.log`/`response.json` pairs, one per
subprocess. The cyclic builds' generated top instantiates
`u_implementation_cyclic` (section 3).

## 3. Accepted build-identity change: the cyclic weight-delivery instance

In the cyclic MVAU configurations the weight-delivery instance is now
`u_implementation_cyclic`; `0d700b1ab` named it `u_weights`. The name derives
from the node's declaration path: the cyclic delivery is the candidate
`implementation.cyclic` of the `implementation` Decision, and `netlist` names
every module `u_<node>` with `.` joined as `_`. The instance name is a consumed
input of the composed module, so it changes those configurations'
`module_build_fingerprint`. **This change is accepted.**

| Configuration | `0d700b1ab` | landed |
|---|---|---|
| `external` | `9973c6fb…2c190` | unchanged |
| `cyclic-block` | `7c2110dd…5a31c` | `36d7f9e0…664aa` |
| `fifo-external` | `09d35f61…11e12` | unchanged |
| `fifo-cyclic` | `468292e6…e5066` | `24f47e03…820c0` |
| `padded-output` | `58ef50a5…94866` | unchanged |
| `pumped-dsp58` | `68ff1342…49413` | unchanged |

With the one instance renamed back (`fingerprints_renamed.py` patches
`streams._wire` to map `u_implementation_cyclic` to `u_weights`), all six are
bit-identical to `0d700b1ab`: the rename is the only build-identity difference.
No MVAU decision key, node key, node kind or top-level ABI port name changed
(`evidence/mvau-keys.diff`, `evidence/mvau-abi-ports.txt`, from the spike, whose
kernel code landed unchanged). Consumers
that hard-code the instance (`tests/kernels/test_mvau_delivery_choice.py`,
`src/finn/kernels/README.md`) already use the new name; a cache holding
artifacts keyed by the old cyclic fingerprints will simply miss.

## 4. R6: duplicate findings in merged results

**Defect** (the design-graph spike's R6, `DESIGN.md` section 6 and open
question 3). One reason could reach a merge by two routes and be reported twice:
a view that reads a value and also requires it (`@view(requires=(kitchen.cost,))`
reading `self.kitchen.cost`) listed `kitchen.finish`'s blocker twice in its
`accepted_result`; a refusal reached through a view's output and an obligation
(`Aligned.factor` in `test_design_graph.py`) was listed twice; and a node whose
arguments are two aliases of one blocked value listed its blocker twice.

**Fix.** `results.merged_findings(answers)` concatenates the answers' findings
and keeps each identical finding once, in first-seen order
(`dict.fromkeys`; `Finding` is a frozen, hashable dataclass, so the same reason
compares equal). The three merge sites use it: the view/constraint reducer's
`_unresolved` and `_rejected` (`results.py`) and a node's blocked inputs
(`_blocked`, `_runtime.py`). Findings that differ in anything (owner, kind,
code, message, details or causes) are all kept, and per-obligation results in
an assessment keep their own findings. Result precedence is unchanged.

**Tests.** New `tests/core/space/test_merged_findings.py` (5): the helper keeps
distinct owners, messages and causes; the reducer reports a blocker and a
refusal reached through output and obligation once, while the obligation's own
result is untouched; a view that reads and requires its inputs lists each
blocker once; a refusal through output and two obligations is reported once
per owner; a derived output over two aliases of one open Decision lists it
once. The four behavioural tests fail when the old concatenation is restored
(checked by monkeypatching `merged_findings`). `test_design_graph.py`'s `Aligned` test,
which compared a set and carried a comment about the duplicate, now asserts the
exact single finding. No other test asserted duplicates. The question left
open is whether an obligation the output already reads should be refused as
redundant.

## 5. Coordination: impact on other work touching this layer

These notes are for the efforts named; neither document was edited.

### 5.1 `docs/kernel-artifact-integration-2026-09-25/SPEC.md`

The SPEC (committed at `599a31c67`, authored against `240c33a11`) predates the
declarative Space. Its architecture (kernel owns valid output; artifacts
independent of Space; external association) is unaffected, but several
sections name Space surface that changed or no longer exists:

| SPEC section | Relies on | Now |
|---|---|---|
| Intro, link to `../../../scratchpad/space/DESIGN.md` | the generic Space design | [`docs/space-declarative-2026-09-26/DESIGN.md`](DESIGN.md) in this repository |
| Outcome diagram, §1 code, §1 "Calling a view returns its accepted value" | callable views: `kernel.build_requirements()`, `point.build_requirements()` | a view **reads** as its accepted value: `point.build_requirements` (raises `ValueUnavailableError` carrying the result when not accepted). `BoundView`, `Space.view()` and `accepted()` are removed |
| §1 "Drivers using `query()` or `inspect()`" | `point.build_requirements.query()` / `.inspect()` on a bound view | `point.query(MVAU.build_requirements)` and `point.inspect(MVAU.build_requirements).accepted_result`; into a child `point.child.inspect(Child.view)`; `field(view)` is a `BoundValue` |
| §1 "Parents consume accepted child products through ordinary view calls or existing accepted references" | `self.child.view()` in a method; `accepted(child.view)` / `.accepted(KEY)` in a class body | `self.child.view` in a method; `child.view` in a class body is a reference typed as the accepted value, usable as a formal, `Present` source or `requires=` obligation |
| "No artifact API may require a live configuration, **bound view**, Space handle ..." | `BoundView` | the principle stands; the type is gone, so "bound view" now means `BoundValue` |
| §1 typed view references, `ViewKey` exports | `ViewKey` + `exports` | unchanged; the new graph primitives `Members(KEY)` and `Users(KEY)` gather exports as `Located` values |
| Baseline paragraph: "reusable `CyclicDelivery`, stream contracts, canonical traversals, and stream composition" | `Stream(Subspace[StreamLink])`, `TopInput`, `compose(instances=, connections=)`, `connected(stream)` in `finn.kernels.streams` | streams are ordinary Spaces that kernels reference: `Stream(spec=, port=)`, `BufferedStream` (transport Decision), a kernel's reference inputs and `PORTS` export, `ends = Users(PORTS)`; composition is `netlist(modules, streams, module=, producer=)` over `Members(MODULE)` and located connections. `TopInput` and `compose` are gone; a missing side becomes the boundary named by the stream's `port` |
| "MVAU external/cyclic behavior" (Compatibility) | `SubspaceChoice` for weight delivery | `implementation: CyclicDelivery \| None = Decision(values={"external": None, "cyclic": cyclic})`, a Decision over nodes; keys unchanged; numeric behaviour re-validated (section 2) |
| Compatibility: "Preserve the Space API, result precedence, ordinary view calls, choice ownership ... No generic engine change is required" | the pre-landing Space API as the baseline to preserve | the baseline moved: result precedence is unchanged, but the API is the declarative one (`design_space`, annotated `Param`/`Decision`, views as values), choice ownership now includes pins (a pinned key disappears, listed by `inspection.pinned`) and replacements; R6 changed only duplicate findings. Re-baseline §1 and §6 against this landing before implementing |
| Compatibility: "Physical instance names ... remain inputs where consumed" | instance naming | holds; the one intended change (`u_weights` → `u_implementation_cyclic`) is recorded in section 3, and its cause is the declaration path |
| §6 association: "public Space inspection APIs"; "not Space handles, callbacks, compiled models, or configurations"; "Saved selections omit parameters" | `inspection.explain`, `SpaceModel`, `Selection` | `inspection.explain` evidence nodes now list bypassed aliases in `EvidenceNode.via`; `inspection.provenance()`/`pinned()` are new and useful for an association's summary; `SpaceModel` is `Model`; a persisted selection holding a pinned key is refused as **stale** on decode |
| Implied driver setup (facts and choices) | `compile_space(F)` returning a `SpaceModel`, `model.bind(facts)`; `finn.kernels.configure.configure(space_type, facts, choices)` | `design_space(MVAU(pe=..., ...))` opens the design space, facts being the root node's typed formals (the spike first called it `configure`, and renamed it); `compile_space` is the internal `compile_model`, `SpaceModel` is `Model`; the kernels' helper is `finn.kernels.configure.commit(point, choices)`, which commits choices by key and refuses pinned keys as stale |
| Declarations of members (implicit in every kernel example) | `Param(T)`, `Decision(T, ...)`, `Subspace(F, ...)`, `SubspaceChoice`, `ScopeBuilder` | annotated members, `area: int = Param()`, `Param(required=False)`; a child is a family call `F(...)` in the class body; a `Param` annotated with a family is a reference input; a choice of child nodes is a Decision over nodes, read through `selected(decision)` or attribute access |
| Typed references and keys (§1 "explicit typed view references") | `ValueKey`, `DecisionRef`, `AcceptedViewRef` (`Subspace.accepted(KEY)`), `ChoiceView` | removed; a member key is typed as its value (`with_choices({MVAU.pe: 2})`, `field(MVAU.pe) -> BoundDecision[int]`); a view is keyed by its declaration `View[T]`; exported views are still `ViewKey`s |

Nothing in sections 2, 3 or 5 (value semantics, declaration surface, parameter
spelling, RTL/HLS preparation) depends on the changed surface.
`default_semantics(ModuleBuildRequirements)` is still how kernels declare
their build views. Section 4 (nested generated modules) should build on
`netlist`'s `u_<node>` naming rather than the removed `compose`.

### 5.2 External-resources merge (`integration/er-kpe`)

The analysis in the memory note `project_external_resources_merge.md` executed
a merge of `origin/feature/external-resources` with this branch at `599a31c67`
into `integration/er-kpe` (`8ab9c9d78`, unpushed). This landing affects it:

- **Re-merge conflicts.** A trial `git merge-tree` of `integration/er-kpe` with
  the landed branch reports textual conflicts in 11 files:
  `src/finn/core/space/_configuration.py`, `declarations.py`,
  `tests/core/space/test_declarations.py`, `test_inspection.py`,
  `tests/kernels/helpers.py`, `test_dotp.py`, `test_flat_kernels.py`,
  `test_installed_package.py`, `test_mvau_assembly.py`,
  `tests/kernels/typing/axi_stream_types.py`, `space_descriptor_types.py`.
  They are ER's edits (`typing_extensions` → `typing`, `finnlib_root()` and
  `vivado_simulator()` from `finn.resources`, the 3.12 `__set_name__` test fix,
  installed-package dependencies) meeting this landing's rewrites of the same
  files. Resolve them in favour of this landing's code, reapplying ER's edits.
- **`typing_extensions`.** ER dropped it from the dependencies (its
  `test_installed_package.py` asserts no `typing*` requirement) and imports
  `Self` from `typing`. The landed engine also imports `dataclass_transform`
  from `typing_extensions` (`_configuration.py`); in the re-merge it must come
  from `typing` (3.11+) as well, or the installed package fails to import.
- **Python version.** The landing is validated only on Python 3.10 (the kernel
  venv and the Space CI). ER targets 3.11+ and plans 3.12; re-run both gates
  there after the re-merge. The one known version-sensitive test,
  `test_declarations.py`'s `__set_name__` wrapping (3.12 no longer wraps the
  error in `RuntimeError`), is one ER already fixed and is among the conflicts.
- **`tests/kernels/helpers.py`.** ER added `finnlib_root()` and
  `vivado_simulator()` to it; this landing removed `compile_space` from its
  imports (the engine no longer exports it). Keep ER's helpers on the landed
  file.
- **Validation baselines.** The note's counts (Space gate 301, kernel gate 758)
  predate this landing; expect the counts in section 2 after a re-merge. Its
  XSI check (MVAU packed, external/cyclic, depth-2 FIFO) is a subset of the
  sweep in section 2.
- **Unaffected:** runtime dependencies (no new third-party import; greenlet,
  jinja2, msgspec and pyslang as before), FinnLib pin and layout, the
  `finn.parked` exclusion, the executor seam, portability of build products.
  The instance-name change (section 3) alters two cyclic MVAU artifact keys;
  it does not affect `finn.resources`.

## 6. Open items carried forward

- The design record's open questions (`DESIGN.md` section 10): a view named in
  its own class body needs a `cast` (R30); the path form of `inspect` and
  `field` is not statically exact (R28, R31); whether a redundant obligation
  should be refused (the rest of R6); statically untyped `requires=` (R32);
  the declared domain as a contract; pinning a Decision over nodes; persisted
  vs in-memory selections; which values a refusal names; `Users` through
  forwarding composites and keyed gathers; typing `Const` as its value; path
  assignment through a reference input; a separate XSim gate target.
- **The query/search-tools pass** (`DESIGN.md` section 3): a presence-aware
  gather `Collect`, `Present` as `Collect(..., exactly_one=True)`, structural
  queries as plain Python over declarations, `Members`/`Users` as
  compositions, and how a class body names its own structure.
- **Cross-snapshot cache reuse** (R7): one local edit still re-runs the whole
  graph (the scale probe's `callbacks re-run` column).
- **Preparation cost:** iteration 3 made preparing a model 16–18 % slower than
  iteration 2 (layering, provenance, annotation dispatch, the collapse pass;
  `DESIGN.md` at `426882411`, section 8); iteration 4 left it unchanged. Not
  addressed here.
- The robust MVAU task (`docs/robust-mvau-2026-09-26/SPEC.md`) starts after
  the kernel audit revises its scope.
