# Lane record: A3b, the RTL checker names parameters without values

Branch `lane/rtl-param-names`, worktree `finn-ka-rtl-checker`, from
`a361697be`; `deps/` fetched with `fetch-repos.sh` (FinnLib `d03f2fc`).
Wave 1, Lane B. `RECORD.md` and `PLAN.md` are not edited here.

## What landed

Commits (fast gates green before each, below):

1. `ecaad8080` **A name is established before its value.**
   `artifacts/rtl.py`: `extract` no longer declines a module because a
   parameter's value is neither an integer nor a string. The parameter is
   still reported, by name, with the value `None`. Type parameters are
   reported the same way (they were silently skipped). The unknown-override
   refusal (a binding naming an undeclared parameter) is unchanged and still
   declines. `tests/kernels/conformance.py`: `_check_rtl` extracts once and
   compares the ports with `check_against_rtl` (what `check_abi` does), and
   returns every declared parameter name, valued or not. So the pin check and
   the `parameters()`-name check now run for thresholding and eltwise.
2. `36036a62f` **Two diagnostics tolerated only where they are confined.**
   `artifacts/rtl.py`: `TOLERATED_WITHIN` maps a diagnostic code to the
   syntax constructs it is tolerated inside, and it is declined anywhere else
   (details and justification below). This covers the two slang elaboration
   declines, packed dotp at SIMD > 1 and transpose. `TOLERATED_DIAGNOSTICS`
   is unchanged.

No decision key, ABI, or kernel changes. New public name: `TOLERATED_WITHIN`,
`ExtractedModule.unestablished`.

## The representation

`ExtractedModule.parameters` and `local_parameters` stay
`tuple[tuple[str, value], ...]` in declaration order, with the value typed
`int | str | None`. `None` means "declared, value not established": an
unpacked array, a real or shortreal, a type parameter, or an integer with
unknown bits. The names in `parameters` are all established. The value
`None` is never filled in, and `unestablished` lists the names that carry it
(declared and local). I chose this over a separate names tuple for three
reasons: one entry per parameter keeps each name next to what is known about
its value; the declaration order and the declared/local split come for free;
and mypy `--strict` forces every reader of a value to handle `None`.

`check_abi`, and the harness's comparison, read `extracted.ports` only:
names, directions, and the widths slang resolved under the binding. No
parameter value enters the pin or width comparison, established or not. The
name check reads names only.

## Tolerated only inside a construct (commit 2)

Each entry is a code together with the constructs that confine it. Probes
(`/tmp/laneB/sv`, reproduced as tests in `test_rtl.py`) showed that a plain
`TOLERATED_DIAGNOSTICS` entry would be wrong for both codes. Elsewhere, each
of them *does* reach a module-level value:

| Code | Tolerated inside | Why it cannot reach a port or a module parameter there | The same code elsewhere (declined, tested) |
|---|---|---|---|
| `ConstEvalFunctionInsideGenerate` (FinnLib `add_multi.sv:45`: a localparam of a generate block built by a function declared in that block; LRM 13.4.3; Vivado accepts it) | `IfGenerate`, `LoopGenerate`, `CaseGenerate` | Ports and module parameters are never declared inside a generate block. A constant inside one reaches module level only through a hierarchical name, which slang refuses in a constant expression (`ConstEvalHierarchicalName`, not tolerated). A generate condition only chooses which blocks exist | `localparam Q = g.f();` at module level raises only this code (probe), and slang leaves `Q` `<unset>`. It is outside any generate construct, so it is declined |
| `UsedBeforeDeclared` (FinnLib `inner_shuffle.sv:294` and `:314`: nets read in continuous assignments above their declaration; `xelab -relax` accepts it) | `ContinuousAssign`, `AlwaysBlock`, `AlwaysCombBlock`, `AlwaysFFBlock`, `AlwaysLatchBlock`, `InitialBlock`, `FinalBlock` | These read nets and variables. No constant expression can depend on those, so no port width or parameter value can either (a constant declared inside a procedural block is local to it). Function bodies are excluded because a function may be a constant function | `localparam A = B; localparam B = 5;` raises the same code, and slang leaves `A` `<unset>`: declined, while the tolerated `assign` in the same module is not reported |

How it is checked: the tolerance applies only when the diagnostic's location
(buffer and offset) falls inside the source range of one of the named
constructs, found by walking the syntax trees
(`SyntaxNode.visit(lookup_table=...)`). A location in another buffer (a macro
expansion, say) is inside no construct, so it declines. The trees are walked
only when a candidate code is among the errors. Tests show both FinnLib
modules report exactly these diagnostics (`_diagnosed`: `add_multi.sv:45`;
`inner_shuffle.sv:294, 314`) and bind only because of this rule: with
`TOLERATED_WITHIN` emptied, both decline (checked by hand).

## The decline table, re-measured

Every `test_the_kernel_conforms[*]` case is a Python run with `-W always`.
There are 34 samples: `thresholding-lanes-in-order` (2 samples) came after
A1's count of 32.

| Kernel (case) | Samples | Before (`a361697be`) | After commit 1 | After commit 2 |
|---|---|---|---|---|
| dotp INT8 (dense, depthwise) | 8 | checked | checked | checked |
| memstream | 4 | checked | checked | checked |
| dotp packed | 4 | 1 checked; interior, largest, adapter declined (`add_multi.sv:45`) | same | **checked** |
| thresholding | 4 | declined (`THRESHOLDS`, array) | **checked** | checked |
| thresholding-rows-first | 4 | declined (`THRESHOLDS`) | **checked** | checked |
| thresholding-lanes-in-order | 2 | declined (`THRESHOLDS`) | **checked** | checked |
| eltwise | 4 | declined (`B_SCALE`, real) | **checked** | checked |
| transpose | 4 | declined, all 4, SIMD 1 included (`inner_shuffle.sv:294`) | same | **checked** |
| **bound** | **34** | **13** | **27** | **34** |

The wrong-order Python tests (`ChannelsFirst` 4 samples, `LanesReversed` 2)
went from 6 declined to 0 after commit 1.

Unestablished values, observed per module after commit 2 (never supplied):
`thresholding_axi`: `THRESHOLDS`; `eltwise`: `B_SCALE`; `inner_shuffle`:
localparams `RD_INIT_PAT`, `REV_RD_INIT_PAT`, `RD_PERM_PAT`,
`REV_RD_PERM_PAT`, `RD_ADDR_INIT` (unpacked arrays). None for `dotp_axi`,
`memstream_axi`.

### `--strict-rtl`

Python tests in `tests/kernels/test_conformance.py`, run with `--strict-rtl`:

| At | Result | Failing |
|---|---|---|
| `a361697be` (before) | 8 failed, 10 passed, 11 skipped | `test_the_kernel_conforms[dotp-packed, eltwise, thresholding, thresholding-lanes-in-order, thresholding-rows-first, transpose]`, `test_a_wrong_order_passes_every_python_check[lane-order, loop-order]` (A1 recorded six, before `lanes-in-order` and the lane-order proof existed) |
| `ecaad8080` | 2 failed, 18 passed, 11 skipped | `test_the_kernel_conforms[dotp-packed]` (`add_multi.sv:45`), `test_the_kernel_conforms[transpose]` (`inner_shuffle.sv:294`) |
| `36036a62f` | **20 passed, 11 skipped** | none |

Without `--strict-rtl`, `-W always` at `36036a62f`: 20 passed, 11 skipped,
0 `RtlDeclined` warnings (27 at `a361697be`, 7 at `ecaad8080`).

## Parameter-name mismatches

None. After commit 2, the name check ran for all 34 samples, plus the 6
wrong-order samples, and every kernel's `parameters()` keys equal the module's
declared parameter names: `dotp_axi` 13, `thresholding_axi` 15, `eltwise` 10,
`inner_shuffle` 5, `memstream_axi` 6. I observed this by instrumenting
`_check_model` (`/tmp/laneB/instrument.py`), not only through tests passing.

The name check is shown to bind by the tests themselves:
`test_parameters_must_name_every_module_parameter` is now parametrized over
memstream without `RAM_STYLE` (as before), eltwise without `B_SCALE` (the
real-valued parameter itself), and thresholding without `DEEP_PIPELINE`.
Before this lane, the last two would have declined, warned and passed.

**Observation (not a mismatch).** Omitting `THRESHOLDS` itself cannot show
the check, because slang refuses `thresholding_axi`'s *default* for it
(`thresholding_axi.sv:29`, `'{default: '{default: '{default: '0}}}`:
"invalid target type 'logic' for assignment pattern"). A binding without
`THRESHOLDS` therefore declines. Every kernel binding supplies `THRESHOLDS`,
so no conformance sample meets it. Recorded, not handled.

## Gates, as observed

Vivado off `PATH`, `FORCE_COLOR` unset, kernel venv. Baseline at `a361697be`:
Space 454, kernels 809 + 25 skipped, graph 4 + 2, dataflow 40.

| Gate | `ecaad8080` | `36036a62f` |
|---|---|---|
| `check-space.sh` | 454 passed; ruff, mypy clean | 454 passed; ruff, mypy clean |
| `tests/kernels` | 817 passed, 25 skipped, 7 warnings (the 7 `RtlDeclined`: packed dotp x 3, transpose x 4) | 822 passed, 25 skipped, no warnings |
| `tests/graph` | 4 passed, 2 skipped | 4 passed, 2 skipped |
| ruff format, ruff check, mypy (`finn.kernels`, `finn.graph`, typed tests) | clean | clean |
| `check-dataflow-design.sh` | 40 passed; ruff, mypy clean | 40 passed; ruff, mypy clean |

New tests: `ecaad8080` +8 (`test_rtl.py` +6: array, real, every kind of
value that is neither an integer nor a string, pins agree, wrong pin refused
next to an unestablished value (width, case, eltwise width), unknown override
still declines; `test_conformance.py` +2 from parametrizing the name check).
`36036a62f` +5 (`test_rtl.py`: construct names are real `SyntaxKind`s, packed
dotp and inner_shuffle bind with exactly their diagnostics, generate-function
tolerance confined, use-before-declare tolerance confined).

**XSim** (optional for this lane): the change is Python-side. Two cases ran
from a `git archive` snapshot of `36036a62f` (`/tmp/laneB-xsim-36036a62f`,
its own `deps/finnlib` and `deps/qonnx`), with Vivado 2025.2 on `PATH`:
`test_the_kernel_conforms_in_xsim[thresholding]` PASSED (223.7 s), and
`test_the_kernel_conforms_in_xsim[transpose]` PASSED (222.0 s). transpose's
strict known failures (`TRANSPOSE_BURSTY`) still settle exactly: `_check_rtl`
now binds `inner_shuffle` ahead of each simulation and changes nothing in it.
The other XSim cases were not run.

## Deviations

- The harness extracts once, then compares the ports with
  `check_against_rtl`, instead of calling `check_abi` and then `extract` (two
  elaborations of the same closure). The comparison is the one `check_abi`
  makes. `check_abi` keeps its API and tests.
- Type parameters are now reported by name (value `None`). Before, they were
  skipped, so a binding for one would have been refused as undeclared. No
  FinnLib top in the conformance cases has one; a synthetic test covers it.
- The task allowed "leave them as declines" for the two slang elaboration
  failures. They are handled instead, by the location-scoped rule above (its
  own commit, so it can be dropped on its own).

## Remaining declines

None among the conformance samples. The checker still declines, by design:
an unbound parameter without a default; a binding naming an undeclared
parameter; interface ports; unresolved port widths; any error not tolerated,
including the two codes above anywhere outside their constructs; and
`thresholding_axi` bound without `THRESHOLDS` (slang refuses its default).
