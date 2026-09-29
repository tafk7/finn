# P0 report: five probes before A1

Date: 2026-09-29. Base `032312b36` (plan committed). `src/` is unchanged: probe 1's
engine rule was applied only for the gate runs and reverted; it is kept here as
[`p1_output_semantics.patch`](p1_output_semantics.patch).

| # | Question | Answer |
|---|---|---|
| 1 | `T \| Rejected` infers `T`'s semantics | **Yes**: 9 added lines in `_signatures.py`; every gate is unchanged with it applied. 88 of the 115 `semantics=` can go, but D2 is what frees only 38 of them. The other 50 can go today |
| 2 | A kernel's `schedule` reads extents bound from its own ports; a disagreement is a `Rejected` | **Yes**, `kernel-extents`. The view case deviates from D3: a view binds nothing and is only checked |
| 3 | `pe = Decision(domain=divisors_of(extent_of(c)))` | **Inline: no** (refused when the model is linked). **Named member: yes** (`channels = extent_of(c)`): the domain enumerates, commit works, and a parent can pin or narrow it by key |
| 4 | A producer reports its element with its output stream absent; a mismatched stream refuses it | **Yes**, with the D5 port (element = `dtype`, placed or idle). The mismatch is refused twice (`stream-tensor` and `stream-element`) |
| 5 | Flat commit of a fold whose domain reads an absent stream's extent | **Engine: no.** **The plan's fallback works**: the RTL's own bound while unplaced, the divisors once placed |

## How to run

```
cd docs/kernel-authoring-2026-09-29/p0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../../src:../../../tests \
    /home/tkeller/prj-kernels/.kernel-venv/bin/python -m pytest -v -p no:cacheprovider .
# 29 passed (probes.out)
cd ../../..
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests <venv python> docs/kernel-authoring-2026-09-29/p0/count_semantics.py
bash docs/kernel-authoring-2026-09-29/p0/run_gates.sh <label>        # -> p0/gates-<label>.log
```

Files:

- `_bind.py`: a probe copy of D3's `bind_extents` and of D4's base helpers (`extents`, `extent_of`, and port enumeration from the class).
- `_pool.py`: the accpool probe kernel.
- `test_p1…test_p5`: one file per probe.
- `count_semantics.py` / `.out`: the count for probe 1.
- `run_gates.sh`: every gate step, run without `set -e`.

## P0.1: the engine rule (D2)

**Question.** Can `output_semantics` infer `default_semantics(T)` for a union of one value
type and the engine's result markers (`Rejected`, `Inapplicable`, `Unresolved`), as
`_answer_value_type` does for `QueryResult[T]`? Explicit `semantics=` must still override
and still be checked. Protocols and genuine unions must still need it.

**Result: yes.** The change (`p1_output_semantics.patch`) adds a helper, `_marked_value_type`.
It strips the markers from the return annotation after the `QueryResult` branch, so the
rest of `output_semantics` sees `T`: it infers the default, checks an explicit override
against `T`, and still refuses Protocols and multi-value unions.

**Evidence.** `test_p1_semantics_rule.py` installs the same rewrite as a wrapper around
the base function. It passes on the base `src/` (8 passed). With the patch in `src/`,
the two tests about the base rule skip themselves.

| Case | Base | D2 |
|---|---|---|
| `-> BeatSequence \| Rejected`, `-> Tensor \| Inapplicable \| Rejected`, `-> tuple[Index, ...] \| Rejected`, no `semantics=` | `DefinitionError: …needs explicit semantics=` | infers `BeatSequence` / `Tensor` / `tuple`; the value evaluates; the odd case returns `Rejected(code="odd")` |
| `@derived(semantics=BEAT_SEQUENCE) -> BeatSequence \| Rejected` | `BEAT_SEQUENCE` | `BEAT_SEQUENCE` (it still overrides) |
| `@derived(semantics=TENSOR) -> BeatSequence \| Rejected` | **accepted silently**: a union skips the check | `DefinitionError: …BeatSequence is incompatible with Tensor semantics` |
| `-> QONNXDataType \| Rejected`, no `semantics=` | refused | `Protocol outputs require explicit semantics=`; with it, accepted |
| `-> Integer \| None`, `-> int \| str \| Rejected` | refused | `…needs explicit semantics=` |
| `@constraint -> bool \| Rejected` | `bool` | `bool` (unchanged) |

**Gates with the rule applied** (`gates-d2.log`) against the base (`gates-base.log`):

| Step | D2 | Base |
|---|---|---|
| `check-space.sh` (pytest, ruff, mypy) | 448 passed, exit 0 | 448 passed, exit 0 |
| `tests/kernels` | 46 failed, 744 passed, 15 skipped | 46 failed, 744 passed, 15 skipped |
| `tests/graph` | 2 failed, 4 passed | 2 failed, 4 passed |
| ruff format, ruff check, mypy `finn.kernels` + `finn.graph` | exit 0 | exit 0 |
| `check-dataflow-design.sh` | 40 passed, exit 0 | 40 passed, exit 0 |

Every failure is a missing `deps/` file: 23 + 15 `FileNotFoundError …/deps/finnlib/…`,
14 `BuildError: copied source …/deps/finnlib/… cannot be read`,
7 `XSIM 43-4316 Can not find file: …/deps/finnlib/…`, and 1 missing `deps/qonnx/src`.
The two runs' sorted `FAILED` lists (48 tests) are **identical** (`diff` empty). The `.log` files are gitignored, so `gates-summary.txt` keeps both summaries and the failed list.

The gate collects every family in `finn.kernels` and `finn.dataflow` under the stricter
check. So no existing explicit `semantics=` on a marked union contradicts its annotation.

**Count** (`count_semantics.out`). There are 115 `semantics=` in `finn.kernels` and
`finn.dataflow`. 113 are on members (derived, view, Param, Decision). The other two are
QONNX: `int_to_fp32.py:40` (a `Const`) and `datatypes/domains.py:134` (a `domain(...)`).

| Bucket | Sites | What |
|---|---|---|
| redundant today | 40 | Plain `T` annotation with default-equivalent semantics; no D2 needed. `CLOCKING` 8, `INDICES` 7, `STAGES` 4, `TENSOR` 3, `SCHEDULE` 2, … |
| redundant with D2 | 31 | `T \| Rejected` with default-equivalent semantics. `TENSOR` 4, `SCALAR_ENCODING` 3, `TRANSPORT` 2, `INPUT_GEN_FACTS` 2, `VPC_FACTS` 2, `PLAN`, `AXI_STREAM`, `STREAM_CONTRACT`, … and inline `default_semantics(int/float/str/…)` |
| differs only in name/snapshot | 17 | `BEAT_SEQUENCE` 14, `TRAVERSAL` 3: removable per D2 once A2 confirms the identity dump. 7 are on `T \| Rejected` (need D2) and 10 on plain `T` |
| stays | 27 | QONNX 19 (Protocol), `INTEGER_VECTOR` 3, `INTEGER_TENSOR` 3, `THRESHOLD_TABLE` 1 (custom recognition), `INTEGER_POLICY` 1 (`Integer \| None`) |

**Deviation from the plan.** D2 says the explicit uses the annotation "now implies" are
"most of the 115 (`BEAT_SEQUENCE`, `TENSOR`, `INDICES`, `CLOCKING`, `SCHEDULE`, `PLAN`,
`TRAVERSAL`, ...)". The total is right: 88 of 115 are removable. The attribution is not:

- **50** are removable on today's engine: 40 plain, plus the 10 plain `BEAT_SEQUENCE`/`TRAVERSAL` (those 10 still wait on A2's identity-dump check).
  These include every `CLOCKING`, `INDICES` and `SCHEDULE`.
- **38** need D2: the 31, plus the 7 marked `BEAT_SEQUENCE` uses.

D2 also closes a gap the plan does not mention. Today an explicit `semantics=` on a union
return is never checked against the annotation.

**Consequence for A2.**

- The rule is the 9-line patch here.
- Removal is 88 edits.
- Constants that also key a `ViewKey` stay as constants; only their `semantics=` uses go:
  `TIEOFFS_SEMANTICS`, `CONTROL_SEMANTICS`, `CONNECTION_SEMANTICS`, `PARTS_SEMANTICS`,
  `MODULE_REQUIREMENTS`, `EXPORTED_SEMANTICS`.
- The `QueryResult[T]` branch still requires `semantics=`. No site in the two packages
  uses it, so it is out of scope.

## P0.2: extent binding read by the kernel's schedule (D3, D4)

**Question.** Can a kernel's `schedule` read extents bound from its own ports' static
`index` and its streams' tensor shapes? The binding must turn a disagreement into a
`reject(...)`, not an exception. It must also handle the view case and an axis addressed
by an `Affine`.

**Result: yes.** One deviation for views, below.

**Evidence** (`test_p2_extent_binding.py`, 7 passed):

- **Plain binding.**
  - The ports are enumerated from the class: `node_record` over the MRO's members whose
    family is a `ScheduledPort`, giving `('x', 'y')`. The graph is never read.
  - `extents` reads each placed port's `index`, `reshaped` and `stream.tensor.shape`, and
    the ports' `sequence` reads the schedule, which reads `extents`. There is no cycle.
    `(1, 4, 8)`/`(1, 8)` gives `{b: 1, s: 4, c: 8}`.
  - With `pe = 4`, x presents 8 beats × 4 lanes and y 2 × 4.
- **Disagreement.** x `(1, 4, 6)` against y `(1, 8)`:
  - `query(extents)` gives `Rejected(code="kernel-extents", message="c is 6 (x axis 2) and 8 (y axis 1)")`.
  - The same finding reaches `channels` and `build_requirements`.
  - `commit(..., {"pool.pe": 2})` raises the helper's `ValueError: … pool.extents: kernel-extents: c is 6 …`
    (the configure helper's contract for a refused point).
  - Rank mismatch: `x: 3 indices for a rank-2 tensor`, the same code.
- **View case (the real `PackedDotpKernel`).** Setup: `form=DENSE`, `reshape_activations=True`,
  x `(2, 3, 4)`, w `(12, 4)`, y `(2, 4)`.
  - The ports are `x (m, k) reshaped`, `w (k, n)`, `y (m, n)`.
  - `bind_extents` gives `{k: 12, n: 4, m: 2}`, equal to today's `rows`, `outputs` and
    `reduction` getters.
  - x `(2, 3, 5)` is refused: `x: a (2, 3, 5) tensor cannot be viewed as (2, 12)`.
- **Affine axis.**
  - On its own, `(oh * 2 + kh, c)` binds only `c` and is refused: `x: kh has no extent`.
  - Given `{oh: 3, kh: 3}` it binds, and it is checked: reach 6 < 7.
  - A window kernel `X[oh*2 + kh, c] -> Y[oh, kh, c]` binds `oh` and `kh` from its output,
    so x's axis 0 contributes nothing. It presents x `(7, 4)`.
  - With `H = 6` it is refused: `x axis 0: kh + oh*2 reaches 6, beyond extent 6`.

**Deviation (D3, view case).** D3 says a view port "binds from the view's axes and checks
`prod(view) == prod(shape)`". A view has no shape of its own, though: today `view =
tuple(schedule.extent(axis) for axis in index)`, so its extents are its indices' extents.
A reshaped port therefore **binds nothing**. After binding it is checked: its indices must
be bound by other ports, and the sizes must agree. For dotp the other ports bind them
(`m` from y, `k` from w). A reshaped port whose indices no other port binds is refused
(`x: m has no extent`), unless explicit `extents=` are given.

**Consequence.**

- For A3, D3's view bullet should read "binds nothing; checked: every index bound
  elsewhere, same size".
- For A4:
  - `extents` enumerates ports statically, as `_bind.port_names` does.
  - It skips idle ports.
  - It returns `reject("kernel-extents", …)` from the `Refused` of `bind_extents`.
- The window case confirms R3's mitigation: an index only an `Affine` addresses needs
  another port or `extents=`.

## P0.3: the fold domain `divisors_of(extent_of(c))` (D4, G0.4)

**Question.** Does `pe: int = Decision(domain=divisors_of(extent_of(c)))` enumerate and
commit, and can a parent pin it by key? Here `extent_of(c)` is a derived member reading
the derived `extents`.

**Result.**

- **Inline: no.** Linking refuses it:
  `DefinitionError: pool.pe: Derived (declared at _bind.py:…) is not a member of this scope; name it as a class attribute, or supply a formal with it`.
- **The fallback works:** one line, `channels = extent_of(c)`, then
  `pe = Decision(domain=divisors_of(channels))`. That is the spelling the plan's target
  sketch already uses.

**Evidence** (`test_p3_fold_domain.py`, 5 passed). On x `(1, 4, 8)`:

- `field(Pool.pe).candidates()` gives `(1, 2, 4, 8)`, and `undecided` lists `pool.pe`.
- `commit {"pool.pe": 4}` gives schedule folds `((c, 4),)`; 3 is refused (`outside`).
- **Pin by key** in the parent body: `pool.pe = 4`.
  - `inspection.pinned` gives `["pool.pe"]`, and it is absent from `inspection.choices`.
  - The schedule folds by 4.
- **Narrow:** `pool.pe = Decision(values=(2, 4))` gives candidates `(2, 4)`.
- A pin outside the domain (`pool.pe = 3`) gives `Rejected` `domain-membership` owned by
  `pool.pe`.

**Consequence for A4 and A7.**

- `extent_of(index)` is a factory returning a `Derived`, and it must be named in the class
  body. The link-time message names the fix. A7's guide should state it.
- `extent_of` returns `reject("kernel-extents", "<i> is bound by no placed port")` for an
  unbound index, which is what P0.5 meets flat.

## P0.4: a producer's element without its output stream (D5)

**Question.** Does a kernel whose output port states `dtype=<derived from facts and input
elements>` report that element with its output stream absent? Does a stream refuse a
tensor of another element (`stream-tensor`)?

**Result: yes**, with the D5 port. `ProducerPort` subclasses today's `ScheduledPort` and
overrides `element` to `ScalarEncoding.admit(self.dtype)`, placed or idle: 3 lines.

**Evidence** (`test_p4_producer_element.py`, 5 passed). `AccPool.sum_dtype` reads
`x.element` and `pixels = extent_of(s)`.

- **Output unplaced.**
  - `y.idle is True`, and `extents` binds `{b: 1, s: 4, c: 8}` from x alone.
  - `sum_dtype == INT7` (INT4 × 4 pixels spans [−32, 28]), and `y.element == ScalarEncoding(INT7)`.
- **Placed on INT7:** `Stream.connection` is `Available`, and y presents 4 beats.
- **Placed on INT8:** `Stream.connection` is `Rejected` with two findings:
  - `stream-tensor`: "pool.y carries INT7; the stream carries INT8" (`dataflow/stream.py` `well_formed`)
  - `stream-element`: "stream-element: INT7 cannot feed INT8" (`kernels/physical/contract.py:145`)
- **Today's contrast.** A placed plain `ScheduledPort` with the same `dtype=` takes the
  stream's INT8 and connects with no refusal. This is the silent second authority that D5
  removes.
- **Why the rule holds.** A `sum_dtype` that reads its own `y_stream` gives an unplaced
  `y.element` of `Unresolved(input-unsupplied, owner="pool.y_stream")`.

**Consequence for A6 and D1.**

- The D5 element rule is the 3-line override.
- D1's "outputs unplaced, every output element stated" check is expressible: build with the
  outputs unplaced and require `Available` on each output's `element`.
- A6 should decide whether one mismatch deserves two codes. `stream-element` in the
  physical contract may be subsumed by `stream-tensor`.
- R6 does not arise for accpool.

## P0.5: a flat fold whose domain reads an absent stream's extent (G0.4b)

**Question.** Can an eltwise-like kernel with no streams commit a fold Decision whose
domain reads a stream's extent? If not, does the fallback work: the RTL's own bound
unplaced, the divisors placed?

**Result.**

- **The engine cannot commit it.** Enumeration is refused and commit raises the helper's
  `ValueError`.
- **The fallback works.**

**Evidence** (`test_p5_flat_fold.py`, 4 passed):

- **`divisors_of(extent_of(c))`, flat.**
  - `extents == {}`.
  - `candidates()` gives `Rejected(kernel-extents, "c is bound by no placed port")`.
  - `commit {"pe": 2}` raises `ValueError: Elt choices are not accepted: channels: kernel-extents: c is bound by no placed port`.
- **The domain read straight off the absent stream** (`self.lhs_stream.tensor.shape[-1]`):
  - `candidates()` gives `Unresolved(input-unsupplied, owner="lhs_stream")`.
  - Commit raises `…lhs_stream: input-unsupplied…`.
- **Fallback** `fold_domain(c)`: `domain(accepts, candidates, extents=BoundKernel.extents)`.
  It reads the `extents` mapping, which is available and empty when flat, not `extent_of`.
  - Unplaced, `accepts` is `1 <= pe < 2**32`. `candidates()` is an advisory
    `Rejected(fold-unplaced)`, which the engine allows ("enumeration is advisory").
  - `commit {"pe": 3}` works flat: `parameters() == {"PE": 3}`. Both idle ports have
    `lane_count == 3` through `idle_lanes=pe`, which previews G0.3.
  - `pe = 0` and `pe = 2**32` are refused (`domain-membership: candidate is outside the domain`).
  - Placed on `(2, 8)`, the candidates are `(1, 2, 4, 8)`: `pe = 4` presents 4 lanes, and
    `pe = 3` is refused.

**Consequence for A5.**

- Eltwise's fold domain is a domain over `extents`, not `divisors_of(extent_of(…))`.
  Unplaced it enumerates nothing, so flat tests commit the fold explicitly
  (`point_for(..., pe=2)`), as G0.4 already says.
- Kernels whose `parameters()` need extents (dotp) keep `divisors_of(extent_of(…))` and are
  refused flat anyway (`kernel-extents`, G0.3).
- Thresholding keeps divisors of its table's channel count (G0.4a), known flat.
- Whether `fold_domain` becomes a base helper, and under what name, is decided in A4/A5.
  It is a domain, not a new family.

## Deviations and corrections to the plan

1. **D2 attribution** (P0.1). 50 of the 88 removable `semantics=` are already redundant
   today: every `CLOCKING`, `INDICES` and `SCHEDULE`, plus 10 plain
   `BEAT_SEQUENCE`/`TRAVERSAL`. D2 frees 38. D2 also newly checks an explicit override on
   a marked union, which today goes unchecked.
2. **D3 view case** (P0.2). A reshaped port binds nothing. It is checked (indices bound
   elsewhere, same size), instead of "binds from the view's axes".
3. **D4 `extent_of`** (P0.3). It must be a named class member. Inline use inside
   `divisors_of(...)` is refused when the model is linked.
4. **G0.4b fallback** (P0.5). Needed: the engine cannot commit flat. The fallback domain
   reads `extents`, not `extent_of`.
5. **D1 fold sampling.** D1 says sampling reads each fold's domain "through the engine
   (`compatible_cases`)". `compatible_cases` serves Decisions over nodes only
   (`settling.py:46`). Scalar folds enumerate through `point.field(<fold>).candidates()`,
   as used in P0.3 and P0.5. For extent-free kernels built flat that enumeration is
   advisory and empty, so D1 samples those folds placed.
6. **Environment.** The P0 probes need no `deps/`, but the kernel gate does. 46
   `tests/kernels` and 2 `tests/graph` tests read FinnLib or qonnx sources and fail in
   this worktree, identically with and without the D2 rule. `fetch-repos.sh` is needed
   before A1 gates, not only before XSim.

## Review (2026-09-29)

Reproduced independently before acceptance:

- The 29 probe tests pass on the base.
- The patch applied in a scratch worktree:
  - `test_p1` passes against the real patch with its wrapper fixture disabled (6 passed, 2 skipped as designed).
  - `check-space.sh` gives 448 passed, with ruff and mypy clean.
  - `tests/kernels` gives the same 46 `deps/`-only failures as `gates-summary.txt`.
- `count_semantics.py` reproduces `count_semantics.out` exactly.

Accepted, with corrections 1–6 folded into PLAN.md. One change to the probe's approach: `Kernel.extents` in
`src` collects each port's access through an exported view (`ACCESS`, read with `Members(ACCESS)` as the
base reads `Members(PINS)`), not by walking the class with the private `_nodes.node_record`. The patch is
11 inserted lines (9 besides blanks).
