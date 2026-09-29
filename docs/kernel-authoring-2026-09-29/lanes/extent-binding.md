# Lane A: extent binding (A3, plan D3)

Wave 1, lane A. Branch `lane/extent-binding`, worktree `finn-ka-extents`, from
`a361697be`; `deps/` fetched with `fetch-repos.sh`. Scope: D3 only, a value-level
function in `finn.dataflow.schedule`. No Space code and no kernel changes;
wiring it into the kernel base is D4 (A4).

## What landed

Commit: `451c4747b` (code and tests); this record is the commit after it.

- **`finn.dataflow.schedule.Access`**, a frozen dataclass:

  ```python
  @dataclass(frozen=True)
  class Access:
      name: str                          # the port, in refusals
      shape: tuple[int, ...]             # the tensor it reads
      index: tuple[Index | Affine, ...]  # one expression per axis (per view axis if reshaped)
      reshaped: bool = False             # reads a row-major view of shape
  ```

  It normalizes `shape` and `index` to tuples and checks them (a nonempty name,
  positive extents, `Index`/`Affine` entries, a bool). A malformed value raises
  `ValueError`/`TypeError`; everything the binding refuses raises `Refused`.

- **`bind_extents(accesses, extents=None) -> dict[Index, int]`**, raises
  `Refused`. `extents` are the author's (`bound_schedule(extents=...)`).
  1. The given extents are checked (positive integers) and bound first.
  2. Structure, for every access: a reshaped access reads plain indices; any
     other has one expression per tensor axis (`x: 3 indices for a rank-2 tensor`).
  3. Binding: each axis of a non-reshaped access addressed by a plain index
     binds it. A plain index is an `Index`, or an `Affine` of one index with
     coefficient one (the module's own definition). An index seen with two
     extents is refused, naming both places: `k is 6 (x axis 1) and 4 (w axis 0)`,
     or `k is 4 (given) and 8 (x axis 1)`.
  4. Checking: every index any access reads has an extent
     (`x axis 0: kh has no extent (no axis addresses it alone and none is given)`).
     A reshaped access's view (its indices' extents) has its tensor's size
     (`x: a (2, 3, 5) tensor cannot be viewed as (2, 12)`, the message
     `Schedule.present` uses). Any other non-plain axis stays inside its axis:
     `reach = sum(c * (extent(i) - 1)) < extent`
     (`x axis 0: kh*2 + oh*2 reaches 6, beyond extent 6`).
  5. It returns the given extents plus the bound ones.

- Both are exported in `__all__`. The module docstring of `schedule.py` gains a
  paragraph on binding and coverage; the `finn.dataflow` package docstring's
  `schedule` bullet names the function.

- **`tests/dataflow/test_extents.py`**, 21 tests:
  - the rules: plain, an index shared across axes and ports, rank mismatch,
    an unbound index, affine, given extents, view, the value's own checks;
  - coverage and the too-wide probe;
  - the S0 roster (9 members, parametrized) and two roster limits.

Mutation check (scratch, reverted): disabling the disagreement check fails 3
tests, the reach check 2, the view size check 1, and letting a view bind 2.

## Gates (Vivado off `PATH`, `FORCE_COLOR` unset), as observed on `451c4747b`'s tree

| Gate | Result |
|---|---|
| `check-dataflow-design.sh` | 61 passed (baseline 40, +21); ruff format, ruff check, mypy (`finn.dataflow`, `tests/dataflow`) clean |
| `check-kernels.sh` | Space 454 passed, ruff and mypy clean; `tests/kernels` 809 passed, 25 skipped; `tests/graph` 4 passed, 2 skipped; ruff format, ruff check and mypy clean. Identical to the `a361697be` baseline. The run prints `RtlDeclined` warnings (`dotp_axi` via `add_multi.sv`, `inner_shuffle.sv`). They are warnings, not failures, from code this lane does not touch. |

## The S0 roster

The roster is `docs/stream-model-2026-09-27/sketches/s0_roster.py` with
`indexed_ports.py` beside it (not under the scratchpad's `open/`, where the lane
prompt placed it). It writes its nests over split levels (`nf`, `p`). Here each
member is written over whole indices with folds, as `tests/dataflow/test_schedule.py`
does. For each member the test binds the accesses and compares the result with
the expected extents. It then builds a `Schedule` over the bound extents,
presents every port, and checks whether the traversal covers its tensor.

| Member | Accesses | Given | Binds | Covers |
|---|---|---|---|---|
| dense MatMul | x `(m, k)`, w `(k, n)`, y `(m, n)` | — | yes | all |
| per-channel MatMul | x `(m, k, n)`, w `(k, n)`, y `(m, n)` | — | yes | all |
| per-channel as a dense view | x `(M, K, N)` read reshaped as `(m, k)`, w `(K·N, N)`, y | — | yes (x binds nothing; checked) | all |
| tiled MVU | x `(mt·T + t, k)`, w `(k, n)`, y `(mt·T + t, n)` | `mt`, `t` | **only with both given** | all (with exact given extents) |
| SWG | x `(oh·S + kh·D, ow·S + kw·D, c)`, y `(oh, ow, kh, kw, c)` | — | yes: y binds all, x is checked | y; **x not** (see below) |
| SWG, FINN layout | x `(b, window…)`, y `(b, oh, ow, kh·KW·C + kw·C + c)` of shape `(1, OH, OW, KH·KW·C)` | `kh`, `kw` | yes | y; x not |
| thresholding | x `(r, c)`, y `(r, c)`, plus its `(sets, channels, thresholds)` table as `(g, c, j)` (bound, not scheduled) | — | yes; the table's channels are checked against the streams' | all |
| eltwise, broadcast operands | lhs `(r, c)`, rhs `(c,)`, `(1, C)` as `(0, c)`, `(R, 1)` as `(r, 0)`, result `(r, c)` | — | yes | all |
| transpose | x `(i, j)`, y `(j, i)` | — | yes | all |

Findings:

- **Every roster member is expressible and binds.** Two of them need extents
  given.
- **SWG image: checked, rightly not covered.** With stride 2 and dilation 2 on
  7 rows, the windows read rows {0, 2, 4, 6}. The check is `reach < extent`,
  not coverage; the test pins it (`test_the_roster_s_windows_are_checked_not_covered`).
  D3's coverage statement holds for bound axes only. That is what it says, but
  the "too-wide tensor settles silently" finding is closed only for axes a
  plain index addresses.
- **Tiled MVU: binds only with `mt` and `t` both given, and `mt` needs `M`.**
  No axis addresses `mt` or `t` alone. `t = T` is a choice, but `mt = M / T`
  must be computed from a tensor's shape: the extent getter D3 is meant to
  remove. It also leaves a hole. Given `{mt: 1, t: 3}` on 6 rows, the reach
  check passes (2 < 6) and x covers half its rows
  (`test_the_tiled_mvu_binds_only_with_its_tiles_given`). See open question 1.
- **FINN's SWG layout needs `kh`, `kw` given.** They are the kernel size, a fact
  of the kernel. With them, y's flattened last axis is an affine checked
  exactly (reach `KH·KW·C − 1`), and `b`, `oh`, `ow`, `c` bind from plain axes.
  Written instead as a reshaped y, it would bind nothing and need every window
  index given; the affine spelling is the better one.
- **A broadcast axis of extent one** is the empty expression `Affine(())`
  (`0 * r` builds the same). It binds nothing and passes the check (reach 0 < 1).
  No new value is needed for numpy-style broadcast.
- **Thresholding PE > C** (rows folded into lanes) would be `r = rf·(PE/C) + rs`,
  an affine that binds nothing. It is out of scope by G0.4a and not in the table.
- **Cross-check on the real dotp** (scratch script, not committed). The
  `PackedDotpKernel`'s own `x`, `w`, `y` declarations (`form=DENSE`,
  `reshape_activations=True`, x `(2, 3, 4)`, w `(12, 4)`, y `(2, 4)`) give
  `{k: 12, n: 4, m: 2}`. That equals its `rows`, `outputs`, `reduction` getters
  `(2, 4, 12)`. Unreshaped x `(2, 12)` gives the same. x `(2, 3, 5)` is refused:
  `x: a (2, 3, 5) tensor cannot be viewed as (2, 12)`.

## The `Access` shape chosen

P0's frozen dataclass `Access(name, shape, index, reshaped=False)`, unchanged
in fields; it now validates itself.

**Deviation from the plan's sketch** (`Access = tuple[Sequence[int], Sequence[Index | Affine]]`).
- The refusals must name the port, and the view case needs `reshaped`; a
  2-tuple carries neither.
- `default_semantics(Access)` works as a view value (checked: recognizes,
  equal to its snapshot, equal to its deepcopy), so D4's `ACCESS` key can be
  `ViewKey("access", default_semantics(Access))`.

## Deviations from D3

1. **`Access` is a named dataclass**, not the 2-tuple (above).
2. **The explicit extents are the second argument of `bind_extents`**
   (`extents=`), bound first and checked like an axis (`k is 4 (given) and 8 (x axis 1)`).
   D3's signature has one argument, though its bullets name the explicit extents.
3. **"Several indices" is read as "not a plain index"**: a single index with a
   coefficient other than one (`oh * 2`, a subsampling read) and the empty
   expression also bind nothing and are checked by reach. Binding from `oh * 2`
   would have to pick between ceil and floor; checking is exact.
4. **A reshaped access binds nothing and is checked** (P0 correction 2, as the
   plan already reads). It must read plain indices; a non-plain index is refused
   by the binding, as `ScheduledPort` refuses it today. One difference:
   `ScheduledPort` requires an `Index` value, while the binding also accepts
   a coefficient-one `Affine`.
5. **A given extent that is not a positive integer is a `Refused`, not a
   `ValueError`.** The kernel base then turns it into `kernel-extents`, rather
   than crashing on an extent derived from a Param.
6. **An index is "no axis binds it" only if some access reads it.** An index in
   a schedule's `beats` that no access reads and nothing gives is not seen by
   `bind_extents`. `bound_schedule` must refuse it (below).

## For D4 (the port and base step)

- **Message names.** `Access.name` is the port's name in refusals. P0 used the
  kernel member name (`x`), which `Members(ACCESS)` gives as `Located.node`. A
  port exporting its own `Access` knows only its bus name (`s_axis_input`).
  Either name the access from `Located.node` in the base
  (`dataclasses.replace(located.value, name=located.node)`) or accept bus names
  in messages. Choose one.
- **`bound_schedule` must restrict to its beats.** `Schedule(extents, beats=...)`
  requires the beats to order each extent's index exactly once. The binding can
  return indices the schedule does not iterate: a table bound beside the
  streams (thresholding's `j`), or given extents. So `bound_schedule` should
  take `{i: bound[i] for i in beats}`. It should refuse a beat with no extent as
  `kernel-extents`, not let `Schedule` raise `ValueError`.
- **Rank-generic kernels** (eltwise, thresholding, transpose read `(…, C)`). An
  access's `index` must match its tensor's rank, so their leading indices
  (`a0..`) must be generated from the stream's rank. Flattening the leading
  axes through a reshaped view instead would bind nothing: every leading index
  would need another port or a given extent. The `index` Param then depends on
  the stream's shape, and the `ACCESS` export still reads Params and the
  stream's tensor only, so no cycle.
- **An idle port exports no access**, as planned. With no access, an index is
  simply unread by the binding. It is refused only if some other access reads
  it (see deviation 6 for the schedule side).
- **Coverage is structural only for plain axes and views.** D1's conformance
  coverage check stays meaningful for windowed and tiled ports.

## Open questions

1. **Tiles.** Should an affine axis with exactly one index nothing else binds
   bind it, when the axis admits exactly one extent?
   - Example: `mt·T + t` with `t = T` given, over `M` rows, gives
     `mt = (M − 1 − reach_rest) / T + 1`, refused unless it divides exactly.
   - Gain: it removes the tiled MVU's getter and closes its coverage hole.
   - Risk: SWG must not be affected. Its `oh` is always bound from the output,
     so the rule never fires there.
   - It is an extension of D3, not built here.
2. **Message naming** (D4 above): member name or bus name.
3. **The table access.** Should thresholding's table take part in binding in
   A5, as a `(g, c, j)` access?
   - It would replace the `shape[-1] != channels` half of the
     `threshold-stream-form` check (`thresholding.py:245`).
   - The table is a Param, not a port, and the set port keeps `sequence=`. So
     the access would come from the kernel itself (an extra access beside
     `Members(ACCESS)`), not from a port's export.
