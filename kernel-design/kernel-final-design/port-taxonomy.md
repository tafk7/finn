# Port taxonomy — the vocabulary the composition stitch binds against

*The durable spec for `space/ports.py` + `space/stitch.py` (built + venv-validated
2026-07-18). This is the model that makes the composition stitch — FINN's ~270-line
imperative `code_generation_ipi` Tcl god-method — a trivial, op-agnostic resolver.
Companion to `param-delivery-design-space.md` (the three parameter coordinates) and
`generality-gaps.md` (this attacks gap G1, the port model).*

## The one idea

A composed kernel is a **GRAPH of role-tagged hardware blocks**; composition = wiring
ports by role. An emit no longer just produces files — it **declares the ports** its
block exposes, and an op-agnostic resolver wires complementary roles into nets.

The move that makes the resolver trivial: **separate the PHYSICAL protocol (`kind`)
from the SEMANTIC binding key (`role`).**

- Brainsmith's port model is **kind-only** (three interface categories: AXI-Stream /
  AXI-Lite / Control). It cannot tell an activation stream from a weight stream — both
  are `kind=AXIS`.
- The finn-proto prototype keyed by **protocol** (`s_axis`/`m_axis`/`aximm`/`axilite`/
  `ap_none`), so activations and weights collapsed into `s_axis` together, recovered
  only by a hardcoded `in1_V` string match. `ap_none` was an untyped junk-drawer for
  every non-stream signal (FINNLoop's `done_if` lived there).
- **Keying by role is the fix.** A weight stream and a data stream are the same `kind`
  (AXIS) but different `role` (WEIGHT_SINK vs DATA_IN); the resolver binds on role.

## The type

`Port = (direction, kind, role, pin, index, shape|width, boundary)` — a frozen
dataclass. The resolver reads ONLY `(direction, kind, role, index, width)` to DECIDE a
binding; `pin` appears only inside emitted command strings, never in a conditional.

### direction
`IN | OUT`, from the block's own perspective.

### kind — the physical protocol (how the wires talk)
`AXIS | AXIMM | AXILITE | CLOCK | RESET | SIGNAL`.
- `AXIS` — AXI4-Stream (TDATA/TVALID/TREADY): the dataflow + weight fabric.
- `AXIMM` — AXI4 memory-mapped **master** (address/burst to a memory controller). This
  is what makes `fetch_weights` special — it has an AXIMM port + an address plan — NOT
  a "chip location". (See "Why OFFCHIP is dropped".)
- `AXILITE` — AXI4-Lite slave (control/config registers).
- `CLOCK` / `RESET` — clock and reset pins (broadcast, not pairwise-bound).
- `SIGNAL` — a bare sideband wire (the prototype's `ap_none`, now typed).

### role — the semantic binding key (what the port is FOR)
The vocabulary the resolver binds against, and the thing topology can REASSIGN.

**BUILT (exercised by the shipped MVAU + memstream emits):**
`DATA_IN`, `DATA_OUT`, `WEIGHT_SINK`, `WEIGHT_SOURCE`, `CONFIG`, `CLOCK`, `RESET`.

**DOCUMENTED / INERT (the full target vocabulary — a single enum entry each, so adding
one later is trivial, not resolver surgery):**
`INDEX_SINK` / `INDEX_SOURCE` (MLO set-selection stream), `MEMORY_MASTER` (an AXIMM to
off-chip storage), `STATUS` (a sideband status bit, e.g. FINNLoop `done`).

### index
Disambiguates multiple ports of the same role: weights = 0, thresholds = 1; `DATA_OUT`
0/1/… for a multi-output op. **This is what answers the cardinality question
structurally** — "one weight stream" is not baked into the resolver; N weight streams
are just N indices.

### shape-by-role
Only DATA/WEIGHT roles carry a folded tensor `shape`; every other role carries a bit
`width` or nothing (`shape=None`). Enforced in `Port.__post_init__`. This makes iodma's
"folded-shape getter RAISES on the AXI-MM side" a **category error the type forbids**,
not a runtime bug to handle. (In the built MVAU emits the DATA/WEIGHT ports currently
carry `width` only — the point exposes stream widths, not folded shapes, as derived
fields; shape is the documented target the type already admits.)

### boundary
`True` = this port **exports to the enclosing region** (a dataflow-graph edge or a
top-level pin) rather than binding to a sibling inside the composed kernel. **This is
load-bearing in the resolver, not decorative:** only *non-boundary* ports are eligible
for pairwise binding. Without it, a compute⊗compute chain (each cell has a DATA_IN and
DATA_OUT at index 0) would see 2 sources / 2 sinks and bind ambiguously; `boundary`
distinguishes the region's external edges from the one internal A.out→B.in edge.

## The three clarifications that sharpened this

1. **DATA vs WEIGHT is a ROLE (provenance/ownership), not a protocol.** At the wire
   level both are identical AXI-Streams (the prototype wires the weight net with a plain
   `WSTRM_TDATA/TVALID/TREADY` triple, `matrixvectoractivation_sip.py:197-229`). The
   difference is semantic: DATA connects kernel-to-kernel along the dataflow graph (an
   external region edge); WEIGHT is a param stream a kernel OWNS internally via a
   delivery sub-kernel. The role tells the resolver WHICH binding it is. **It can FLIP
   with topology**: external `mem_mode` makes `in1_V` a DATA_IN boundary port, not a
   WEIGHT_SINK (the prototype does exactly this at `sip:332-334`).

2. **"OFFCHIP" is dropped as a role** — a misleading location word for a soft core in
   programmable logic. The real distinction is `kind`: `AXIMM` (a memory-mapped master
   issuing address/burst transactions) vs `AXIS` (in-fabric streaming). DDR-vs-HBM is
   irrelevant to the kernel model. "Within the kernel scope, offchip just means a port
   that leaves the region" — which is what `boundary` + `kind=AXIMM` already say.

3. **The prototype had a real-but-shallow port model to learn from.**
   `get_verilog_top_module_intf_names` (`finn-proto/.../kernels/kernel.py:171-196`)
   returns a dict keyed by protocol, AXIS ports as `(name, width)` indexed by node
   input/output order, width-0-means-absent. Good bones (arity + width-0-absent — KEEP
   these conventions); two flaws we fix (keyed by protocol not role; `ap_none`
   junk-drawer → typed SIGNAL/STATUS).

## The resolver (`space/stitch.py`) — trivial by design

`stitch(cells, region_name) -> IPICommands` where a `Cell = (instance, module, ports)`:
1. `create_bd_cell` per cell.
2. For each complementary role pair in `STANDARD_BINDINGS`
   (`WEIGHT_SOURCE↔WEIGHT_SINK`, `DATA_OUT↔DATA_IN`, `INDEX_SOURCE↔INDEX_SINK`), bind a
   SOURCE output to the SINK input sharing its `index`, across cells, requiring
   compatible `kind` + equal `width`. Only **non-boundary** ports are eligible.
   Ambiguity (>1 source or sink per index) is a hard error, not a silent pick.
3. Broadcast CLOCK/RESET to every cell.
4. Export every still-unbound port up as a region boundary pin
   (`make_bd_intf_pins_external` for interface kinds, `make_bd_pins_external` for a bare
   signal). **An embedded compute's unbound WEIGHT_SINK exports here** — the "no
   delivery sibling" case, handled structurally with no branch.

**The op-agnostic guard (a test asserts it):** the resolver source contains no concrete
pin-name literal (`in1_V`, `m_axis_0`), no op type (`mvau`/`MVAU`/`implementation`), no
`mem_mode`. That is what makes it handle N-memory / compute⊗compute / zero-param shapes
for free.

## Why this is the crux, not a close-out

The cardinality of "one compute + one memory" is an **MVAU-shaped assumption**, not a
law. Build the stitch declaratively against `(role, index)` + `boundary` and the "one
of each" special case generalizes to the graph:
- **2-weight-by-index** (weights + thresholds, gap G2): both bind by index — proven by
  a synthetic test.
- **2-compute / 0-memory** (shuffle: RTL-inner→HLS-outer; gap G4): DATA_OUT→DATA_IN
  binds, no memory cell — proven by a synthetic test.
- **0-compute / 1-memory** (standalone streamer, IODMA): a lone WEIGHT_SOURCE exports up.

## Built vs. target

| Piece | Status |
|---|---|
| `Port` type, `Kind`/`Role`/`Direction` enums, `STANDARD_BINDINGS` | **built** (`space/ports.py`) |
| `Artifacts.ports` + `Artifacts.merge` | **built** (`space/artifacts.py`) |
| Ports declared on `emit_mvau_rtl` / `emit_mvau_hls` / `emit_memstream` | **built** |
| `stitch()` resolver + `Cell` | **built** (`space/stitch.py`) |
| `emit_composed()` (dispatch both pools, merge, stitch) | **built** (`fixtures/mvau/compose_emit.py`) |
| Synthetic cardinality proofs (2-weight, 2-compute/0-memory) | **built** (`tests/test_stitch.py`) |
| Elaborate the stitched composed top (Vivado xvlog/xelab) | Phase 3 (Docker) |
| rtlsim the composed MVAU end-to-end (the prize) | Phase 4 (Docker) |
| `INDEX_*` / `MEMORY_MASTER` / `STATUS` roles wired to real emits | future (inert today) |
| DATA/WEIGHT folded `shape` populated (vs width only) | future (type already admits) |

## Lineage

- `kernel-model.md` T3/T4 (RoleBinding, Region-owns-nets, STANDARD_BINDINGS) — designed,
  never built; this builds the Level-1 subset.
- finn-proto `kernel.py:171-196` — arity + width-0-absent conventions, upgraded to
  `(kind, role, index)`.
- brainsmith `kernels.md` — the three interface categories, upgraded from kind-only to
  kind × role.
