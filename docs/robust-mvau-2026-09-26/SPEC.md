# Task spec: a robust MVAU on the declarative Space model

Date: 2026-09-26. Status: **specified, not started.** The kernel audit that
follows the Space landing will revise the scope and order below; start work
only after that revision.

## 1. Goal

Build an MVAU that covers the design space of FINN's baseline MVAU (HLS and RTL
variants) that FinnLib can realize, as a composite on the landed declarative
Space model. The current MVAU was the stress case that drove the Space
redesign, and it covers only a thin slice:

- one compute core, `dotp_axi`;
- replay through `replay_buffer`;
- external or cyclic weights;
- a direct or FIFO weight stream.

"Robust" means four things:

- **coverage:** the baseline variants that FinnLib can realize are expressible
  as decisions and nodes;
- **reuse:** delivery, replay and adapters are reusable families, not MVAU
  internals;
- **attribution:** every refusal names its node or stream;
- **evidence:** numeric simulation for every configuration class.

## 2. Inputs

- **Previous audit** (the baseline to orient against):
  [`scratchpad/open/kernel-roster-map/`](../../../scratchpad/open/kernel-roster-map/)
  (`SUMMARY.md`, `ROSTER.md`, `DELIVERY.md`, `STREAMS.md`, `REVALIDATION.md`).
  In particular, see *Decision 1* (delivery placement), *Decision 2* (robust MVAU
  scope) and claims V01–V45.
- **The kernel audit following the landing.** It remaps every kernel family onto
  the landed model and supersedes the parts of the roster it touches.
- **The model:** the landed `finn.core.space`, i.e. declarations, `design_space`,
  `Decision` over nodes, references, `Users`/`Members`, overrides and provenance.
  See the landed Space documentation, and
  `docs/kernel-status-2026-09-25/STATUS.md` §5 for the earlier robust-MVAU list.

## 3. Scope

### 3.1 In scope (ordered by dependency; the audit may reorder)

| # | Capability | Realization (FinnLib or kernel-local) | Notes |
|---|---|---|---|
| R1 | **Clock and reset as a referenced Space** | a `Clock` / `ClockDomain` family referenced by kernels; the netlist drives pins from it | Removes name-based routing (`"clk2x" in name`); the prerequisite for pumped compute and multiple domains |
| R2 | **Delivery family parameterized by form**, placed by visibility | `CyclicDelivery` over any `Traversal`; `pack(operand, form)` owns the image | Roster Decision 1 option C: inside when the edge is invisible, a boundary stream when external. Generalizes to VVAU, EW and thresholds (V01) |
| R3 | **Replay as a choice** | `replay_buffer` (RTL) or `input_gen` with a `Level(d)` marker | A `Decision` over nodes. `input_gen` also serves the SWG/conv composition (V35) |
| R4 | **dotp mode as a choice**: packed or soft-vector | FinnLib wrapper split (MVU wrapper split, bit-equivalent to the fused golden) | Needs the FinnLib split landed on a remote |
| R5 | **Fused activation**: MVAU → Thresholding | `thresholding_axi`, with the internal table | An optional node. Fix the census thresholding contract first (V13: `V·NF` threshold beats, not `NF`) |
| R6 | **Runtime-writable weights** (AXI-Lite) | `hls/memstream.hpp`, array on `s_axilite` | Delivery case "decoupled-writable"; staticness is a requirement-tier fact, not a memory axis |
| R7 | **Multi-set delivery with a set-index sideband** | none today: design it | Generalizes to EW-C5 and MLO. The index is one per set (memstream) or one per beat (thresholding) (V10) |
| R8 | **Adapters on streams** | DWC / lane regroup / reorder kernels, chosen by `classify()` | A `Decision` over adapter nodes inside a stream, like the FIFO transport. Ports publish the traversal families they support |
| R9 | **VVAU as a reuse check** | the same families, `ACTIVATION_BROADCASTING=0`, a marker generator instead of replay | Proves the families aren't MVAU-shaped. Needs the SWG→VVAU lane order re-derived (E-048) |

### 3.2 Deferred

Deferred because FinnLib has no parts for them, or on purpose:

- tiled MVAU with TH>1 (C4);
- dynamic weights;
- `external_mem`/fetch;
- MMV / OUT_TILED;
- MX datatypes (a later `BlockEncoding`);
- data fan-out;
- a padded AXIS child feeding another child.

### 3.3 Out of scope

- ONNX/QONNX integration and the graph adapter;
- the `finn.dataflow` model pass;
- the query/search-tools pass. MVAU work should *record* the queries it wanted,
  as input to that pass;
- reuse of cached results across snapshots.

## 4. Constraints

- **The engine stays generic.** Streams, clocks, AXIS and adapters live in
  `finn.kernels`. If an increment needs an engine change, write that up as a
  separate proposal.
- **Every capability is a node, a `Decision` over nodes, or a reference.**
  Nothing is special-cased in the netlist.
- **Keys are stable.** Existing persisted decision keys and top-level ABI names
  must not change without a recorded reason.
- **FinnLib dependencies** are pinned to commits on a remote (see V30).
- Follow the repository `CLAUDE.md`:
  - run RTL work in the background;
  - one simulation per process;
  - no concurrent `run-docker.sh`;
  - an unlicensed synthesis that is skipped is not a pass.

## 5. Evidence required per increment

1. **Tests:** Space-level tests for new decisions and references, including
   attribution and presence; kernel tests for contracts and wiring.
2. **Numeric XSI** for every new configuration class, with free and stalled
   output. For example, today's runs are
   `python -m kernels.rtlsim.mvau_assembly_numeric` in the environment of
   `docs/kernel-package-extraction/rtl/VALIDATION.md`: 20 direct plus 8 through a
   FIFO.
3. **Fingerprints** of unchanged configurations are unchanged
   (`docs/space-graph-composition-2026-09-25/fingerprints.py`).
4. **Out-of-context synthesis**, where an increment claims a resource or
   inference effect (`rom_style`, `ram_style`, DSP use). Record licensing skips
   as skips.
5. **Both gates** green, with XSim executed.
6. **A coverage table**, maintained across increments: each baseline MVAU
   attribute or variant, mapped to its Space coordinate (Param, Decision or
   node), or marked deferred with the reason.

## 6. Increment plan (a draft, to be revised by the audit)

| Increment | Content | Depends on |
|---|---|---|
| I0 | The coverage table (§5.6), from baseline FINN MVAU attributes and the audit | audit |
| I1 | R1: clock and reset | none |
| I2 | R2: delivery by form, placement by visibility | I1 |
| I3 | R3: replay choice | I1 |
| I4 | R5: fused thresholding (with the V13 contract fix) | I2 |
| I5 | R4: dotp mode choice | FinnLib split on a remote |
| I6 | R6: AXI-Lite writable weights | I2 |
| I7 | R7: multi-set delivery | I2, I6 |
| I8 | R8: stream adapters | I2, I3 |
| I9 | R9: VVAU reuse check | I2, I3 (and E-048) |

Each increment is landable on its own and ends at a review gate.

## 7. Open questions for the audit to answer

1. For each kernel family, which parts of the roster still hold on the landed
   model, and which workarounds disappeared?
2. Where does replay live for conv: in MVAU, or in the SWG nest (V35)? That
   decides whether R3 is an MVAU choice or a composition choice.
3. Is `thresholding_axi`'s internal table sufficient for the fused case, or does
   thresholding need a delivery family too (V40, V42)?
4. What is the current remote status of the FinnLib commits R3–R5 need
   (`replay_buffer`, the MVU wrapper split)?
5. Which baseline MVAU attributes have no FinnLib realization at all? Those set
   the deferred list definitively.
