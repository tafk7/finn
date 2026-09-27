# Kernel audit after the declarative Space landing

Date: 2026-09-26. Status: **audit only**. It changes no source code and starts no
increment.

**Revisions audited:**

| Tree | Revision |
|---|---|
| Kernels | `6aa0383cf` on `feature/kernel-package-extraction` (the landing is `b244ec8be..6aa0383cf`) |
| FinnLib | `b17eae6a`. `deps/finnlib` is a symlink to the shared clone `prj-kernels/finnlib`. |
| Baseline custom ops | `src/finn/custom_op/fpgadataflow/`, in the same checkout |

**Inputs:**

- `docs/robust-mvau-2026-09-26/SPEC.md`;
- `docs/space-declarative-2026-09-26/{DESIGN,LANDING}.md`;
- `scratchpad/space/{DESIGN,AUTHORING}.md`;
- `scratchpad/open/kernel-roster-map/*`;
- `docs/kernel-status-2026-09-25/STATUS.md`.

**Supporting files:**

- [`COVERAGE.md`](COVERAGE.md): the SPEC §5.6 coverage table, plus short tables for the other six families.
- [`ROSTER-RECONCILIATION.md`](ROSTER-RECONCILIATION.md): V01–V45, Decisions 1–3 and open questions 1–10.
- [`probes/`](probes/): six probe scripts, their transcript (`probes-output.txt`) and the FinnLib remote-status transcript.

**Labels.** **[V]** means verified by reading the code at the cited line or by
running it. **[I]** means inferred. Paths are relative to the repository root
unless abbreviated:

| Prefix | Path |
|---|---|
| `k/` | `src/finn/kernels/` |
| `fpd/` | `src/finn/custom_op/fpgadataflow/` |
| `FL/` | FinnLib at `b17eae6a` |

**What was run:**

- `pytest tests/kernels tests/core/space`, with the Xilinx tools removed from
  `PATH` so that the XSim tests skip: **1177 passed, 9 skipped** (the 9 XSim
  tests) in 110 s. This matches the landing's 423 + 763 count less the 9
  simulations.
- Probes P1–P6 ([`probes/probes-output.txt`](probes/probes-output.txt)).
- `git fetch --all` in FinnLib, read-only.
- No XSim, no synthesis and no Docker were run.

## 1. Key findings

1. **The landed model fits MVAU well, and two roster decisions are now realized.**
   - Streams are nodes that kernels reference, and `Users`/`Members` gather them.
   - A Decision over nodes whose `None` case leaves a boundary
     (`k/mvau.py:276-278`, `k/streams.py:253-256`) *is* Decision 1 option C
     ("placement by visibility") for external/cyclic delivery.
   - The S1 stream contract of Decision 3 is implemented: `StreamSpec`,
     `StreamContract`, `compatibility` and `Composition.connect`.
   - R2's core is already done: `CyclicDelivery` takes any `Traversal`, and
     `pack(form, …)` owns the image (`k/delivery.py:48-73`, `k/physical/forms.py:364`). [V]
2. **Only 3 of 11 kernel families speak the stream idiom:** `DotpAxiKernel`,
   `ReplayBuffer` and `CyclicDelivery`.
   - Thresholding, Eltwise, InputGenerator and IntToFp32 are still flat,
     standalone codegen kernels with no stream references or `PORTS`.
     MemStreamHLS produces HLS requirements, which cannot be placed in a
     netlist at all.
   - The one eltwise + cyclic composition is hand-wired with the `Composition`
     API in a test (`tests/kernels/test_stream_contract.py:305-345`), not
     declared as a Space.
   - So every robust-MVAU item that touches thresholding (R5), `input_gen`
     (R3) or eltwise starts with a migration. [V]
3. **Soundness defect: consumer ports adopt the stream's form.** dotp builds each
   port contract from `stream.spec` and never checks the lane count or order
   against its own PE/SIMD (`k/dotp.py:302-317`).
   - A one-lane activation stream feeding a SIMD=2 dotp composes into a
     netlist. So does a transposed weight walk (probe P5).
   - MVAU is correct only because MVAU's derived specs are correct
     (`k/mvau.py:214-244`).
   - This blocks honest reuse (R9 VVAU), adapters (R8) and the E-048 lane-order
     check. [V]
4. **Attribution defect (R16): one kernel refusal is reported by every stream it
   sits on.** A single bad weights dtype makes `replayed`, `weight_stream` and
   `results` all refuse with `compute.weights_type.family` (probe P2).
   - The cause: `Users(PORTS)` gathers one `Ports` record per kernel
     (`k/streams.py:178-193`, `:240-243`).
   - SPEC §1 counts attribution as part of "robust". [V]
5. **The largest missing piece is the non-stream interface, not the clock.**
   `netlist` drives only stream pins, clocks and resets, and skips every bus pin
   (`k/streams.py:453-454`). Validation then requires every child input to be
   driven (`k/physical/validation.py:185-208`). As a result:
   - Thresholding's always-present `s_axilite` and `s_axis_set` buses (58 input
     bits) cannot be driven, tied off or exported, so thresholding cannot be
     composed at all (probe P4).
   - A padded AXIS child cannot feed another child (`k/physical/contract.py:153-159`,
     probe P3), which is exactly dotp → thresholding.
   - R5, R6 and R7 all depend on these. [V]
6. **The FinnLib pin is split three ways.** Every kernel source path will move
   on the next bump.
   - The kernels need flat `rtl/*.sv` at `b17eae6a`, which has the dotp queue
     fix and `rtl/replay_buffer.sv`.
   - `fetch-repos.sh:53` pins `dfeafac8` (`origin/feature/replay-buffer-finnlib`).
     That commit has the `rtl/{infra,linalg,…}` layout and no dotp fix.
   - The baseline MVAU/VVAU RTL ops in this same checkout read
     `rtl/infra/replay_buffer.sv` (`fpd/rtl/matrixvectoractivation_rtl.py:208`,
     `:458`).
   - `upstream/dev` (`5306111`) has neither `replay_buffer` nor the dotp fix,
     and has merged `queue` into `fifo`.
   - Neither pinned commit is on upstream ([`probes/finnlib-remote-status.txt`](probes/finnlib-remote-status.txt)). [V]
7. **R4 (dotp mode) has no FinnLib realization.** `dotp_axi` chooses between
   the INT8 and soft-vector cores internally (`FL/rtl/dotp_axi.sv:263-289`).
   - The "MVU wrapper split" exists only in FINN's `finn-rtllib`, on
     `tafk7/finn` `origin/feature/dataflow-kernel` (as `mvu_vvu_axi_base_{head,tail}.svh`
     plus `mvu_vvu_axi_{packed,softvec}.sv`). It is not in FinnLib.
   - Recommendation: defer R4. [V]
8. **Coverage (COVERAGE §1).** The kernel MVAU covers:
   - MW, MH and repetitions;
   - PE and SIMD as divisor Decisions;
   - INT/UINT activations and signed weights on DSP;
   - external and read-only cyclic delivery, with `rom_style`;
   - pumped compute;
   - a FIFO on the weight stream.

   No FinnLib realization exists in any form for `TH>1`, MMV, `dynamic`,
   `external_mem`/MLO, `pumpedMemory` or multi-set delivery. Binary/xnor,
   `lut`, embedded weights, unsigned weights and the writable memstream exist in
   FinnLib only as HLS. [V]
9. **Replay for conv (SPEC §7 Q2) is a composition choice.** MVAU should own a
   replay slot as a Decision over nodes. The "the upstream SWG replays" case
   belongs to a conv composite and is deferred.
   - Thresholding needs no delivery family (Q3).
   - Fused thresholding needs stream migration, the non-stream-interface
     increment and a padding-disposal rule. [V/I]
10. **Model friction is moderate and local.** The costliest points are:
    - an unused stream is *refused* rather than absent, so optional streams need
      explicit `when=` guards (`k/streams.py:250-251`);
    - `@derived` + `View(...)` pairs instead of same-body view edges (R30);
    - tuple value semantics that force `cast` and positional indexing;
    - string keys in the adapter although typed handles work (probe P6). [V]

## 2. Per-family findings

**Idioms of the landed model:**

| Code | Idiom |
|---|---|
| **S** | streams as referenced nodes, with a `PORTS` export |
| **DN** | a Decision over nodes |
| **H** | a candidate handle |
| **A** | an anchored spec |
| **C** | per-member constraints and groups |
| **VV** | views read as values |
| **F** | reuse through a factory returning fresh declarations |

| Family | Formals (value-annotated) | Decisions | Views / exports | Stream references and `ports` | Delivery or placement | Composes by | Uses well | Workarounds and pre-redesign shapes |
|---|---|---|---|---|---|---|---|---|
| **`MVAU`** (`k/mvau.py:159-294`) | `repetitions`, `matrix_width`, `matrix_height: int`; `activation_dtype`, `weights_dtype: QONNXDataType` (value semantics); `target_dsp: DspBlock`; `segment_length: int`; `weights: IntegerTensor` (optional) (`:172-179`) | `pe`, `simd` (divisor domains, `:180-181`); `implementation` over nodes (`:276-278`); nested: `compute.compute_pumping`, `implementation.cyclic.rom_style`, `weight_stream.transport` (+`fifo.buffer.depth`, `ram_style`) | derived: `result_type`, `folding`, the four specs, `weight_period`. Views: `structure` (COMPOSED, `requires=(dimensions, modules, streams)`, `:283-290`) and `build_requirements` (`:292-294`) | owns four `Stream` nodes (`:248-251`) with ports `in0_V`/`in1_V`/`out0_V`. `modules = Members(MODULE)`, `streams = Members(CONNECTION)` (`:280-281`) | `implementation`: `None` makes `weight_stream` the boundary `in1_V`; `cyclic` places the handle `cyclic` (`:273-275`) | `netlist(Members, Members)` (`:285-290`) | S, DN, H, A, C, VV | The `mvau_assembly` adapter uses string keys and returns a bundle (`:300-368`). The module name and producer id are strings built from `selected(implementation)` (`:279`, `:288-289`). The `dimensions` group reads the `folding` refusal only to gate the view (`:199-204`). |
| **`Stream`, `BufferedStream`, `StreamFifo`, `netlist`** (`k/streams.py`) | `Stream.spec: StreamSpec` (anchored, `:232`); `port: str` (optional, `:233`) | `BufferedStream.transport: _Direct \| StreamFifo` over nodes (`:304-308`); `StreamFifo.buffer.depth` inline (`:141`) | `endpoints`, `compatible`, `connection` (exported under `CONNECTION`, `:236-295`); `BufferedStream.stage = View(transport.stage)` (`:308`) | `ends = Users(PORTS)` (`:234`) | a one-sided stream is a boundary, named by `port` (`:253-256`) | `netlist` wires `Members(MODULE)` through `Members(CONNECTION)` (`:325-409`) | S, DN, A, C; the subclass re-sourcing `stage` through the decision is idiomatic | **Clocks by name:** `_CLOCKING` literal (`:97`); `_top_abi` gathers `ap_clk`/`ap_clk2x`/`ap_rst_n` by name (`:412-443`); `_drive` routes by the substring `"clk2x"` and skips every bus pin (`:446-463`, probe P1). Instance names are string-munged node paths (`:320-322`, `:339`). The FIFO instance name is a literal (`:383`). `Stage` is identity-only (one spec on both sides, `:145-152`). An unused stream is refused (`:250-251`). A composition-level refusal is a single `stream-composition` code (`:355-356`). |
| **`DotpAxiKernel`** (`k/dotp.py:77-340`) | `pe`, `simd`, `segment_length: int`; `target_dsp`; three dtypes (`:88-96`); three **optional** stream references (`:102-104`) | `compute_pumping` (`:92`) | scalar nodes via the `integer_scalar` factory (`:97-99`); AXIS port nodes via `axi_stream` (`:105-107`); `support` group (`:196-204`); `build_requirements = View(codegen, …)` (`:206-295`); `interfaces` (a tuple, `:297-300`); `activation_port`, `weights_port`, `result_port`, `ports`; exports `MODULE`, `PORTS` (`:340`) | three references; `ports` keyed by the reference name (`:331-338`) | — | via streams | F, C, S | **Adopts the stream's form** (`:302-317`, P5). `_port(index)` indexes a tuple by position (`:302-305`). Clock names are literals (`:311`, `:259`). `ACTIVATION_BROADCASTING=1`, `NARROW_WEIGHTS=0` and `FORCE_BEHAVIORAL=0` are pinned (`:229-234`). Geometry and width checks are duplicated in `codegen` (`:208-216` vs `:116-130`). Signed weights are checked twice (`:98` scalar vs `:152`), which yields two findings for one cause (P2). Standalone use leaves `ports` unresolved but still listed. |
| **`ReplayBuffer`** (`k/streaming.py:115-165`) | `input_stream`, `output_stream: Stream` (required); `sequence_length`, `replay_count: int` | — | `contracts` (derived, `default_semantics(tuple)`, `:128-138`); `input_port`, `output_port` (via `cast`, `:141-146`); `ports`; `build_requirements` | two references | — | via streams | S; its **output contract is derived from its input**, which is the idiom a consumer should follow | Tuple semantics plus `cast`. Opaque `ReadyValidStream` with literal `clk`/`rst` pins (`:58-66`, `:70-86`). `ofin` is left unused. |
| **`CyclicDelivery`** (`k/delivery.py:48-100`) | `dtype`; `form: Traversal`; `values: IntegerTensor`; optional `output_stream` (`:52-57`) | `rom_style` (`:58`) | `image` (`pack`), `output` (a cyclic contract), `build_requirements`, `ports` (exports `:100`) | one optional reference | a candidate of MVAU's `implementation`; also placeable beside a consumer (`tests/kernels/test_declared_streams.py:54-80`) | via a stream | S, F, H; a **producer-owned form** | Integer only (`:53`), so it cannot deliver FLOAT32 eltwise constants. The kernel-local `cyclic_stream.sv`. Optional reference for dual use. |
| **`FifoKernel`** (`k/fifo.py:45-136`) | `word_bits`, `depth: int` | `ram_style` (`:60`) | `storage`, `interfaces` (a tuple), `build_requirements` | none (used as a `Stage`) | the `BufferedStream` fifo candidate | through `StreamFifo.stage` | DN (inside a stream) | Flat kernel with opaque words. The capacity model mirrors the pinned `fifo.sv`, which upstream has since merged with `queue` [I: model may drift]. |
| **`ThresholdingAxiKernel`** (`k/thresholding.py:69-315`) | `input_dtype`, `threshold_dtype`, `thresholds: ThresholdTable`, `bias`, `pe`, `depth_trigger_{bram,uram}` | `use_axilite`, `deep_pipeline` (`:97-98`) | `result_dtype`, `implementation_supported`, `build_requirements` | **none**: no `StreamSpec`, no `PORTS` | internal table (V40) | **cannot compose** (P4) | C | Fully pre-redesign in spirit. Streams and buses are built inside `build_requirements` (`:250-287`). `s_axis_set` and `s_axilite` are always present. `pe` is a Param, not a folding Decision. `FPARG` is pinned to 0 (`:234`). |
| **`EltwiseKernel`** (`k/eltwise.py:62-189`) | `operation`, `pe`, `lhs_dtype`, `rhs_dtype`, `b_scale`, `target_dsp` | — | typed native ports via `native_stream` (`:85-89`); `interfaces` (a tuple); `build_requirements` | **none** | — | only hand-built in a test | F, C | No stream references or `PORTS`. The int MUL width limit is missing (COVERAGE §4). |
| **`InputGeneratorKernel`** (`k/input_generator.py:41-135`) | `word_bits`, `frame_words`, `extents`, `strides` | `ram_style` (`:75`) | `interfaces` (a tuple, `olst` LOOP_END of width D, `:95`), `build_requirements` | **none** | — | not composable; a multi-bit marker has no contract rule (`k/physical/contract.py:81`) | C | Opaque words (no element or lanes), so it cannot be the replay alternative in R3 without migration. |
| **`IntToFp32Kernel`** (`k/int_to_fp32.py:30-58`) | `input_dtype`; `result_dtype = Const(FLOAT32)` | — | `build_requirements` | none (combinational) | — | not a stream kernel | F | `Const` stays typed `Const[T]` in its own body (DESIGN §12). |
| **`MemStreamHlsKernel`** (`k/memstream_hls.py:35-105`) | `element_dtype`, `depth` | — | `build_requirements -> HlsSourceRequirements` | none | a candidate for R6 | **not netlistable**: no pin ABI | C | The AXI-Lite array exists, but it cannot enter `netlist`. |
| **Scalars, ports, `configure`, `Kernel`** | `Scalar.dtype`, `IntegerScalar.{signedness,min_bits}` (`k/datatypes/scalar.py:78-151`); `TypedStream` Params (`k/physical/ports.py:60-81`) | — | `encoding = View(candidate, requires=(admission,))` | — | — | factories | F, C, VV; subclassing for admission | `commit` rebuilds a key-to-reference map per call (`k/configure.py:40-62`). `Kernel.capabilities()` is a member-kind filter (`k/base.py:32-35`). |

**Partially migrated in spirit** (the items SPEC Q1 asks to flag):

- **Name-based clock and reset routing.** `k/streams.py:97`, `:412-463`; dotp
  literals `k/dotp.py:259`, `:311`. The netlist gives an unpumped MVAU a top
  `ap_clk2x` input with a `Data` role (P1). That matches the baseline wrapper
  (`finn-rtllib/mvu/mvu_vvu_axi_wrapper.v:60`), but the clock is identified only
  by its name. [V]
- **The `ports` convenience record.** One export per kernel (`k/streams.py:178-193`)
  causes the attribution defect (P2). [V]
- **Positional and tuple views.** `interfaces` tuples in dotp, fifo, eltwise and
  input_gen; `ReplayBuffer.contracts` with `cast`. [V]
- **Literal names.** The FIFO instance `u_{stream}_fifo` (`k/streams.py:383`).
  The native pin names and FinnLib paths are fine as kernel facts. [V]
- **String keys in the adapter.** `mvau_assembly` (`k/mvau.py:342-353`) and
  `configure.commit`; typed handles work (P6). [V]
- **Kernels outside the stream idiom.** Thresholding, Eltwise, InputGenerator;
  eltwise composition exists only through the raw `Composition` API in a test. [V]
- **Callable views still in documents.** The artifact-integration SPEC
  (`docs/kernel-artifact-integration-2026-09-25/SPEC.md:30`, `:84`,
  `point.build_requirements()`) and STATUS §4 (`:182`). [V]
- **Stale README text.** `k/README.md:121` (`TopInput`), `:126` (`compose`) and
  `:129` (`Present(in1_V, cyclic.output)`) describe pre-landing constructs.
  Nothing in `k/` uses `Present` or `LocatedParam`. [V]
- **Factories that bypass declarations.** None found. `integer_scalar`,
  `axi_stream` and `native_stream` return fresh node declarations, which is the
  sanctioned reuse form. The `_Folding` dataclass (`k/mvau.py:107-141`) is a
  value, not a bypass. [V]

## 3. Roster reconciliation (summary)

The full table is in [`ROSTER-RECONCILIATION.md`](ROSTER-RECONCILIATION.md).

**Counts over V01–V45:**

| Status | Count |
|---|---:|
| holds, or holds (baseline) | 27 (V13 among them, which holds but is obsolete for kernels) |
| changed | 9 |
| resolved | 2 (V27, V29) |
| obsolete | 6 |
| not rechecked | 1 (V14, not load-bearing) |

**Changes that matter for planning:**

- **V30.** The FinnLib pin is split (finding 6).
- **V26.** Upstream has reorganized the FinnLib layout.
- **V38.** The padding question moves to the child-padding rule.
- **V11.** The period-divides-pass check is now generic.
- **V13.** It holds in the census but does not block R5.

**Decisions:**

- Decision 1 option C is realized for external/cyclic. The cases that need
  control buses or sidebands remain open.
- Decision 3 (S1) is realized, except for clock identity and multi-bit markers.

**Open questions:**

- Q2 and Q6 are resolved.
- Q1 has changed (it needs a human decision).
- Q9 is now checkable once dotp owns its form.

## 4. Answers to the robust MVAU SPEC §7

**Q1. Which parts of the roster hold, and which workarounds disappeared?**
See §2 and §3.

These roster workarounds are gone:

- `_wire_mvau` hand wiring, now checked `connect`;
- `SubspaceChoice`, now a Decision over nodes;
- `assemble_streams`, now `Members` + `Users`;
- the MVAU-bound `_Traversal`, now `Traversal` + `pack`;
- "typed ports carry no markers", now `Every` rules;
- the implicit period/pass alignment, now `_presented`.

These remain:

- clocks by name;
- the per-kernel `Ports` record;
- flat kernels;
- no non-stream interfaces;
- no adapter insertion;
- forms supplied by the composite rather than owned by the consumer. [V]

**Q2. Where does replay live for conv?** It is **both**, at different levels.
[V for the facts, I for the recommendation]

- **Inside MVAU**, replay should become a Decision over nodes:
  `replay: ReplayBuffer | InputGenerator = Decision(...)`.
  - `replay_buffer` is what the XSim sweep validates today.
  - `input_gen` with an NF level must first be migrated to the stream idiom.
    It also needs multi-bit marker rules, because `olst[d]` asserts every
    `prod(extents[d:])` beats. That makes it a family of `Every(k)` rules, one
    per bit, and `StreamContract` accepts only 1-bit markers today
    (`k/physical/contract.py:81`).
- **In conv, the SWG nest absorbs the NF loop**, as FinnLib `conv2d.sv` does
  (V35). That is a **composition choice**: a conv composite that places the SWG
  and MVAU, with MVAU's replay slot set to "none", so that `replayed` becomes
  MVAU's input boundary. Two costs follow:
  - That boundary would carry a marker (TLAST). Baseline inter-op AXIS has none
    (V37), so the "none" case must not be offered to a standalone MVAU.
  - The now-unused `activations` stream would be refused as `stream-unused`
    (`k/streams.py:250-251`) unless it is guarded with `when=`.
- **So R3 is an MVAU choice between `replay_buffer` and `input_gen`.** The
  upstream case is deferred to the conv-composite pass.

**Q3. Is `thresholding_axi`'s internal table enough for the fused case?**
**Yes.** Thresholding needs no delivery family. [V]

- No FinnLib thresholding consumes a threshold stream (V40, V42). The internal
  table covers static tables and AXI-Lite-writable single-set tables.
- Multi-set needs a per-input-beat set index (V10). That is an R7 sideband, not
  delivery.

Four other things block the fused case:

1. `ThresholdingAxiKernel` has no stream references or `PORTS`, so it must be
   migrated.
2. dotp's padded AXIS result cannot feed a child (P3), so the padding rule must
   allow disposal toward a sink that ignores padding.
3. The always-present `s_axis_set` and `s_axilite` buses need a tie-off or
   export (P4).
4. The fused `activated` stream and the bare `results` boundary are alternatives
   and need `when=` guards.

V13 is not a blocker.

**Q4. What is the remote status of the FinnLib commits?** [V]
(`probes/finnlib-remote-status.txt`)

- **`replay_buffer` (R3):**
  - Flat at `b17eae6a`, on `origin/kernel-contract-refinement-20260925`.
  - `rtl/infra/` at `dfeafac8`, on `origin/feature/replay-buffer-finnlib`.
  - Both copies are byte-identical, both are on the personal fork
    (`tkeller/finnlib`), and neither is on `upstream/dev` (`5306111`), which
    has no `replay_buffer` at all.
- **dotp queue fix:** only on `b17eae6a`. `dfeafac8` and `upstream/dev` both
  still have `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH`.
- **MVU wrapper split (R4):** not in FinnLib on any branch. It exists in FINN
  `finn-rtllib/mvu/` on `tafk7/finn` `origin/feature/dataflow-kernel`.
  FinnLib's `dotp_axi` still forks internally.
- **Thresholding (R5):** `thresholding_axi` and `axilite` are on
  `upstream/dev`, under `rtl/nonlin` and `rtl/infra`.

**Q5. Which baseline MVAU attributes have no FinnLib realization?**
[V] (COVERAGE §1)

- **None at all:** `TH>1`, MMV/OUT_TILED, `mem_mode=dynamic`,
  `mem_mode=external_mem` (fetch, `mlo_max_iter` index, `address_offset`),
  `pumpedMemory`, a multi-set memstream (`SETS`).
- **HLS only:** a writable memstream (`hls/memstream.hpp`), embedded weights,
  `resType=lut`, binary/bipolar/xnor, unsigned weights.

These set the deferred list. The HLS-only group becomes reachable only with an
HLS netlisting path (a pin ABI for `HlsSourceRequirements`).

## 5. Defects and limits

| Id | Item | Status | Evidence |
|---|---|---|---|
| D1 | **FinnLib pin split** | **open, new** | Finding 6 above. The kernel `CopiedSource` paths are flat (`rtl/dotp_axi.sv`, …). Upstream moved them to `rtl/{arith,infra,linalg,nonlin,shape}` and merged `queue` into `fifo` (upstream `21a3c45`). `eltwise.sv` now instantiates `fifo` [V]. A bump changes source identities, so fingerprints change with a recorded reason. |
| D2 | **Consumer ports adopt the stream form** | **open, new** | `k/dotp.py:302-317`; P5 [V] |
| D3 | **Refusal attribution across a kernel's streams (R16)** | open (deferred engine question) | P2; `k/streams.py:240-243` [V]. It can be mitigated in the kernels without an engine change [I]. |
| D4 | **No non-stream interfaces** (control bus export, tie-off, sideband) | **open, new as a named item** | P4; `k/streams.py:446-463`; `k/physical/validation.py:185-208` [V] |
| D5 | **A padded AXIS child cannot feed a child** | open | P3. `compatibility` refuses (`k/physical/contract.py:153-159`), because `UnusedOutput` disposes whole pins only (`k/physical/structure.py:95-97`) and `ignored_top_input_bits` covers the top only. The refusal is over-strict when the sink's padding policy is `IGNORE_ON_RECEIVE` (`k/physical/ports.py:39-57`) [V]. |
| D6 | **No data fan-out** | open (deferred) | A stream refuses a second consumer (`stream-users`, `k/streams.py:244-249`). Validation refuses a data fan-out (`k/physical/validation.py:217`). FinnLib has `hls/dup.hpp` and `rtl/stream_tap.sv` as candidate kernels [V]. |
| D7 | **Forms declared, not negotiated** | open | The composite authors every spec (the anchoring rule, `k/streams.py:20-22`). Ports do not publish supported forms. `classify` names adapters (`k/physical/forms.py:276-308`), but nothing inserts them, and `Stage` is identity-only (`k/streams.py:145-152`) [V]. D2 makes this worse: consumers do not even check. |
| D8 | **Clock domains** | partial | Domains are top pin *names*, recorded by `drive` (`k/physical/composition.py:73-92`). The check compares names (`:103-119`). Derived clocks exist only as ABI `ClockAlignment` (`k/dotp.py:264`). There is no domain node, no multi-domain composite and no CDC [V]. |
| D9 | **V13** (thresholding threshold-beat count) | census still wrong; **not a kernel blocker** | `scratchpad/proofs/baseline-finn-design-census/families/thresholding.md:236` still says `NF`. The kernel has no threshold stream [V]. |
| D10 | **E-048** (SWG→VVAU lane order) | open, now expressible | FinnLib VVU field order is `simd*PE+pe`, not reversed (`FL/rtl/dotp_axi.sv:109-120`). It can be declared as a `Traversal`, and `classify` handles a permutation for free, but only after D2 is fixed [V]. |
| D11 | **V38** (padding at eltwise) | moved into D5 | Native-to-native is exact. The padded AXIS boundary appears only at the top, which `connect` handles (`k/physical/composition.py:146-165`) [V]. |
| D12 | Accumulator dtype cannot be supplied | open (design) | `result_type` is derived only (`k/mvau.py:183-188`). A graph adapter applying value-based narrowing has nowhere to put the narrower dtype [V]. For writable weights, narrowing must stay off. |
| D13 | Eltwise int MUL width limit missing | open, small | Baseline asserts ≤24/23 bits (`fpd/rtl/elementwise_binary_rtl.py:115-119`); the kernel admits 128 bits (`k/eltwise.py:57`) [V]. |
| D14 | Documents drift | open, small | The kernel README (`:121-129`) and the artifact-integration SPEC (`:30`, `:84`) show pre-landing constructs [V]. |
| D15 | `deps/finnlib` is a symlink to a shared working clone | risk | `ls -la deps` → `/home/tkeller/prj-kernels/finnlib`. Another session that checks out a different commit there silently changes this checkout's sources [V]. |

## 6. Queries wanted (input for the query and search tools pass)

| # | Where | Hand-written structure walk, or the query it wants |
|---|---|---|
| W1 | `k/configure.py:40-62`; `tests/kernels/test_mvau_delivery_choice.py:64`, `:171`, `:414`, `:451` | Keyed lookup of decisions and choices: `{item.key: item.reference for item in inspection.decisions(point)}`, rebuilt on every call |
| W2 | `tests/kernels/test_mvau_delivery_choice.py:127`, `:352`, `:445`; `tests/kernels/test_mvau_assembly.py:57`, `:79` | An instance by **node**: `structure.instances[2]` and `[-1]` are positional. Wants "the instance realizing node `implementation.cyclic`". |
| W3 | `tests/kernels/test_mvau_assembly.py:69`, `:97`, `:110`; `tests/kernels/test_stream_contract.py:363`, `:414`; `tests/kernels/test_mvau_delivery_choice.py:438` | Wires filtered by pin names. Wants "the wires of stream `weight_stream`" (a stream-to-wire provenance query). |
| W4 | `k/streams.py:240-243` | A per-port keyed gather (one entry per `(user, input)` port, with its own result). This fixes D3. |
| W5 | `k/streams.py:412-443` | A walk over every child's ABI for clock and reset names. With R1 this becomes `Users(CLOCK)` on a clock node, or a `Collect`. |
| W6 | `k/streams.py:236-259` | "Is this stream a boundary?" is a count of present users per side. Placement by visibility (R2, R6, R7) and the replay "none" case want it as a query, and would also make an unused stream *absent* rather than refused. |
| W7 | `k/base.py:32-35`; `tests/kernels/test_dotp.py:92`, `:104`, `:116`; `tests/kernels/test_axi_stream_declaration.py:149`, `:189`, `:219` | Members filtered by kind or scope (views, params). Wants `children(space, exporting=K)` / members-by-kind. |
| W8 | (none today) | All transport slots in a composite, for FIFO sizing, and all streams whose `compatible` refusal is `stream-form`, for adapter insertion under R8. |
| W9 | `k/mvau.py:279`, `:288-289` | The selected candidate's key used as a string. `selected()` works; it records the need for structured names. |
| W10 | `k/streams.py:320-322`, `:339` | Instance naming by `str(node).replace(".", "_")`. Wants a structured path on `Located.node`. |
| W11 | `docs/space-declarative-2026-09-26/mvau_keys.py`, `docs/space-graph-composition-2026-09-25/fingerprints.py` | Evidence scripts walk inspection for keys and nodes. They want a stable "dump the structure" query. |
| W12 | R2/R9 delivery slot (planned) | "The consumer port that this delivery candidate feeds" is `Users` on the delivered stream. It is needed to derive the delivery form from the consumer instead of passing `weight_period` twice (`k/mvau.py:227-237`, `:273-275`). |

## 7. Model friction (for kernel authoring)

| # | Friction | Code references | Cost |
|---|---|---|---|
| M1 | **A view named in its own body is `View[T]`** (R30). No kernel feeds a same-body view into a formal: kernels declare a `@derived` (value-typed in the body) and then a `View(...)` over it. [I] Why they do this is inferred; the decorator form `@view(requires=...)` also exists. | `k/dotp.py:206` + `:295`; `k/streams.py:289` + `:294`; `k/physical/axi_stream.py:135-148`; `k/datatypes/scalar.py:87-92` (`candidate` + `encoding`) | Two names per product and more nodes; the idiom is non-obvious. No `cast` for R30 appears in `k/` [V]. R30 will bite when a composite passes a child's accepted view into a sibling in the same body; J6 is an example (thresholding reading dotp's result contract). [I] |
| M2 | **Tuple value semantics** carry no element types | `k/streaming.py:128-146` (`cast(StreamContract, self.contracts[0])`); `k/dotp.py:297-305` (`interfaces[index]`) | casts and positional coupling [V] |
| M3 | **`Users(key)` returns one export per user** (R16) | `k/streams.py:178-193`, `:240-243`; P2 | attribution (D3) [V] |
| M4 | **Optional streams**: an unused stream is refused, and a boundary needs its `port` declared up front | `k/streams.py:233`, `:250-251`; `tests/kernels/test_declared_streams.py:175-185` | fused thresholding and replay "none" need `when=` guards and duplicate port names [V/I] |
| M5 | **The anchoring rule** makes consumers read the spec, and nothing makes them check it | `k/streams.py:20-22`; `k/dotp.py:302-317`; P5 | D2 [V] |
| M6 | **Candidates declared inline have no class-attribute handle**, so adapters fall back to string keys | `k/streams.py:305-307`; `k/mvau.py:342-353`; P6 (`MVAU.weight_stream.transport.fifo` is a `ChoiceMemberRef`) | string keys in the adapter [V] |
| M7 | **`composite()` families are untyped at the call** | `tests/kernels/test_kernel_extensions.py:313-327` (`type: ignore[call-arg]`, `getattr`) | matters for generated kernels (loop nests, VVAU variants built as data) [V] |
| M8 | **Dual-use kernels** carry optional stream references | `k/dotp.py:102-104`; `k/delivery.py:57` | `ports` is unresolved when a kernel stands alone but is still listed by `capabilities()`. This is STATUS §3's "ports are formals" in a milder form. [V] |
| M9 | **`Const` is typed `Const[T]` in its own body** | `k/int_to_fp32.py:36-38` | minor [V] |
| M10 | **`requires=` is untyped** (R32) | `k/mvau.py:283` | a wrong obligation is caught only at link [V] |
| M11 | **`LocatedParam` is unused.** It would fit sibling relations (for example "the thresholding input dtype equals the dotp result dtype") | grep: no use in `k/` | not friction; noted because the SPEC asked [V] |
| M12 | **The engine is not at fault for clocks and buses**: they belong in `finn.kernels` by layering (SPEC §4), but the netlist is the only place that knows them | `k/streams.py:412-463` | R1 and D4 are kernel work, not engine work [V] |

None of M1–M12 requires an engine change for the increments below. M3 and M4/W6
are the two that the query pass could remove at the root.

## 8. What the landed model already gives the robust MVAU (verified)

- **Placement by visibility**: the Decision over nodes with a `None`
  candidate, plus one-sided streams (`k/mvau.py:276-278`).
- **A reusable, form-parameterized delivery**: `k/delivery.py`.
- **A checked replay contract**: the replay's output is derived and checked
  against the consumer, and marker rules are wired (`k/streaming.py:89-112`,
  `k/physical/contract.py:144-151`).
- **Per-stream refusals, with independent streams settling independently**:
  `tests/kernels/test_declared_streams.py:100-114`.
- **Atomic structural switching and sparse replay**: the README example and
  `tests/kernels/test_mvau_delivery_choice.py`.
- **Transport slots as Decisions over stage nodes**: `k/streams.py:304-308`.
  This is the template for R8 adapters.

## 9. Revised increment plan (revises SPEC §3 and §6; no increment started)

**Principles behind the revised order:**

- Fix contract soundness and attribution before adding families that rely on
  them.
- Build the missing interface kind (control buses, sidebands, tie-offs) once,
  before the three items that need it.
- Pull VVAU forward, because it is the cheapest proof that the families are not
  MVAU-shaped.
- Pin FinnLib before depending on more of it.

| # | Increment | Content | Depends on | Why here | Replaces |
|---|---|---|---|---|---|
| **J0** | FinnLib consolidation (**human decision**) | One FinnLib commit on a remote with the upstream layout, `rtl/infra/replay_buffer.sv` and the dotp queue fix. Update every kernel `CopiedSource` path, and `queue`→`fifo` for eltwise. Align `fetch-repos.sh` and `deps/finnlib`. Re-run the fingerprints (a recorded source-identity change) and the full XSim sweep. | human decision | Every later increment adds FinnLib sources. The pins disagree today (D1, D15). | SPEC §4 "pinned to a remote" |
| **J1** | Coverage table | Delivered: [`COVERAGE.md`](COVERAGE.md) §1. Maintain it per increment. | — | SPEC §5.6 | I0 |
| **J2** | Port-contract soundness and per-port attribution (**new**) | dotp checks lanes against SIMD, PE·SIMD and PE, and declares its required weight tile relative to its activation form. `_port` is keyed, not positional. `ReplayBuffer` contracts are named, not tupled. Streams refuse only on their own port entry, via a kernel-local `Ports` whose entries carry per-port results, or one export per role. The duplicate dotp checks are removed. P2 and P5 become regression tests. | none | D2 and D3. Robustness means attribution, and every reuse relies on consumers owning their requirements. No engine change. | new |
| **J3** | Clock and reset as a referenced Space | This is R1. Clock/domain nodes are referenced by kernels. The netlist drives pins from them. Domain identity is by node. The top ABI names stay stable, including the unpumped `ap_clk2x`, unless a decision says otherwise (H3). | none (parallel with J2) | D8. A prerequisite for bus clock association (J4), pumped memory and multi-domain designs. | I1 |
| **J4** | Non-stream interfaces (**new**) | Control buses (AXI-Lite) exported through the composite, with clock association. Declared constant tie-offs for optional inputs. A sideband stream kind for set indices. Slice-level padding disposal toward sinks that ignore padding. All of it lives in `finn.kernels` as nodes or Decisions, with no netlist special-casing. | J3 | D4 and D5. It unblocks R5, R6 and R7 at once. | new |
| **J5** | VVAU reuse check | This is R9, moved earlier. dotp gets a VVU mode (broadcasting as a Param or derived value, PE·SIMD activation lanes, DSP58-only). A VVAU composite reuses `CyclicDelivery`, `ReplayBuffer(replay_count=1)` as the marker source (`FL/rtl/replay_buffer.sv:107-113`) and `BufferedStream`. The delivery slot (`None` \| `CyclicDelivery`) is extracted into a reusable factory, which completes R2. The VVU activation form `simd*PE+pe` is declared (E-048 as a form). | J2 (J0 for DSP58 sources) | It needs no new interface kind, and it is the cheapest proof that the families are not MVAU-shaped. | I9, and the remainder of I2 |
| **J6** | Fused thresholding | This is R5. `ThresholdingAxiKernel` is migrated to stream references, `PORTS` and a `StreamSpec`. MVAU gets an optional thresholding node that splices `results` → `activated`, with `when=`-guarded streams. Set and AXI-Lite are tied or exported via J4. V13 is not a blocker. Numeric XSim covers the fused output. | J2, J4 | Q3: the internal table suffices | I4 |
| **J7** | Replay choice | This is R3. `replay` becomes a Decision over nodes {`replay_buffer`, `input_gen`}. `InputGeneratorKernel` is migrated to an element, lanes and ports. `StreamContract` gains multi-bit marker rules and per-bit wiring. **Key stability**: the `replay` key and `u_replay` instance (H3). The conv "none" case is deferred. | J2, J3 | Q2 | I3 |
| **J8** | AXI-Lite writable weights | This is R6. It needs an **RTL writable memstream** (**human decision**: port baseline `finn-rtllib/memstream` into FinnLib, a kernel-local resource, or HLS→RTL ABI projection of `hls/memstream.hpp`). It enables `ram_style=ultra`. | J4, J0 | No FinnLib RTL part exists (Q5) | I6 |
| **J9** | Multi-set delivery with a set-index sideband | This is R7. New RTL (per-set index), the J4 sideband and EW-C5 generality. | J4, J8 | Design needed | I7 |
| **J10** | Stream adapters | This is R8. `Stage` is generalized to form-changing adapters. `vpc` (width conversion) and `input_gen` reorder kernels are added. Insertion is a Decision over adapter nodes on a stream, with parameters from `classify`. | J2, J7 | Without J2 nothing mismatches to adapt | I8 |
| — | dotp mode choice | This is R4. **Deferred**: FinnLib has no split. At most, expose the selected core as a derived view for estimation. | FinnLib change | Finding 7 | I5 |

**Deferred (confirmed or added):**

- `TH>1`; MMV/OUT_TILED; dynamic; `external_mem`/fetch/MLO index;
  `pumpedMemory`;
- data fan-out, which needs a `dup`/`stream_tap` kernel and a relaxed
  `stream-users` rule;
- the conv composite with upstream replay;
- MX;
- the HLS compute path (binary/xnor, `lut`, unsigned weights, embedded
  weights), which needs a pin ABI for `HlsSourceRequirements`;
- valid-only cores;
- multi-domain CDC.

**Suggested landing order:** J0 ∥ (J2 ∥ J3) → J4 → J5 → J6 → J7 → J8 → J9 → J10.
J5 can land before J4. Each increment ends at a review gate with the SPEC §5
evidence.

### Needs a human decision

- **H1. FinnLib consolidation (J0).** Decide which commit and remote. Decide
  whether to open upstream PRs for `replay_buffer` and the dotp queue fix.
  Decide whether to adopt the upstream subdirectory layout now; that changes
  every kernel source identity and every fingerprint. Decide whether
  `deps/finnlib` should remain a symlink to a shared clone (D15).
- **H2. The source of the RTL writable memstream (J8).**
- **H3. Key and ABI stability for structural changes:**
  - `replay` becomes a Decision (key and instance name);
  - a fused-thresholding `activated` stream and its `out0_V`;
  - the unpumped top `ap_clk2x` pin.
- **H4. Per-port attribution (J2).** Mitigate it in the kernels now, or wait for
  the engine's keyed gather in the query pass.
- **H5. The fate of R4.** Defer it, or port the MVU split into FinnLib's `dotp_axi`.
- **H6. The reordering.** Moving VVAU (R9) ahead of fused thresholding (R5), and
  inserting J2 and J4 as new increments.
