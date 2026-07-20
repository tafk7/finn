# KernelOp Integration Strategy — Parallel vs Integrated vs Adapter, Resolved

*How the new KernelOp substrate replaces the FINN HW backend **in place**. The third
of the three as-is/decision documents: `hw-backend-model.md` models the SUBSTRATE
being replaced (HWCustomOp + HLS/RTL backends); `consumer-surface-model.md` models the
CONSUMER SURFACE the replacement must present (the 20-step build flow + 37
transformations + 8 analysis passes); THIS document decides the integration shape that
carries one onto the other. Feeds goal G4 in `../kernel-final-design/KERNEL_REFACTOR_PLAN.md`.*

---

## 1. The question

Three candidate shapes for replacing the subsystem:

- **Parallel** — a new node type + forked build flow; kernels never go through the
  HWCustomOp interface.
- **Integrated (naive wrap)** — retain `HWCustomOp` as the base class, delete the
  HLS/RTL mixins, put the engine inside.
- **Adapter** — satisfy the HWCustomOp *interface* the flow calls, without inheriting
  the HWCustomOp *implementation*.

The consumer census (`consumer-surface-model.md`) settles this — not as a single global
choice, but as a decision **keyed to pipeline depth**. The reason the global framing is
wrong: the census shows the op-contract surface is **tiered and grows monotonically with
pipeline depth** (§3.2 there), and the tiers before and after `specialize_layers` (build
step 6) are structurally different consumers. A strategy chosen once for the whole flow
is answering a question the flow does not ask uniformly.

## 2. The decision: depth-keyed, hinged on step 6

**Share the frontend, replace the selection seam, adapter the backend tail.** All three
candidate shapes are correct — each for the region of the pipeline it fits.

| Pipeline region | Steps | Contract tier | Strategy | Why |
|---|---|---|---|---|
| **Frontend** | 1–5 | Tier 0–2 (base ABC + flat config nodeattrs + indexed dtype getters) | **Integrated / shared** | These operate on the *abstract* HW node. `convert_to_hw` (step 4) reads **source**-op attrs and writes the abstract node directly — it never calls a KernelOp method. Steps 4–5 touch only the base qonnx ABC + config nodeattrs + dtype-from-Context. Old ops and kernels are indistinguishable here; no fork, no duplication. |
| **Selection seam** | 6 | Tier 2 → Tier 4 transition | **Replaced** | `specialize_layers` is the god-switch we already indicted (the `if optype==` feasibility ladder). Our resolve engine already **is** its replacement: feasibility as per-bundle predicates, selection as `resolve`. Not adaptered — deleted. §3 below. |
| **Backend tail** | 7–20 | Tier 3 (estimate) + Tier 4 (full build) | **Adaptered** | A thin adapter projects `resolve`/`emit`/`stitch` onto the legacy method names the tail calls (`get_exp_cycles`, `node_res_estimation`, `code_generation_ipgen`, `code_generation_ipi`, `prepare_rtlsim`, `get_verilog_top_module_intf_names`). A mixed graph works because classic `_hls/_rtl` leaves and new kernels answer the **same** calls. |

The governing principle for the tail (census §4): **the adapter is a projection of the
resolved Point onto the legacy op-instance API — every method answers from
resolved-config-as-data, never by reading state another pass stashed on the node.** The
concrete adapter width is fixed (census §4a): ~30 methods + the two untyped accessors +
one property + five capability flags.

This is not a compromise between the three candidate shapes — it is what the tiered
contract dictates. "Parallel" is never needed for the head (the frontend is genuinely
shared). "Adapter" is never the whole answer (the tail only). And the seam is a
replacement, which is neither.

## 3. `specialize_layers` is the point of complete divergence

Build step 6 is where the design deliberately breaks from FINN, and the census gives
three independent reasons it is **unsalvageable, not adaptable**:

1. **It IS the god-switch.** `SpecializeLayers` is the central `if optype == "MVAU"/
   "VVAU"/...` RTL-feasibility ladder (`hw-backend-model.md` §2) — the exact
   base-knows-subclasses inversion the whole effort exists to remove. Adaptering it would
   preserve the defect.
2. **It is where the contract surface doubles** (census §3.1, R11). Everything left of
   step 6 operates on the abstract HW node (base ABC + config); everything right requires
   a concrete `_hls/_rtl` leaf resolvable to a backend mixin. The op surface roughly
   doubles across this one step.
3. **It does not delegate feasibility to the op.** It reaches into per-op-family knobs
   (`inWidth/outWidth`, `noActivation`, `binaryXnorMode`, `lhs_style/rhs_style`, `narrow`)
   to choose an impl. Our engine already inverts this: `mvau_dsp_packed` carries a
   "requires DSP58 ∧ w≤8 ∧ a≤9" **predicate** the resolve engine evaluates. The device
   feasibility the FINN-prototype had *no home for* is data here.

**The mapping that falls out:**

```
convert_to_hw (step 4)      → produce an UNRESOLVED kernel (design space declared, no Point committed)
steps 4–5 (Tier 0–2)        → operate via base ABC + config nodeattrs + dtype-from-Context (SHARED, no fork)
step 6  specialize_layers   → REPLACED BY resolve(): commit the `implementation` axis → a (partial) Point   ◄── divergence
steps 7–20 (Tier 3–4)       → adapter projects resolve/emit/stitch onto legacy method names
```

**Caveat — separate the selection logic from the mechanical plumbing.** `specialize_layers`
has tendrils: `transpose_decomposition` (step 10) re-invokes it on shuffle sub-ops;
`target_fps`/`minimize` recurse into FINNLoop bodies calling it. Those are node-naming /
SDP-template mechanics, separable from the selection decision. The *selection logic* is
unsalvageable; the plumbing around it is harness cleanup, not a counterargument to
replacing it.

## 4. Data still lives on the node — but the kernel structure is the authority

We keep storing per-node state as ONNX nodeattrs, but access/mutation is abstracted
behind the kernel's typed data structures. This is partly **chosen** (ergonomics,
control) and partly **forced**:

**Forced by R10 (census §4c).** `NodeLocalTransformation` — which drives `PrepareIP`,
`HLSSynthIP`, `PrepareRTLSim`, `PrepareCppSim` — deep-copies the model and **pickles each
`NodeProto` to `mp.Pool` workers**. So per-node state must survive as serialized
nodeattrs, or be reconstructable from them in a fresh process. This yields the invariant
that makes "abstract behind kernel structures" safe:

> **The kernel data structure holds no authority the nodeattrs don't.** It is a typed,
> cached *view* over serializable state, reconstructed as a pure function of
> (nodeattrs + graph + Context) in the worker.

That is exactly the frozen-Point-resolved-from-Context model already built. Honor it and
R10 disappears — no non-picklable handle ever lands on the node.

**Three kinds of nodeattr, three dispositions** (the abstraction is NOT uniform — census
§2, §4b):

| Kind | Examples | Disposition |
|---|---|---|
| **Genuine config** | folding `PE`/`SIMD`, `mem_mode`, dtypes | Point-owned. Kernel structure is the source; the nodeattr is its projection. (Your model, clean.) |
| **Side-channel** (one pass stashes for another) | `cycles_estimate`, `gen_top_module` | Adapter serves the *read* from a **live derivation** (`get_nodeattr("cycles_estimate")` → fresh `get_exp_cycles()`), and accepts the *write* as a harness-cache no-op (R7). If the kernel treats the stashed write as durable, hermeticity breaks. |
| **Ordering / handoff** (filesystem/build pointers) | `code_gen_dir_ipgen`, `ipgen_path`, `rtlsim_so` | Harness-owned. `emit()` returns Artifacts; it does not stash paths on the node. Adapter answers `get_nodeattr` for these from the harness build context. |

So: store on the node, abstract behind kernel structures — yes — but the kernel structure
is a *view*, the nodeattr is the serialization substrate beneath it, and
side-channel/handoff attrs are answered by **derivation, not storage**.

## 5. Why this de-risks the migration (mixed graphs)

Because the frontend is shared and the tail answers a fixed method set, a
**partially-migrated graph** — some nodes classic `_hls/_rtl`, some KernelOp-backed —
rides every step: both kinds present identical Tier 0–3 surfaces through step 6, then
each resolves to something the tail can call. The census names the failure surface
precisely (§4c): the three couplings that are **duck-typed today** — `op_type.startswith`
for weight emit (R2), `hasattr` for bit-width minimize (R3), and pervasive op_type-prefix
dispatch (R1) — will **silently skip** a conforming-but-incomplete kernel (unminimized
resources, missing runtime weights, unstitched IP; no error). The migration rule that
falls out:

> **Convert every silent skip into an explicit capability query before the corresponding
> KernelOp class ships.** (R1 — replacing op_type-string dispatch with capability queries
> — is the single largest structural obligation.)

## 6. Open decision that shapes G4 — resolution staging

The current flow **stages** the design space across steps: `implementation` at step 6,
folding (`PE`/`SIMD`) at steps 7–9, bit-width at step 9. Our engine models all of these as
axes in **one** space. So when `resolve` replaces `specialize_layers`, does it fire:

- **Incremental per tier** — commit `implementation` at step 6, folding at 7–9 as further
  assignments; partial Points, which the guarded-axis engine already supports natively; or
- **Atomically deferred** — steps 6–9 gather config, a single `resolve` fires later.

**Leaning incremental:** it matches the flow's existing staging, keeps the mixed-graph
steps uniform (each step commits the axes it owns), and the guarded-axis / partial-Point
model handles this without new machinery. But it is a genuine fork — it determines what
"replace `specialize_layers`" actually emits (a partially-resolved kernel vs a config
accumulator) — and should be decided before G4 is rewritten. **Deferred to the staging
discussion.**

## 7. What this changes in the plan

To fold into `../kernel-final-design/KERNEL_REFACTOR_PLAN.md` (next pass):

- **G4 is re-specified** from "wrap everything in an adapter" to **"share the frontend
  (steps 1–5), replace the selection seam (step 6, `resolve` supersedes
  `specialize_layers`), adapter the backend tail (steps 7–20)."**
- **M1 re-scopes to the estimate-only tier.** The census isolates a shallow sub-contract
  (through step 11: `get_exp_cycles` + `node_res_estimation`, **no codegen/IP/rtlsim**) —
  `estimate_only_dataflow_steps`. A kernel that resolves + estimates but does not emit
  rides the whole shallow flow. This is a far cheaper, well-defined first milestone than
  "MVAU to bitstream," and it exercises the shared-frontend + seam-replacement without the
  Tier-4 adapter surface.
- **The adapter method list (census §4a) becomes G4's concrete acceptance surface**, and
  the risk register (census §4c, R1–R12) becomes the migration's ordering — R1/R2/R3
  (the silent-skip couplings) gated before any weight-bearing kernel ships; R11
  (`specialize_layers` feasibility delegation) as the seam itself.

*Cross-refs: `hw-backend-model.md` (substrate), `consumer-surface-model.md` (consumer
surface — §3 tiers, §4a adapter list, §4c risk register), `../kernel-final-design/KERNEL_REFACTOR_PLAN.md`
(G4, milestones).*
