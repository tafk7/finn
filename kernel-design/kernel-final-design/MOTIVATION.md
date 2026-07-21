# Why Replace the FINN HW Backend — Motivation

*The case for the KernelOp effort: what is structurally wrong with the current
`HWCustomOp` + `HLSBackend`/`RTLBackend` system, and what the two real prior
systems (the FINN kernel prototype on `feature/kernel_flow`, and brainsmith) prove
about the fix. This is the stable "why"; the "how" lives in
`KERNEL_REFACTOR_PLAN.md`. Grounded in the as-is analysis
(`../finn-hw-backend-analysis/`), `../../kernel-prototype-profile/`, and
`../../brainsmith-profile/` + `../../compare-brainsmith/`.*

---

## 1. The problem: the abstraction does not model what varies

FINN's HW backend works, and ships real silicon. But its structure fights every
change we now want to make — multi-param folding, new memory topologies (MLO,
fetch-weights), transformer ops, clean composition. The problem is not bugs; it is
that **the abstraction does not model the things that actually vary.** The as-is
audit (`hw-backend-model.md`, a per-op census of all 27 op families: 174 confirmed
hack claims, 331 hacks, 106 hermeticity violations, 0 refuted) pins this down. Five
findings define the motivation — four about axes the model lacks, one about the
conventions it has but never declares or enforces.

### 1.1 The specialization axis is mis-named — it models language, not microarchitecture

A concrete op is `(OpBase, Backend)` — the agnostic op mixed with `HLSBackend` *or*
`RTLBackend` — and specialization (`specialize_layers`) picks the backend. On its
face this models *what the op computes* × *what language emits it*. But **the
language axis is a proxy for the wrong thing.** The decision that actually matters
at specialization is *which microarchitecture* realizes the kernel — a distinct
hardware design with its own resource/latency/feasibility profile. HLS-vs-RTL is
just one microarchitectural fork that happens to align with a language boundary; the
model mistakes the incidental (language) for the essential (microarchitecture).

The tell is what happens when a single language holds *more than one*
microarchitecture. MVU's RTL is not one design — it ships a soft-vectorized core
(`mvu.sv`) and a DSP58 INT8-packed core (`mvu_vvu_8sx9_dsp58.sv`), genuinely
different microarchitectures with different feasibility (the packed core needs DSP58
∧ w≤8 ∧ a≤9). Because the 2-axis model has no slot for "which microarchitecture,"
that choice is **buried in a hardcoded `generate genINT8/genSoftVec` fork inside one
shared wrapper** (`mvu_vvu_axi.sv`), which duplicates `mvu.sv`'s own lane-count math
to intercept the packed case (the source's own `@todo`). A real design axis is
demoted to an addendum branch because the taxonomy can't name it. (Our own model
corrects this: the axis is `implementation` — the realization pool — with `language`
a *derived* property of the chosen microarchitecture; see `design-space-model.md`
§1.2.1.)

And there is a second axis the model omits entirely: **weight/parameter delivery** —
`mem_mode ∈ {embedded, decoupled, external, dynamic, MLO fetch-weights}` — orthogonal
to both compute and microarchitecture, smeared by hand across every method that
should be delivery-agnostic. Together these two gaps (unmodeled microarchitecture,
unmodeled delivery) drive a **4–6× override tax**: the nominal contract is ~12
methods; MVAU overrides 40, VVAU 38, Thresholding 30, almost all of it re-branching
on an axis the abstraction cannot name (`hw-backend-model.md` §5).

You feel this directly as the RTL owner: MVAU's `code_generation_ipi` is ~236 lines
of imperative Tcl in the *agnostic* base that branches on `mem_mode`, hardcodes pin
names, calls `instantiate_ip()` (which exists only on the leaves), and splices the
memstream weight-streamer in by hand. The op that *produces* that 270-line IPI is
exactly the op whose delivery axis has nowhere to live.

### 1.2 The substrate carries op-specific knowledge

`HWCustomOp.generate_hdl_{memstream,fetch_weights,dynload}` live *above* the
op/backend split, yet branch on a hardcoded op-type allowlist
(`["MVAU_hls","MVAU_rtl","VVAU_hls","VVAU_rtl","Thresholding_hls"]`,
`hwcustomop.py:310`) and call `calc_wmem`/`calc_tmem`/`ram_style` — methods that live
only on the concrete subclasses. This is a deliberate maintainer choice: it lets the
shared substrate host common weight-delivery HDL generation across the ops that need
it, part of the same specialization progression the backend mixins serve. It is a
reasonable trade *for that model* — but it is a trade. The cost is that the substrate
is no longer op-agnostic: adding, renaming, or removing a weight-bearing op means
editing the shared base, and a new op that isn't on the allowlist is silently skipped
rather than served. The census ranks this the largest blocker to clean substitution
precisely because the coupling is structural, not accidental.

The selection layer (`specialize_layers.py`) carries the same pattern: a central `if
optype == "MVAU"/"VVAU"/...` ladder that reaches into each op's private node
attributes to decide feasibility. Again intentional — it centralizes the
specialization decision — and again the same trade: per-op knowledge lives in a
shared switch rather than in the ops, so the two copies can drift (the inline
`bitwidth() >= 4` pre-gate vs `_mvu_rtl_possible`'s `>= 2`).

**We reject the trade, not the intent.** The specialization progression is a real
goal; the way to reach it without op-specific substrate knowledge is to make the
design space declarative — each op *carries* its own feasibility as data (a
Predicate), and the substrate resolves rather than branches. That keeps the
progression the maintainers wanted while making the substrate genuinely op-agnostic.

### 1.3 One method name, two contracts (`execute_node`)

The functional/golden reference and the backend rtlsim-execute collide on the single
name `execute_node` under Python MRO. Every clean op family patches it with a
hand-written re-dispatch shim (the census counts ~21). It also *hid a real
correctness bug*: VVAU's `rtlsim_multi_io` was mis-indented inside the per-input
loop. When two different contracts wear one name, bugs hide in the ambiguity.

### 1.4 Codegen is inelegant: mutable side-channel, file duplication, no source model

Two problems compound here — a contract problem and a source-management problem.

**The contract is a mutable side-channel.** The four HLS abstracts
(`global_includes`/`defines`/`docompute`/`blackboxfunction`) look like pure
functions but each mutates a shared `self.code_gen_dict` that an orchestrator
populates *in order* then clears. The real contract is "four ordered side-effecting
steps," un-declarable and un-testable in isolation. HDL is produced by untyped
`$KEY$` `str.replace` — which silently no-ops on a renamed token. With pervasive
ambient state (34 `FINN_ROOT` reads, `finnxsi` as an import-time singleton), no op
can be built, faked, or tested in isolation.

**The source model is worse than the contract.** Each RTL op welds itself to exact
`finn-rtllib/<subtree>` paths and literal filenames, `shutil.copy`s the same shared
`.sv` into every node's codegen dir, and duplicates the source list 3× per op with
inconsistent ordering; the `os.listdir`-scan variants pick "the last matching file"
non-deterministically. There is **no first-class model of a source file, its
provenance, or how sources compose** — sources are ad-hoc strings copied around. This
is exactly what our exploratory MVU-wrapper split (`../finn-mvu-split/`,
`feature/mvu-wrapper-split`) ran into: physically splitting the monolithic
`mvu_vvu_axi.sv` into a shared `base_head/base_tail.svh` + per-microarchitecture
`packed.sv`/`softvec.sv` (the §1.1 fix in RTL form) immediately needs proper source
tracking and composition — which core → which shared includes, per resolved
Implementation — that the current copy-strings model cannot express. Our typed-emit
foundation begins to address this (Artifacts as data, not copied files), but a full
source/provenance/composition model is **long-tail work** the MVU split surfaces and
does not yet finish.

### 1.5 The standards are silent, unenforced, and undocumented

The findings above are about axes the model *lacks*. This one is about the
conventions it *has* but never writes down or checks. FINN's real contract is far
larger than its declared one — the three ABCs name ~15 methods, but the load-bearing
interface is those plus ~15 undeclared host methods/attrs, several defined only on
the concrete leaf (`hw-backend-model.md` §2: "the undeclared host-object interface").
The ABCs describe roughly half the true interface, and **the missing half is exactly
the part that carries the coupling.** A contributor learns the real contract by
tracing call sites, not by reading a declaration — and because nothing enforces it,
an op can quietly violate it with no error. Three concrete standards, each silent:

- **Datatypes.** The declared contract is "return the port's `DataType`." In practice
  `hwsoftmax.get_output_datatype` hard-returns `FLOAT32` and ignores its argument;
  LayerNorm's datatype attributes are *decorative* — the RTL wrapper hardcodes
  `[31:0]` on both TDATA ports and the HLS side hardcodes `TO=float`, while
  `get_instream_width` computes from the attr, so fp32 is fused across three layers
  with **no single source of truth** and the declared width contract is silently
  overridden by the emitted hardware. Nothing flags the divergence. (Our fix makes the
  single source of truth *per fact*, but that source has three distinct **provenances** —
  the split matters and is spelled out in `kernelop-tensor-block-stream.md` §5.2: input
  dtypes are GIVEN by the graph; a param dtype is a graph-GIVEN ceiling the kernel may
  narrow only when the delivery topology makes it static; accumulator/output dtypes are an
  op-DERIVED envelope (a function of the givens and the BLOCK reduction length) that a
  backend may OVERRIDE when its microarchitecture clamps it. One source per fact, but
  given-vs-derived-vs-realized are different owners.)
- **Input ports / the `ind` parameter.** Nominally `ind` selects a port index. But its
  *meaning* is per-op: on MVAU `get_instream_width(1)` is the weight stream; on
  Requant `get_instream_width` returns 0 for `ind != 0` purely to signal "this input
  isn't actually a stream." The same parameter is overloaded into "which port" and
  "is this a stream at all," and which reading applies is undocumented — you infer it
  from the op you happen to be in.
- **Interface descriptors.** `get_verilog_top_module_intf_names()` must return a dict
  with an exact key set (`clk`/`rst`/`clk2x`/`axilite`/`aximm`/`m_axis`/`s_axis`/
  `ap_none`) whose stream entries are positionally-indexed `(name, width)` tuples —
  an "undeclared structural sub-contract" (`consumer-surface-model.md` §1) that
  stitching depends on verbatim. Return the wrong shape and you get a mis-wired block
  design, not an error.

**Weight/parameter delivery (§1.1) is the paradigm case of this failure too**, not
just of the missing axis: even where the `mem_mode` convention exists it is a silent
standard threaded by hand through ~12 sites, with no declared, checkable definition of
what a *valid* delivery configuration is — so an incompatible combination is emitted,
not rejected.

Our fix makes the standard the machine-checkable artifact: the design space (Axes /
Derived / Predicates) **is** the declared, enforced, documented contract. `resolve`
either produces a legal Point or an *explained* illegality; datatypes are `Derived`
with one source of truth; ports are typed with a declared role and index (see
`port-taxonomy.md`); an illegal parameter-delivery combination is a Predicate failure
with a reason, not a silently mis-emitted design.

### The through-line

All five are the same shape from two directions. Where the hardware *varies*
(microarchitecture, parameter delivery, feasibility), the model lacks an axis to name
it, so the variation is hand-coded into a method, a shared switch, or a `generate`
fork. Where a convention *does* exist (datatypes, port indexing, interface shape,
delivery config), it is silent and unenforced, so it is learned by tracing call sites
and violated without error. Both reduce to one thing: **the design space of a kernel
is real but never made explicit** — neither declared as data nor checked by the
machine. None of this is carelessness; each is a defensible choice for the model the
maintainers built. That single observation is the whole motivation; everything in the
plan follows from making the design space an explicit, enforced, documented artifact.

---

## 2. What "good" looks like — evidence from two real systems

We are not theorizing about the fix. Two *real, shipping* systems each solved a
different half, and the profiles tell us exactly what to keep and what to avoid.

### 2.1 The FINN kernel prototype — the structural fix, and the trap

The prototype (`feature/kernel_flow`, `src/finn/kernels/`) replaced the god base
class with a **frozen value object** — single inheritance, zero abstract methods,
zero op knowledge (`kernel.py` greps clean for
`op_type|mem_mode|memstream|execute_node`). Against the top-10 pressure points it
retired ~4 outright and credibly plumbed the rest (`prototype-vs-baseline.md`).

**The strength — weight delivery became a first-class composed subkernel.** A
standalone frozen `MemstreamRTL` owns its own HDL and is injected *only when* the
topology calls for it — and the proof it is a genuine shared axis, not MVAU-private,
is that **the same class is reused verbatim by Thresholding**. The 236-line
branch-on-`mem_mode` IPI becomes composition + config. This is the §1.1
parameter-delivery axis finally given a home. The `execute_node` collision, the
mutable side-channel, and the hidden `vivado` third backend all become *structurally
impossible* rather than patched.

**The limitation — it deleted capability rather than modeling it.**
Device-dependent RTL feasibility (`_mvu_rtl_possible` DSP48-vs-DSP58, Versal) had
*no home* in its constraint contract (a bare `Callable[[Kernel],bool]` with no
`fpgapart`), so it was **dropped, not moved** — MVAU shipped with no RTL variant at
all. MLO/fetch-weights was never ported. Its selection `cost_fn` was stubbed;
viability was discovered by try-construct-and-`print()`. (The stubbed `cost_fn` now has
a resolved answer: cost is a three-tier structure — an engine generic default
(trip-count product over resolved stream shapes), an op-level `cost_model` needed only
where the BLOCK reduction structure makes that default wrong (MVU's product-across-
reduction), and a per-Implementation override for microarchitecture deviations
(pipeline fill, drain); see `kernelop-tensor-block-stream.md` §5.2.) And by walking away
from `HWCustomOp` entirely it fell **out of the FINN ecosystem** — the 20-step build
flow was never integrated. The lesson: composition-as-a-value-object is the right
skeleton, but **feasibility and the design space itself must be modeled data, or
they get thrown away** — and a clean break that abandons the build flow forfeits the
migration path.

### 2.2 Brainsmith — the declarative fix, and the inherited leak

Brainsmith (`KernelOp(HWCustomOp, ABC)`) took the other route: keep FINN's
substrate, but **derive** behavior from a declared schema instead of coding it per
op. Its decisive, shipping-code contributions (`four-way-synthesis.md`):

- **The entire 8-method folded-shape/width contract is derived once, generically,
  from declared data.** A kernel author overrides one `build_schema()` and writes
  *zero* folding code.
- **Legal folds are a derived fact**, not hand-written divisibility: the
  `divisors(gcd(dims where a token appears))` rule, with a shared token across
  interfaces automatically becoming one coupled knob. This *unifies* fold math that
  even the paper "ideal" left per-op. The folding model underneath this — brainsmith's
  **TENSOR → BLOCK → STREAM** hierarchy — is the mechanism our G1 folding rides
  (`kernelop-tensor-block-stream.md`): TENSOR from the graph, BLOCK = how the op's
  *math* segments the tensor (the reduction quantum), STREAM = elements/cycle.
- **Design-space exploration is a first-class kernel contract** —
  `design_space`/`design_point`/`get_valid_ranges` + navigable ordered parameters +
  sweeps. **No other system models this at all.** For a transformer-DSE future this
  is the single most valuable capability in the whole comparison.

**The limitation — it inherits the leaks it can't dissolve.** Because it *retained*
`HWCustomOp`, it drags the substrate defects along (base-class leak, the
`execute_node` diamond, `code_gen_dict`), its escape hatch stays reachable (RoPE
bypasses to raw HWCustomOp), and its backend-viability check is *still* a per-op `if
optype==` god-switch — it declared the DSE space but not backend feasibility. It
dissolves the derivation-layer defects while inheriting the substrate ones.

**One place we deliberately *reject* brainsmith's design, not just its substrate.**
Brainsmith's schema OWNS the folding/tiling and datatype resolution (`build_schema` on
one shared object; the backend is a subclass that inherits it). We invert that: because
our backends are pool members (not subclasses), and because **STREAM tiling IS the
BLOCK→STREAM lowering — an RTL realization detail, not op math** — tiling and
value-optimized datatype narrowing are **Implementation-owned**, while the op owns only
the BLOCK structure (reduction topology) and the datatype *contract*. This is the
BLOCK=op / STREAM=backend line, developed in `kernelop-tensor-block-stream.md` §5.1–5.2.
We keep brainsmith's declarative, DSE-rich *body*; we move tiling ownership down to the
backend where the pool model puts it.

### 2.3 The synthesis that defines the goal

The two real systems are complementary halves of one better system, and the four-way
synthesis states the convergence directly: **brainsmith's proven, declarative,
DSE-rich *body* on the prototype's inheritance-free *skeleton*.** The single most
valuable lesson, in the synthesis's own words:

> Design-space exploration belongs *inside* the kernel abstraction, and structure is
> better declared than coded.

That is the thesis, in one sentence:

> **Make the design space of a kernel — every legal way it can be built, and the
> composition of a kernel with its parameter-delivery and container sub-kernels —
> first-class declarative DATA, resolved against a device context into legal build
> points, on a substrate that carries zero op-specific knowledge.**

Everything in §1 falls out of that: microarchitecture and parameter delivery become
named axes (coordinates in the space, not `generate` forks or `mem_mode` smears);
op-specific knowledge leaves the substrate (each op carries its own feasibility as a
Predicate, so the substrate resolves rather than branches); the `execute_node`
collision is gone (reference and emit are separate slots); codegen is typed data with
a real source model (not a mutable side-channel over copied files); and every
convention becomes machine-checkable (the space is the declared, enforced contract —
`resolve` returns a legal Point or an explained illegality, so a silent standard can
no longer be violated without error).

**But — the prototype's cautionary tale is that the skeleton alone is not enough.**
Brainsmith keeps the ecosystem and pays with inherited leaks; the prototype sheds the
leaks and pays by leaving the ecosystem. The plan (`KERNEL_REFACTOR_PLAN.md`) takes
the third path: the clean skeleton *and* the build-flow integration — an adapter over
a knowledge-free substrate, so we get brainsmith's interoperability without
brainsmith's inherited base.

*Cross-refs: `../finn-hw-backend-analysis/hw-backend-model.md` (the substrate
census), `KERNEL_REFACTOR_PLAN.md` (goals, integration strategy, staging).*
