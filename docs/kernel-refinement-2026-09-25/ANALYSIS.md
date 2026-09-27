# Kernel refinement: initial analysis, 25 September 2026

The subsequent implementation is described in [REVIEW.md](REVIEW.md).
The findings below retain the source state of the initial analysis.

The next pass should make each kernel a truthful, inspectable hardware building
block, using the refined Space engine to expose distinctions that still live in
RTL generate branches, Python procedures, or prose. It should also improve the
hardware where that gives the boundary a simpler and stronger contract.

This is an analysis and a small set of executable probes, not a production
refactor. FINN was inspected at `dc7980873`; the adjacent FinnLib checkout was
at `f774412`. Production Python, RTL, and HLS were not changed. Existing
untracked FinnLib experiments were left intact.

The starting architecture is sound. Kernel identity does not imply a stream;
Space owns configuration and acceptance; detached requirements own the artifact
handoff. Dotp owns a physical reduction engine while MVAU supplies folding and
bounded arithmetic. Opaque transport words remain distinct from numerical
datatypes. HLS source interfaces remain distinct from synthesized RTL ABIs.
These are useful boundaries to preserve.

The criteria come from [Space's design](../../../scratchpad/space/DESIGN.md)
and the [dataflow motivation](../../../scratchpad/dataflow/MOTIVATION.md):
implementation identity, parameter delivery, feasibility, and source composition
should have explicit owners. This work can address those concerns without
introducing DataflowOp, Regions, graph transformations, or graph scheduling.

**1. Establish the actual FinnLib dependency first.**

The new kernel package already targets FinnLib, rather than importing the legacy
HLS/RTL libraries. However, its declared sources target the reorganized
`dfeafac81cd2a6da27e647ee03915ade5532186e` revision recorded in the earlier
[authoring results](../kernel-authoring-pass/RESULTS.md). There is no `deps`
directory in this worktree, and that revision is absent from the adjacent
FinnLib Git object database.

The probe found 15 distinct literal FinnLib source paths in the kernel modules;
all 15 are absent at their declared paths in the adjacent checkout. Fourteen
have same-basename files under its flat `rtl/` or `hls/` directories.
`rtl/infra/replay_buffer.sv` has no such counterpart. Same basenames do not
establish compatibility.

Consequently, simply passing the adjacent checkout as `roots["finnlib"]` cannot
build these kernels. Updating path strings alone would still leave replay
unresolved and would silently mix the older local corrections with another
source revision. The first implementation step should explicitly select and
record a compatible revision/layout, or perform a deliberate port to the
adjacent one. The artifact layer already fingerprints source bytes; that
prevents stale artifact reuse but does not prove the Python contract agrees
with a new RTL revision.

Local `resources/dotp_axi.sv` and `resources/axilite.sv` carry pinned corrections.
The adjacent dotp wrapper still has the older output-capacity expressions.
Those corrections should eventually belong to FinnLib and its regression
suite. Avoid a growing collection of indefinitely forked library modules in
the Python package. Source-set declarations can be shared per hardware
component while keeping the existing explicit roots and dependency closure;
there is no need for an operation-name registry or filename-search fallback.

**2. Make architecture selection explicit, and keep hardware safety local.**

[`dotp.py`](../../src/finn/kernels/dotp.py) is the highest-value example.
It exposes pumping but the wrapper selects between soft-vectorized `dotp`
and packed `dotp_8sx9_dsp58` from widths, target, and a repeated lane-count
formula. The choice affects legal types, segmentation, pipeline behavior, and
resource use. A single `VERSION` parameter does not identify that choice.

One concrete consequence is the rejection of `UINT9` activations with small
signed weights on DSP58: the automatically selected packed path cannot provide
the ninth sign bit. A separately selectable soft-vectorized path is a plausible
way to admit these values, since its DSP B port is wider. This needs a numeric
hardware probe before claiming support; the current rejection should remain
until then.

I recommend explicit compute implementations, either separate native adapters
or an explicit implementation parameter, with one Python family able to select
between them. `SubspaceChoice` and typed exported views fit that family once
both implementations have their own real contracts. Pumping belongs to the
chosen compute/clocking configuration. Segmentation should be applicable to
the architecture that implements it. Currently positive segment lengths are
preserved even when ignored by DSP48, and DSP58 validation also covers cases
whose selected soft-vectorized core does not use segmentation.

This does not mean moving lane slicing, pipeline registers, credit counters,
or backpressure machinery into Python. In particular, the output wrapper must
remain safe for every legal input frame sequence without relying on a graph
scheduler's promised average rate. The local correction already reserves
capacity for pipeline outputs and delayed backpressure. Consolidate the
selected core's timing facts and its wrapper's capacity calculation within
FinnLib, for example through shared SV elaboration functions. Python may expose
a checked projection for inspection; it should not independently invent the
capacity required for correctness. A tighter workload-dependent buffer can be
a later explicit profile with enforceable assumptions.

`NARROW_WEIGHTS` should stay disabled unless a value-range contract proves the
excluded minimum never occurs. That fact does not follow from an `INTn` dtype.
An immutable initializer can supply evidence; an arbitrary external weight
stream needs a stated caller obligation and suitable validation.

**3. Consolidate datatype policy independently of transport.**

The QONNX value boundary is valuable: it preserves canonical identity,
distinguishes TERNARY from INT2, detaches mutable objects, and does not collapse
floating formats into widths. Keep it.

The reusable policy layer is incomplete. `datatypes/domains.py` provides
integer constraint factories, mostly reached through the AXI declaration,
while dotp, eltwise, thresholding, conversion, and memstream repeat integer
family and width checks. `_local_domain` in `physical/axi_stream.py` has an
`isinstance(Integer)` special case to rebind dynamic bounds. A growing scalar
type library should not require edits inside the stream implementation.

Also, the current `DatatypeDomain` is a factory for view constraints, not a
Space decision Domain. These have different roles:

- Supplied dtypes are facts, checked by constraints on accepted products.
- Chosen dtypes are decisions, admitted through membership and optionally
  enumerated for exploration.
- Width, signedness, bounds, and C++ representation are consequences.
- Joint arithmetic rules depend on several dtypes, the operation, and sometimes
  the selected hardware architecture.

Use common pure type-policy predicates with adapters for both constraints and
decision domains. Start with the existing ordinary integer profile, signed
integer profile, exact FLOAT32, and their needed unions. Add a reusable scalar
scope only where it gives useful derived facts and accepted scalar views.
This should work for a combinational converter and an HLS memory as naturally
as for an AXI stream. Output dtypes also need policy checks when callers choose
them; input/output direction should not decide who may declare constraints.

Keep numerical range, storage encoding, and arithmetic semantics separate.
Exact bounded reduction, wrap/truncate behavior, threshold input saturation,
and round-toward-zero conversion are not conveyed by bit width alone.
`MVAU.exact_result_dtype` is a useful first reusable bounded-reduction function;
avoid introducing a universal numerical type solver before additional kernels
need it. Likewise, nonstandard FinnLib floating formats require evidence of
encoding and arithmetic agreement, not just matching QONNX names or widths.

The refined engine exposes two concrete authoring gaps:

- `IntToFp32Kernel(INT0)` and `IntToFp32Kernel(UINT0)` raise
  `NativeEvaluationError` while constructing the zero-width `ival` pin.
  They should be ordinary unsupported-configuration results. A view's
  constraints do not act as imperative guards around its output callback.
  Build callbacks should consume accepted scalar/interface products, or
  explicitly return a rejection before constructing an invalid value.
- Thresholding with a supplied BIPOLAR input cannot report its known type
  rejection until `use_axilite` is chosen. Its monolithic support function
  reads that decision first. Split independent shape, dtype, table, bias,
  memory, and configuration checks so their evidence settles independently.

The probe also accepted `DotpAxiKernel(pe=2**32)`. The native `PE` is an
`int unsigned`; this is outside its representable range. Audit parameter
lowering and intermediate native arithmetic, beyond testing positivity.
Domain admission, semantic constraints, and safe construction each need
their own deliberate place.

**4. Share stream constructs without erasing native interfaces.**

There are currently several authoring forms for similar transport:

| Kernel | Current representation |
|---|---|
| Dotp | Scoped, typed `AxiStreamInterface` |
| Thresholding | Dtype Params and detached `AxiStream` values inside build |
| FIFO, input generator, eltwise | Individual unpadded ready/valid pins |
| Replay and cyclic stream | Unpadded words grouped as `StandardProtocol.AXIS` |
| MVAU | Local `_axis`, padding, field-slicing, and handshake-wiring helpers |
| IntToFp32 | Plain combinational pins, correctly without a stream |

`AxiStream` currently means a homogeneous low-field-first beat rounded to a
byte boundary, with optional one-bit TLAST. That is a useful specific profile.
It should not become the compulsory representation of every native stream.

The common construct underneath should describe ready/valid transfer, its
payload layout, sidebands, clock/reset, and explicit physical signal mapping.
Payloads can be opaque words or typed packed elements. AXI-specific lowering
can then impose byte alignment and supported AXI members. A native 13-bit
word stream remains 13 bits; a multi-bit `olst` remains a loop-completion
marker. Grouping native handshake pins should not silently assert all AXI
packaging requirements.

There are two different kinds of behavior to describe. A port can say that
payload and markers remain stable under stalls and what reset discards.
The kernel must describe relationships across ports: dotp joins activation
and weight transfers; activation last terminates a nonempty reduction;
thresholding joins set selectors when multiple sets are active; replay emits
each sequence several times and distinguishes sequence end from final replay.
These relationships cannot be reduced to a `last: bool` on one interface.

Small detached behavior records, initially tailored to these kernels, are
enough. They need not carry tensor-coordinate maps or a cycle scheduler.
Expose only timing guarantees that are established: combinational dependencies,
initiation interval under stated conditions, fixed latency where applicable,
and minimum guaranteed buffering. Keep measured synthesis/timing evidence
separate. Such records would already support better local composition checks
and test drivers, and later provide a clean boundary for DataflowOp.

MVAU is a good consumer for this abstraction: replace duplicated padding and
manual per-field wiring with explicit compatible-stream connections that lower
to the existing validated `PhysicalStructure`. Crossing encodings, reordering
fields, adapting clocks, or converting marker meanings still requires an
explicit adapter.

**5. Requested configuration is not always effective implementation.**

The FIFO probe compiled and simulated the adjacent native `fifo.sv` with
Vivado 2025.2. With `DEPTH=2`, `RAM_STYLE="ultra"`, and continuously stalled
output, it reported:

```text
effective_style=shift accepted_while_stalled=5
```

This agrees with RTL's forced shallow shift-register path and minimum internal
storage. It is not a newly discovered FIFO bug; the Python docstring already
acknowledges native rounding and overrides. The missing piece is inspectability.
The kernel cannot currently return effective backing or guaranteed capacity,
and several distinct choices describe the same implementation.

Give callers an explicit meaning for the selection: either it is a preference
whose effective implementation is exposed, or it is a strict implementation
choice whose unavailable cases are refused. Keep automatic selection as policy
that resolves to a concrete implementation. Target resource requirements, such
as URAM availability, should be explicit without requiring every kernel to know
a complete FPGA platform. Distinguish native backing selection from what
synthesis ultimately infers.

The same discipline applies to input-generator storage. The current loop-nest
parameters are an appropriate native boundary. Python should lower a caller's
traversal into those parameters and explain footprint implications. The RTL
should own buffer release, wraparound, and refill safety. Its
`INIT_MAX_OCCUPANCY` includes write-ahead requirements, not just a mathematical
live-range minimum; do not replace it with a smaller analytical bound without
testing refill and stalls. No reason has emerged to move arbitrary tensor
lowering into the kernel itself.

Thresholding offers two worthwhile FinnLib repairs already documented by the
earlier authoring pass: complete multi-set configuration addressing and correct
signed output-width/bias arithmetic. Its native narrowing saturation should
remain explicit in the numerical contract. After repairs, replace workaround
refusals and width-emulation code with the intended arithmetic rules and
boundary regressions. Keep parameter-image packing tied to the actual memory
layout. It could later support an explicit initializer artifact as an
alternative to very large inline SV literals.

**6. Complete the local composition and HLS stories incrementally.**

MVAU's child consumes accepted dotp requirements, but `assemble()` is an
ordinary method. The declared `dimensions` constraint group is not attached
to a parent accepted view, and validation is repeated in `_Traversal` during
assembly. Invalid dimensions are still checked there; this is a missed
inspection opportunity rather than a demonstrated invalid build.

An accepted assembly-plan view could expose folding, stream counts,
component selection, and wiring obligations before artifact execution.
External and cyclic delivery can own their own scoped requirements and
initializer obligations. Large weight data can remain an explicit artifact
input; it need not become a choice or drag file I/O into Space. Preserve
`assemble(weights)` as a driver convenience over that accepted plan if useful.
This is physical assembly, not a resumption of the parked dataflow layer.

For HLS, preserve the honest source-versus-RTL distinction. `HlsInterface` and
`HlsSourceRequirements` are currently small string-oriented records, with
control and interface directives also written in the template. Introduce
structured scalar/vector/flit and control descriptions only as additional HLS
kernels need them, and derive C++ spellings/includes/directives consistently.
The MemStream wrapper's start/auto-restart requirement should be machine
readable. A truly free-running deployment top is a possible separate profile;
its AXI-Lite initialization and restart behavior need synthesis and simulation.

FinnLib's HLS dotp is a particularly useful next comparison kernel: it has
separate input, accumulator, and result types, explicit vectors/flits, and a
free-running top. Comparing it with RTL dotp would test the proposed common
numeric and stream contracts while preserving genuinely different arithmetic
and timing capabilities. It should not be introduced merely to create a
generic HLS-versus-RTL selector.

The recommended order is:

1. Resolve and record the FinnLib source boundary; retain/upstream corrections.
2. Repair small admission and diagnostic gaps, including native integer bounds.
3. Factor scalar policy and the native ready/valid stream description, using
   IntToFp32, FIFO, and dotp as contrasting consumers.
4. Make dotp implementation selection explicit and consolidate its RTL timing
   and buffering knowledge. Validate both cores, pumping, and segmentation.
5. Expose an accepted MVAU assembly plan and reuse stream connection lowering.
6. Refine thresholding and add one substantive HLS comparison to test whether
   the abstractions generalize.

Each step should leave a useful, self-contained kernel. A large framework
rewrite before these concrete examples would obscure the questions this pass
needs to settle.

Validation for this analysis: **203 focused Python tests passed**, with the
source-materialization test deliberately deselected because its dependency
checkout is unavailable. The separate native FIFO experiment passed.
No full kernel gate, new dotp simulation, HLS synthesis, or place-and-route
claim is made. Historical validation counts remain historical evidence.

Reproduce the probes with this checkout and a Python environment containing
the refined Space dependencies plus QONNX:

```bash
PYTHONPATH=src:/home/tkeller/prj-kernels/qonnx/src \
  /home/tkeller/prj-kernels/.self-space-greenlet-py310-20260924/bin/python \
  docs/kernel-refinement-2026-09-25/probe.py \
  --finnlib /home/tkeller/prj-kernels/finnlib --rtl
```

The `--rtl` option requires `xvlog`, `xelab`, and `xsim` on PATH. It leaves
tool work products in a temporary directory which is cleaned after execution.
The Python observations and source inventory also run without `--rtl`.

```bash
PYTHONPATH=src:tests:/home/tkeller/prj-kernels/qonnx/src \
  /home/tkeller/prj-kernels/.self-space-greenlet-py310-20260924/bin/python \
  -m pytest -q --confcutdir=tests/kernels \
  tests/kernels/test_datatypes.py tests/kernels/test_axi_stream_declaration.py \
  tests/kernels/test_migrated_simple.py tests/kernels/test_dotp.py \
  -k 'not sources_materialize'
```
