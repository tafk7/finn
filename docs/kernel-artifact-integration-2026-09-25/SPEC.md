# Kernel and artifact integration refactoring

Status: specification for implementation; implementation is pending.

Date: 2026-09-25. Authoring baseline:
`240c33a1163a77b2c4c3f62067134686d19a0715` in `finn-kernels-extraction`.

This specification records the agreed responsibility split and the bounded
refactoring work needed to align kernel authoring and artifact generation.
It is an integration specification, not a replacement for the generic
[Space design](../../../scratchpad/space/DESIGN.md).

The governing principle is **elegant simplicity**: use the existing accepted
view API, ordinary detached values, and the existing artifact machinery. Add
an abstraction only where concrete source-generation or composition work
requires it.

## Outcome and scope

A kernel is responsible for producing a valid build specification. Artifact
generation consumes that specification and produces its declared files, with
content-based caching and deduplication. Neither subsystem imports the other.

```text
Kernel authoring
  Hardware meaning, supplied facts, implementation choices, support rules
                 |
                 | uses generic Space evaluation and accepted views
                 v
       kernel.build_requirements()
                 |
                 | valid, detached build specification
                 v
Artifact generation
  Explicit roots -> frozen inputs -> identity -> cache/generation
                 |
                 v
       Verified files and packages
```

The pass includes:

1. Making the existing accepted-view handoff explicit and testing it.
2. Retaining the current value-semantics policy without a normalization or
   equality overhaul.
3. Simplifying the artifact declaration surface and parameter formatting.
4. Supporting a bounded composition of generated child modules.
5. Sharing source-preparation mechanics between RTL and HLS.
6. Connecting build provenance and diagnostics through an external association.

The baseline includes reusable `CyclicDelivery`, stream contracts, canonical
traversals, and stream composition. Preserve those developments. They do not
remove the artifact lowering restrictions on generated children, copied-source
flattening, or data slots. See the
[composition record](../kernel-composition-2026-09-25/REVIEW.md) and
[stream contract record](../kernel-stream-contract-2026-09-25/REVIEW.md).

## Architectural ownership

| Concern | Owner |
|---|---|
| Partial configurations, dependencies, decision admission, evaluation, and accepted views | `finn.core.space` |
| Hardware semantics, supported configurations, stream meaning, and valid kernel outputs | Kernel authoring and physical composition |
| Internal consistency of a detached build specification | Artifact declaration types |
| Source resolution, template closure, naming, build identity, storage, and packaging | Artifact processing |
| Tool invocation and actual execution results | External execution layer |
| Source occurrence, configuration correspondence, and association with an artifact | Caller or a small kernel-side driver |

`finn.core.space` remains independent of kernels, artifacts, and hardware
datatypes. `finn.kernels.artifacts` remains independent of Space, concrete
kernels, and the physical semantic model. Kernel authoring may use both.

No artifact API may require a live configuration, bound view, Space handle,
assessment, or kernel object. Its inputs are detached values and explicit
content/source services. Any caller using both APIs belongs outside artifacts.

## 1. The kernel owns valid output

The public build view returns a specification satisfying the kernel's
applicable hardware rules, or produces the existing unavailable-result behavior.
The ordinary call is the handoff:

```python
requirements = point.build_requirements()
prepared = prepare_module_build(
    requirements,
    roots=roots,
    template_roots=template_roots,
    blobs=store,
)
artifact = materialize_module_sources(prepared, store)
```

Calling a view returns its accepted value. Drivers using `query()` or `inspect()`
must consume the accepted result. `output_result` remains a diagnostic
intermediate, not the public build product.

```text
View inspection
  Computed output:     a structurally valid recipe can exist
  Kernel constraints:  the configuration can still be unsupported
  Accepted result:     the recipe is returned only when obligations pass
```

Space already supports this behavior. Preserve the following:

- View constraints govern acceptance; they are not universal execution gates
  for the output callback.
- Outputs can become available independently. No global requirement to finish
  every choice or accept every view is introduced.
- Parents consume accepted child products through ordinary view calls or
  existing accepted references.
- Known refusals remain inspectable while other obligations are unresolved.
- A body that cannot construct a value for particular inputs uses a checked
  prerequisite or an explicit rejection path. Artifact exceptions must not
  substitute for expected kernel support decisions.

Artifact processing still validates its own contract: source availability,
declared template inputs, renderer/format compatibility, output layout,
manifest identity, and content integrity. It does not duplicate hardware rules
or infer whether an independently supplied specification came from Space.

Do not add an `AcceptedRequirements` wrapper, acceptance certificate, proof
token, second acceptance pass, or mandatory integration adapter. Independently
constructed specifications remain legitimate artifact inputs.

Use explicit typed view references for generic callers. Shared `ViewKey`
exports may be factored into a kernel-owned module if an actual composition
consumer benefits. A capability registry, automatic discovery by method name,
and a mandatory build view on the `Kernel` base are outside scope. A producer
may be a plain Space such as MVAU.

Acceptance criteria:

- Accepted kernel products reach preparation through the ordinary API.
- A rejected or unresolved build view stops that driver path before source
  preparation or store writes begin.
- Raw diagnostics remain inspectable without becoming the driver's input.
- A specification remains usable after its configuration is discarded, and
  artifact generation also works without constructing Space.

## 2. Keep value handling explicit

The earlier proposal for a requirements-specific Space semantics adapter,
type-sensitive equality changes, deep normalization, and copy avoidance is
**not part of this pass**.

Retain `default_semantics(ModuleBuildRequirements)` and existing defensive
copying. Authors supply values conforming to declared types, including immutable
tuples where the build contract requires them.

Do not introduce implicit list-to-tuple conversion, numeric coercion, datatype
substitution, or a generic value-repair layer. Preserve existing supported input
behavior. A new runtime check needs a concrete contract failure to address;
where warranted, prefer a useful rejection over silent reinterpretation.

Two observations motivated the earlier proposal but do not establish a current
build correctness defect:

- Python equality can equate `True` and `1`, while exact artifact fingerprints
  preserve their types. Artifact keys still distinguish them; no current build
  omission caused by that equality was demonstrated.
- Cached view reads still copy results. A profile confirmed the work, but did
  not establish a bottleneck in the intended application workload.

Do not add `__deepcopy__` shortcuts or change equality in this pass. Reconsider
copy avoidance only with workload evidence and a justified immutable contract.

Canonical artifact encoding remains authoritative for identity. Existing type
tags and ordering rules for unordered option tables remain. Explicit parameter
conversion to an RTL spelling, specified below, also remains. These defined
representation operations do not create a new policy of correcting author input.

## 3. Simplify declarations and canonical parameter formatting

### Lightweight declaration surface

Provide a stable artifact declaration surface for requirement types, source
contributions, renderer contract identities, and small canonical value helpers
needed by authors. Prefer the existing declaration modules over a new umbrella
framework.

Concrete kernels and detached physical declarations must be able to name build
contracts without importing rendering, store, or packaging implementations.
Declaring a rendered source should not require importing `artifacts.build`
solely for its renderer identity.

Processing consumes the same declaration objects. Keep one runtime identity
per type; do not create declaration/processing duplicates. Existing processing
facades may retain compatible re-exports.

### One parameter spelling implementation

Authors currently write a typed parameter table and repeat its string conversion
for `ModuleABIRequirements`. Physical lowering has another scalar formatter.
Provide one artifact-owned public conversion operation and use it consistently
in kernel authorship and module instantiation emission.

An explicit helper from the typed table to canonical RTL bindings is sufficient.
A factory may remove additional demonstrated repetition, but no new parameter
language or builder hierarchy is required.

The operation must:

- Preserve existing supported scalar spellings, including Boolean and enum
  parameters and explicitly authored RTL strings/literals.
- Respect existing parameter-table ordering.
- Let authors supply each parameter value once and derive its ABI spelling
  from that same table.
- Retain consistency checks for independently supplied tables.
- Produce unchanged HDL for existing supported examples.

Do not reinterpret a string as a number or a quoted RTL literal as a Python
value. Complex array/fragment emission remains explicit at the source boundary.

Acceptance criteria:

- Declaring kernel requirements does not transitively load the artifact store
  or source-rendering implementation solely for declaration values.
- Retained import paths resolve to the same public type objects.
- Scalar, array, enum, and Boolean examples retain their emitted spellings;
  inconsistent duplicate representations still fail.
- Existing fixed-source and generated-source examples preserve source bytes
  and identities except for individually documented contract changes.

## 4. Support nested generated-module composition

### Bounded use case

Support a generated component, such as an MVAU wrapper, as the child of another
generated module. Demonstrate reuse of the same generated child by two parents,
alongside existing copied-source children.

This extends the fixed-name/copied-source composition profile. It does not turn
Space's evaluation graph into an artifact build graph.

```text
Accepted child and parent specifications
                  |
                  v
Prepare child source closures and concrete module names
                  |
                  v
Resolve parent's logical child references
                  |
                  v
Prepare parent wrapper and declared source dependencies
                  |
                  v
Materialize child and parent artifacts through the existing store
```

### Required contract

1. Kernels/composers return detached build specifications. Authoring does not
   read source files, render templates, or consult an artifact store.
2. A child with a generated name can be referenced by a stable local identity
   before preparation determines its concrete symbol. Do not publish an
   invented concrete module name and later reinterpret it.
3. Preparation resolves the child symbol from frozen inputs before final parent
   instance text is generated. Prepared child identity is usable without
   rerunning its kernel view.
4. Parents declare the child source/build dependencies they consume. Reuse
   preserves libraries, flags, defines, source order, and content identity.
   Different specializations must not collide by symbol or staged filename.
5. A final parent source package contains its complete declared compilation
   closure. Resolve internal child references before export; the recipient must
   not need the originating store to recover implicitly omitted sources.
6. Changing a consumed child implementation/source changes the relevant parent
   identity. Changing only parent wiring preserves reuse of an unchanged child.
7. Preparation erases checkout locations from prepared identities. Later
   rendering consumes the prepared closure, including required child blobs.

Keep semantic wiring validation in physical composition. Artifact processing
may consume a small detached representation of names, instances, or recipes,
but may not import `physical.structure`, stream semantics, or Space. Reuse
requirements, derivations, references, and source closure code. Introduce only
the data needed by the worked example; no general-purpose DAG framework or
callback-based deferred kernel evaluation.

Record the chosen declaration shape and an end-to-end example in the change
description. New type names and the precise representation of logical child
references are implementation choices, constrained by the contract above.

### Data slots

Never drop a data slot while composing children. This increment requires
generated children over copied/rendered source closures. Until an explicit
composed data-binding contract is implemented, children with data slots remain
a clear refusal. Preserve existing independently supported slot behavior.

Embedded initialization, such as cyclic ROM parameter values, remains a consumed
input. Do not remove it from identity in pursuit of extra sharing.

Acceptance criteria:

- A generated MVAU child nests in a parent wrapper and elaborates with its full
  source closure.
- Repeated use reuses the child definition; distinct specializations coexist
  without naming or staging collisions.
- Parent-only changes reuse children; consumed child changes affect parent keys.
- Missing dependencies, symbol/metadata collisions, and unsupported data slots
  are diagnosed before incomplete publication.
- Output remains usable after checkout-facing inputs disappear, provided its
  declared blobs and referenced artifacts remain available.

## 5. Share RTL and HLS source preparation

Keep distinct output contracts:

- `ModuleBuildRequirements` describes a known RTL module interface.
- `HlsSourceRequirements` describes C++ sources and an HLS function interface.
  It does not establish synthesized RTL pins, latency, or resources.

Factor the common preparation actually needed by both: resolving copied
contributions, freezing source/template bytes, recording consumed arguments and
compile metadata, validating closure, declaring outputs, and using the store.
Language-specific interface and realization information stays separate.

```text
RTL module requirements             HLS function requirements
           \                         /
            +-- common source preparation --+
                            |
                  Frozen content and recipes
                            |
                  Source artifact materialization
                            |
                  Representation-specific next step
```

The HLS path gains a prepared value and derivation before source rendering.
Rendering then uses frozen blobs rather than rereading a checkout. Include paths,
language standards, source order, and declared template inputs reach the identity
of the stage that consumes them.

Use the existing self-contained renderer rules where applicable. Any necessary
difference must be explicit in the renderer contract; do not retain a less
constrained HLS path simply because its current helper reads templates directly.

Preserve `render_hls_sources()` as a thin convenience function where practical,
delegating to the shared implementation. Do not maintain two independent
definitions of HLS source generation.

HLS synthesis, installation discovery, scheduling, and extraction of synthesized
RTL interfaces remain outside this pass. A universal requirements class with
optional fields for every representation is unnecessary.

Acceptance criteria:

- MemStream HLS uses shared preparation and retains its C++ interface, complete
  header closure, relative include layout, and source bytes.
- A prepared bundle renders after its original roots are unavailable.
- Source, template, argument, and consumed compile-setting changes invalidate
  the appropriate artifact; checkout relocation alone does not.
- Store reuse and declared-layout validation apply to HLS source artifacts.
- RTL retains known ABIs; HLS does not acquire guessed RTL interfaces merely
  to satisfy a shared API.

## 6. Associate explanations and builds externally

Provide a small caller-owned association between an accepted product and the
artifact produced from it. Use existing evidence and fingerprints where useful:

```text
Caller occurrence + accepted view explanation
                       |
                       v
             Exact requirements fingerprint
                       |
                       v
             Prepared-input fingerprint
                       |
                       v
                Artifact reference
```

A record may contain a caller-supplied origin, a view label, a detached summary
of acceptance/dependency evidence, and the applicable requirements/prepared/
artifact references. It records correspondence; it is not an acceptance token.

Record construction belongs outside artifacts and uses public Space inspection
APIs. Artifact generation does not need an association to operate. A standalone
specification may omit Space evidence entirely.

Keep the implementation small: a detached record, a construction/example path,
and an explicit caller-controlled persistence representation suffice. Do not
add a database, global registry, automatic event tracking, or serialized
evaluator graph. Persist selected values and labels, not Space handles,
callbacks, compiled models, or configurations.

Support many origins for one artifact and multiple products from one origin.
Occurrence labels, evidence, selections, and timestamps must not enter canonical
artifact manifests or build-key preimages.

Keep the identities and diagnoses distinct:

- Space evidence explains an evaluation's reads/obligations; it does not
  establish a filesystem/template/tool input closure.
- An artifact derivation identifies a declared build; it does not prove kernel
  support or record an entire design configuration.
- Saved selections omit parameters; they are not complete build requests.
- Requirements fingerprints, prepared fingerprints, source keys, and output
  digests retain separate purposes. No universal configuration hash or shared
  Space/artifact cache is introduced.

Kernel refusal, unresolved choices, missing sources, invalid templates, corrupt
store content, and execution failures remain distinguishable. A caller can add
origin/view context to a build error while preserving its stage and cause.
Do not translate every build failure into a kernel rejection.

Acceptance criteria:

- Different callers can retain different associations for the same artifact
  without changing its key, files, or canonical manifest.
- A persisted association explains the accepted-product/artifact correspondence
  without retaining its configuration.
- A missing source can be attributed to its originating request while retaining
  its artifact-stage error and cause.
- Generation remains functional with association recording disabled or absent.

## Compatibility and baseline preservation

Before implementation, record actual source/dependency revisions and capture
representative outputs. The authoring baseline may have advanced; preserve
unrelated work and account for new accepted behavior.

Preserve the Space API, result precedence, ordinary view calls, choice ownership,
optional-input behavior, sparse selections, and copying policy. No generic
engine change is required by this specification.

Preserve scalar/port admission, stream contracts, traversal meaning, padding,
clock/reset relations, MVAU external/cyclic behavior, and reusable delivery.
Do not replace semantic compatibility with equal widths or matching filenames.

Declaration cleanup preserves public type identity and existing preimages.
Requirement/contribution classes currently carry compatibility `__module__`
values used by some fingerprints. Moving a file must not accidentally change
those identities.

Use explicit producer/schema versions for new composition/HLS contracts. Where
an intentional change affects an existing name, descriptor, key, or manifest,
record its cause and expected scope. Do not reuse entries validated under a
different completion contract.

Physical instance names and module symbols remain inputs where consumed.
Diagnostic occurrence names are external provenance. Excluding provenance does
not authorize removing meaningful physical names from source identity.

## Implementation sequence

1. **Baseline and handoff:** capture representative outputs, document public
   build-view responsibility, and add focused independence/handoff tests.
2. **Declaration cleanup:** expose lightweight contracts and one parameter
   spelling operation; migrate authors/lowering and prove existing parity.
3. **Shared preparation:** extract common RTL/HLS mechanics, migrate HLS, and
   preserve the distinct product types.
4. **Nested composition:** add the smallest declaration extension required by
   the generated-child example; use shared preparation and the existing store.
5. **External association:** connect accepted-product evidence and artifact
   references through the small caller-owned record.
6. **Integration review:** update examples and record compatibility, source-byte,
   identity, type-checking, and applicable hardware evidence.

These are reviewable increments of the same pass. Sections 3-6 are implementation
deliverables. The value-semantics overhaul, automatic discovery, composed data
slots, and tool execution are not.

## Verification and completion

Use meaningful integration examples rather than tests that only restate helpers.

| Area | Required evidence |
|---|---|
| Independence | Import boundaries; artifacts without Space; detached products without their configurations |
| Acceptance | Supported, rejected, and unresolved views; useful raw diagnostics; accepted child propagation |
| Existing behavior | FIFO, dotp, MVAU external/cyclic, reusable delivery, stream composition, and MemStream HLS |
| Declarations | One type identity; lightweight imports; parameter spellings; representative unchanged source bytes |
| Composition | A real generated child; reuse/specialization; full closure; symbol and metadata refusal cases |
| HLS | Frozen preparation, root-independent rendering, identity sensitivity, store reuse, complete headers/includes |
| Associations | Multiple origins sharing an artifact; detached persistence; stage-specific attribution |
| Packaging | Complete source closure and preserved supported ABI/descriptor round trips |

Run `scripts/check-kernels.sh` with the project's suitable Python, Ruff, and
mypy environment; it includes the independent Space gate. Use focused checks
throughout and verify installed-package behavior after declaration/resource
changes.

Source-generation/composition changes also require appropriate elaboration and
numerical/sequence checks against pinned native sources, including backpressure
where relevant. Record tools, versions, commands, and outcomes. Report unavailable
or failing hardware checks explicitly; source-only checks do not establish
hardware behavior. Prior analysis/pass results are not implementation validation.

Completion requires delivered sections 1 and 3-6, adherence to section 2,
preserved independence, recorded intentional compatibility changes, updated
examples, and validation of the implemented scope. State remaining limitations
precisely rather than representing them as completed features.

## Explicit exclusions

- Space evaluator redesign, global validity, or changed acceptance order.
- Acceptance wrappers, certificates, or provenance-based authorization.
- Hidden author-value normalization, equality overhaul, or copy optimization.
- A universal build-spec class, generic build graph, or capability registry.
- Automatic insertion of new stream adapters or new hardware support policies.
- Vendor execution backends, HLS synthesis, or external scheduling.
- Porting parked dataflow/graph consumers or changing nodeattr persistence.
- Unrelated store repairs and native RTL changes without evidence that they are
  necessary for the specified integration behavior.
