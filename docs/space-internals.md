# Maintaining Space

The [author guide](design-space.md) describes the supported language. This guide
explains the implementation boundaries and the contracts to preserve when
changing them. Supported imports come from `finn.core.space` and its public
service modules: `inspection`, `selections`, `codecs`, and `extensions`. Other module paths and compiled records are implementation
details.

## Definitions, prepared models, and snapshots

Space has three distinct lifetimes:

1. **Declarations** describe an authored family. Their identity matters:
   inherited references still identify an effective member after an override.
   Python descriptors preserve the distinction between class references and
   instance values.
2. **A prepared model** owns immutable node, scope, and choice tables. Preparation
   validates the complete definition before publishing the class-local cache.
   Repeated construction of the same root family reuses that model.
3. **A snapshot** owns frozen parameter values, admitted assignments, a lock,
   and a local evaluation cache. Each scoped `Space` object stores that snapshot
   and its scope index. Parent and child wrappers share one snapshot. A successor
   shares frozen facts and the model, but gets independent assignments and caches.

There is no global model registry or placement cache. Discarding dynamically
created families and configurations can release their models, values, and caches.

## Finding the implementation

| Change | Start here |
|---|---|
| Add or understand a declaration/reference | `declarations.py` |
| Change typed instance access or construction | `_configuration.py` |
| Change inheritance and effective member rules | `collection.py` |
| Change callback signatures or annotations | `_signatures.py` |
| Change child suppliers or nested parameter exposure | `_bindings.py` |
| Change preparation caching/finalization | `compiler.py` |
| Change node lowering or generated identities | `_linker.py`, `ir.py` |
| Change cycle detection/order | `_graph.py` |
| Interpret a reference against frozen scopes | `references.py` |
| Bind inputs, read a field, or navigate a child | `occurrence.py` |
| Change atomic choice replacement | `_changes.py` |
| Change node evaluation or public value copying | `_runtime.py` |
| Change native suspension, cleanup, or cancellation | `_execution.py` |
| Change result aggregation | `results.py` |
| Change value recognition, copying, or equality | `semantics.py`, `domains.py` |

The remaining modules implement the public services named above. Codecs consume
private selection entries so that they can validate and detach payloads at the
codec boundary without first creating redundant public entry copies.

## Preparation

`compile_space` invokes one linker, then publishes a `SpaceModel` only after
linking and constructor validation succeed. The linker performs these phases:

1. Collect each reachable family once. Collection builds effective member and
   alias maps, checks overrides, interprets annotations, and records guards and
   exports. It does not evaluate authored computations or domain callbacks.
2. Normalize placement bindings through a preparation-local `PlacementPlans`.
   Guard collection, ownership checks, and allocation consume the same plans.
   Literal snapshot adapters run here to detach definition inputs. They are
   pure value transformations, distinct from computations.
3. Reject recursive family placement, then allocate occurrence IDs and scope
   membership maps. Repeated templates have distinct occurrence IDs. After
   allocation, scope identities remain stable.
4. Apply nested parameter bindings from inner placements to outer placements.
   Only deliberately exposed parameters can be rebound. Each supplier keeps the
   scope where it was authored; a reference alias does not acquire edit rights.
5. Lower members, guards, choices, and integer expressions into the node table.
   A function view gets an explicit raw-output node. Selected exports keep
   potential edges to every case, although execution demands only one case.
6. Order known dependencies, report precise cyclic components, and check linked
   value semantics. Cross-scope types are authoritative at this stage.
7. Validate integer expression operands and freeze the model's indexes. Expressions
   remain ordinary lazy runtime nodes, including expressions over constants.

The graph routine operates on integer adjacency and diagnostic names; it handles
both family recursion and node dependencies without recursive Python calls.
It cannot discover reads hidden inside arbitrary method bodies. Reached dynamic
cycles belong to evaluation.

To add a declaration kind, examine its descriptor, collection policy, allocation
kind, member lowering, `Node.dependencies`, and runtime frame. Add a behavioral
example that exercises its ownership and guarded/inactive behavior. A new class
hierarchy or generic registration system is not required.

## Reading a value

A descriptor delegates to `occurrence.read_value`. That boundary checks the
snapshot and resolves the reference using frozen scope maps. A driver read asks
the evaluator for the node. A read inside an authored callback records a demand
and either uses a cached evaluation or suspends the callback until the node is
resolved. Cache hits still count as observed dependencies.

`_runtime._frame` contains node semantics. It yields a prerequisite node index
or a `Call`, then returns an `Evaluation`. `_execution.run` drives those frames
iteratively and runs authored calls on native continuations:

| Request | What resumes the requester |
|---|---|
| Engine frame yields a node index | The prerequisite's `QueryResult` |
| Engine frame yields `Call` | A returned value, blocked-read outcome, or failure |
| Native descriptor read suspends | The prerequisite's complete `Evaluation` or failure |

The dispatcher and evaluator have a narrow, intentional mutual dependency.
Their local imports implement this protocol; a service registry would add
another mechanism without improving it.

Integer task identities are cacheable nodes. Domain enumeration/membership
operations use unique task identities so their answers cannot overwrite the
decision's cached value. Failures are local to a dispatch run; successful
evaluations belong to the snapshot cache. Public reads detach cached payloads
through their declared semantics before returning them.

## Result precedence

These policies are deliberately separate:

| Context | Reduction |
|---|---|
| Applicability | A false outer guard prevents inner demands and produces inapplicability |
| Required callback inputs | Unresolved, then rejected, then inapplicable; preserve findings of the highest-precedence variant |
| Constraint group | Unresolved, then rejected, otherwise accepted; inactive members do not refuse the group |
| Readiness | Unresolved obligations block readiness; settled refusals remain visible in member results |
| View acceptance | Applicability, readiness, inactive output, refusal, then raw output |

Only constraints and domain predicates interpret a Boolean false as refusal.
A derived Boolean false is an ordinary value. All callback inputs require values.
An omitted optional Param blocks dependent computation; drivers inspect the
unresolved outcome directly. Callbacks can still return semantic results.

Status inspection and configuration revision are driver operations. Inside a
callback, catching an unavailable read or forbidden operation cannot turn it
into a successful fallback. Recognition, equality, and snapshot hooks cannot
read configuration state, even if their code catches the access error.

## Revising choices

Configuration replacement follows this boundary order:

```text
validate all request identities and keys
  -> recognize all supplied values
  -> snapshot all supplied candidates
  -> admit demanded decisions in a private trial
  -> publish only if the complete transaction succeeds
```

Replacement builds a complete revised assignment set, revalidating retained
choices as well as new ones. Clearing a selector does not silently remove its
case's commitments. Every retained and new candidate passes admission in a trial
that starts with no trusted assignments.

The trial's decision frame performs admission before exposing a candidate as a
value. Dependents never receive an inadmissible candidate, even when the relevant
dependency is discovered through a method read. Failed trials do not publish a
successor. Equal updates preserve receiver identity; child revisions return the
same authored child type over the revised root snapshot.

`ChangeRequest` denotes concrete `Change` objects in a heterogeneous batch; it
is not an extension protocol. The batch annotation erases the payload type so
nested factory calls can infer their own `T`. Each `Change[T]` and its factory
retain precise typing.

## Sparse persistence

A Selection owns one model reference and sorted entries containing an owning node
index and a detached value. Capture reads assignments without evaluating unrelated
work. Public entries derive stable keys and handles from the model; internal code
does not retain or repeatedly reconstruct complete DecisionInfo records. Schema
records similarly retain node indices and codecs. Selector keys are authored choice
keys, never generated node names.

Edit configurations and capture the result. Restore requires a root with an empty
assignment set, checks that precondition and model identity before adapters run,
and delegates to ordinary batched replacement. It returns ConfigurationResult.
Queried bases and fully cleared configurations are eligible; configured receivers
are rejected rather than merged or overwritten. Empty replay is an identity no-op.

Codec decode validates the complete document structure before invoking decoders.
Nominal values, portable identity, codec round trips, and defensive snapshots stay
at the boundary. A structurally valid document may contain stale case commitments;
restore must refuse them atomically under the newly bound facts. Portable documents
retain their existing family/version, key, codec/version, and value fields.

## Failure completion

Callback failures return through suspended reads so Python `finally` blocks and
context managers can finish. Cleanup may demand previously uncached values.
The first error stays primary and secondary cleanup errors/nonvalues remain
ordered. Control-flow exceptions retain their original identity, with structured
cancellation details attached.

Task completion includes constructing the result, caching it, closing the engine
frame, retiring the task, removing its active identity, and delivering its outcome.
Interruption between these steps must still drain waiting callbacks. Keep these
steps inside the dispatcher failure boundary when changing them.

`test_native_cleanup.py` injects interruptions at the actual transition lines,
using a shared locator in `_native_support.py`. It does not require production
test hooks. `test_native_execution.py` covers access rules and dynamic cycles;
`test_native_admission.py` covers admission, scaling, concurrency, and reclamation.
The suite-wide fixture checks that evaluation context is restored after each test.

## Validation

The standalone gate needs no QONNX, ONNX, FPGA toolchain, or kernel dependencies:

```bash
python -m pip install -r requirements-space-test.txt
bash scripts/check-space.sh
```

It runs behavioral and typing fixtures, the executable author guide, strict
package/test typing, and Ruff lint/import sorting/formatting. The Space workflow
runs this gate on Python 3.10 and 3.12. Space source and tests use Ruff formatting;
the rest of the repository retains its existing Black/isort configuration.

For kernel consumers and installed-wheel boundaries, run
`scripts/check-kernels.sh` in the repository's kernel environment. It includes
the standalone gate and requires the pinned QONNX and FinnLib checkouts under
`deps`, plus the kernel testing dependencies. FinnLib currently defaults to an
internal fork in `fetch-repos.sh`; the public standalone workflow does not claim
to validate that private dependency or toolchain-specific tests.

Before changing allocation, admission, or dispatch costs, compare
`scripts/benchmark-space.py --suite all --output <report.json>` on the same
machine and interpreter. Its structural work and reclamation assertions are
stronger regression signals than a single timing. Deep explicit/native chains,
wide fan-in, dependent batches, and discarded configuration populations all
exercise different contracts.
