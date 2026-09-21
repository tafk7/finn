Unified Kernel authoring
========================

The experimental dataflow stack uses one domain abstraction for reusable
implementations: ``finn.dataflow.model.kernel.Kernel``. A Kernel is a normal
``Space`` and uses the same declarations, immutable points, nested occurrences,
readiness checks and constraints as every other Space. The framework lives in
``finn.dataflow.model``; ``finn.dataflow.kernels`` is the implementation
library containing concrete leaves and families.

The detached logical value model is under ``finn.dataflow.model.logical``.
Generic physical structure, layout and lowering values are under
``finn.dataflow.model.physical``. Public logical operands are checked references
over actual Region operands, with explicit coordinate maps and optional port
presentations. These lower layers do not import concrete Kernels or source
operations. There is no mandatory cross-view witness or relation capture.

Leaf Kernel
-----------

A leaf declares typed inputs and a ``RegionDeclaration``. The Kernel helper
adds the standard logical and physical views. ``ModuleParameter`` values feed
the detached module requirements without introducing another configuration
object::

   class ExampleLeaf(Kernel):
       id = "example_leaf"
       version = "1"

       width = Input(int)
       region = RegionDeclaration(
           family="example.copy",
           version="1",
           construct=make_region,
           width=width,
       )
       WIDTH = ModuleParameter(width)

       @classmethod
       def component_abi(cls, parameters):
           return ComponentABI("example_leaf", (), (("WIDTH", str(parameters["WIDTH"])),))

``kernel.logical`` returns a ``RegionResult``. ``kernel.physical`` is evaluated
independently and returns ``ModuleBuildRequirements`` when the leaf can be
built. A logical-only leaf states ``physical_unavailable`` explicitly.

Composite Kernel
----------------

A composite uses ``KernelChoice`` and explicit ``NetworkEdge`` and
``NetworkBoundary`` declarations. The same ``Kernel`` base creates its logical
``NetworkResult``; there is no separate Design authoring or runtime layer::

   class ExamplePipeline(Kernel):
       id = "example_pipeline"
       width = Input(int)

       producer = KernelChoice(Subspace(Producer, width=width))
       consumer = KernelChoice(Subspace(Consumer, width=width))

       stream = NetworkEdge(
           producer.output("out"),
           EdgeSink(consumer.input("in")),
       )
       source = NetworkBoundary(producer.input("in"))
       result = NetworkBoundary(consumer.output("out"))

Children may themselves return ``RegionResult`` or ``NetworkResult``. Nested
networks are qualified by use path, so schedules and identities remain
independent. A compatible plain ``Space`` may be used as a child when it exports
``logical_result`` with the standard logical-result value semantics and declares
a ``logical`` projection.

Public operands and focused queries
-----------------------------------

``LogicalView`` and ``PhysicalView`` are named Projection helpers. Each declares
its own output, applicability, readiness and constraint groups. Logical acceptance
checks its relevant graph/interface coherence. Physical acceptance means code
generation is ready under its declared inputs and supported configuration; it
does not claim simulation, synthesis, timing or liveness checks were performed.
An absent or unresolved optional query does not block an unrelated query.

``PublicOperandDeclaration`` associates a stable semantic role with independently
assessed datatype/domain facets and a complete logical export. The full logical
query validates export-to-body type, domain, coverage and presentation agreement.
Narrow facets consume the same declarations and only their actual dependencies.
A required value without a stream port remains an operand without a fabricated
endpoint. Multiple presentations require an explicit stream presentation choice.
Canonical ordered beats and OneToOne pass rules still apply after lowering.

The matrix family exposes activation ``[R,K]``, weights ``[K,N]`` and result
``[R,N]``. Kernel owns any transpose to private ``W[N,K]`` and internal child
qualification. A graph adapter only supplies its own coordinate convention, such
as flattening leading activation dimensions. Changing private child names does
not change the Op operand binding.

Native graph hydration
----------------------

The DataflowOp ONNX graph is the durable sparse design state. ``op.hydrate(model)``
returns the normal bound DataflowOp Space occurrence, with frozen current inputs.
There is no separate use record or bound-Op hierarchy. ``space.resolve_implementation()``
returns the existing Answer, including Unresolved before an implementation choice.
Queries and normal Space successors never alter old points or read live mutable
graph state. ``space.operand_type("result")`` may resolve before unrelated folding/target facts.
Known raw values cannot bypass the relevant support constraints.

Native keys bind explicitly to occurrence-aware Decisions, including selectors.
Omitted choices remain uncommitted; false and zero remain explicit values.
``target_op.save_space(model, proposal)`` extracts only the proposed selected
values, checks compatible family/schema/key codecs, rehydrates current target
facts and validates the choices before writing atomically. Valid choices can be
reused from another origin or after a source change. The proposal's old weights,
shapes, semantic attributes and build facts are never copied back. Omitted
physical facts remain unavailable unless supplied explicitly for a relevant claim.

Native schema versions and sparse choices describe interpretation; a historical
source fingerprint is not required persisted state or save authorization.
Deferred graph-effect plans still use actual read-set/rollback checks because
their writes were computed earlier. Checked updates project accepted result types.
Producer annotations are derived caches: fresh producer
contracts govern downstream use, and changed types invalidate dependent caches
without changing independent consumer choices. Unavailable types do not fall
back to FLOAT.

``InferDataflowMatMul`` is a bounded entry for unfused nonsparse integer MatMul
with authenticated fixed weights. It derives a sufficient exact accumulator
requirement from the canonical range model and checks the family/source type
contract before replacement. It selects no folding, implementation or hardware
target. Rejected or unresolved admission leaves the ordinary node unchanged.
Shape/datatype inference over existing DataflowOps is a separate consumer of
the same canonical rules.

Artifact use and scope
----------------------

Local physical capture and preparation use detached generation requirements.
Node-bound artifact authorization additionally checks the exact current bound Op,
current relevant source/target facts, required public/physical bindings, requested
inputs, preparation receipts, manifests and stored contents. Equal complete build
inputs can share an artifact across distinct valid node uses.

Selected-ONNX construction, snapshots, publication, rebind and expansion transform
APIs are retired. There is no second persistent selection store or expansion
backend. Native schema versions whose meaning changed are explicitly rejected;
saved user files are not silently migrated. Scheduling/FIFO validation, broad
fusion/routing and an extensive RTL authorship-validation campaign remain future
work.

Reusable matrix profiles, type rules and implementations live under
``finn.dataflow.kernels.matmul``. Source decoding, initializer capture, native
hydration and graph effects remain in the compiler-owned ``finn.dataflow.ops``.
