Unified Kernel authoring
========================

The experimental dataflow stack uses one domain abstraction for reusable
implementations: ``finn.dataflow.kernels.Kernel``. A Kernel is a normal
``Space`` and uses the same declarations, immutable points, nested occurrences,
readiness checks and constraints as every other Space.

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

Optional capabilities
---------------------

``LogicalView``, ``PhysicalView`` and ``RelationView`` are named Projection
helpers. Each view declares its own output, applicability, readiness and
constraint groups. An absent or unresolved optional view does not block an
unrelated logical or physical query.

Operations own source interpretation and selected ONNX construction. A
``DataflowOp`` exposes its chosen implementation through ``selected_kernel()``;
the selected Kernel optionally declares ``selected_construction``. Persisted
choice paths use the ``kernel`` namespace. Pre-unified ``design__*`` attributes
and older native schema versions are rejected rather than migrated implicitly.
