Adding a DataflowOp
===================

``DataflowOp`` is FINN's model-aware bridge from a source graph operation to a
closed family of logical dataflow designs. It is separate from
``HWCustomOp``. A source operation owns graph/build projection and persisted
design decisions; each ``DataflowDesign`` owns one flat ``DataflowNetwork``;
physical ``Kernel`` candidates implement named placements in that Network.

The final contributor stack is:

.. code-block:: text

   DataflowOp
       -> DataflowDesign inventory
           -> one selected flat DataflowNetwork
               -> configured Kernel placements
                   -> physical elaboration and artifacts

A one-Region implementation is still represented by a singleton Network.
Production operations always resolve to ``NetworkRef``; a bare Region is never
an operation result.

Public imports
--------------

Use the declaration-time authoring façade when defining an operation or
design:

.. code-block:: python

   from finn.dataflow.authoring import (
       DataflowDesign,
       DataflowDesignScope,
       DataflowOp,
       NodeAttrCodec,
       OpDesign,
       Ref,
       finite,
       reject,
       unresolved,
   )

Use the evaluation-time façade when selecting or inspecting a resolved
operation:

.. code-block:: python

   from finn.dataflow.design import Engine, DesignPoint, NetworkRef, ResolvedDataflowOp

Physical implementations use:

.. code-block:: python

   from finn.dataflow.kernels import Kernel, KernelScope

Do not construct raw engine declarations in an operation package, inspect a
compiled ``DesignSpaceSpec`` to recover handles, or reconstruct ``Ref`` values
from paths. ``Scope``, ``DataflowDesignScope``, and the inventory compiler own
that bookkeeping.

Declaring the operation problem
-------------------------------

An ``OpDesign`` declares graph-, analysis-, target-, and build-owned facts once
for the operation family. Projection supplies live values from the attached
``ModelWrapper`` and build context. The static declaration graph must not read
an ONNX node or cache facts from a mutable model.

Always obtain an operation through its owning model:

.. code-block:: python

   operation = model.get_customop_wrapper(node)
   resolved = operation.resolve_dataflow(build_config)

``DataflowOp.wants_model`` is true. A wrapper created with bare
``getCustomOp(node)`` may inspect attributes, but graph projection, hydration,
and resolution reject it because tensor shapes, datatypes, initializers, and
graph relationships belong to the live model.

Declaring a design
------------------

A design declares exactly one Network and its physical placements. The Network
is semantic: it contains Regions, edges, and boundaries, not RTL components or
tool settings.

.. code-block:: python

   class ExampleDesign(DataflowDesign):
       id = "example"
       version = "1"

       @classmethod
       def define(cls, design: DataflowDesignScope[ExampleInputs]) -> None:
           lanes = design.choice("lanes", int, domain=finite((1, 2, 4)))
           compute = design.region(
               "compute",
               node_id="compute",
               dependencies={"lanes": lanes},
               evaluate=construct_example_region,
               computation=EXAMPLE_COMPUTATION,
           )
           design.singleton_network(compute)
           design.kernels(
               "compute",
               covers=(compute,),
               candidates=(ExampleKernel,),
               inputs=ExampleKernelInputs(compute.region, compute.computation, lanes),
           )

The operation's ``DataflowDesignInventory`` owns design selection, active
placement metadata, constraint/readiness aggregation, and configured Kernel
identities. Operation code may add genuinely operation-specific properties or
constraints, but must not scan ``spec.decisions`` or assemble a second metadata
inventory beside the authoritative one.

Conditional input supply
------------------------

An operation-level input-supply declaration can apply to every design that maps
the same source operand. The generic authoring layer handles conditional
Region, edge, boundary, and placement attachment. An inactive supplier is
``Absent``; it is not a missing active placement.

The MVAU operation owns the current concrete policy:

.. code-block:: text

   weight supply = external | finn_rtl_memstream

The generic mechanism does not discover suppliers or adapters globally. MVAU's
``weight``/``delivery``/``weights`` identities, cyclic parameter Region,
initializer association, and ``SETS=1`` memstream wrapper remain MVAU-local.

Declaring a Kernel
------------------

A ``Kernel`` is a reusable physical implementation of an already-declared
computation/interface contract. It owns physical choices, coverage
constraints, parameters, source manifests, and component elaboration. It does
not choose logical topology, source policy, or graph occurrence.

.. code-block:: python

   class ExampleKernel(Kernel):
       id = "example_rtl"
       version = "1"

       @classmethod
       def define_design(cls, kernel: KernelScope[ExampleKernelInputs]) -> None:
           kernel.covers_region(
               "compute",
               kernel.inputs.region,
               kernel.inputs.computation,
           )
           kernel.parameter("LANES", kernel.inputs.lanes)
           kernel.source("finn", "path/to/example.sv")

       @classmethod
       def elaborate(cls, configured: Kernel):
           return (...,)

Compiled Kernel declarations and candidate-selection records are private
implementation values. A contributor supplies Kernel classes to a design
placement and uses typed handles returned by the Kernel definition for any
Kernel-owned choices.

Graph admission and physical feasibility
----------------------------------------

Three questions remain separate:

``semantic recognition``
   Does the graph describe the source operation?

``graph-stage build admission``
   For at least one applicable design/supply trial, does every active placement
   retain a candidate after evaluating all graph-answerable Kernel coverage
   constraints?

``resolved physical feasibility``
   With target, build, design, supply, and Kernel choices known, do all selected
   Kernel coverage constraints pass?

Target/build-dependent constraints are deferred during graph inference rather
than treated as passed or failed. Both stages use the same Kernel-owned
predicates; do not copy datatype or width checks into an operation or
transformation.

Persistent decisions
--------------------

Use ``NodeAttrCodec.integer``, ``boolean``, ``string``, or ``finite_enum`` for
each operation-scope decision that persists. Attribute presence means a
decision is committed. QONNX optional defaults are storage defaults, not design
choices.

``commit_dataflow_assignments`` adds commitments and
``replace_dataflow_assignments`` replaces the full persisted selection. Both
validate and encode the complete request before mutating the node. A rejected
request leaves its serialized bytes unchanged.

The node stores a stable scope ID, family ID/version, problem fingerprint, and
the v6 decision attributes. Regions, Networks, configured Kernels,
elaborations, and artifacts are recomputed. v5/v10 persistence is rejected
explicitly; v6/v11 is the supported schema pair.

MVAU reference operation
------------------------

The canonical operation import is:

.. code-block:: python

   from finn.dataflow.ops.mvau import MVAUDataflowBuildContext, MvauDataflowOp

``MVAUDataflowBuildContext`` wraps a normal FINN build configuration when the
invocation also supplies runtime-writable-weight policy, an external weight
``BeatSequence``, or an initialized-URAM capability override.

MVAU offers two semantic designs: production ``dot_product`` and semantic-only
``batch_interleaved``. Production hardware uses the reusable DotpAxi,
ReplayBuffer, and FINN RTL memstream Kernels from ``finn.dataflow.kernels``.
The old Provider/semantic-Kernel framework is not a compatibility path and is
not importable. Legacy FINN HWCustomOps remain available only as independent
comparison oracles.

Testing a contribution
----------------------

``finn.dataflow.testing.assert_dataflow_op_conforms`` exercises model
attachment, deterministic projection, transactional writes, save/reload
hydration, stale-problem rejection, and the Network-only result boundary.

.. code-block:: python

   from finn.dataflow.testing import (
       DataflowOpConformanceCase,
       assert_dataflow_op_conforms,
   )

   assert_dataflow_op_conforms(
       DataflowOpConformanceCase(
           model=model,
           node_name="example0",
           operation_type=ExampleDataflowOp,
           config=build_config,
           complete_assignments=complete,
           rejected_assignments=rejected,
           reload_path=tmp_path / "example.onnx",
       )
   )

Operation-specific tests must additionally pin exact Region/Network values,
source associations, configured Kernel identities and parameters, artifact
identities/layouts, and numerical/tool behavior for any changed physical path.
