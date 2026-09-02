Adding a DataflowOp
===================

``DataflowOp`` is FINN's model-aware bridge from a source graph operation to a
closed family of logical dataflow designs. It is separate from ``HWCustomOp``.
The contributor-facing stack has three classes:

.. code-block:: text

   DataflowOp
       -> DataflowDesign
           -> Kernel

``DataflowRegion`` and ``DataflowNetwork`` are immutable semantic IR values,
not additional selectable layers. A Design declares exactly one Network, and a
one-Region Design uses the same Network-shaped result as every other Design.

Public imports
--------------

Operation and Design authors use ``finn.dataflow.authoring``. Kernel authors
use ``finn.dataflow.kernels`` together with the declaration objects re-exported
from ``finn.dataflow.authoring``:

.. code-block:: python

   from finn.dataflow.authoring import (
       Attribute,
       Choice,
       ClosedDesigns,
       DataflowDesign,
       DataflowOp,
       InputTensor,
       Kernels,
       Network,
       Persist,
       Region,
       UsesDesign,
       constraint,
       derived,
   )
   from finn.dataflow.kernels import Kernel, PhysicalComponent

``Scope``, ``Ref``, ``OpDesign``, ``DataflowDesignScope``, ``KernelScope``,
compiled declarations, inventories, coverage records, and placement records
are compiler implementation details. They are intentionally absent from the
public façades. Do not reconstruct a handle from a path or inspect a compiled
``DesignSpaceSpec`` to find declarations.

Operation example
-----------------

An operation class declares source operands, attributes, derived facts,
constraints, its closed Design inventory, and persistent decisions in one
place. Conditional operands use the same condition for arity, projection,
initializer policy, and downstream applicability.

The complete non-MVAU forcing example is
``ChannelwiseAffineDataflowOp``. It has an optional bias operand and two Designs,
and uses the generic projection, fingerprint, persistence, and lifecycle code:

.. literalinclude:: ../../../tests/dataflow/channelwise_affine_op.py
   :pyobject: ChannelwiseAffineDataflowOp
   :language: python

Concrete operations do not implement graph/build mapping dictionaries,
fingerprinting, persistence envelopes, or engine start/commit/resolve loops.
The generic occurrence API is:

.. code-block:: python

   operation = model.get_customop_wrapper(node)
   point = operation.hydrate_dataflow_point(build_config)
   operation.commit_dataflow_assignments(build_config, assignments)
   resolved = operation.resolve_dataflow(build_config)
   realization = operation.realize_dataflow(build_config)
   physical = operation.compose_dataflow(build_config)

``resolve_dataflow`` returns one ``ResolvedDataflowOp`` containing the selected
Design id, Network, logical source association, source-scope id, immutable
point, and private compiled-operation identity. There is no nested
``NetworkRef`` wrapper.

Design example
--------------

A Design class owns its choices and constraints, Region declarations, one flat
Network, source-to-semantic mappings, Kernel placements, and optional physical
composer. Selecting the Design does not force every Design- or Kernel-local
choice to be resolved.

The production MVAU ``DotProductDesign`` is a complete multi-placement example:

.. literalinclude:: ../../../src/finn/dataflow/ops/mvau/designs/dot_product.py
   :pyobject: DotProductDesign
   :language: python

Its ``replay`` and ``compute`` placements cover different Network nodes. The
operation-owned weight-supply policy may add a ``delivery`` placement without
changing the core Design identity. ``BatchInterleavedDesign`` demonstrates a
semantically valid Design with no production compute Kernel: logical resolution
succeeds while physical realization reports an explicit limitation.

A Design's ``PhysicalComposition`` declaration names the composer and every
additional fact it may read. The composer receives a restricted context with
the resolved Network, logical association, exact realization, declared facts,
source-scope identity, and immutable Kernel-origin projections. It never
receives an ``Engine`` or ``DesignPoint``.

Kernel example
--------------

A Kernel class declares exactly what it covers, its physical choices and
constraints, emitted parameters, and source manifest. Binding supplies the
resolved Regions and edges and produces the configured Kernel instance used by
elaboration.

``DotpAxiKernel`` is the complete reusable example. Its class-local declaration
includes an implementation-local pumping choice, exact computation coverage,
target-dependent constraints, all RTL parameters, and its ordered FinnLib
source list:

.. literalinclude:: ../../../src/finn/dataflow/kernels/dotp_axi.py
   :pyobject: DotpAxiKernel
   :language: python

``Kernel.elaborate`` receives only the configured Kernel. It cannot read a raw
design point or source graph, and every emitted parameter is audited against a
declared value.

Lifecycle and ownership
-----------------------

The similarly named values represent different stages:

``class declaration``
   Immutable relative templates in an Op, Design, or Kernel class. The same
   Design or Kernel can be rebound under several namespaces without mutation.

``compiled metadata``
   Private bound handles, projection plans, codecs, inventory, placements, and
   one flat ``DesignSpaceSpec``.

``DesignPoint``
   An immutable problem snapshot plus sparse choices. It may be partial.

``ResolvedDataflowOp``
   A selected Design's resolved Network and logical source association.

``configured Kernel``
   One Kernel family bound to exact Region/edge values and declared parameters.
   It holds no unrestricted point.

``DesignRealization``
   Proof that active placements cover the whole selected Network exactly.

``physical composition``
   The selected Design's operation-specific components, interfaces,
   connections, boundaries, and physical association ledger.

``artifact state``
   Downstream identities, source closure, derivations, storage, packaging, and
   tool receipts. These are not design-space state.

Persistence and provenance
--------------------------

The adapter compiler owns one portable codec registry and one problem
fingerprint implementation for every operation. Persistence records identify
the family, family schema version, format version, source scope, complete
problem fingerprint, and sparse assignments. Incompatible family or format
versions are rejected explicitly. MVAU publishes ``mvau-dataflow-op-v7`` with
persistence format ``1``; v6 and the retired v11 graph-metadata transport are
not reinterpreted.

Logical source association maps source occurrences and tensors to semantic
operands and coordinates. It contains no Kernel ids, placement names,
Kernel-local decisions, component ids, artifact keys, or physical local-state
destinations. Those facts belong to realization and physical provenance.

Artifact boundary
-----------------

The adapter-side MVAU handoff projects configured Kernels and composition into
immutable physical/provenance and artifact-identity values before artifact
processing. The separately owned artifact-substrate integration owns ABI,
source closure, derivations, lifecycle, storage, packaging, and execution
receipts; this adapter migration does not modify that package. Semantic
association remains separate from reusable artifact identity.

Testing a contribution
----------------------

``finn.dataflow.testing.assert_dataflow_op_conforms`` checks model attachment,
deterministic projection, transactional writes, save/reload hydration,
stale-problem rejection, and the Network-only result boundary. Operation tests
must additionally pin semantic values, source associations, configured Kernel
parameters, generated text, artifact identity, and relevant numerical/tool
behavior.

Run the complete adapter software gate with:

.. code-block:: bash

   PYTHON_BIN=.venv/bin/python ./scripts/check-dataflow-design.sh
