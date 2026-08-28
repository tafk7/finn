Adding a DataflowOp
===================

``DataflowOp`` is FINN's model-aware source-operation bridge for the new
dataflow design stack. It is separate from ``HWCustomOp``: a logical operation
describes the source tensor function and assembles a static design family,
while Kernel definitions describe complete logical regions and implementation
providers perform physical elaboration and artifact generation.

Use the public authoring imports:

.. code-block:: python

   from finn.dataflow.authoring import DataflowOp, NodeAttrCodec

A subclass supplies a stable family ID and version, one node-independent
``DesignSpaceSpec``, the selected-result and source-association paths, and an
explicit mapping from persistent decision paths to ONNX attributes. The MVAU
reference implementation is ``finn.dataflow.ops.mvau_op.MvauDataflowOp`` and
is registered in the ``finn.custom_op.dataflow`` ONNX domain.

A minimal complete subclass has this shape. ``TINY_SPEC`` and ``TinyPaths``
are ordinary declarations built with ``finn.dataflow.design``; the result
property must produce a ``RegionRef`` or ``NetworkRef``.

.. code-block:: python

   class TinyDataflowOp(DataflowOp):
       @classmethod
       def dataflow_family_id(cls):
           return "example.tiny"

       @classmethod
       def dataflow_family_version(cls):
           return "1"

       @classmethod
       def build_design_space_spec(cls):
           return TINY_SPEC

       @classmethod
       def result_path(cls):
           return TinyPaths.RESULT

       @classmethod
       def source_association_path(cls):
           return TinyPaths.SOURCE_ASSOCIATION

       @classmethod
       def decision_nodeattrs(cls):
           return {TinyPaths.LANES: NodeAttrCodec.integer("dataflow_lanes")}

       def project_graph_problem(self):
           model = self._attached_model()
           return {TinyPaths.SHAPE: tuple(model.get_tensor_shape(self.onnx_node.input[0]))}

       def project_build_problem(self, config):
           return {TinyPaths.CLOCK_NS: float(config.synth_clk_period_ns)}

       def make_shape_compatible_op(self, model):
           return self.make_const_shape_op(model.get_tensor_shape(self.onnx_node.input[0]))

       def infer_node_datatype(self, model):
           model.set_tensor_datatype(
               self.onnx_node.output[0],
               model.get_tensor_datatype(self.onnx_node.input[0]),
           )

       def execute_node(self, context, graph):
           context[self.onnx_node.output[0]] = context[self.onnx_node.input[0]].copy()

       def verify_node(self):
           if len(self.onnx_node.input) != 1 or len(self.onnx_node.output) != 1:
               raise ValueError("TinyDataflowOp requires one input and one output")

Register it through the custom domain's public export list:

.. code-block:: python

   # finn/custom_op/dataflow/__init__.py
   from my_package.tiny import TinyDataflowOp

   __all__ = ["TinyDataflowOp"]

The ONNX ``op_type`` must match the class name and the node domain must resolve
to that Python package.

Model-aware construction
------------------------

Always obtain a dataflow operation from its owning ``ModelWrapper``:

.. code-block:: python

   op = model.get_customop_wrapper(node)
   resolved = op.resolve_dataflow(build_config)

``DataflowOp.wants_model`` is true. A wrapper created with bare
``getCustomOp(node)`` can inspect attributes, but projection, hydration, and
resolution reject it because tensor shapes, datatypes, initializers, and graph
relationships belong to the live model.

Static family and live problem
------------------------------

``build_design_space_spec`` must depend only on class-level declarations. FINN
validates and caches that space once per subclass. Per-node evaluation is
separate:

.. code-block:: text

   DataflowOp subclass
       -> static DesignSpaceSpec

   attached node + ModelWrapper + DataflowBuildConfig
       -> current ProblemInstance

   physically present decision attributes
       -> sparse assignments

   space + problem + assignments
       -> RegionRef | NetworkRef

Implement ``project_graph_problem`` for graph-owned facts and
``project_build_problem`` for target and invocation facts. Projection is
read-only and is repeated on demand; do not cache facts from a mutable model
without an explicit model revision identity.

Persistent decisions
--------------------

Use ``NodeAttrCodec.integer``, ``boolean``, ``string``, or ``finite_enum`` for
each decision that persists at operation scope. Attribute presence means that
a decision is committed. QONNX's optional attribute default is only a storage
default and is never treated as a design choice.

Use ``commit_dataflow_assignments`` for additive commitments and
``replace_dataflow_assignments`` for a complete replacement. Both validate the
entire request through the engine and encode every value before changing the
node. A rejected request leaves the serialized node unchanged.
``clear_dataflow_assignments`` removes only dataflow decisions and their
identity metadata.

The node records a stable scope ID, family ID, family version, and problem
fingerprint beside its sparse choices. Hydration rejects a changed family,
graph fact, target fact, or invocation fact instead of silently interpreting
old choices against a different problem. Regions, networks, constraints,
readiness results, elaborations, and artifacts are always recomputed and are
never stored as node attributes.

Scope identity exists independently of selection. A lowering pass should put a
new ``dataflow_scope_id`` on the node when it creates the logical operation, or
call ``initialize_dataflow_scope_id`` explicitly before evaluation. Clearing or
replacing choices preserves that ID. An ordinary protobuf copy intentionally
retains the same identity; a rewrite that creates a semantically distinct clone
must call ``renew_dataflow_scope_id``, which assigns a fresh ID and clears the
copied selection. Renaming ``node.name`` does not change the scope identity or
invalidate a selection.

MVAU build context
------------------

The MVAU operation accepts a normal ``DataflowBuildConfig`` when target part
and clock are sufficient. ``MVAUDataflowBuildContext`` wraps that configuration
when an invocation also supplies runtime-writable-weight policy, an external
weight ``BeatSequence``, or an explicit initialized-URAM capability override:

.. code-block:: python

   from finn.custom_op.dataflow import MVAUDataflowBuildContext

   context = MVAUDataflowBuildContext(
       build_config,
       runtime_writable_weights=True,
       external_weight_sequence=weight_sequence,
   )
   resolved = op.resolve_dataflow(context)

These values remain problem facts. PE, SIMD, region declaration, parameter
topology, binding, pumping, RAM style, and connection topology remain explicit
decisions stored through the MVAU codec inventory.

Testing a contribution
----------------------

``finn.dataflow.testing.assert_dataflow_op_conforms`` exercises the common
operation lifecycle: model attachment, deterministic read-only projection,
absent-versus-default behavior, rejected-write atomicity, partial and complete
save/reload hydration, stale-problem rejection, and
``RegionRef | NetworkRef`` result discipline. A new operation should run this
harness in addition to operation-specific exact schedule, requirement,
availability, and ``BeatSequence`` tests.

.. code-block:: python

   from finn.dataflow.testing import (
       DataflowOpConformanceCase,
       assert_dataflow_op_conforms,
   )

   result = assert_dataflow_op_conforms(
       DataflowOpConformanceCase(
           model=model,
           node_name="tiny0",
           operation_type=TinyDataflowOp,
           config=build_config,
           complete_assignments={TinyPaths.LANES: 4},
           rejected_assignments={TinyPaths.LANES: 3},
           reload_path=tmp_path / "tiny.onnx",
           stale_config=changed_build_config,
       )
   )

Standard ONNX lowering, fusion discovery, Kernel binding contributions, and
implementation providers are separate extension points. A ``DataflowOp`` must
not import legacy hardware operations, vendor tools, elaborators, or artifact
builders.
