.. _nw_prep:

*****************
Graph Preparation
*****************

Graph preparation takes the network as :ref:`brevitas_export` leaves it to the
streamlined graph the kernel path converts to KernelOps. It is the first phase of a
kernel-path build, ``phase_graph_preparation``
(:py:mod:`finn.builder.kernel_build_steps`), and its configuration is the build's
``preparation`` (:py:class:`finn.transformation.prepare.GraphPreparation`). It reads
no target: what preparation makes of a network does not depend on the board, part or
clock it is built for.

The phase is a sequence of sub-phases, each a builder step that a build's ``steps``,
``start_step`` and ``stop_step`` can name. The transforms that do the work stay where
they are (:py:mod:`finn.transformation.qonnx`, :py:mod:`finn.transformation.streamline`
and QONNX's); :py:mod:`finn.transformation.prepare.phase` states the order and the
options. The sub-phases run in this order:

``step_prepare_import`` (P0, import)
    QONNX's cleanup: shapes inferred, constants folded with the Quant nodes kept,
    parameters made unique, nodes and tensors named. ``override_inpsize`` sets the
    batch size (an integer) or the input's shape (a list).

``step_prepare_quantization`` (P2, quantization lowering)
    :py:mod:`finn.transformation.qonnx.convert_qonnx_to_finn`: quantized weights folded
    to integers, Quant activations of up to ``max_multithreshold_bit_width`` bits
    (8 by default) to MultiThreshold nodes; then the tidy-up, which ends in
    QONNX's datatype inference.

``step_prepare_io`` (P1, the network's inputs and outputs)
    The ``preprocessing`` model merged ahead of the network (an ONNX file, for
    instance :py:class:`finn.util.pytorch.ToTensor` exported by Brevitas, which scales
    8-bit pixels to [0, 1]); the graph input's ``input_datatype`` stated (``UINT8`` for
    pixels), the domain the network is prepared for; the ``topk`` largest outputs'
    indices selected by a TopK node appended to the graph. It follows P2: the
    preprocessing model is lowered on its own and merged into the lowered network.

``step_prepare_streamlining`` (P3, streamlining)
    The ``streamlining`` recipe, a list of transforms by name
    (:py:data:`finn.transformation.prepare.RECIPE_TRANSFORMS`) run in order. The
    default absorbs scalar scales and biases into the thresholds that follow them
    (``Streamline``, see `this paper <https://arxiv.org/pdf/1709.04060.pdf>`_), lowers
    convolutions to ``Im2Col`` and ``MatMul``, rewrites bipolar MatMuls as
    ``XnorPopcountMatMul``, and folds the last scale and bias into the label select.

``step_prepare_topology`` (P4, topology)
    The ``topology`` recipe: by default MaxPool made NHWC, transposes absorbed into
    MultiThresholds and consecutive transposes collapsed. Then the graph is tidied,
    every tensor's layout inferred, and unused tensors removed.

``step_prepare_containers`` (P6, containers)
    Every integer tensor held exactly by its container: an export holds its tensors in
    float32, exact for integers up to 2**24. Where a tensor's producer reaches past that
    (a wide layer's partial sums), the integer region ONNX's type constraints tie it to
    is widened to float64, exact up to 2**53, with a Cast where an integer enters float
    arithmetic or a quantizer's output enters the region; float arithmetic, the graph's
    inputs and its outputs keep their containers. Past 2**53 the checkpoint refuses the
    graph, naming the tensor.

``step_prepare_checkpoint`` (P7, checkpoint)
    The prepared graph checked before anything converts it
    (:py:mod:`finn.transformation.prepare.checkpoint`), described below.

A build from a model that is already prepared names its steps from
``phase_kernel_path`` on, or starts there (``start_step``).

For TFC, whose images are 8-bit and whose output is the top label:

.. code-block:: python

    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.builder.kernel_build_config import KernelBuildConfig, KernelVerificationStepType
    from finn.platform import TargetRequest
    from finn.transformation.prepare import GraphPreparation

    cfg = KernelBuildConfig(
        output_dir="output",
        target=TargetRequest(board="Ultra96", period_ns=5.0, shell="pynq"),
        preparation=GraphPreparation(
            preprocessing="preproc.onnx", input_datatype="UINT8", topk=1
        ),
        verify_steps=[KernelVerificationStepType.GRAPH_PREPARATION_PYTHON],
    )
    build_dataflow_cfg("tfc_w2a2.onnx", cfg)

In ``kernel_build_config.json`` the same preparation reads
``"preparation": {"preprocessing": "preproc.onnx", "input_datatype": "UINT8", "topk": 1}``;
a key the phase does not have, or a transform no recipe may name, is refused when the
configuration is read.

The checkpoint
==============

What the kernel path relies on, checked on every build without executing the graph;
a violation stops the build, each named:

* **Structure.** Every tensor has a known shape (``shape-unknown``), and every tensor a
  node at a KernelOp's anchor reads or writes carries a datatype annotation
  (``annotation-absent``): an absent annotation is never read as FLOAT32. An
  initializer is exempt; its values are its statement.
* **Soundness, by bound.** Each node's integer output annotation holds the range its
  op computes from its inputs' annotations and its initializers' values
  (``annotation-unsound``). An op without a rule
  (:py:data:`finn.transformation.prepare.BOUND_RULES`) is counted as unbounded in the
  report, never assumed sound.
* **Exactness.** Each integer tensor's container holds every value its producer
  computes on the way, a MatMul's partial sums included, exactly
  (``container-inexact``): float32 up to 2**24, float64 (a region P6 widened) up to
  2**53.

What the phase could not settle is reported, not refused: nodes with a float output
(``float-remains``), Transposes (``transpose-remains``), and nodes no KernelOp anchors
on between nodes that one does (``host-between-predicted``).

With ``GRAPH_PREPARATION_PYTHON`` in ``verify_steps``, the prepared graph is also
executed against the export with P0 and P1 alone applied, on seeded draws from the
input's annotation (its extremes, random values, and corners): every output must be
equal, element for element, and every integer annotation must hold the values computed
there (:py:mod:`finn.harness.preparation`). The phase computes otherwise than the
export only where it declares it
(:py:data:`finn.transformation.prepare.DEVIATIONS`): ``topk-affine`` (the label
select's values lose the scale folded into it; its indices are the export's),
``sign-at-zero`` (a Sign turned threshold gives +1 at exactly 0) and
``threshold-float32`` (the export computes the scales and biases absorbed into
thresholds in float32, the phase computes and stores the thresholds in float64). The
predicates that tell where a deviation explains a difference belong to the test
harness, so a build reports every difference (``equivalence-unexplained``); the
kernel gate checks TFC against its export with them, and fails on a difference none
explains.

The check also runs the export with its float32 tensors held in float64: an integer
the export computes otherwise in float32 is where the trained network itself rounds and
the prepared graph, its containers exact, does not. Each such tensor is named
(``export-rounds``, a limitation): it may explain a difference, and fails nothing.

In a build this check is a verification, as ``PARTITION_PYTHON`` is: its differences
from the export are reported, not refused. The log prints ``Verification for
graph_preparation_python : FAIL``, one line per finding code and the declared value
deviations that might explain them, ``report/graph_preparation.json`` keeps them under
its equivalence, and the build continues. A value drawn outside its tensor's integer
annotation (``annotation-unsound``) is no difference from the export but the prepared
graph's own annotation proved wrong, and the kernels' widths and accumulators trust
it: it is reported and logged as the others are, and refuses the graph beside the
checkpoint's blockers, named.

The report
==========

``report/graph_preparation.json`` records each sub-phase's nodes removed and added by
op type and the annotations and containers it changed, the checkpoint's findings (each
with its kind, code, owner, message and details, as ``report/kernel_ops.json`` records
conversion's) and how many integer outputs it bounded, and the equivalence's draws,
findings and ``export_rounds`` when asked. The build's log has one line per sub-phase and one per finding code.
