############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding — the multi-threshold activation kernel: WHAT it is, and how FINN sees it.

The op-definition file (mirrors FINN's ``thresholding.py``, declarative). The shared
design space (axes/derived/predicates) lives in ``shared.py``; each ``impl_*.py`` backend
declares the HOW for one compute core (HLS baked-ROM, RTL binary-search). This file adds
the two assemblies the backends could not: the ``DataflowOp`` (identity + pool + delivered
parameters) and the FINN ``DataflowOp`` wrapper.

Tensor-name convention for the Context this schema resolves against:
    "inp"         the activation input tensor   (inputDataType, dynamic)
    "thresholds"  the threshold tensor          (thresholdDataType + initializer VALUES)
    "out"         the output tensor             (graph output dtype)

The ``thresholds`` interface is the op's ONE parameter interface. It is a SEPARABLE,
static-schedule memory in the HLS backend (a baked ``thresh.h`` ROM read by the
output-channel loop → the ``embedded``/constant topology) and a FUSED, data-dependent
memory in the RTL backend (the binary-search ``.dat`` scatter, addressed by the runtime
comparison outcome). Both resolve to the ``embedded`` (constant) topology → demand None,
no memstream cell; the HLS/RTL difference is purely each compute emit's baked artifact.
"""

from __future__ import annotations

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.kernels.ir import DataflowOp, TransformationResult
from finn.kernels.engine.constraints import ShapeRank
from finn.kernels.engine.attr import attr
from finn.kernels.engine.derived import Derived
from finn.kernels.model.kernel import InterfaceSchema
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL

from .names import (  # noqa: F401
    COMPUTE_STREAM,
    INPUT,
    OUTPUT,
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    THRESHOLDS,
)

from .impl_hls import hls_bundle as hls_backend
from .impl_rtl import rtl_bundle as rtl_backend
from .shared import (
    _num_channels,
    _num_steps,
    _threshold_datatype,
    _tmem,
    _unsigned_input_nonneg_thresholds,
)


# =============================================================================
# INTERFACES — the ONNX-facing arity + direction.
# =============================================================================


def thresholding_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure. ``inp`` and
    ``out`` iterate the spatial/vector count (``1``) and hold the channel dim in-block
    (``FULL``); ``thresholds`` is the whole ``(NumChannels, numSteps)`` matrix in one
    block. PE folds the channel dim (out position 1); the threshold block folds with it."""
    return (
        InterfaceSchema(INPUT, Direction.IN, block=[1, FULL]),          # (n_vecs, NumChannels)
        # The rank-2 requirement is a PER-PORT structural constraint, not an op predicate:
        # NumChannels and numSteps are both read positionally off this shape, so a non-2D
        # tensor is unresolvable rather than merely illegal. Declaring it here is what makes
        # it fire BEFORE the deriveds that index the shape — an op_predicate runs after them,
        # so a 1-D tensor raised IndexError instead of reporting the real reason. Same
        # mechanism MVAU already uses for its optional threshold port.
        InterfaceSchema(
            THRESHOLDS, Direction.IN, block=[FULL, FULL],  # (NumChannels, numSteps)
            constraints=(ShapeRank(THRESHOLDS, 2),),
        ),
        InterfaceSchema(OUTPUT, Direction.OUT, block=[1, FULL]),
    )


# =============================================================================
# FINN WRAPPER — ThresholdingDataflowOp(DataflowOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# The interface↔node-slot binding is the op's own interface list (inp=0, thresholds=1,
# out=0 — declaration order); no separate PortSpec (F9).


class ThresholdingDataflowOp(DataflowOp):
    """Thresholding (multi-threshold activation) — an op composing two cells.

    The class body is the design space that used to be a separate `DataflowKernel` container
    (F6). Only the COMPUTE cell's pool (HLS / RTL) is written here; the threshold interface
    becomes a second, delivery cell because both backends name it in ``mem_modes``, derived
    by `DataflowOp.__init_subclass__`. So the compiled space has two selection roots
    (``backend``, ``parameters.thresholds.topology``). Both backends consume thresholds in
    embedded mode, so delivery resolves to the ``embedded`` topology (no memstream cell).
    """

    # -- the design space ---------------------------------------------------
    name = "Thresholding"
    interfaces = thresholding_interfaces()
    pool = (hls_backend(), rtl_backend())
    # EMPTY, and every removal was measured:
    #
    # ``numSteps`` -> Derived (below). Its only LEGAL value was the Context one: its default
    # already read ``tensor_shape(THRESHOLDS)[1]``, and a predicate rejected anything else. An
    # axis whose domain is a singleton determined by Context is a derivation wearing an Axis
    # costume — the same correction ``NumChannels`` already had, for the same reason. The
    # predicate went with it: it was not a legality rule, it was a consistency check on a
    # duplicate.
    #
    # ``numInputVectors`` -> DELETED. It was INERT: nothing in this package read it, and
    # pinning it changed no resolved value (folded shapes track the TENSOR). Baseline FINN
    # needs it because it RECONSTRUCTS shapes from nodeattrs (``tuple(vecs + [ich])``,
    # thresholding.py:211); we source shapes from the live graph, so the reconstruction input
    # is dead weight. MVAU already treats it as Context (``geometry.py``: ``nvec`` from the
    # input tensor's leading dims) and publishes no such nodeattr — the two ops disagreed
    # about one quantity and MVAU was right.
    #
    # ``ActVal`` -> kernel_attrs (a node CONSTANT; it is the one value here the graph really
    # does not hold once infer absorbs the MultiThreshold).
    #
    # NOT here either, and deliberately: the ``PE`` fold dial and the ``NumChannels``
    # divisibility rule are GENERATED by the tiling engine from each backend's declared
    # ``stream=COMPUTE_STREAM``. They were hand-written while the stream was declared but
    # never wired, which also forced ``NumChannels`` to be carried as a pseudo-axis purely
    # to feed ``divisor_axis``. It is a Context fact, so it is now a Derived (below).
    op_axes = ()

    op_derived = (
        # NumChannels is a CONTEXT fact (the threshold tensor's channel extent), not a
        # choice. It was an Axis only because divisor_axis("PE", "NumChannels") needed to
        # read it off the point; with PE generated from the declared stream, it can be what
        # it always was. Emit reads point.NumChannels, so the key stays.
        Derived("NumChannels", _num_channels),
        # numSteps is the threshold tensor's STEP extent — the sibling fact to NumChannels,
        # and it took the same route: an Axis whose default read Context, whose domain was a
        # singleton, and whose "legality" predicate only checked it had not been overridden
        # to something inconsistent. Emit reads point.numSteps, so the key stays.
        Derived("numSteps", _num_steps),
        Derived("TMEM", _tmem, deps={"NumChannels", "PE"}),
        # No outputDataType derived: the output dtype IS the graph output dtype (the trivial
        # DatatypeSpec — None → graph fallback), resolved uniformly like every other output's
        # derived_dtype. The out port declares no derived_dtype; no second mechanism here.
        # thresholdDataType is Context-only while the narrowing is disabled for FINN parity
        # (see _threshold_datatype's TODO), so it declares NO dep. Re-enabling the narrowing
        # means reading the storage owner's published ParamDatatype again, which is a
        # parameters-pool derived — restore ``deps={param_datatype_key(THRESHOLDS)}`` with it
        # so the unified topo-sort orders this after the publisher.
        Derived("thresholdDataType", _threshold_datatype),
    )
    # instream_width/outstream_width are GONE: they are the singular-stream pair
    # resolution-phases.md §4 dissolved, hand-reintroduced here because this op bypassed the
    # tiling engine. The generated per-interface ``stream_width.<iface>`` replaces them.

    op_predicates = (_unsigned_input_nonneg_thresholds,)

    # The absorbed MultiThreshold's out_bias — infer bakes it (op.py) and emit reads it off
    # the Point (emit_hls.py, emit_rtl.py). No graph home once the frontend node is gone,
    # which is what makes it an Attr rather than a Context fact. See engine/attr.py.
    kernel_attrs = (attr("ActVal", "int", lambda v: isinstance(v, int), 0),)

    # -- Seam A: frontend claim (mirror of InferThresholdingLayer) ---------------------

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Whether ``node`` is a STANDALONE ``MultiThreshold`` this kernel can claim.
        Mirrors ``InferThresholdingLayer`` (convert_to_hw_layers.py:159) as explicit
        preconditions (return False, not assert).

        Precedence: a ``MultiThreshold`` fed by a ``MatMul`` is the FUSED activation the
        MVAU kernel absorbs — it must NOT be claimed standalone. Since MVAU precedes this op
        in the pool and removes the consumer during its own inference, that node is normally
        gone before we reach it; the explicit producer check makes the ordering robust even
        against a stale node-list snapshot.

        WHY THESE CHECKS STAY IN PYTHON (unlike MVAU's, which became IsStatic/SparsityFree
        constraints): every one below is a FRONTEND-GRAPH fact, not a design-space fact. The
        producer identity, the tensor layout and ``out_scale`` describe the shape of the
        pattern this op claims — the op's own business — not what a backend can build. A
        constraint is the right home only for the latter, because its whole purpose is to let
        a NEW BACKEND widen the claim; no backend will ever make an NCHW layout claimable.

        The dtype check is the interesting case, and it deliberately did NOT migrate to
        ``DatatypeSupport``. Two reasons, both checked rather than assumed:
          1. It is a UNION of kinds (integer OR fixed OR float32/16). ``DatatypeSupport`` is
             one kind plus a bitwidth range, so expressing it would need a custom callable —
             a closure on the port, which is no more declarative than the closure here.
          2. Both backends have IDENTICAL envelopes and there is no verified per-backend gate
             (a fabricated one was previously falsified — see the package ``__init__`` and
             ``scratchpad/reference/toy-vs-brainsmith-thresholding.md`` A1). Empirically this
             check rejects only ``SCALEDINT`` among FINN's dtypes, so declaring a backend
             gate would be inventing a distinction no backend actually makes.
        Migrating it would move a frontend-pattern fact into the design space AND fabricate
        backend knowledge to do it. When a Thresholding backend appears whose dtype envelope
        genuinely differs, THAT is the moment to declare a gate — with a real case behind it.
        """
        if node.op_type != "MultiThreshold":
            return False
        producer = model.find_producer(node.input[0])
        if producer is not None and producer.op_type == "MatMul":
            return False

        # Pattern-shape dtype admissibility (mirrors InferThresholdingLayer), NOT a
        # buildability gate — see the docstring.
        idt = model.get_tensor_datatype(node.input[0])
        tdt = model.get_tensor_datatype(node.input[1])
        idt_ok = idt.is_integer() or idt.is_fixed_point() or idt in ["FLOAT32", "FLOAT16"]
        tdt_ok = tdt.is_integer() or tdt.is_fixed_point() or tdt in ["FLOAT32", "FLOAT16"]
        if not (idt_ok and tdt_ok):
            return False

        # The slice claims NHWC/2-D layouts only; NCHW would need a layout-conversion node
        # (FINN's :194-206), deferred. A None layout (plain 2-D matmul activations) is fine.
        from qonnx.core.data_layout import NCHW

        if model.get_tensor_layout(node.input[0]) == NCHW:
            return False

        # out_scale must be 1 for HW conversion (FINN :215).
        if getCustomOp(node).get_nodeattr("out_scale") != 1.0:
            return False
        return True

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int
    ) -> TransformationResult:
        """Build the unresolved ``finn.kernels`` Thresholding node replacing this
        ``MultiThreshold``. Thin per F2′: it re-points the SAME input/threshold tensors and
        bakes ONLY ``ActVal`` (the ``out_bias`` residual). NumChannels/numSteps/PE/
        numInputVectors and all dtypes stay derived live from Context.
        """
        actval = int(getCustomOp(node).get_nodeattr("out_bias"))
        kernel_node = helper.make_node(
            "Thresholding",
            [node.input[0], node.input[1]],
            [node.output[0]],
            domain="finn.kernels",
            name="Thresholding_" + node.name,
            ActVal=actval,
        )
        return TransformationResult(nodes_to_insert=[kernel_node], nodes_to_remove=[node])

    def get_folding_axes(self):
        """PE folds the channel dim NumChannels (the threshold tensor's leading extent),
        read straight off the Context."""
        ctx = self._context()
        channels = ctx.tensor_shape(THRESHOLDS)[0]
        return {"PE": int(channels)}
