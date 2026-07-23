############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the matrix-vector activation kernel: WHAT it is, and how FINN sees it.

This is the one op-definition file (mirrors FINN's ``matrixvectoractivation.py``, but
declarative). A backend author reads it top to bottom; each ``impl_*.py`` bundle in this
package declares only the HOW for one compute core. Source of truth for each
axis/derived/predicate (file:line into real FINN):
``kernel-design/kernel-final-design/mvau-design-space.md``.

Sections (what each replaces in the classic FINN MVAU):
    1. CONSTANTS         tensor names + pool-member identities
    2. INTERFACES        the ONNX-facing arity + direction  (FINN: node in/out wiring)
    3. OP DESIGN SPACE   op_axes/op_derived/op_predicates — the shared BLOCK structure
                         (FINN: get_nodeattr_types + the shape/dtype getters' math)
    4. COMPUTE TILING    the BLOCK->STREAM lowering shared by the pool
    5. DEMAND            the compute->memory demand stage of the supply waterfall
    6. COST              rough op-level get_exp_cycles  (FINN: get_exp_cycles)
    7. ASSEMBLY          mvau_kernel() / mvau_schema()  (FINN: the class itself)
    8. FINN WRAPPER      MvauKernelOp(KernelOp)         (FINN: MVAU(HWCustomOp))

Tensor-name convention for the Context this schema resolves against:
    "inp"      the activation input tensor   (inputDataType, dynamic)
    "weights"  the weight tensor             (weightDataType + initializer VALUES)
    "out"      the output tensor             (outputDataType, unless derived)
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.kernels.adapter import KernelOp, PortSpec
from finn.kernels.primitives.spec_helpers import smallest_datatype_for_range
from finn.kernels.space import (
    FULL,
    Derived,
    Direction,
    Illegal,  # noqa: F401  (kept available for callers/tests)
    Interface,
    Kernel,
    KernelSchema,
    Role,
    Schema,
    discrete_axis,
    fixed_axis,
    predicate,
    predicate_axis,
    stream_width_key,
)
from finn.kernels.ops._dsp_rtl import VERSION  # noqa: F401  (re-exported for bundles)
from finn.kernels.ops.parameters import ParamDemand, parameters_pool, parameters_schema
from finn.kernels.ops.parameters.names import (
    ALL_MODES,
    TOPOLOGY_MODE,
    demand_key,
    runtime_writeable_key,
    topology_key,
)

from .registry import build_pool


# =============================================================================
# 1. CONSTANTS
# =============================================================================

# Pool-member identities (values of the root `implementation` axis). A new backend
# defines its own name in its own bundle file; these three are the built-ins.
MVAU_HLS = "mvau_hls"
MVAU_DSP_SOFTVEC = "mvau_dsp_softvec"
MVAU_DSP_PACKED = "mvau_dsp_packed"

# Context tensor names this schema resolves against.
WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"


# =============================================================================
# 2. INTERFACES — the ONNX-facing arity + direction (no tiling; tiling is impl-owned)
# =============================================================================


def mvau_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure (the math). No
    semantic role: whether ``weights`` is a stored parameter or a live activation emerges
    from graph context (initializer?) at resolve time. Stream folding (SIMD/PE) is impl-owned.

    The block reads as the matmul: ``inp`` iterates its vector count (``1``) and holds the
    reduction dim MW in-block (``FULL``); ``weights`` is the whole matrix ``(MW, MH)`` in one
    block; ``out`` iterates vectors and holds MH. ``weights`` is an ORDINARY interface — no
    WidthOnly, no special flag; its PE·SIMD stream is just a 2-D fold of a 2-D block."""
    return (
        Interface("inp", Direction.IN, block=[1, FULL]),        # (n_vecs, MW)
        Interface("weights", Direction.IN, block=[FULL, FULL]),  # (MW, MH)
        # dtype_source="outputDataType": the stream width uses the derived output type
        # (= accDataType under noActivation), not the raw graph dtype — so the generated
        # stream_width.out matches the emit-side value.
        Interface("out", Direction.OUT, block=[1, FULL], dtype_source="outputDataType"),
    )


# =============================================================================
# 3. OP DESIGN SPACE — the shared BLOCK structure (axes / derived / predicates).
#    Everything every MVU has, regardless of the chosen compute core. An impl
#    bundle never edits this; it resolves against it.
# =============================================================================

# -- small guard/helper functions (named, not lambdas, for legible tracebacks) --


def _has_activation(p) -> bool:
    return p.noActivation == 0


def weights_may_change(p) -> bool:
    """Weights are not statically known — accDataType/weightDataType must use
    worst-case bounds rather than actual values (base:482-498). This is the one
    CROSS-COORDINATE coupling from the parameters subsystem back into the compute
    dtype derivations: staticness (coordinate C) is decided by the composed
    ``parameters.*`` fields. Reads with ``.get`` so it is safe on a point where the
    parameters pool is absent (a future param-free op) — absent ⇒ statically known.

    This is a FORK, not a cycle (interface-supply-waterfall.md §3): one given fact
    (staticness) fans out to two consumers — compute dtype (up) and the memory reload port
    (down). A future M2 increment relocates staticness to the requirement tier both READ,
    at which point this helper reads that given instead of the parameters axis.

    Increment-1 topologies are ``embedded`` (static) and ``decoupled`` (static unless
    runtime-writable). The external / dynamic / MLO staticness sources return when
    those topologies land (each will be its own ``parameters.<iface>.topology`` value)."""
    return bool(p.get(runtime_writeable_key(WEIGHTS), 0))


def _matrix_dim(idx):
    def default(p, ctx):
        return ctx.tensor_shape(WEIGHTS)[idx]

    return default


def _num_input_vectors(p, ctx):
    # The input tensor's leading (non-reduction) dims — FINN's numInputVectors. The last
    # dim is MW (the reduction), so the vectors are everything before it. [1] for a plain
    # (1, MW) FC input; e.g. [1, H, W] for a conv-as-matmul.
    return list(ctx.tensor_shape(INPUT)[:-1])


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


# -- op-level SHARED axes — present under every implementation ------------------


def op_axes():
    return (
        # MW/MH are NO LONGER axes — they are BLOCK extents (the matmul's reduction +
        # output dims) declared on the interfaces' `block`, and the PE/SIMD fold DIALS the
        # engine derives from each impl's `stream`. MW/MH survive only as emit-facing
        # migration aliases (see op_derived below).
        # --- activation / threshold cluster ----------------------------------
        discrete_axis("noActivation", {0, 1}, 0),
        predicate_axis(
            "ActVal",
            "int",
            lambda v: isinstance(v, int),
            0,
            guard=_has_activation,
            deps={"noActivation"},
        ),
        discrete_axis(
            "ram_style_thresholds",
            {"auto", "block", "distributed"},
            "auto",
            guard=_has_activation,
            deps={"noActivation"},
        ),
        discrete_axis("binaryXnorMode", {0, 1}, 0),
        # numInputVectors is NOT an axis — it IS the input tensor's leading (non-reduction)
        # dims (FINN: get_normal_input_shape = numInputVectors + [MW]). Derived from the
        # block/tensor below (an emit-facing migration alias), same as MW/MH.
        # F4: mlo_max_iter is a per-node iteration count, unbounded non-neg int.
        # The `64` in the old {0..64} domain was n_max_layers (a fabric-wide MLO
        # table size, hwcustomop.py:378) — a different entity that does not bound
        # this axis (hwcustomop.py:317-319, mlo_max_iter unbounded).
        predicate_axis("mlo_max_iter", "nonneg int", _is_nonneg_int, 0),
        # --- weight-delivery cluster: MOVED OUT to the `parameters` pool ------
        # mem_mode/ram_style/runtime_writeable_weights/pumpedMemory/dynamic_input used
        # to live here as the "reserved composition seam". They are now the
        # `parameters` subsystem (ops/parameters/), composed into the MVAU schema
        # via `compose(...)` under the `parameters.*` namespace. mem_mode is gone:
        # being the `decoupled` topology IS "internal_decoupled". The cross-coordinate
        # couplings (memstream geometry, the pumpedMemory/fold gate) are contributed in
        # section 5 below (appended to the op schema before compose).
    )


# -- op-level SHARED derived — computed for every implementation ----------------


def _wmem(p, ctx):
    return p.MW * p.MH // (p.PE * p.SIMD)


def _tmem(p, ctx):
    return p.MH // p.PE if p.noActivation == 0 else 0


def _acc_datatype(p, ctx):
    # base:469-527 — worst-case type bounds when weights may change, actual weight
    # VALUES when static. The canonical data-dependent Derived.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    weights = ctx.initializer(WEIGHTS)
    if p.binaryXnorMode == 1 and weights is not None:
        weights = 2 * weights - 1
    if weights_may_change(p) or weights is None:
        lower = wdt.min() * np.ones((p.MW, p.MH))
        upper = wdt.max() * np.ones((p.MW, p.MH))
        lo_r = calculate_matvec_accumulator_range(lower, idt)
        hi_r = calculate_matvec_accumulator_range(upper, idt)
        acc_min = min(min(lo_r), min(hi_r))
        acc_max = max(max(lo_r), max(hi_r))
    else:
        acc_min, acc_max = calculate_matvec_accumulator_range(weights, idt)
    return smallest_datatype_for_range(float(acc_min), float(acc_max))


def _weight_datatype(p, ctx):
    # base:529-549 — VALUE_OPTIMIZED narrow, only when weights are statically known.
    weights = ctx.initializer(WEIGHTS)
    if weights is None or weights_may_change(p):
        return ctx.tensor_datatype(WEIGHTS)
    w_min = float(weights.min())
    w_max = float(weights.max())
    if w_min < 0:
        extreme = w_min if abs(w_min) > w_max else -w_max - 1
        return DataType.get_smallest_possible(extreme)
    return DataType.get_smallest_possible(w_max)


def _output_datatype(p, ctx):
    # base:517 — outputDataType = accDataType when noActivation, else the graph dtype.
    if p.noActivation == 1:
        return _acc_datatype(p, ctx)
    return ctx.tensor_datatype(OUTPUT)


def _identity_geometry_derived():
    """EMIT-MIGRATION ALIASES — MW/MH/numInputVectors/WMEM/TMEM. Emit-facing extents
    sourced from the block/weight shapes (NOT named op axes), so Tier-4 emit_hls/emit_rtl
    (which read point.MW/MH/WMEM/TMEM) is untouched. New code should read extents from the
    interface block shapes directly; these retire when emit is reworked (NOT this task).
    Order matters: MW/MH precede WMEM/TMEM, which read them (resolve computes deriveds in
    list order)."""
    return (
        Derived("MW", _matrix_dim(0)),
        Derived("MH", _matrix_dim(1)),
        Derived("numInputVectors", _num_input_vectors),
        Derived("WMEM", _wmem),
        Derived("TMEM", _tmem),
    )


def _identity_dtype_derived():
    """THE REAL DATATYPE CONTRACT (base:469-549) — accDataType/weightDataType/outputDataType.
    Data-dependent derivations: they read the actual weight VALUES when static, worst-case
    bounds otherwise. ``weights_may_change`` is the one static→dtype coupling that gates
    them (a FORK, not a cycle — see its docstring)."""
    return (
        Derived("accDataType", _acc_datatype),
        Derived("weightDataType", _weight_datatype),
        Derived("outputDataType", _output_datatype),
    )


def op_derived():
    # PURE IDENTITY. Two groups, concatenated (order load-bearing — geometry aliases precede
    # the dtype contract; WMEM/TMEM read MW/MH). NOT here, by design:
    #   * `stream_width.<iface>` — the tiling engine generates one per interface from each
    #     impl's `stream` folds (the `out` interface's dtype_source="outputDataType" gives it
    #     the accumulator type under noActivation).
    #   * `weight_stream_width` — a per-TOPOLOGY fact (0 embedded / demand bit_rate decoupled)
    #     owned by the parameters pool.
    #   * `parameters.demand` — the DEMAND stage of the supply waterfall (`demand_schema()`),
    #     a composed sub_schema between the compute pool and the parameters pool (it reads the
    #     compute pool's resolved `stream_width.weights`, only present AFTER compute tiling).
    return _identity_geometry_derived() + _identity_dtype_derived()


# -- op-level SHARED predicates — one kind; provenance is what each reads --------


# The divisibility predicates (MH%PE==0, MW%SIMD==0) are NOT hand-written here — the
# tiling engine generates one per Fold spec from COMPUTE_TILING. The op declares only
# the block-structural math facts below.


# The `pumpedMemory => not(PE==SIMD==1)` gate and the `ram_style=ultra & not versal
# => runtime_writeable=1` URAM gate live in the parameters subsystem: the URAM gate
# is self-contained in the decoupled topology bundle; the pumpedMemory/fold gate is
# cross-coordinate (reads the compute fold) and is contributed in section 5.


@predicate("weight initializer must exist unless params are not statically known")
def _weights_present(p, ctx):
    # Weights must exist as an initializer unless the parameters subsystem says they
    # are not statically known (runtime-writable / external / dynamic / MLO). Reads
    # the composed staticness via weights_may_change (parameters.*), with .get safety
    # for a future param-free op (no parameters pool ⇒ still requires an initializer).
    if ctx.initializer(WEIGHTS) is None:
        if not weights_may_change(p):
            return "weight initializer required unless params are not static (base:782)"
    return None


# NOTE (audit F2, DROPPED): a `bipolar x bipolar => nonneg thresholds` predicate used
# to live here, but it checked the scalar `ActVal` (the threshold activation's bias,
# base:156) whereas FINN's assertion is over the THRESHOLD TENSOR VALUES
# (`orig_thres_matrix >= 0`, base:578) — a different object. The Context models
# inp/weights/out; thresholds are NOT a first-class Context tensor, so the correct
# action is to drop it. Reinstate when thresholds become a Context tensor.


def op_predicates():
    return (_weights_present,)


# =============================================================================
# 4. COMPUTE TILING — the BLOCK->STREAM lowering shared by all three compute impls.
# =============================================================================

# They fold identically — SIMD folds the reduction dim MW on the activation (last axis),
# The STREAM folding shared by all three compute impls: SIMD folds the reduction dim MW
# (inp position 1, weights position 0), PE folds the output dim MH (out position 1, weights
# position 1). weights is a 2-D fold of its 2-D block → PE·SIMD tile/cycle. Positional over
# each interface's `block`. Declared once here; a tiled backend overrides only the weights
# entry (deliver PE/TH along MH). The engine DERIVES the SIMD/PE dials (divisor domains),
# divisibility, and widths from these — none hand-written.
COMPUTE_STREAM = {
    INPUT: [1, "SIMD"],
    OUTPUT: [1, "PE"],
    WEIGHTS: ["SIMD", "PE"],
}


# =============================================================================
# 5. DEMAND — the compute→memory DEMAND stage of the supply waterfall.
#    The op no longer reaches into memstream: the DEMAND stage publishes ONE
#    realization-free ParamDemand (what the compute core consumes of its weight
#    interface), sourced from the RESOLVED interface geometry (stream_width.weights +
#    block extents), NOT from named backend dials. The selected parameters topology
#    sizes ITS OWN geometry from it (impl_decoupled.py). This is the compute→memory
#    demand channel (param-delivery-design-space.md §4 Level-1 / §7 Q1;
#    interface-supply-waterfall.md Q1): the memory backend owns memstream
#    width/depth/sets/init_file + the pumped/URAM gates + weight_stream_width; the
#    demand stage owns only the (realization-free) demand.
# =============================================================================


def _demand_for(iface):
    """The compute core's DEMAND on parameter interface ``iface``, as a resolve closure.

    Sourced from the RESOLVED interface geometry — NOT from named backend dials
    (PE/SIMD/WMEM). This is the supply-waterfall's DEMAND stage: it reads what the COMPUTE
    stage produced (``stream_width.<iface>``, the tiling engine's per-interface resolved
    width) and the block extents, so a tiled/systolic/packed backend that folds the
    interface in its own terms works automatically (interface-supply-waterfall.md Q1).

    Returns ``None`` (nothing to deliver) in two emergent cases:

    * **no initializer** — the interface is a live activation (e.g. dynamic matmul operand
      B), not a stored parameter; it streams in like any dataflow edge, the delivery pool
      sizes to nothing, the port exports as a boundary. Dissolves the dynamic-matmul case
      with no special flag (resolution-phases.md: a Context-reading fact is a phase-3
      closure).
    * **constant consumption mode** — the selected topology bakes the parameter into the
      compute core (``embedded`` = the ``constant`` mode, consumption-mode-delivery.md);
      there is no stream to size, so no demand. The topology domain was itself already
      filtered to the compute backend's consumable modes (see ``_topology_domain``).
    """
    width_key = stream_width_key(iface)

    def compute(p, ctx):
        if ctx.initializer(iface) is None:
            return None
        if TOPOLOGY_MODE[p[topology_key(iface)]] == "constant":
            return None
        width_bits = int(p[width_key])  # resolved stream width (PE*SIMD*wbits)
        elem_bits = ctx.tensor_datatype(iface).bitwidth()
        parallelism = width_bits // elem_bits  # elements/cycle, in the backend's own fold
        block = ctx.tensor_shape(iface)  # the block extents (MW, MH for weights)
        depth = _prod(block) // parallelism  # words/set = WMEM, from geometry not p.WMEM
        return ParamDemand(
            parallelism=parallelism,
            elem_bits=elem_bits,
            depth=depth,
            cadence=1,  # weights consumed once per layer; thresholds per-activation (later)
        )

    return compute


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


def _topology_domain(iface):
    """A domain override for ``parameters.<iface>.topology`` that keeps only the storage
    topologies whose CONSUMPTION MODE the selected compute backend can consume for this
    interface (consumption-mode-delivery.md §2c).

    The compute backend is master: it declares, per interface, which modes it consumes
    (``Backend.consumes``); delivery provisions to that. An interface the backend says
    nothing about is PERMISSIVE (both modes) — so the default is today's full topology set
    (embedded default), and nothing regresses until a backend declares a restriction. Reads
    the resolved ``implementation`` (a phase-3 cross-coordinate coupling — hence appended by
    the op before ``compose``, where both the compute pool and the parameters pool are in
    scope; backend.py:189-192)."""
    topologies = tuple(b.name for b in parameters_pool(iface))

    def legal(p):
        # Defensive read: during real resolve `implementation` is fixed before this axis;
        # under a bare probe point (nodeattr-registry typing) it is absent → permissive
        # (all modes), which is exactly the no-restriction default.
        impl = p.get("implementation") if hasattr(p, "get") else None
        backend = {b.name: b for b in mvau_pool()}.get(impl)
        modes = (backend.consumes.get(iface) if backend else None) or ALL_MODES
        return tuple(t for t in topologies if TOPOLOGY_MODE[t] in modes)

    return lambda p, ctx: frozenset(legal(p)), legal


def _topology_default(iface, base_default, legal):
    """The topology axis's default, guarded to the consumable modes: keep the pool's own
    first-registered default (``embedded``) when the selected backend can consume it, else
    fall to the first in-domain topology (a stream-only backend defaults to the first
    streamer). Prevents an out-of-domain default from making an unpinned topology illegal."""

    def default(p, ctx):
        allowed = legal(p)
        d = base_default(p, ctx)
        return d if d in allowed else (allowed[0] if allowed else d)

    return default


def demand_schema(iface) -> Schema:
    """The DEMAND stage of the supply waterfall as a derived-only schema for one parameter
    interface, composed BETWEEN the compute pool and the parameters (memory) pool. It exists
    as its own compose stage — not in ``op_derived`` — because the demand reads the tiling
    engine's resolved ``stream_width.<iface>`` (and the resolved topology mode), both
    generated by earlier compose stages and so only on the point AFTER them (resolve
    computes deriveds in schema order: op_derived → compute-pool tiling → THIS → parameters
    pool). Encoding COMPUTE→DEMAND→MEMORY as compose order makes the waterfall structural."""
    return Schema(axes=(), derived=(Derived(demand_key(iface), _demand_for(iface)),), predicates=())


# =============================================================================
# 6. COST — no op-level override needed.
# =============================================================================
#
# MVAU's cost IS the reduction product nf·sf·n_vecs, and it falls out of the generic
# max-over-interfaces floor for FREE now that `weights` is a proper 2-D block streamed
# SIMD·PE: its stream-cycle count is MW·MH/(SIMD·PE) = sf·nf — the largest interface term,
# times the input's n_vecs leading dims. So MVAU declares NO cost_model (the old override
# only existed because weights was modelled as a width-only port skipped by the floor).


# =============================================================================
# 7. ASSEMBLY — the full MVAU design space as a Kernel (and as a bare Schema).
# =============================================================================


def mvau_shared():
    """The op-level shared (axes, derived, predicates) — everything every MVU has,
    independent of the composed parameters couplings. Used by tests that assemble a
    bare ``pool_schema`` directly."""
    return op_axes(), op_derived(), op_predicates()


def mvau_pool():
    """The registered MVAU implementations (flat peers), in registration order."""
    return build_pool()


def _params_subschema(iface) -> Schema:
    """The parameters pool for ``iface`` with its topology root axis's domain overridden to
    the consumption-mode guard (``_topology_domain``): only topologies the selected compute
    backend can consume for this interface remain selectable. This is the cross-coordinate
    coupling the op owns — it reads BOTH the compute ``implementation`` and the parameters
    topology, so it is applied HERE (the op has both pools in scope), not inside the
    standalone ``parameters_schema`` (backend.py:189-192).

    The domain reads the resolved ``implementation``; ``compose`` appends this topology axis
    after the compute pool's ``implementation`` axis (which is first), and ``_topo_sort``
    preserves input order among independent axes, so resolve always fixes the compute backend
    before this domain runs. (A ``deps={"implementation"}`` declaration would be more explicit
    but cannot live on the STANDALONE parameters schema, where ``implementation`` is absent,
    so the ordering is relied on, exactly as the existing merged param axes rely on the
    topology root resolving before them.)"""
    from dataclasses import replace

    schema = parameters_schema(iface)
    root = schema.axes[0]  # pool_schema emits the root topology axis first
    domain, legal = _topology_domain(iface)
    guarded = replace(
        root, domain=domain, default=_topology_default(iface, root.default, legal)
    )
    return replace(schema, axes=(guarded,) + tuple(schema.axes[1:]))


# The parameter interfaces this op delivers through the parameters pool. Only ``weights``
# is live (initializer-backed) this increment; a thresholds interface joins here once it is
# a first-class Context tensor — one more (demand_schema, _params_subschema) pair, no engine
# change (consumption-mode-delivery.md §6).
_PARAM_INTERFACES = (WEIGHTS,)


def mvau_kernel() -> Kernel:
    """The full MVAU design space as a :class:`Kernel` — the WHAT-owning op node.

    The compute pool (``implementation``: HLS / DSP-softvec / DSP-packed) with impl-owned
    tiling, composed — PER parameter interface — with that interface's DEMAND stage and then
    its PARAMETERS pool (``parameters.<iface>.topology``). The ``sub_schemas`` order IS the
    supply waterfall COMPUTE→DEMAND→MEMORY: the demand stage reads the compute pool's
    resolved ``stream_width.<iface>`` and publishes ``parameters.<iface>.demand``; the
    selected delivery topology (whose domain is guarded to the backend's consumable modes)
    then sizes its own memstream geometry + ports from that demand — the op no longer brokers
    memstream realization. The getters project from a resolved point via the impl
    ``stream``."""
    sub_schemas = ()
    for iface in _PARAM_INTERFACES:
        sub_schemas += (demand_schema(iface), _params_subschema(iface))
    return Kernel(
        identity=KernelSchema(
            name="MVAU",
            interfaces=mvau_interfaces(),
            op_axes=op_axes(),
            op_derived=op_derived(),
            op_predicates=op_predicates(),
        ),
        pool=mvau_pool(),
        sub_schemas=sub_schemas,
    )


def mvau_schema() -> Schema:
    """The full MVAU design space as a resolve ``Schema`` — delegates to
    :func:`mvau_kernel` (identical assembly). Kept as the name emit/composition tests
    resolve against."""
    return mvau_kernel().schema()


# =============================================================================
# 8. FINN WRAPPER — MvauKernelOp(KernelOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# Binds the kernel to a FINN node: three ports (activation in, weights in, activation
# out) mapping the graph's tensor slots to the kernel's inp/weights/out interfaces.
# Registered as ``MVAUKernel_hls`` in the ``finn.custom_op.fpgadataflow.hls`` domain
# (see the hls custom_op dict) so ``is_hls_node`` sees it and the estimate analyses run
# — while the real compute impl is chosen by the ``implementation`` nodeattr, not the
# domain (consumer-surface-model.md R11).

_PORTS = (
    PortSpec(iface="inp", direction="in", index=0, role=Role.DATA_IN),
    PortSpec(iface="weights", direction="in", index=1, role=Role.WEIGHT_SINK),
    PortSpec(iface="out", direction="out", index=0, role=Role.DATA_OUT),
)


class MvauKernelOp(KernelOp):
    """MVAU (matrix-vector activation) as a Kernel-backed FINN op."""

    def kernel(self):
        return mvau_kernel()

    def ports(self) -> tuple[PortSpec, ...]:
        return _PORTS

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        # MVAU's outputDataType is a resolved derived: the graph dtype when forwarding,
        # or the weight-derived accumulator type under noActivation. Read it off the
        # point so infer propagates the exact (possibly narrowed) type.
        if index == 0 and "outputDataType" in point:
            return point["outputDataType"]
        return super()._output_datatype_from_point(kernel, ctx, point, index)

    def get_folding_axes(self):
        """The folding dials this op exposes, each mapped to its resolved max value —
        the capability SetFolding queries instead of op_type prefix-matching
        (consumer-surface-model.md R1). SIMD folds the reduction dim MW, PE the output
        dim MH; both are context-derived (from the weights shape), so we read them off a
        resolved point rather than from a stored nodeattr."""
        _, _, point = self._point()
        return {"SIMD": int(point.MW), "PE": int(point.MH)}
