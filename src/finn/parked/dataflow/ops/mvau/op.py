# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU as one source node over a closed set of Kernels.

MvauSpace declares frozen graph facts, public operand bindings and sparse choices
using the ordinary Space machinery. Kernels own folding, Regions, topology and
internal value maps. The small MvauDataflowOp adapter supplies model-aware QONNX
construction and execution, and owns a separate immutable ``space`` occurrence.
No live model is retained by that Space or by any Kernel evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

from finn.kernels._engine import ABSENT, Absent, Decided
from finn.parked.dataflow.analysis.integer_dot import (
    DotProductBounds,
    analyze_integer_dot_product,
    IntegerSupportReport,
    InvocationScope,
    NumericalFinding,
    RuntimeWeightPromise,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.parked.dataflow.kernels.matmul.base import (
    AccumulationMode,
    DspBlock,
    MvauComputationProfile,
    MatmulInterface,
    computation_profile,
    matrix_result_requirement,
)
from finn.kernels.space.declarations import (
    ConstraintGroup,
    CanonicalValueCodec,
    Problem,
    Subspace,
    SubspaceChoice,
    allow_absent,
    constraint,
    derived,
    reject,
    reject_all,
)
from finn.parked.dataflow.ops.mapping import CoordinateMapping
from qonnx.analysis.tensor_value_summary import TensorValueSummary  # type: ignore[import-not-found]
from finn.parked.dataflow.ops.base import (
    DataflowOp,
    DataflowOpError,
)
from finn.parked.dataflow.ops.mvau.computation import execute_mvau
from finn.parked.dataflow.ops.mvau.numerics import (
    check_mvau_integer_support_from_operands,
    execute_mvau_integer,
    mvau_integer_premise_from_operands,
)
from finn.parked.dataflow.ops.space import DataflowSpace
from finn.parked.dataflow.ops.source import SourceNode, SourceOperand
from finn.parked.dataflow.ops.binding import ChoiceBinding, ImplementationBinding, OperandBinding
from finn.parked.dataflow.kernels.matmul.batch_interleaved import BatchInterleavedKernel
from finn.parked.dataflow.kernels.matmul.dot_product import DotProductKernel
from finn.parked.dataflow.ops.schema import (
    Attribute,
    BuildFact,
    DatatypeAttribute,
    OpInput,
    OpOutput,
)


def _target_dsp(build: Any) -> DspBlock:
    value = build.target_dsp
    return value if isinstance(value, DspBlock) else DspBlock(str(value))


def _target_dsp_canonical(value: DspBlock) -> dict[str, object]:
    """Preserve the pre-relocation structural enum encoding."""

    return {
        "enum": "finn.dataflow.kernels.dotp_axi.DspBlock",
        "value": value.value,
    }


def mvau_profile(source: SourceNode) -> MvauComputationProfile:
    """One reading's computation profile, without an occurrence.

    The formula the derived member uses, reachable from the unbound side --
    ``execute_node`` runs on a wrapper QONNX built and has no design space, and
    two spellings of "what does this node compute" is exactly the drift the
    derivation exists to prevent.
    """

    return computation_profile(
        no_activation=bool(source.attributes["no_activation"]),
        binary_xnor=bool(source.attributes["binary_xnor"]),
        activation_type=source.operand("activation").datatype,
        weight_type=source.operand("weight").datatype,
    )


def origin_nodes(source: SourceNode) -> tuple[str, ...]:
    """The original nodes fused into this one, from the comma-separated list.

    The spelling FINN's transformations already write.  Parsed here rather than
    at every reader, and empty entries are dropped so a trailing comma is not a
    node called ``""``.
    """

    raw = str(source.attributes.get("source_nodes", ""))
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _runtime_writable(build: Any) -> bool:
    """Whether this build writes the matrix at runtime.  Absent means no."""

    return bool(getattr(build, "runtime_writable_weights", False))


def _runtime_weight_range_contract(build: Any) -> bool | None:
    """The caller's promise about a runtime matrix's range, if they made one.

    ``None`` is not ``False``: "no contract" and "a contract that says the
    minimum is used" both prevent narrowing, but only one of them is a
    statement, and a later reader that wants to warn about the first must be
    able to tell them apart.
    """

    value = getattr(build, "runtime_weight_range_contract", None)
    return None if value is None else bool(value)


def _ignored_runtime_weight_promise(build: Any) -> RuntimeWeightPromise | None:
    """Retain the provisional build-field spelling without consuming its value."""

    del build
    return None


def _numerical_rejection(report: IntegerSupportReport) -> object:
    if not report.findings:
        return True
    return reject_all(
        reject(item.code, item.message, values=dict(item.values)) for item in report.findings
    )


class MvauSpace(DataflowSpace):
    """One matrix-vector node, projected onto the unified Space stack.

    **Ownership.**  Each restriction below is written where something can argue
    for it, rather than copied down from an earlier implementation:

    ``weight`` is rank 2
        Matrix-vector multiplication is defined on a matrix.  An operation
        constraint, because it is the operation's own definition.

    the output shape follows the activation and the matrix
        Likewise the operation's own contract, and the thing
        ``make_shape_compatible_op`` and the graph effects both derive from.

    the activation may be constant
        Deliberately *not* restricted.  A constant activation is
        mathematically valid, and the previous ``NoInitializer`` was inherited
        rather than argued for.

    a rank-1 activation is one repetition
        Accepted, and deliberately not restricted anywhere.  A shape ``(W,)``
        whose extent matches the matrix width is a single vector through the
        matrix -- ``repetitions`` evaluates to 1 by the same formula every
        other rank uses, and the Kernels need nothing special to build it.

        This paragraph previously claimed the opposite: that a rank-1
        activation had no applicable Kernel and that
        ``WeightedDotProductKernel`` rejected it. No such rejection existed,
        and none was added to make the sentence true -- a restriction has to be
        argued from the mathematics or from a Kernel's structure, and neither
        argues for one here.
    """

    family: ClassVar[str] = "finn.dataflow.mvau"
    family_version: ClassVar[str] = "1"
    schema_version: ClassVar[int] = 8

    implementation_binding = ImplementationBinding(("kernel",))
    operand_bindings = (
        OperandBinding("activation", "activation", 0, adapter=CoordinateMapping.FLATTEN_LEADING),
        OperandBinding("weight", "weights", 1, adapter=CoordinateMapping.IDENTITY),
        OperandBinding(
            "output", "result", 0, output=True, adapter=CoordinateMapping.FLATTEN_LEADING
        ),
    )
    choice_bindings = (
        ChoiceBinding("kernel__case", ("kernel",), "case"),
        ChoiceBinding(
            "kernel__dot_product__compute__kernel", ("kernel", "dot_product", "compute"), "kernel"
        ),
        ChoiceBinding("kernel__dot_product__pe", ("kernel", "dot_product"), "pe"),
        ChoiceBinding("kernel__dot_product__simd", ("kernel", "dot_product"), "simd"),
        ChoiceBinding(
            "kernel__dot_product__weight_supply", ("kernel", "dot_product"), "weight_supply"
        ),
        ChoiceBinding(
            "kernel__dot_product__compute__dotp_axi__compute_pumping",
            ("kernel", "dot_product", "compute", "dotp_axi"),
            "compute_pumping",
        ),
        ChoiceBinding(
            "kernel__dot_product__compute__dotp_axi_embedded__compute_pumping",
            ("kernel", "dot_product", "compute", "dotp_axi_embedded"),
            "compute_pumping",
        ),
        ChoiceBinding("kernel__batch_interleaved__pe", ("kernel", "batch_interleaved"), "pe"),
        ChoiceBinding("kernel__batch_interleaved__simd", ("kernel", "batch_interleaved"), "simd"),
        ChoiceBinding(
            "kernel__batch_interleaved__interleave", ("kernel", "batch_interleaved"), "interleave"
        ),
        ChoiceBinding(
            "kernel__batch_interleaved__compute__dotp_axi_batch_interleaved__compute_pumping",
            ("kernel", "batch_interleaved", "compute", "dotp_axi_batch_interleaved"),
            "compute_pumping",
        ),
    )

    # -- the source schema ----------------------------------------------------

    activation = OpInput(index=0, operand="X", correspondence=CoordinateMapping.FLATTEN_LEADING)
    weight = OpInput(index=1, operand="W", correspondence=CoordinateMapping.IDENTITY)
    #: Present exactly when the node fuses an activation.  Optional rather than
    #: conditional-by-declaration: presence is *emergent* -- it is read from the
    #: graph -- and the agreement between it and ``no_activation`` is a
    #: constraint that can be reported, not a schema rule that makes the node
    #: unreadable.
    threshold = OpInput(index=2, optional=True)
    output = OpOutput(index=0, operand="Y", correspondence=CoordinateMapping.FLATTEN_LEADING)

    #: The three attributes FINN's graphs already carry, under the names they
    #: already have.  ``no_activation`` defaults to ``True`` because a plain
    #: matrix-vector node is the common case and because that is what every
    #: existing graph means by omitting it.
    no_activation = Attribute(bool, default=True, onnx="noActivation")
    binary_xnor = Attribute(bool, default=False, onnx="binaryXnorMode")
    activation_bias = Attribute(int, default=0, onnx="ActVal")

    #: The accumulator width, and therefore the output's datatype when nothing
    #: is fused.  A source-semantic *attribute*, not a reading of the output
    #: annotation: the value reaches Region construction and the physical
    #: realization, so it must be part of the problem's identity.  Reading it
    #: back off the tensor the operation itself writes would make a fact the
    #: design space depends on something the design space is authoritative for
    #: -- and, kept out of the fingerprint to make repair possible, would let
    #: recorded choices be silently wrong.
    accumulator_type = DatatypeAttribute(default="INT32", onnx="accDataType")

    @derived(bool, summary=allow_absent(weight.value_summary), datatype=weight.datatype)
    def weight_excludes_minimum(*, summary: object, datatype: QONNXDataType) -> object:
        if not isinstance(summary, TensorValueSummary) or summary.minimum is None:
            return reject("weight-summary-absent", "weight extrema are unavailable")
        return bool(summary.minimum != datatype.min())

    # Required source authority, including for fused-threshold scale/bias.
    output_type = DatatypeAttribute(onnx="outputDataType")

    #: Which original graph nodes were fused into this one.  Carried because it
    #: is provenance nothing else records: once several nodes become one
    #: logical MVAU, the lineage exists only here.
    source_nodes = Attribute(str, default="", onnx="dataflow_source_nodes")

    target_dsp = BuildFact(
        DspBlock,
        accessor=_target_dsp,
        canonical=CanonicalValueCodec("dataflow.structural", 1, _target_dsp_canonical),
    )
    #: Whether the matrix is written at runtime, and what range the caller
    #: promises it will hold.  Build facts rather than node attributes: both
    #: are decisions of the surrounding build, not properties of the graph.
    runtime_writable_weights = BuildFact(
        bool, accessor=_runtime_writable, default=False, required=False
    )
    runtime_weight_range_contract = BuildFact(
        bool, accessor=_runtime_weight_range_contract, required=False
    )
    runtime_weight_promise = BuildFact(
        RuntimeWeightPromise,
        accessor=_ignored_runtime_weight_promise,
        required=False,
    )
    clock_period_ns = BuildFact(float, accessor=lambda build: float(build.synth_clk_period_ns))

    invocation_scope = Problem(
        InvocationScope,
        canonical=CanonicalValueCodec(
            "finn.dataflow.invocation_scope.external_identity",
            1,
            lambda _scope: {"identity": "recorded-separately"},
        ),
    )

    @classmethod
    def _additional_problem_values(
        cls, source: SourceNode, *, scope_id: str
    ) -> Mapping[Problem[Any], object]:
        stable_scope = scope_id or f"source-node:{source.domain}:{source.node_name}"
        return {cls.invocation_scope: InvocationScope(stable_scope)}

    # -- what the composition below reads -------------------------------------

    @derived(tuple, shape=weight.shape)
    def matrix(*, shape: tuple[int, ...]) -> object:
        """The validated matrix shape every other derivation reads.

        The one place the rank is checked, and everything downstream depends on
        *this* rather than on the raw shape.  A ``Derived`` cannot rely on a
        sibling constraint having run first -- the engine offers no such
        ordering -- so ``shape[1]`` on a rank-1 weight would raise an
        EvaluationError before the rank constraint ever produced its finding.
        """

        if len(shape) != 2:
            return reject(
                "mvau-weight-not-a-matrix",
                f"matrix-vector multiplication needs a rank-2 matrix; this one is {shape}",
                values={"shape": list(shape)},
            )
        if any(extent <= 0 for extent in shape):
            # Checked here for the same reason the rank is: everything
            # downstream divides by the width, and a zero would raise before
            # any constraint got to say what was wrong.
            return reject(
                "mvau-degenerate-extent",
                f"a matrix needs positive extents; this one is {shape}",
                values={"shape": list(shape)},
            )
        return shape

    @derived(int, shape=matrix)
    def matrix_width(*, shape: tuple[int, ...]) -> object:
        return shape[0]

    @derived(int, shape=matrix)
    def matrix_height(*, shape: tuple[int, ...]) -> object:
        return shape[1]

    @derived(int, shape=activation.shape, width=matrix_width)
    def repetitions(*, shape: tuple[int, ...], width: int) -> object:
        if not shape or any(extent <= 0 for extent in shape):
            return reject(
                "mvau-degenerate-extent",
                f"an activation needs positive extents in every dimension; got {shape}",
                values={"shape": list(shape)},
            )
        total = 1
        for extent in shape:
            total *= extent
        return total // width

    @derived(
        MvauComputationProfile,
        activated=no_activation,
        xnor=binary_xnor,
        activation_type=activation.datatype,
        weight_type=weight.datatype,
    )
    def profile(
        *, activated: bool, xnor: bool, activation_type: object, weight_type: object
    ) -> object:
        """What this node computes, on both of the axes that decide it.

        Derived once and read by everything -- the execution, the Kernels'
        applicability, any later parity record -- so a consumer that asked
        ``noActivation`` directly could not come to disagree with it.  The
        operand datatypes are dependencies because the accumulation genuinely
        depends on them: two BIPOLAR operands are a popcount whatever the
        attributes say.
        """

        return computation_profile(
            no_activation=activated,
            binary_xnor=xnor,
            activation_type=cast(Any, activation_type),
            weight_type=cast(Any, weight_type),
        )

    @derived(
        IntegerSupportReport,
        activation=activation,
        weight=weight,
        accumulator=accumulator_type,
        output=output_type,
        profile=profile,
        scope=invocation_scope,
        runtime_writable=allow_absent(runtime_writable_weights),
    )
    def source_numerical_report(
        *,
        activation: SourceOperand,
        weight: SourceOperand,
        accumulator: QONNXDataType,
        output: QONNXDataType,
        profile: MvauComputationProfile,
        scope: InvocationScope,
        runtime_writable: object,
    ) -> IntegerSupportReport:
        if profile.accumulation is not AccumulationMode.INTEGER or profile.fuses_activation:
            return IntegerSupportReport(None, ())
        if not activation.datatype_established or not weight.datatype_established:
            return IntegerSupportReport(
                None,
                (
                    NumericalFinding(
                        "integer-logical-datatype-annotation",
                        "plain-integer execution requires explicit logical datatypes "
                        "on both inputs",
                    ),
                ),
            )
        return check_mvau_integer_support_from_operands(
            activation,
            weight,
            accumulator_datatype=accumulator,
            output_datatype=output,
            invocation_scope=scope,
            runtime_writable=runtime_writable is not ABSENT and bool(runtime_writable),
            runtime_promise=None,
            target_max_bits=64,
        )

    @derived(
        IntegerSupportReport,
        activation=activation,
        weight=weight,
        accumulator=accumulator_type,
        output=output_type,
        profile=profile,
        scope=invocation_scope,
        runtime_writable=allow_absent(runtime_writable_weights),
        target=target_dsp,
    )
    def numerical_support(
        *,
        activation: SourceOperand,
        weight: SourceOperand,
        accumulator: QONNXDataType,
        output: QONNXDataType,
        profile: MvauComputationProfile,
        scope: InvocationScope,
        runtime_writable: object,
        target: DspBlock,
    ) -> IntegerSupportReport:
        if (
            profile.accumulation is not AccumulationMode.INTEGER
            or profile.fuses_activation
            or not activation.datatype_established
            or not weight.datatype_established
        ):
            return IntegerSupportReport(None, ())
        from finn.parked.dataflow.ops.mvau.numerics import target_accumulator_bits  # noqa: PLC0415

        return check_mvau_integer_support_from_operands(
            activation,
            weight,
            accumulator_datatype=accumulator,
            output_datatype=output,
            invocation_scope=scope,
            runtime_writable=runtime_writable is not ABSENT and bool(runtime_writable),
            runtime_promise=None,
            target_max_bits=target_accumulator_bits(target),
        )

    @derived(
        bool,
        excludes_minimum=allow_absent(weight_excludes_minimum),
        runtime_writable=allow_absent(runtime_writable_weights),
    )
    def effective_narrow_weights(*, excludes_minimum: object, runtime_writable: object) -> object:
        """Whether the weights can be stored one bit narrower.

        Two sources, and which one applies is a property of the *weights*, not
        a preference.  A matrix baked in at build time is judged by its own
        values. A runtime-writable matrix has no authoritative fixed values, so
        it uses datatype-only sizing. Reading initializer analysis for such a
        node would narrow hardware on the strength of values that may be
        overwritten.

        Missing or inapplicable narrower facts are ``False``; numerical support
        independently falls back to the complete logical datatype range.
        """

        writable = runtime_writable is not ABSENT and bool(runtime_writable)
        if writable:
            return False
        return excludes_minimum is not ABSENT and bool(excludes_minimum)

    @constraint(shape=matrix)
    def weight_is_a_matrix(*, shape: tuple[int, ...]) -> object:
        """Restates the refusal ``matrix`` already made, as a *verdict*.

        The derivation refuses so nothing downstream computes on a malformed
        shape; the constraint exists so ``op.dataflow`` reports it as a
        rejection rather than only as an unresolved Network.
        """

        del shape
        return True

    @constraint(activation=activation.shape, width=matrix_width)
    def activation_matches_the_matrix(*, activation: tuple[int, ...], width: int) -> object:
        if activation and activation[-1] == width:
            return True
        return reject(
            "mvau-activation-width-mismatch",
            f"the activation's last dimension {activation[-1:]} does not match the "
            f"matrix width {width}",
            values={"activation": list(activation), "width": width},
        )

    #: The output annotation is deliberately *not* here.  A stale annotation is
    #: what ``graph_effects`` repairs, and a constraint that refused it would
    #: refuse the very projection whose acceptance is required to commit the
    #: repair -- so the node could never be fixed.  It is a reconciliation
    #: difference; see ``expected_outputs``.
    @constraint(present=threshold.present, activated=no_activation)
    def threshold_present_iff_activated(*, present: bool, activated: bool) -> object:
        """The operand list and the attribute must agree about what this node is.

        Either disagreement is a real fault and neither is repairable here: a
        node that says it fuses an activation and supplies no thresholds cannot
        be executed, and one that supplies thresholds while claiming it does
        not fuse would silently ignore them.  The check is on the operation
        because it is the operation's own mathematics -- no Kernel has an
        opinion about it.
        """

        if present is not activated:
            return True
        return reject(
            "mvau-threshold-presence-mismatch",
            "a fused-activation MVAU needs a threshold operand and a plain one must not "
            f"have it; noActivation={activated} with threshold present={present}",
            values={"no_activation": activated, "threshold_present": present},
        )

    @constraint(operand=allow_absent(threshold), height=matrix_height)
    def threshold_shape_supported(*, operand: object, height: int) -> object:
        """One threshold row per output channel, when there are thresholds at all.

        Absence-tolerant rather than conditional: a plain MVAU has nothing to
        check here, and that is an ordinary "yes" rather than a constraint that
        does not exist.
        """

        if operand is ABSENT:
            return True
        shape = tuple(cast(SourceOperand, operand).shape)
        if len(shape) == 2 and shape[0] == height:
            return True
        return reject(
            "mvau-threshold-shape",
            f"a threshold operand is one row per output channel: expected {height} rows in a "
            f"rank-2 tensor, got {shape}",
            values={"shape": list(shape), "matrix_height": height},
        )

    @constraint(no_activation=no_activation, output=output_type, accumulator=accumulator_type)
    def unactivated_output_matches_accumulator(
        *, no_activation: bool, output: QONNXDataType, accumulator: QONNXDataType
    ) -> object:
        return matrix_result_requirement(
            no_activation=no_activation, output=output, accumulator=accumulator
        )

    @constraint(report=source_numerical_report, profile=profile)
    def source_integer_numerically_supported(
        *, report: IntegerSupportReport, profile: MvauComputationProfile
    ) -> object:
        if profile.accumulation is not AccumulationMode.INTEGER or profile.fuses_activation:
            return True
        return _numerical_rejection(report)

    source_accepts = ConstraintGroup(
        unactivated_output_matches_accumulator,
        weight_is_a_matrix,
        activation_matches_the_matrix,
        threshold_present_iff_activated,
        threshold_shape_supported,
        source_integer_numerically_supported,
    )
    type_source_accepts = ConstraintGroup(
        weight_is_a_matrix,
        activation_matches_the_matrix,
        threshold_present_iff_activated,
        threshold_shape_supported,
        source_integer_numerically_supported,
    )

    # -- the composition ------------------------------------------------------

    @derived(
        DotProductBounds,
        activation=activation,
        weight=weight,
        accumulator=accumulator_type,
        output=output_type,
        profile=profile,
        scope=invocation_scope,
        runtime_writable=allow_absent(runtime_writable_weights),
    )
    def integer_bounds(
        *,
        activation: SourceOperand,
        weight: SourceOperand,
        accumulator: QONNXDataType,
        output: QONNXDataType,
        profile: MvauComputationProfile,
        scope: InvocationScope,
        runtime_writable: object,
    ) -> object:
        if profile.accumulation is not AccumulationMode.INTEGER or profile.fuses_activation:
            return Absent()
        try:
            premise = mvau_integer_premise_from_operands(
                activation,
                weight,
                accumulator_datatype=accumulator,
                output_datatype=output,
                invocation_scope=scope,
                runtime_writable=runtime_writable is not ABSENT and bool(runtime_writable),
                runtime_promise=None,
            )
            return analyze_integer_dot_product(premise)
        except ValueError as error:
            return reject("mvau-integer-premise", str(error))

    family_interface = Subspace(
        MatmulInterface,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation.datatype,
        weight_type=weight.datatype,
        accumulator_type=accumulator_type,
        output_type=output_type,
        computation_profile=profile,
        integer_bounds=integer_bounds,
    )
    interface_binding = ImplementationBinding(("family_interface",))

    kernel = SubspaceChoice(
        {
            "dot_product": Subspace(
                DotProductKernel,
                repetitions=repetitions,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
                activation_type=activation.datatype,
                weight_type=weight.datatype,
                accumulator_type=accumulator_type,
                # This core drives the accumulator straight out, so the output
                # type *is* the accumulator type.  Derived onto the tensor, not
                # read back off it.
                output_type=output_type,
                narrow_weights=effective_narrow_weights,
                computation_profile=profile,
                numerical_support=numerical_support,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
                initializer_present=weight.initializer_present,
                integer_bounds=integer_bounds,
            ),
            "batch_interleaved": Subspace(
                BatchInterleavedKernel,
                repetitions=repetitions,
                matrix_width=matrix_width,
                matrix_height=matrix_height,
                activation_type=activation.datatype,
                weight_type=weight.datatype,
                accumulator_type=accumulator_type,
                output_type=output_type,
                narrow_weights=effective_narrow_weights,
                computation_profile=profile,
                numerical_support=numerical_support,
                target_dsp=target_dsp,
                clock_period_ns=clock_period_ns,
                integer_bounds=integer_bounds,
            ),
        },
    )

    # -- the projections ------------------------------------------------------

    # -- what this operation is authoritative for -----------------------------

    def expected_outputs(self) -> dict[str, tuple[tuple[int, ...] | None, Any]]:
        source = self.source
        use = self
        datatype, domain = use.operand_type("result"), use.operand_domain("result")
        shape = None
        if isinstance(domain, Decided):
            # Public matrix rows flatten the source leading coordinates.
            shape = (*source.operand("activation").shape[:-1], domain.value.extents[-1])
        return {"output": (shape, datatype.value if isinstance(datatype, Decided) else None)}

    # -- executing the source semantics ----------------------------------------


class MvauDataflowOp(DataflowOp):
    space_type = MvauSpace

    def execute_node(self, context: Any, graph: Any) -> None:
        """Compute this node in ONNX, on both of its semantic axes."""

        del graph
        source = self.space.source
        node = self.space.node_snapshot()
        thresholds = (
            context[node.input[2]] if source.has("threshold") and len(node.input) > 2 else None
        )
        profile = mvau_profile(source)
        if profile.accumulation is AccumulationMode.INTEGER and not profile.fuses_activation:
            answer = self.space.answer(self.space_type.source_numerical_report)
            if not isinstance(answer, Decided):
                raise DataflowOpError(
                    "integer MVAU execution is numerically unsupported",
                    getattr(answer, "findings", ()),
                )
            report = answer.value
            writable = self.space.answer(self.space_type.runtime_writable_weights)
            runtime = isinstance(writable, Decided) and bool(writable.value)
            try:
                result = execute_mvau_integer(
                    activation=context[node.input[0]],
                    weights=context[node.input[1]],
                    support_report=report,
                    fixed_initializer=(
                        None if runtime else source.operand("weight").initializer_value
                    ),
                )
            except ValueError as error:
                raise DataflowOpError(str(error)) from error
        else:
            result = execute_mvau(
                activation=context[node.input[0]],
                weight=context[node.input[1]],
                thresholds=thresholds,
                profile=profile,
                output_type=cast(Any, source.attributes["output_type"]),
                activation_bias=int(cast(int, source.attributes["activation_bias"])),
            )
        expected = self.space.expected_outputs()["output"][0]
        if expected is None:
            raise DataflowOpError(f"{node.name} cannot state the shape of its own output")
        context[node.output[0]] = result.reshape(expected)


__all__ = ["MvauSpace", "MvauDataflowOp", "mvau_profile", "origin_nodes"]
