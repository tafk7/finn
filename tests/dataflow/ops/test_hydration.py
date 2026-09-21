# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact immutable uses and partial source-independent public type queries."""

import pytest
import numpy as np
from dataclasses import replace
from onnx import helper
from qonnx.core.datatype import DataType

from dataflow.ops.test_dataflow_op import _mvau_model, _replay_model
from dataflow.ops.mvau.test_source_semantics import (
    _model as _semantic_mvau_model,
    _bound as _semantic_mvau_bound,
    _configure_mvau_point,
    MATRIX_HEIGHT,
)
from finn.dataflow._engine import Absent, Decided, Unresolved, RequestError
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.binding import ImplementationBinding, ChoiceBinding
from finn.dataflow.space.declarations import (
    Decision,
    Subspace,
    Space,
    ConstraintGroup,
    Readiness,
    Projection,
    constraint,
    derived,
)
from finn.dataflow.kernels.replay import ActivationReplayKernel
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.native import (
    serialize_choices,
    FINGERPRINT_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    NativeAttribute,
)
from finn.dataflow.ops.replay.op import ActivationReplayOp


def test_partial_mvau_type_is_common_family_contract_without_target_or_selection():
    model = _mvau_model()
    use = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    assert isinstance(use.resolve(), Unresolved)
    assert use.operand_type("result") == Decided(DataType["INT32"])
    assert use.operand_domain("result").value.extents == (2, 4)
    assert serialize_choices(use.root) == {}
    assert use.root.expected_outputs()["output"] == ((2, 4), DataType["INT32"])


def test_fixed_replay_type_and_domain_do_not_choose_folding():
    model = _replay_model()
    use = ActivationReplayOp(model.graph.node[0]).hydrate(model)
    assert isinstance(use.resolve(), Decided)
    assert use.operand_type("result") == use.operand_type("activation")
    assert use.operand_domain("result").value.extents == (8, 8)
    assert serialize_choices(use.root) == {}


def test_successors_keep_old_point_and_refuse_mixed_or_inactive_occurrences():
    model = _mvau_model()
    original = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    selected = original.commit({"kernel__case": "dot_product"})
    assert isinstance(original.resolve(), Unresolved)
    kernel = selected.require_implementation()
    next_use = selected.commit({"kernel__dot_product__pe": 1})
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        next_use.require_implementation(kernel)
    sibling = selected.root.kernel.alternative("batch_interleaved")
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        selected.require_implementation(sibling)
    assert not isinstance(
        ImplementationBinding(("kernel", "batch_interleaved")).resolve(selected.root), Decided
    )
    other = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    with pytest.raises(DataflowOpError, match="association"):
        selected.successor(other.root)


def test_choice_nominal_types_and_inactive_branch_refuse():
    model = _mvau_model()
    use = MvauDataflowOp(model.graph.node[0]).hydrate(model).commit({"kernel__case": "dot_product"})
    with pytest.raises((TypeError, ValueError)):
        use.commit({"kernel__dot_product__pe": True})
    with pytest.raises((TypeError, ValueError, RequestError)):
        use.commit({"kernel__batch_interleaved__pe": 1})


def test_native_partial_roundtrip_uses_explicit_keys():
    model = _mvau_model()
    use = MvauDataflowOp(model.graph.node[0]).hydrate(model).commit({"kernel__case": "dot_product"})
    values = serialize_choices(use.root)
    values[FINGERPRINT_ATTRIBUTE] = NativeAttribute("s", use.root.local_problem_fingerprint)
    values[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", use.root.schema_version)
    model.graph.node[0].attribute.extend(value.proto(key) for key, value in values.items())
    restored = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    assert restored.resolve().value.__class__ is use.resolve().value.__class__
    assert serialize_choices(restored.root) == serialize_choices(use.root)
    assert restored.operand_type("result") == use.operand_type("result")


def test_use_cannot_substitute_its_authored_implementation_interface_or_operands():
    model = _mvau_model()
    use = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    with pytest.raises(DataflowOpError, match="contradicts"):
        replace(use, implementation=ImplementationBinding(("family_interface",)))
    with pytest.raises(DataflowOpError, match="contradicts"):
        replace(use, interface=ImplementationBinding(("kernel",)))
    with pytest.raises(DataflowOpError, match="contradicts"):
        replace(use, operands=(replace(use.operands[0], role="weights"), *use.operands[1:]))
    with pytest.raises(DataflowOpError, match="contradicts"):
        replace(use, choices=())


class TwinReplay(ActivationReplayOp):
    mirror = Subspace(
        ActivationReplayKernel,
        repetitions=ActivationReplayOp.repetitions,
        matrix_width=ActivationReplayOp.matrix_width,
        matrix_height=ActivationReplayOp.matrix_height,
        activation_type=ActivationReplayOp.activation.datatype,
    )
    enabled = Decision(bool, values=(False, True))
    offset = Decision(int, values=(0, 1))
    choice_bindings = (
        *ActivationReplayOp.choice_bindings,
        ChoiceBinding("mirror_pe", ("mirror",), "pe"),
        ChoiceBinding("mirror_simd", ("mirror",), "simd"),
        ChoiceBinding("enabled", (), "enabled"),
        ChoiceBinding("offset", (), "offset"),
    )


def test_repeated_definition_choices_and_false_zero_remain_distinct():
    model = _replay_model()
    use = TwinReplay(model.graph.node[0]).hydrate(model)
    selected = use.commit({"kernel__simd": 2, "mirror_simd": 4, "enabled": False, "offset": 0})
    encoded = serialize_choices(selected.root)
    assert encoded["kernel__simd"] == NativeAttribute("i", 2)
    assert encoded["mirror_simd"] == NativeAttribute("i", 4)
    assert encoded["enabled"] == NativeAttribute("i", 0)
    assert encoded["offset"] == NativeAttribute("i", 0)
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        selected.require_implementation(selected.root.mirror)
    assert serialize_choices(use.root) == {}
    persisted = dict(encoded)
    persisted[FINGERPRINT_ATTRIBUTE] = NativeAttribute("s", selected.root.local_problem_fingerprint)
    persisted[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", selected.root.schema_version)
    model.graph.node[0].attribute.extend(value.proto(key) for key, value in persisted.items())
    restored = TwinReplay(model.graph.node[0]).hydrate(model)
    assert serialize_choices(restored.root) == encoded
    assert restored.root.answer(TwinReplay.enabled).value is False
    assert restored.root.answer(TwinReplay.offset).value == 0


def test_fixed_conditional_target_preserves_unresolved_and_inactive_answers():
    class Leaf(Space):
        value = Decision(int, values=(0,))

    class Parent(Space):
        enabled = Decision(bool, values=(False, True))
        child = Subspace(Leaf, when=enabled)

    root = Parent.start({})
    binding = ImplementationBinding(("child",))
    assert isinstance(binding.resolve(root), Unresolved)
    assert isinstance(binding.resolve(root.assign(Parent.enabled, False)), Absent)
    assert isinstance(binding.resolve(root.assign(Parent.enabled, True)), Decided)


def test_optional_rejecting_op_query_does_not_gate_source_or_public_type():
    class OptionalCheck(ActivationReplayOp):
        @derived(int)
        def diagnostic_value():
            return 1

        @constraint(value=diagnostic_value)
        def optional_rejects(*, value):
            return value == 0

        optional_accepts = ConstraintGroup(optional_rejects)
        optional_ready = Readiness(properties=(diagnostic_value,), constraints=optional_accepts)
        optional_query = Projection(
            diagnostic_value,
            readiness=optional_ready,
            constraints=optional_accepts,
        )

    model = _replay_model()
    use = OptionalCheck(model.graph.node[0]).hydrate(model)
    assert use.root.assess_source().verdict is True
    assert isinstance(use.operand_type("result"), Decided)
    assert isinstance(use.root.assess_view(OptionalCheck.optional_query).accepted_answer, Absent)


def test_datatype_inference_clears_an_unaccepted_cached_result_type():
    model = _mvau_model()
    model.graph.node[0].attribute.append(helper.make_attribute("accDataType", "INT8"))
    operation = MvauDataflowOp(model.graph.node[0])
    assert isinstance(operation.hydrate(model).operand_type("result"), Absent)
    operation.infer_node_datatype(model)
    assert not any(
        annotation.tensor_name == "output" and entry.key == "finn_datatype"
        for annotation in model.graph.quantization_annotation
        for entry in annotation.quant_parameter_tensor_names
    )
    assert isinstance(operation.hydrate(model).operand_type("result"), Absent)


def test_fused_type_checks_source_thresholds_without_claiming_bare_kernel_support():
    missing = _semantic_mvau_model(no_activation=False, output_type="UINT4")
    missing_use = MvauDataflowOp(missing.graph.node[0]).hydrate(missing)
    assert isinstance(missing_use.operand_type("activation"), Decided)
    rejected = missing_use.operand_type("result")
    assert isinstance(rejected, Absent)
    assert any(f.code == "mvau-threshold-presence-mismatch" for f in rejected.findings)

    thresholds = np.zeros((MATRIX_HEIGHT, 1), dtype=np.float32)
    model = _semantic_mvau_model(no_activation=False, thresholds=thresholds, output_type="UINT4")
    use = MvauDataflowOp(model.graph.node[0]).hydrate(model)
    assert use.operand_type("result") == Decided(DataType["UINT4"])
    chosen = _configure_mvau_point(_semantic_mvau_bound(model))
    logical = chosen.dataflow.accepted_answer
    assert isinstance(logical, Absent)
    assert any(f.code == "mvau-kernel-fuses-no-activation" for f in logical.findings)
