# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact immutable uses and partial source-independent public type queries."""

import pytest
import numpy as np
import importlib.util
from dataclasses import replace
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from parked.dataflow.ops.test_dataflow_op import Build, _mvau_model, _replay_model
from parked.dataflow.ops.mvau.test_source_semantics import (
    _model as _semantic_mvau_model,
    _configure_mvau_point,
    MATRIX_HEIGHT,
)
from finn.kernels._engine import Absent, Decided, Unresolved, RequestError
from finn.parked.dataflow.ops.base import DataflowOp, DataflowOpError
from finn.parked.dataflow.ops.binding import ImplementationBinding, ChoiceBinding
from finn.kernels.space.declarations import (
    Decision,
    Input,
    Subspace,
    Space,
    ConstraintGroup,
    Readiness,
    Projection,
    constraint,
    derived,
    AuthoringError,
)
from finn.parked.dataflow.kernels.replay import ActivationReplayKernel
from finn.parked.dataflow.ops.mvau.op import MvauSpace
from finn.parked.dataflow.ops.native import (
    choice_schema,
    serialize_choices,
    SCHEMA_VERSION_ATTRIBUTE,
    NativeAttribute,
)
from finn.parked.dataflow.ops.replay.op import ReplaySpace
from finn.parked.dataflow.ops.reconstruction import build_space
from finn.parked.dataflow.ops.source import SourceError
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from parked.dataflow.ops.factory import make_op


def test_partial_mvau_type_is_common_family_contract_without_target_or_selection():
    model = _mvau_model()
    use = model.get_customop_wrapper(model.graph.node[0]).space
    assert isinstance(use.resolve_implementation(), Unresolved)
    assert use.operand_type("result") == Decided(DataType["INT32"])
    assert use.operand_domain("result").value.extents == (2, 4)
    assert serialize_choices(use) == {}
    assert use.expected_outputs()["output"] == ((2, 4), DataType["INT32"])


def test_fixed_replay_type_and_domain_do_not_choose_folding():
    model = _replay_model()
    use = model.get_customop_wrapper(model.graph.node[0]).space
    assert isinstance(use.resolve_implementation(), Decided)
    assert use.operand_type("result") == use.operand_type("activation")
    assert use.operand_domain("result").value.extents == (8, 8)
    assert serialize_choices(use) == {}


def test_successors_keep_old_point_and_refuse_mixed_or_inactive_occurrences():
    model = _mvau_model()
    original = model.get_customop_wrapper(model.graph.node[0]).space
    selected = original.commit_choices({"kernel__case": "dot_product"})
    assert isinstance(original.resolve_implementation(), Unresolved)
    kernel = selected.require_implementation()
    next_use = selected.commit_choices({"kernel__dot_product__pe": 1})
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        next_use.require_implementation(kernel)
    sibling = selected.kernel.alternative("batch_interleaved")
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        selected.require_implementation(sibling)
    assert not isinstance(
        ImplementationBinding(("kernel", "batch_interleaved")).resolve(selected), Decided
    )


def test_choice_nominal_types_and_inactive_branch_refuse():
    model = _mvau_model()
    use = model.get_customop_wrapper(model.graph.node[0]).space.commit_choices(
        {"kernel__case": "dot_product"}
    )
    with pytest.raises((TypeError, ValueError)):
        use.commit_choices({"kernel__dot_product__pe": True})
    with pytest.raises((TypeError, ValueError, RequestError)):
        use.commit_choices({"kernel__batch_interleaved__pe": 1})


def test_native_partial_roundtrip_uses_explicit_keys():
    model = _mvau_model()
    use = model.get_customop_wrapper(model.graph.node[0]).space.commit_choices(
        {"kernel__case": "dot_product"}
    )
    values = serialize_choices(use)
    values[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", use.schema_version)
    model.graph.node[0].attribute.extend(value.proto(key) for key, value in values.items())
    restored = model.get_customop_wrapper(model.graph.node[0]).space
    assert (
        restored.resolve_implementation().value.__class__
        is use.resolve_implementation().value.__class__
    )
    assert serialize_choices(restored) == serialize_choices(use)
    assert restored.operand_type("result") == use.operand_type("result")


def test_factory_returns_adapter_and_separate_space():
    model = _mvau_model()
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, DataflowOp)
    assert not isinstance(operation, Space)
    assert isinstance(operation.space, MvauSpace)
    assert not hasattr(operation, "hydrate")
    assert not hasattr(operation.space, "onnx_node")


def test_malformed_authored_source_bindings_refuse():
    class SwappedSources(MvauSpace):
        operand_bindings = (
            replace(MvauSpace.operand_bindings[0], role="weights"),
            replace(MvauSpace.operand_bindings[1], role="activation"),
            MvauSpace.operand_bindings[2],
        )

    model = _mvau_model()
    with pytest.raises(AuthoringError, match="different sources"):
        build_space(SwappedSources, model, model.graph.node[0])


class TwinReplay(ReplaySpace):
    mirror = Subspace(
        ActivationReplayKernel,
        repetitions=ReplaySpace.repetitions,
        matrix_width=ReplaySpace.matrix_width,
        matrix_height=ReplaySpace.matrix_height,
        activation_type=ReplaySpace.activation.datatype,
    )
    enabled = Decision(bool, values=(False, True))
    offset = Decision(int, values=(0, 1))
    choice_bindings = (
        *ReplaySpace.choice_bindings,
        ChoiceBinding("mirror_simd", ("mirror",), "simd"),
        ChoiceBinding("enabled", (), "enabled"),
        ChoiceBinding("offset", (), "offset"),
    )


def test_repeated_definition_choices_and_false_zero_remain_distinct():
    model = _replay_model()
    use = build_space(TwinReplay, model, model.graph.node[0])
    selected = use.commit_choices(
        {"kernel__simd": 2, "mirror_simd": 4, "enabled": False, "offset": 0}
    )
    encoded = serialize_choices(selected)
    assert encoded["kernel__simd"] == NativeAttribute("i", 2)
    assert encoded["mirror_simd"] == NativeAttribute("i", 4)
    assert encoded["enabled"] == NativeAttribute("i", 0)
    assert encoded["offset"] == NativeAttribute("i", 0)
    with pytest.raises(DataflowOpError, match="exact node occurrence"):
        selected.require_implementation(selected.mirror)
    assert serialize_choices(use) == {}
    persisted = dict(encoded)
    persisted[SCHEMA_VERSION_ATTRIBUTE] = NativeAttribute("i", selected.schema_version)
    model.graph.node[0].attribute.extend(value.proto(key) for key, value in persisted.items())
    restored = build_space(TwinReplay, model, model.graph.node[0])
    assert serialize_choices(restored) == encoded
    assert restored.answer(TwinReplay.enabled).value is False
    assert restored.answer(TwinReplay.offset).value == 0


class ReplayEnclosure(Space):
    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    renamed = Subspace(
        ActivationReplayKernel,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation_type,
    )


class RenestedReplay(ReplaySpace):
    kernel = Subspace(
        ReplayEnclosure,
        repetitions=ReplaySpace.repetitions,
        matrix_width=ReplaySpace.matrix_width,
        matrix_height=ReplaySpace.matrix_height,
        activation_type=ReplaySpace.activation.datatype,
    )
    implementation_binding = ImplementationBinding(("kernel", "renamed"))
    choice_bindings = (ChoiceBinding("kernel__simd", ("kernel", "renamed"), "simd"),)


def test_native_semantic_choice_survives_private_implementation_renesting(tmp_path):
    model = _replay_model()
    original = make_op(model)
    selected = original.space.commit_choices({"kernel__simd": 2})
    original.save_space(selected)
    encoded = serialize_choices(original.space)
    original_path = choice_schema(original.space)[0].choice.reference.path

    renamed = make_op(model, space_type=RenestedReplay)
    assert renamed.space.family == original.space.family
    assert renamed.space.schema_version == original.space.schema_version
    assert serialize_choices(renamed.space) == encoded == {"kernel__simd": NativeAttribute("i", 2)}
    renamed_path = choice_schema(renamed.space)[0].choice.reference.path
    assert renamed_path != original_path
    assert renamed.space.require_implementation().answer(ActivationReplayKernel.simd) == Decided(2)
    renamed.save_space(selected)

    path = tmp_path / "renested-replay.onnx"
    model.save(str(path))
    restored_model = ModelWrapper(str(path))
    restored = make_op(restored_model, space_type=RenestedReplay)
    assert serialize_choices(restored.space) == encoded
    assert choice_schema(restored.space)[0].choice.reference.path == renamed_path
    assert restored.space.require_implementation().answer(ActivationReplayKernel.simd) == Decided(2)
    assert restored.space.operand_type("result") == original.space.operand_type("result")
    assert restored.space.operand_domain("result") == original.space.operand_domain("result")


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
    class OptionalCheck(ReplaySpace):
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
    use = build_space(OptionalCheck, model, model.graph.node[0])
    assert use.assess_source().verdict is True
    assert isinstance(use.operand_type("result"), Decided)
    assert isinstance(use.assess_view(OptionalCheck.optional_query).accepted_answer, Absent)


def test_datatype_inference_clears_an_unaccepted_cached_result_type():
    model = _mvau_model()
    model.graph.node[0].attribute.append(helper.make_attribute("accDataType", "INT8"))
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation.space.operand_type("result"), Absent)
    operation.infer_node_datatype(model)
    assert not any(
        annotation.tensor_name == "output" and entry.key == "finn_datatype"
        for annotation in model.graph.quantization_annotation
        for entry in annotation.quant_parameter_tensor_names
    )
    assert isinstance(operation.space.operand_type("result"), Absent)


def test_fused_type_checks_source_thresholds_without_claiming_bare_kernel_support():
    missing = _semantic_mvau_model(no_activation=False, output_type="UINT4")
    missing_use = missing.get_customop_wrapper(missing.graph.node[0]).space
    assert isinstance(missing_use.operand_type("activation"), Decided)
    rejected = missing_use.operand_type("result")
    assert isinstance(rejected, Absent)
    assert any(f.code == "mvau-threshold-presence-mismatch" for f in rejected.findings)

    thresholds = np.zeros((MATRIX_HEIGHT, 1), dtype=np.float32)
    model = _semantic_mvau_model(no_activation=False, thresholds=thresholds, output_type="UINT4")
    use = model.get_customop_wrapper(model.graph.node[0]).space
    assert use.operand_type("result") == Decided(DataType["UINT4"])
    chosen = _configure_mvau_point(
        model.get_customop_wrapper(model.graph.node[0]).set_context(Build()).space
    )
    logical = chosen.dataflow.accepted_answer
    assert isinstance(logical, Absent)
    assert any(f.code == "mvau-kernel-fuses-no-activation" for f in logical.findings)


def test_selected_expansion_api_is_removed_and_not_a_wrong_target_route():
    for name in (
        "selected_graph",
        "selected_snapshot",
        "selected_construction",
        "selected_facts",
        "selected_source_provenance",
        "selected_source_semantics",
        "selected_construction_identity",
        "plan_selected_publication",
        "publish_selected",
        "rebind_selected",
    ):
        assert not hasattr(DataflowOp, name)
    for module in (
        "selected",
        "selected_registry",
        "selected_transform_registry",
        "selected_transforms",
        "selected_verification",
        "mvau.selected",
        "replay.selected",
    ):
        assert importlib.util.find_spec(f"finn.parked.dataflow.ops.{module}") is None


def test_duplicate_source_logical_annotations_refuse_at_source_capture():
    model = _replay_model()
    original = model.graph.quantization_annotation[0]
    model.graph.quantization_annotation.add().CopyFrom(original)
    with pytest.raises(SourceError, match="duplicate logical datatype annotations"):
        model.get_customop_wrapper(model.graph.node[0]).space
