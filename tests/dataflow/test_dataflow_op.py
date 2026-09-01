# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.authoring import DataflowOpError, NodeAttrCodec
from finn.dataflow.design import QualifiedPath
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.op import dataflow_problem_fingerprint
from finn.dataflow.resolution import NetworkRef
from finn.dataflow.testing import DataflowOpConformanceCase, assert_dataflow_op_conforms

from dataflow.synthetic_op import (
    SyntheticDataflowOp,
    SyntheticMode,
    SyntheticPaths,
    ZeroDecisionDataflowOp,
)


def _model(extent: int = 4, op_type: str = "SyntheticDataflowOp") -> ModelWrapper:
    node = helper.make_node(
        op_type,
        ["x"],
        ["y"],
        name="synthetic0",
        domain="dataflow.synthetic_op",
        dataflow_scope_id="synthetic-scope",
    )
    graph = helper.make_graph(
        [node],
        "synthetic",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [extent])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [extent])],
    )
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="dataflow-op-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("dataflow.synthetic_op", 1),
            ],
        )
    )
    model.set_tensor_datatype("x", DataType["INT8"])
    model.set_tensor_datatype("y", DataType["INT8"])
    return model


@dataclass
class _BuildConfig:
    synth_clk_period_ns: float


def _config(clock: float = 5.0) -> _BuildConfig:
    return _BuildConfig(clock)


def _complete_assignments() -> dict[QualifiedPath, object]:
    return {
        SyntheticPaths.LANES: 2,
        SyntheticPaths.ENABLED: True,
        SyntheticPaths.LABEL: "alpha",
        SyntheticPaths.MODE: SyntheticMode.SECOND,
    }


def _wrapped(model: ModelWrapper) -> SyntheticDataflowOp:
    instance = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(instance, SyntheticDataflowOp)
    return instance


def _change_synthetic_shape(model: ModelWrapper) -> None:
    model.set_tensor_shape("x", [8])
    model.set_tensor_shape("y", [8])


def test_mvau_dsp_block_keeps_its_pre_move_problem_identity() -> None:
    assert (
        dataflow_problem_fingerprint({QualifiedPath("target.dsp_block"): DspBlock.DSP58})
        == "cf8ff27c13c04b2049026a5b360a9cf813bbdc3c511cab62639b0f80457e0f7f"
    )


def test_modelwrapper_attaches_exact_model_and_bare_wrapper_rejects_evaluation() -> None:
    model = _model()
    wrapped = _wrapped(model)
    assert wrapped._attached_model() is model

    bare = getCustomOp(model.graph.node[0])
    assert isinstance(bare, SyntheticDataflowOp)
    with pytest.raises(DataflowOpError) as missing:
        bare.problem_instance(_config())
    assert {finding.code for finding in missing.value.findings} == {"dataflow-model-required"}


def test_static_space_cache_is_per_subclass_and_context_independent() -> None:
    class OtherSyntheticDataflowOp(SyntheticDataflowOp):
        spec_build_count = 0

        @classmethod
        def dataflow_family_id(cls) -> str:
            return "test.synthetic.other"

    SyntheticDataflowOp.clear_validated_space_cache()
    OtherSyntheticDataflowOp.clear_validated_space_cache()
    first = SyntheticDataflowOp.validated_design_space()
    assert SyntheticDataflowOp.validated_design_space() is first
    second = OtherSyntheticDataflowOp.validated_design_space()
    assert second is not first
    assert SyntheticDataflowOp.spec_build_count == 1
    assert OtherSyntheticDataflowOp.spec_build_count == 1


def test_reserved_metadata_names_cannot_be_reused_by_a_contribution() -> None:
    class CollidingSyntheticDataflowOp(SyntheticDataflowOp):
        @classmethod
        def dataflow_family_id(cls) -> str:
            return "test.synthetic.collision"

        @classmethod
        def decision_nodeattrs(cls) -> Mapping[QualifiedPath, NodeAttrCodec]:
            codecs = dict(super().decision_nodeattrs())
            codecs[SyntheticPaths.LANES] = NodeAttrCodec.integer(cls.SCOPE_ID_ATTR)
            return codecs

    CollidingSyntheticDataflowOp.clear_validated_space_cache()
    with pytest.raises(TypeError, match="reserved dataflow metadata"):
        CollidingSyntheticDataflowOp.validated_design_space()


def test_absent_defaults_are_unassigned_and_projection_is_read_only() -> None:
    model = _model()
    wrapped = _wrapped(model)
    before = model.model.SerializeToString(deterministic=True)
    first = wrapped.problem_instance(_config())
    second = wrapped.problem_instance(_config())

    assert wrapped.get_nodeattr("dataflow_lanes") == 0
    assert wrapped.read_assignments() == {}
    assert first == second
    assert model.model.SerializeToString(deterministic=True) == before


def test_scalar_codecs_partial_and_complete_reload(tmp_path: Path) -> None:
    model = _model()
    wrapped = _wrapped(model)
    partial = wrapped.commit_dataflow_assignments(
        _config(),
        {
            SyntheticPaths.LANES: 2,
            SyntheticPaths.ENABLED: True,
        },
    )
    assert partial.point.assignments == {
        SyntheticPaths.LANES: 2,
        SyntheticPaths.ENABLED: True,
    }
    model_path = tmp_path / "partial.onnx"
    model.save(model_path)
    partial_reloaded = _wrapped(ModelWrapper(str(model_path)))
    assert (
        partial_reloaded.hydrate_dataflow_point(_config()).assignments == partial.point.assignments
    )

    complete = partial_reloaded.commit_dataflow_assignments(
        _config(),
        {
            SyntheticPaths.LABEL: "alpha",
            SyntheticPaths.MODE: SyntheticMode.SECOND,
        },
    )
    assert complete.point.assignments == _complete_assignments()
    resolved = partial_reloaded.resolve_dataflow(_config())
    assert isinstance(resolved.result, NetworkRef)
    assert resolved.source_association == "synthetic-scope"
    complete_path = tmp_path / "complete.onnx"
    partial_reloaded._attached_model().save(complete_path)
    restored = _wrapped(ModelWrapper(str(complete_path))).resolve_dataflow(_config())
    assert restored.point.assignments == resolved.point.assignments
    assert restored.result == resolved.result
    assert restored.source_scope_id == resolved.source_scope_id


@pytest.mark.parametrize(
    "requested",
    [
        {SyntheticPaths.LANES: 3},
        {SyntheticPaths.LANES: 2, SyntheticPaths.LABEL: "not-in-domain"},
    ],
)
def test_rejected_assignment_transaction_leaves_node_byte_identical(
    requested: dict[QualifiedPath, object],
) -> None:
    model = _model()
    wrapped = _wrapped(model)
    before = wrapped.onnx_node.SerializeToString(deterministic=True)
    with pytest.raises((DataflowOpError, TypeError)):
        wrapped.commit_dataflow_assignments(_config(), requested)
    assert wrapped.onnx_node.SerializeToString(deterministic=True) == before


@pytest.mark.parametrize("change", ["graph", "build"])
def test_hydration_rejects_changed_problem_identity(change: str) -> None:
    model = _model()
    wrapped = _wrapped(model)
    wrapped.commit_dataflow_assignments(_config(), {SyntheticPaths.LANES: 2})
    config = _config()
    if change == "graph":
        model.set_tensor_shape("x", [8])
        model.set_tensor_shape("y", [8])
    else:
        config = _config(3.0)
    with pytest.raises(DataflowOpError) as stale:
        wrapped.hydrate_dataflow_point(config)
    assert {finding.code for finding in stale.value.findings} == {
        "dataflow-selection-problem-mismatch"
    }


def test_hydration_rejects_changed_family_identity() -> None:
    wrapped = _wrapped(_model())
    wrapped.commit_dataflow_assignments(_config(), {SyntheticPaths.LANES: 2})
    wrapped.set_nodeattr(wrapped.FAMILY_VERSION_ATTR, "obsolete")
    with pytest.raises(DataflowOpError) as stale:
        wrapped.hydrate_dataflow_point(_config())
    assert {finding.code for finding in stale.value.findings} == {
        "dataflow-selection-family-mismatch"
    }


def test_raw_decision_attribute_without_identity_metadata_is_not_trusted() -> None:
    wrapped = _wrapped(_model())
    wrapped.set_nodeattr("dataflow_lanes", 2)
    with pytest.raises(DataflowOpError) as incomplete:
        wrapped.hydrate_dataflow_point(_config())
    assert {finding.code for finding in incomplete.value.findings} == {
        "dataflow-selection-metadata-missing"
    }


def test_scope_identity_is_stored_independently_of_node_name() -> None:
    model = _model()
    wrapped = _wrapped(model)
    wrapped.commit_dataflow_assignments(_config(), _complete_assignments())
    original = wrapped.resolve_dataflow(_config())
    scope_id = wrapped.dataflow_scope_id()
    assert isinstance(scope_id, str) and scope_id
    wrapped.clear_dataflow_assignments()
    assert wrapped.dataflow_scope_id() == scope_id
    wrapped.commit_dataflow_assignments(_config(), _complete_assignments())
    assert wrapped.dataflow_scope_id() == scope_id
    wrapped.onnx_node.name = "renamed"
    assert wrapped.dataflow_scope_id() == scope_id
    renamed = wrapped.resolve_dataflow(_config())
    assert renamed.result == original.result

    cloned_model = ModelWrapper(model.model, make_deepcopy=True)
    cloned = _wrapped(cloned_model)
    assert cloned.dataflow_scope_id() == scope_id
    renewed = cloned.renew_dataflow_scope_id()
    assert renewed != scope_id
    assert cloned.dataflow_scope_id() == renewed
    assert cloned.read_assignments() == {}


def test_zero_decision_operation_resolves_with_operation_owned_scope() -> None:
    model = _model(op_type="ZeroDecisionDataflowOp")
    operation = model.get_customop_wrapper(model.graph.node[0])
    assert isinstance(operation, ZeroDecisionDataflowOp)
    resolved = operation.resolve_dataflow(_config())
    assert resolved.point.assignments == {}
    assert resolved.source_scope_id == "synthetic-scope"


def test_scope_identity_can_be_initialized_before_any_assignment() -> None:
    model = _model()
    node = model.graph.node[0]
    scope_attribute = next(
        attribute for attribute in node.attribute if attribute.name == "dataflow_scope_id"
    )
    node.attribute.remove(scope_attribute)
    operation = _wrapped(model)
    with pytest.raises(DataflowOpError) as missing:
        operation.dataflow_scope_id()
    assert {finding.code for finding in missing.value.findings} == {"dataflow-scope-id-missing"}
    assert operation.initialize_dataflow_scope_id("created-before-selection") == (
        "created-before-selection"
    )
    assert operation.read_assignments() == {}


@pytest.mark.parametrize("invalid_scope", ["", 0, False])
def test_scope_identity_rejects_explicit_invalid_values(invalid_scope: object) -> None:
    model = _model()
    node = model.graph.node[0]
    scope_attribute = next(
        attribute for attribute in node.attribute if attribute.name == "dataflow_scope_id"
    )
    node.attribute.remove(scope_attribute)
    operation = _wrapped(model)
    with pytest.raises(ValueError, match="non-empty string"):
        operation.initialize_dataflow_scope_id(cast(str, invalid_scope))


def test_unknown_attributes_are_not_assignments_and_clear_and_replace_are_exact() -> None:
    model = _model()
    wrapped = _wrapped(model)
    wrapped.onnx_node.attribute.append(helper.make_attribute("unrelated", 17))
    wrapped.commit_dataflow_assignments(_config(), _complete_assignments())
    assert QualifiedPath("unrelated") not in wrapped.read_assignments()

    replacement = dict(_complete_assignments())
    replacement[SyntheticPaths.LANES] = 4
    wrapped.replace_dataflow_assignments(_config(), replacement)
    assert wrapped.read_assignments() == replacement
    wrapped.clear_dataflow_assignments()
    assert wrapped.read_assignments() == {}
    assert (
        next(
            attribute.i
            for attribute in wrapped.onnx_node.attribute
            if attribute.name == "unrelated"
        )
        == 17
    )


def test_nodeattr_schema_is_stable_and_contains_no_result_fields() -> None:
    model = _model()
    wrapped = _wrapped(model)
    before = dict(wrapped.get_nodeattr_types())
    wrapped.commit_dataflow_assignments(_config(), {SyntheticPaths.LANES: 2})
    assert dict(wrapped.get_nodeattr_types()) == before
    names = set(before)
    assert "semantic.synthetic.region" not in names
    assert "semantic.synthetic.result" not in names


def test_reference_execution_remains_normal_customop_behavior() -> None:
    model = _model()
    wrapped = _wrapped(model)
    context = {"x": np.arange(4, dtype=np.float32), "y": np.empty((4,), dtype=np.float32)}
    wrapped.execute_node(context, model.graph)
    np.testing.assert_array_equal(context["y"], context["x"])


def test_synthetic_operation_passes_shared_conformance_harness(tmp_path: Path) -> None:
    case = DataflowOpConformanceCase(
        model=_model(),
        node_name="synthetic0",
        operation_type=SyntheticDataflowOp,
        config=_config(),
        complete_assignments=_complete_assignments(),
        rejected_assignments={SyntheticPaths.LANES: 3},
        reload_path=tmp_path / "synthetic-conformance.onnx",
        stale_config=_config(3.0),
        mutate_graph_problem=_change_synthetic_shape,
    )
    result = assert_dataflow_op_conforms(case)
    assert isinstance(result.original.result, NetworkRef)


def test_zero_decision_operation_passes_shared_conformance_harness(tmp_path: Path) -> None:
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=_model(op_type="ZeroDecisionDataflowOp"),
            node_name="synthetic0",
            operation_type=ZeroDecisionDataflowOp,
            config=_config(),
            complete_assignments={},
            rejected_assignments={SyntheticPaths.LANES: 1},
            reload_path=tmp_path / "zero-decision-conformance.onnx",
        )
    )
    assert result.original.point.assignments == {}
    assert isinstance(result.original.result, NetworkRef)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_dataflow_operation_modules_do_not_import_physical_or_legacy_stacks() -> None:
    root = Path(__file__).parents[2]
    base_imports = _imported_modules(root / "src/finn/dataflow/op.py")
    logical_mvau_imports = _imported_modules(root / "src/finn/dataflow/ops/mvau/op.py")
    assert not any(module.startswith("finn.dataflow.mvau") for module in base_imports)
    assert not any(module.startswith("finn.builder") for module in base_imports)
    for imports in (base_imports, logical_mvau_imports):
        assert not any(module.startswith("finn.custom_op.fpgadataflow") for module in imports)
        assert not any(module.startswith("finn.dataflow.mvau.artifacts") for module in imports)
        assert not any(module.startswith("finn.dataflow.mvau.elaboration") for module in imports)
        assert not any(module.startswith("finn.xsi") for module in imports)


def test_public_authoring_import_is_operation_generic() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import finn.dataflow.authoring as a; "
            "assert a.DataflowOp; assert a.NodeAttrCodec; "
            "assert 'finn.dataflow.mvau' not in sys.modules; "
            "assert 'finn.custom_op.fpgadataflow' not in sys.modules",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from finn.custom_op.dataflow import MvauDataflowOp; "
            "assert MvauDataflowOp; "
            "assert 'finn.custom_op.fpgadataflow' not in sys.modules; "
            "assert 'finn.dataflow.mvau.artifacts' not in sys.modules; "
            "assert 'finn.dataflow.mvau.elaboration' not in sys.modules",
        ],
        check=True,
    )
