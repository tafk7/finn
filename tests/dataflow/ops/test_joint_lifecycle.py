# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The combined native-operation to detached-artifact lifecycle."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from onnx import helper  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

import finn.dataflow.designs.design as design_module
import finn.dataflow.ops.mapping as mapping_module
from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.artifacts.derivation import ContentRef, build_key
from finn.dataflow.artifacts.formats import RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.packaging import Target, plan_package
from finn.dataflow.conformance import (
    DataflowOpConformanceCase,
    assert_dataflow_op_conforms,
)
from finn.dataflow.designs import DataflowDesign
from finn.dataflow.kernels import ModuleBuildRequirements
from finn.dataflow.artifacts.build import (
    prepare_module_build,
    module_source_derivation,
    materialize_module_sources,
    portable_module_component,
)
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.kernels.replay_buffer import FINNLIB_ROOT
from finn.dataflow.model import (
    DataflowNetwork,
    NetworkValidationReport,
    RegionInputRef,
    RegionOutputRef,
    exposing_ports,
    validate_network,
)
from finn.dataflow.model.refs import DataflowOperandRef
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.dataflow.ops.inference import InferDataTypes, InferShapes
from finn.dataflow.ops.mapping import CoordinateMapping, External
from finn.dataflow.ops.native import (
    FINGERPRINT_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    SCOPE_ID_ATTRIBUTE,
    NativeAttribute,
    read_attributes,
)
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.reconstruction import bind_operations
from finn.dataflow.ops.replay.op import ActivationReplayOp
from finn.dataflow.ops.schema import Attribute
from finn.dataflow.space import Subspace
from finn.dataflow.space.occurrence import ProjectionAssessment, occurrence_persistable

_projection_fixtures = import_module("dataflow.designs.test_projection_boundary")
MissingSource: Any = _projection_fixtures.MissingSource
RefusingDesign: Any = _projection_fixtures.RefusingDesign
SuppliedDesign: Any = _projection_fixtures.SuppliedDesign

_conformance_fixtures = import_module("dataflow.ops.test_conformance")
Build: Any = _conformance_fixtures.Build
_configure_replay: Any = _conformance_fixtures._configure_replay
_replay_model: Any = _conformance_fixtures._replay_model
_widen_the_activation: Any = _conformance_fixtures._widen_the_activation


@dataclass(frozen=True)
class _ArtifactObservation:
    spec: ModuleBuildRequirements
    source_key: str
    source_paths: tuple[str, ...]
    package_key: str
    generated: tuple[tuple[str, bytes], ...]


class _ResolvedContents:
    def __init__(self, values: Mapping[str, bytes]) -> None:
        self._values = dict(values)

    def get_blob(self, reference: ContentRef) -> bytes:
        return self._values[reference.digest]


def _finnlib_root() -> Path:
    configured = os.environ.get("FINNLIB_ROOT")
    if configured is not None:
        return Path(configured)
    return Path(__file__).parents[3] / "deps" / "finnlib"


def _observe_artifacts(spec: ModuleBuildRequirements) -> _ArtifactObservation:
    root = _finnlib_root()
    with TemporaryDirectory() as temporary:
        store = ArtifactStore(Path(temporary))
        prepared = prepare_module_build(
            spec, roots={FINNLIB_ROOT: root}, blobs=store, template_roots=()
        )
        source = materialize_module_sources(prepared, store)
        component = portable_module_component(prepared, source)
        package = plan_package(
            RtlModuleDirectory(),
            component,
            Target("test-part"),
            RtlModuleOptions(),
            store,
        )
        return _ArtifactObservation(
            spec,
            build_key(module_source_derivation(prepared)),
            source.files,
            build_key(package.derivation),
            package.contents,
        )


def _replay_spec(operation: Any) -> ModuleBuildRequirements:
    kernel = operation.design.kernel("replay")
    assert isinstance(kernel, Decided)
    assert kernel.value.root is operation.root
    physical = kernel.value.physical.accepted_answer
    assert isinstance(physical, Decided)
    return cast(ModuleBuildRequirements, physical.value)


class _NestedActivationReplayOp(ActivationReplayOp):
    root_namespace: ClassVar[str] = "deeply.nested.replay"


def test_native_reload_to_accepted_mapping_and_portable_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = Build()
    model = _replay_model(repetitions=2, matrix_width=8, folds=4)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    before = model.model.SerializeToString(deterministic=True)

    bound = bind_operations(model, build)[0]
    assert isinstance(bound, ActivationReplayOp)
    assert bound.root is bound
    assert bound.design.root is bound
    initial_kernel = bound.design.kernel("replay")
    assert isinstance(initial_kernel, Decided)
    assert initial_kernel.value.root is bound
    assert isinstance(bound.dataflow.accepted_answer, Unresolved)

    configured = _configure_replay(bound)
    assert configured.problem_snapshot is bound.problem_snapshot
    assert model.model.SerializeToString(deterministic=True) == before

    reload_path = tmp_path / "joint-replay.onnx"
    result = assert_dataflow_op_conforms(
        DataflowOpConformanceCase(
            model=model,
            node_name="replay0",
            operation_type=ActivationReplayOp,
            build=build,
            configure=_configure_replay,
            reload_path=reload_path,
            mutate_problem=_widen_the_activation,
        )
    )

    saved = ModelWrapper(str(reload_path))
    attributes = read_attributes(saved.graph.node[0])
    assert attributes["design__pe"] == NativeAttribute("i", 1)
    assert attributes["design__simd"] == NativeAttribute("i", 4)
    assert attributes[SCHEMA_VERSION_ATTRIBUTE] == NativeAttribute("i", 3)
    assert set(attributes) == {
        "neuron_folds",
        SCOPE_ID_ATTRIBUTE,
        FINGERPRINT_ATTRIBUTE,
        SCHEMA_VERSION_ATTRIBUTE,
        "design__pe",
        "design__simd",
    }
    assert "dataflow_state" not in attributes
    assert dict(result.committed.recorded()) == {"design.pe": 1, "design.simd": 4}
    assert result.committed.problem_fingerprint == result.restored.problem_fingerprint

    restored_model = saved.transform(InferShapes())
    restored_model = restored_model.transform(InferDataTypes())
    checked: list[DataflowNetwork] = []

    def counted_validation(network: DataflowNetwork) -> NetworkValidationReport:
        checked.append(network)
        return validate_network(network)

    monkeypatch.setattr(design_module, "validate_network", counted_validation)
    restored = bind_operations(restored_model, build)[0]
    accepted = restored.dataflow.accepted_answer
    assert isinstance(accepted, Decided)
    network = accepted.value
    assert checked == [network]

    mapping = restored.operand_mapping
    assert isinstance(mapping, Decided)
    by_source = {item.source_operand: item for item in mapping.value}
    activation = by_source["activation"]
    expanded = by_source["expanded"]
    assert activation.tensor == "activation"
    assert activation.semantic_operand == RegionInputRef("replay", "X")
    assert activation.placement == External("activation", "replay", "activation_in")
    assert activation.correspondence is CoordinateMapping.FLATTEN_LEADING
    assert activation.source_shape == (2, 8)
    assert activation.semantic_shape == (2, 8)
    assert activation.edge_presented_set.is_empty
    assert not activation.boundary_presented_set.is_empty
    assert activation.unpresented_set.is_empty
    assert expanded.tensor == "expanded"
    assert expanded.semantic_operand == RegionOutputRef("replay", "XR")
    assert expanded.placement == External("expanded", "replay", "activation_out")
    assert expanded.correspondence is CoordinateMapping.IDENTITY
    assert expanded.source_shape == (8, 8)
    assert expanded.semantic_shape == (8, 8)
    committed_mapping = result.committed.operand_mapping
    assert isinstance(committed_mapping, Decided)
    assert mapping.value == committed_mapping.value
    assert restored.operand_mapping == mapping
    assert exposing_ports(network, RegionInputRef("replay", "X"))
    assert checked == [network]

    spec = _replay_spec(restored)
    assert not hasattr(spec, "region")
    kernel = restored.design.kernel("replay")
    assert isinstance(kernel, Decided)
    assert kernel.value.dataflow.accepted_answer == Decided(network.node("replay").region)
    assert dict(spec.parameters) == {"LEN": 2, "REP": 4, "W": 32}
    assert spec.abi.entry_point.value == "replay_buffer"
    persisted = {item.path for item in occurrence_persistable(restored)}
    assert {"design.pe", "design.simd"} <= persisted

    committed_spec = _replay_spec(result.committed)
    assert committed_spec == spec
    committed_artifacts = _observe_artifacts(committed_spec)
    restored_artifacts = _observe_artifacts(spec)
    assert restored_artifacts == committed_artifacts

    nested_model = _replay_model(repetitions=2, matrix_width=8, folds=4)
    nested_model = nested_model.transform(InferShapes())
    nested_model = nested_model.transform(InferDataTypes())
    nested_unbound = _NestedActivationReplayOp(nested_model.graph.node[0])
    nested = _configure_replay(nested_unbound.bind(nested_model, build))
    nested_spec = _replay_spec(nested)
    assert not hasattr(nested_spec, "imported_decisions")
    assert {item.path for item in occurrence_persistable(nested)} == persisted
    assert _observe_artifacts(nested_spec) == restored_artifacts


class _RefusalProbe(DataflowOp):
    family: ClassVar[str] = "test.c2.refusal_probe"
    width = Attribute(int, default=2)

    def operand_references(
        self, network: DataflowNetwork
    ) -> Mapping[str, tuple[DataflowOperandRef, ...]]:
        del network
        raise AssertionError("a refused Network must not ask for source references")


class _KernelRefusalOp(_RefusalProbe):
    family: ClassVar[str] = "test.c2.kernel_refusal"
    design = Subspace(SuppliedDesign, width=_RefusalProbe.width)

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        return cast(DataflowDesign, self.design).dataflow


class _DesignRefusalOp(_RefusalProbe):
    family: ClassVar[str] = "test.c2.design_refusal"
    design = Subspace(RefusingDesign, width=_RefusalProbe.width)

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        return cast(DataflowDesign, self.design).dataflow


class _InvalidTopologyOp(_RefusalProbe):
    family: ClassVar[str] = "test.c2.invalid_topology"
    design = Subspace(MissingSource, width=_RefusalProbe.width)

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        return cast(DataflowDesign, self.design).dataflow


@pytest.mark.parametrize(
    ("operation_type", "width"),
    ((_KernelRefusalOp, 1), (_DesignRefusalOp, 2), (_InvalidTopologyOp, 2)),
)
def test_design_rejections_cross_the_operation_boundary_without_presentation(
    operation_type: type[_RefusalProbe],
    width: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = helper.make_node(
        operation_type.__name__,
        [],
        [],
        domain=DATAFLOW_DOMAIN,
        name="boundary0",
        width=width,
    )
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph([node], "boundary", [], []),
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid(DATAFLOW_DOMAIN, 1),
            ],
        )
    )
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    operation = operation_type(model.graph.node[0])
    bound = operation.bind(model, Build())
    selected = bound.selected_dataflow()
    assert selected is not None
    assert isinstance(selected.output, Decided)
    assert isinstance(selected.accepted_answer, Absent)

    def unexpected_presentation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a refused Network must not reach presentation")

    for name in (
        "exposing_ports",
        "exposing_boundaries",
        "edge_presented_position_set",
        "boundary_presented_position_set",
        "unpresented_position_set",
    ):
        monkeypatch.setattr(mapping_module, name, unexpected_presentation)

    before = model.model.SerializeToString(deterministic=True)
    assert isinstance(bound.dataflow.accepted_answer, Absent)
    assert isinstance(bound.operand_mapping, Absent)
    with pytest.raises(DataflowOpError, match="dataflow projection refuses"):
        bound.commit(model, Build())
    assert model.model.SerializeToString(deterministic=True) == before
