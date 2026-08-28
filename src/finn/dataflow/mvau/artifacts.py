# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed artifact requirements for the first MVAU RTL soft-vector slice."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterator, cast

import numpy as np  # type: ignore[import-not-found]
from onnx import AttributeProto, NodeProto, TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Absent, Decided, Finding, FindingKind, QualifiedPath, Unresolved
from finn.dataflow.mvau.elaboration import (
    MVAUPhysicalElaboration,
    MVAUPhysicalNumericInterface,
    mvau_elaboration_origin,
)
from finn.dataflow.mvau.source import (
    MVAUModelAccessor,
    MVAUResolvedDesign,
    tensor_value_fingerprint,
)
from finn.dataflow.ops.mvau import (
    MVAU_COMPUTE_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
    NetworkRef,
    RegionRef,
)
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
)
from finn.dataflow.mvau_problem import MVAUProblemPaths
from finn.dataflow.region import BeatSequence, NumericElementType

_ARTIFACT_PATH = QualifiedPath("artifact.mvau.rtl_softvec")
_DATAFLOW_SCOPE_ID_ATTR = "dataflow_scope_id"


class MVAUWeightPayloadKind(str, Enum):
    INITIALIZER = "initializer"
    EXTERNAL_RUNTIME = "external_runtime"
    RUNTIME_WRITABLE_LOCAL_STATE = "runtime_writable_local_state"


def _source_scope_matches(node: NodeProto, source_scope_id: str) -> bool:
    if node.name == source_scope_id:
        return True
    attribute = next(
        (item for item in node.attribute if item.name == _DATAFLOW_SCOPE_ID_ATTR),
        None,
    )
    if attribute is None or attribute.type != AttributeProto.STRING:
        return False
    return cast(bytes, attribute.s).decode("utf-8") == source_scope_id


@dataclass(frozen=True)
class MVAUTensorData:
    """Portable dense tensor value required by the covered builder."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32).reshape(self.shape)


@dataclass(frozen=True)
class MVAURTLSourceRequirement:
    id: str
    source_path: str
    relative_path: str


@dataclass(frozen=True)
class MVAURTLGeneratedOutput:
    id: str
    relative_path: str


@dataclass(frozen=True)
class MVAURTLInterfaceRequirement:
    id: str
    component_id: str
    interface_name: str
    logical_width_bits: int
    physical_width_bits: int
    data_signal: str
    valid_signal: str
    ready_signal: str
    beat_sequence: BeatSequence


@dataclass(frozen=True)
class MVAURTLArtifactRequirements:
    """Complete declared inputs to the covered legacy RTL generator."""

    source_scope_id: str
    finn_root: str
    target_fpga_part: str
    clock_period_ns: float
    top_module_name: str
    mem_mode: str
    activation_tensor_id: str
    weight_tensor_id: str
    output_tensor_id: str
    activation_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    activation_datatype: str
    weight_datatype: str
    accumulator_datatype: str
    output_datatype: str
    weight_payload_kind: MVAUWeightPayloadKind
    weight_initializer: MVAUTensorData | None
    parameters: tuple[tuple[str, bool | int | str], ...]
    interfaces: tuple[MVAURTLInterfaceRequirement, ...]
    source_dependencies: tuple[MVAURTLSourceRequirement, ...]
    generated_outputs: tuple[MVAURTLGeneratedOutput, ...]
    elaboration: MVAUPhysicalElaboration

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))
        object.__setattr__(
            self, "interfaces", tuple(sorted(self.interfaces, key=lambda item: item.id))
        )
        object.__setattr__(
            self,
            "source_dependencies",
            tuple(sorted(self.source_dependencies, key=lambda item: item.id)),
        )
        object.__setattr__(
            self,
            "generated_outputs",
            tuple(sorted(self.generated_outputs, key=lambda item: item.id)),
        )


@dataclass(frozen=True)
class MVAUBuiltRTLArtifact:
    """Generated artifact paths plus the requirements-backed legacy model."""

    requirements: MVAURTLArtifactRequirements
    output_directory: str
    source_files: tuple[str, ...]
    generated_files: tuple[str, ...]
    model: ModelWrapper


@dataclass(frozen=True)
class MVAURTLSimulationObservation:
    """Numerical and transaction-order observations from one XSIM run."""

    artifact_identity: str
    oracle: str
    output_shape: tuple[int, ...]
    output_values: tuple[float, ...]
    expected_values: tuple[float, ...]
    output_transactions: tuple[tuple[float, ...], ...]
    expected_transactions: tuple[tuple[float, ...], ...]
    measured_cycles: int

    @property
    def numerical_match(self) -> bool:
        return self.output_values == self.expected_values

    @property
    def output_order_match(self) -> bool:
        return self.output_transactions == self.expected_transactions


@dataclass(frozen=True)
class MVAUStitchedSimulationObservation:
    """Artifact-bound numerical observation of cyclic delivery plus compute."""

    artifact_identity: str
    oracle: str
    semantic_edge_ids: tuple[str, ...]
    output_shape: tuple[int, ...]
    output_values: tuple[float, ...]
    expected_values: tuple[float, ...]

    @property
    def numerical_match(self) -> bool:
        return self.output_values == self.expected_values


class MVAUArtifactError(ValueError):
    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = tuple(
            sorted(findings, key=lambda item: (item.path, item.kind.value, item.code))
        )
        super().__init__(f"MVAU artifact preparation failed with {len(self.findings)} finding(s)")


def _finding(code: str, message: str, path: QualifiedPath = _ARTIFACT_PATH) -> Finding:
    return Finding(FindingKind.REJECTION, code, path, message)


def _require_feasible(resolved: MVAUResolvedDesign, constraint_set: str) -> None:
    assessment = resolved.engine.evaluate_constraint_set(resolved.point, constraint_set)
    if assessment.verdict is True:
        return
    findings: list[Finding] = []
    for path, answer in assessment.answers.items():
        if isinstance(answer, Decided) and answer.value is False:
            findings.append(
                _finding(
                    "mvau-artifact-constraint-violated",
                    f"artifact construction requires feasible {constraint_set!r} constraints",
                    path,
                )
            )
        elif isinstance(answer, (Absent, Unresolved)):
            findings.extend(answer.findings)
    raise MVAUArtifactError(
        tuple(findings)
        or (
            _finding(
                "mvau-artifact-feasibility-incomplete",
                f"artifact construction could not establish {constraint_set!r} feasibility",
            ),
        )
    )


def _datatype_name(element_type: NumericElementType) -> str:
    if element_type.type_id == "bipolar" and element_type.bit_width == 1:
        return "BIPOLAR"
    if element_type.type_id == "binary" and element_type.bit_width == 1:
        return "BINARY"
    if element_type.type_id == "int":
        return f"INT{element_type.bit_width}"
    if element_type.type_id == "uint":
        return f"UINT{element_type.bit_width}"
    if element_type.type_id == "float":
        return f"FLOAT{element_type.bit_width}"
    raise MVAUArtifactError(
        (
            _finding(
                "mvau-artifact-datatype-unsupported",
                f"no QONNX datatype spelling is declared for {element_type!r}",
            ),
        )
    )


def _find_interface(
    elaboration: MVAUPhysicalElaboration, component_id: str, suffix: str
) -> MVAUPhysicalNumericInterface:
    matches = tuple(
        item
        for item in elaboration.numeric_interfaces
        if item.component_id == component_id and item.id.endswith(suffix)
    )
    if len(matches) != 1:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-interface-missing",
                    f"expected one generated interface ending in {suffix!r}",
                ),
            )
        )
    return matches[0]


def _interface_requirement(
    interface: MVAUPhysicalNumericInterface,
    interface_name: str,
    beat_sequence: BeatSequence,
) -> MVAURTLInterfaceRequirement:
    return MVAURTLInterfaceRequirement(
        interface.id,
        interface.component_id,
        interface_name,
        interface.logical_width_bits,
        interface.physical_width_bits,
        interface.data_signal,
        interface.valid_signal,
        interface.ready_signal,
        beat_sequence,
    )


def _artifact_file_requirements(
    finn_root: Path, module_name: str, *, cyclic: bool, initialized: bool
) -> tuple[tuple[MVAURTLSourceRequirement, ...], tuple[MVAURTLGeneratedOutput, ...]]:
    sources = [
        MVAURTLSourceRequirement(
            "compute.template",
            str(finn_root / "finn-rtllib" / "mvu" / "mvu_vvu_axi_wrapper.v"),
            "finn-rtllib/mvu/mvu_vvu_axi_wrapper.v",
        ),
        *(
            MVAURTLSourceRequirement(
                f"compute.library.{index}",
                str(finn_root / "finn-rtllib" / "mvu" / filename),
                f"finn-rtllib/mvu/{filename}",
            )
            for index, filename in enumerate(
                (
                    "mvu_pkg.sv",
                    "mvu_vvu_axi.sv",
                    "replay_buffer.sv",
                    "mvu.sv",
                    "mvu_vvu_8sx9_dsp58.sv",
                    "add_multi.sv",
                )
            )
        ),
    ]
    outputs = [MVAURTLGeneratedOutput("compute.wrapper", f"{module_name}_wrapper.v")]
    if cyclic:
        sources.extend(
            (
                MVAURTLSourceRequirement(
                    "delivery.template",
                    str(
                        finn_root
                        / "finn-rtllib"
                        / "memstream"
                        / "hdl"
                        / "memstream_wrapper_template.v"
                    ),
                    "finn-rtllib/memstream/hdl/memstream_wrapper_template.v",
                ),
                MVAURTLSourceRequirement(
                    "delivery.library.0",
                    str(finn_root / "finn-rtllib" / "memstream" / "hdl" / "memstream.sv"),
                    "finn-rtllib/memstream/hdl/memstream.sv",
                ),
                MVAURTLSourceRequirement(
                    "delivery.library.1",
                    str(finn_root / "finn-rtllib" / "memstream" / "hdl" / "memstream_axi.sv"),
                    "finn-rtllib/memstream/hdl/memstream_axi.sv",
                ),
                MVAURTLSourceRequirement(
                    "delivery.library.2",
                    str(finn_root / "finn-rtllib" / "axi" / "hdl" / "axilite.sv"),
                    "finn-rtllib/axi/hdl/axilite.sv",
                ),
                MVAURTLSourceRequirement(
                    "stitched.simulation_control",
                    str(finn_root / "finn-rtllib" / "sim" / "hdl" / "sim_ctrl.v"),
                    "finn-rtllib/sim/hdl/sim_ctrl.v",
                ),
                MVAURTLSourceRequirement(
                    "stitched.axi_info.component",
                    str(finn_root / "finn-rtllib" / "axi_info" / "component.xml"),
                    "finn-rtllib/axi_info/component.xml",
                ),
                MVAURTLSourceRequirement(
                    "stitched.axi_info.hdl",
                    str(finn_root / "finn-rtllib" / "axi_info" / "hdl" / "axi_info.sv"),
                    "finn-rtllib/axi_info/hdl/axi_info.sv",
                ),
                MVAURTLSourceRequirement(
                    "stitched.axi_info.top",
                    str(finn_root / "finn-rtllib" / "axi_info" / "hdl" / "axi_info_top.sv"),
                    "finn-rtllib/axi_info/hdl/axi_info_top.sv",
                ),
                MVAURTLSourceRequirement(
                    "stitched.axi_info.xgui",
                    str(finn_root / "finn-rtllib" / "axi_info" / "xgui" / "axi_info_top_v1_0.tcl"),
                    "finn-rtllib/axi_info/xgui/axi_info_top_v1_0.tcl",
                ),
                MVAURTLSourceRequirement(
                    "stitched.driver.mdd",
                    str(finn_root / "src" / "finn" / "qnn-data" / "mdd-data" / "finn_design.mdd"),
                    "src/finn/qnn-data/mdd-data/finn_design.mdd",
                ),
                MVAURTLSourceRequirement(
                    "stitched.driver.tcl",
                    str(finn_root / "src" / "finn" / "qnn-data" / "mdd-data" / "finn_design.tcl"),
                    "src/finn/qnn-data/mdd-data/finn_design.tcl",
                ),
            )
        )
        outputs.append(
            MVAURTLGeneratedOutput("delivery.wrapper", f"{module_name}_memstream_wrapper.v")
        )
        if initialized:
            outputs.extend(
                (
                    MVAURTLGeneratedOutput("delivery.initializer", "memblock.dat"),
                    MVAURTLGeneratedOutput("delivery.simulation_weights", "input_1.npy"),
                )
            )
    missing = tuple(item.source_path for item in sources if not Path(item.source_path).is_file())
    if missing:
        raise MVAUArtifactError(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-artifact-source-missing",
                    _ARTIFACT_PATH,
                    "one or more declared RTL library sources do not exist",
                    (("paths", missing),),
                ),
            )
        )
    return tuple(sources), tuple(outputs)


def build_mvau_rtl_artifact_requirements(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    model: MVAUModelAccessor,
    finn_root: str | Path,
) -> MVAURTLArtifactRequirements:
    """Create a self-contained requirements value from selected semantics."""
    readiness = resolved.engine.check_readiness(resolved.point, "artifact_inputs")
    if readiness.ready is not True:
        raise MVAUArtifactError(
            (
                Finding(
                    FindingKind.BLOCKER,
                    "mvau-artifact-inputs-not-ready",
                    _ARTIFACT_PATH,
                    "artifact-affecting decisions, properties, and constraints are incomplete",
                ),
            )
        )
    if elaboration.origin != mvau_elaboration_origin(resolved):
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-elaboration-origin-mismatch",
                    "physical elaboration was not produced from this exact selected point",
                ),
            )
        )
    if elaboration.semantic_result != resolved.result:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-elaboration-mismatch",
                    "physical elaboration does not belong to the selected semantic result",
                ),
            )
        )
    _require_feasible(resolved, "mvau_op_structural")
    _require_feasible(resolved, MVAU_COMPUTE_SELECTION.feasibility_constraint_set)
    if isinstance(resolved.result, NetworkRef):
        _require_feasible(resolved, MVAU_WEIGHT_SUPPLY_SELECTION.feasibility_constraint_set)
    source = resolved.result.source_association
    nodes = tuple(
        node for node in model.graph.node if _source_scope_matches(node, source.source_node_id)
    )
    if len(nodes) != 1:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-source-node-missing",
                    "source scope must identify exactly one node while requirements are built",
                ),
            )
        )
    description = resolved.projection.source_description
    if description is None:
        raise MVAUArtifactError(
            (_finding("mvau-artifact-source-description-missing", "source description is absent"),)
        )
    cyclic = isinstance(resolved.result, NetworkRef)
    initializer = model.get_initializer(description.weight_operand_id)
    expected_initializer_fingerprint = resolved.point.problem.get(
        MVAUDataflowOpPaths.WEIGHT_INITIALIZER_FINGERPRINT
    )
    actual_initializer_fingerprint = (
        None if initializer is None else tensor_value_fingerprint(initializer)
    )
    if actual_initializer_fingerprint != expected_initializer_fingerprint:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-weight-source-mismatch",
                    "weight values do not match the source problem used for selection",
                ),
            )
        )
    runtime_writable = cast(
        bool, resolved.point.problem.get(MVAUProblemPaths.RUNTIME_WRITABLE, False)
    )
    if initializer is None and cyclic and not runtime_writable:
        raise MVAUArtifactError(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-artifact-weight-values-missing",
                    MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE,
                    "cyclic local-state delivery requires initialized or runtime-writable weights",
                ),
            )
        )
    matrix_width = cast(int, resolved.point.problem[MVAUProblemPaths.MATRIX_WIDTH])
    matrix_height = cast(int, resolved.point.problem[MVAUProblemPaths.MATRIX_HEIGHT])
    expected_weight_shape = (matrix_width, matrix_height)
    weight_array = None if initializer is None else np.asarray(initializer, dtype=np.float32)
    weight_payload_kind = (
        MVAUWeightPayloadKind.INITIALIZER
        if weight_array is not None
        else MVAUWeightPayloadKind.RUNTIME_WRITABLE_LOCAL_STATE
        if cyclic
        else MVAUWeightPayloadKind.EXTERNAL_RUNTIME
    )
    if weight_array is not None and tuple(weight_array.shape) != expected_weight_shape:
        raise MVAUArtifactError(
            (_finding("mvau-artifact-weight-shape-mismatch", "weight values do not match MW x MH"),)
        )
    activation_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE],
    )
    weight_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.WEIGHT_ELEMENT_TYPE],
    )
    accumulator_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE],
    )
    output_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUProblemPaths.OUTPUT_ELEMENT_TYPE],
    )
    wrapper_id = f"{source.source_node_id}.compute.wrapper"
    wrapper = elaboration.component(wrapper_id)
    parameters = tuple(
        (name, cast(bool | int | str, value))
        for name, value in wrapper.parameters
        if type(value) in {bool, int, str}
    )
    if len(parameters) != len(wrapper.parameters):
        raise MVAUArtifactError(
            (_finding("mvau-artifact-parameter-type", "covered RTL parameters must be scalar"),)
        )
    parameters += (("TH", 1),)
    if isinstance(resolved.result, RegionRef):
        compute_region = resolved.result.region
        mem_mode = "external"
    else:
        compute_region = resolved.result.network.node("compute").region
        mem_mode = "internal_decoupled"
    activation_port = compute_region.input_interface("activation").port
    weight_port = compute_region.input_interface("weight").port
    output_port = compute_region.output_interface("output").port
    interfaces: tuple[MVAURTLInterfaceRequirement, ...] = (
        _interface_requirement(
            _find_interface(elaboration, wrapper_id, ".activation"),
            "in0_V",
            activation_port.beat_sequence,
        ),
        _interface_requirement(
            _find_interface(elaboration, wrapper_id, ".weight"),
            "in1_V",
            weight_port.beat_sequence,
        ),
        _interface_requirement(
            _find_interface(elaboration, wrapper_id, ".output"),
            "out0_V",
            output_port.beat_sequence,
        ),
    )
    if cyclic:
        network = cast(NetworkRef, resolved.result).network
        delivery_port = network.node("delivery").region.output_interface("weight").port
        delivery_id = f"{source.source_node_id}.delivery.wrapper"
        interfaces += (
            _interface_requirement(
                _find_interface(elaboration, delivery_id, ".weight"),
                "m_axis_0",
                delivery_port.beat_sequence,
            ),
        )
    root = Path(finn_root).resolve()
    if not root.is_dir():
        raise MVAUArtifactError(
            (_finding("mvau-artifact-root-missing", "the declared FINN source root is absent"),)
        )
    ram_style = resolved.point.assignments.get(FINN_RTL_MEMSTREAM_PATHS.ram_style)
    pumped_memory = resolved.point.assignments.get(FINN_RTL_MEMSTREAM_PATHS.pumped_memory)
    parameters += (
        ("RAM_STYLE", "auto" if ram_style is None else cast(Enum, ram_style).value),
        ("RUNTIME_WRITABLE", runtime_writable),
        ("PUMPED_MEMORY", False if pumped_memory is None else cast(bool, pumped_memory)),
    )
    source_dependencies, generated_outputs = _artifact_file_requirements(
        root,
        source.source_node_id,
        cyclic=cyclic,
        initialized=weight_array is not None,
    )
    return MVAURTLArtifactRequirements(
        source.source_node_id,
        str(root),
        elaboration.target_fpga_part,
        elaboration.target_clock_period_ns,
        source.source_node_id,
        mem_mode,
        description.activation_operand_id,
        description.weight_operand_id,
        description.output_operand_id,
        (*description.leading_shape, matrix_width),
        expected_weight_shape,
        (*description.leading_shape, matrix_height),
        _datatype_name(activation_type),
        _datatype_name(weight_type),
        _datatype_name(accumulator_type),
        _datatype_name(output_type),
        weight_payload_kind,
        None
        if weight_array is None
        else MVAUTensorData(
            expected_weight_shape, tuple(float(value) for value in weight_array.flat)
        ),
        parameters,
        interfaces,
        source_dependencies,
        generated_outputs,
        elaboration,
    )


def _materialize_model(requirements: MVAURTLArtifactRequirements) -> ModelWrapper:
    graph_inputs = [
        helper.make_tensor_value_info(
            requirements.activation_tensor_id,
            TensorProto.FLOAT,
            list(requirements.activation_shape),
        ),
        helper.make_tensor_value_info(
            requirements.weight_tensor_id,
            TensorProto.FLOAT,
            list(requirements.weight_shape),
        ),
    ]
    output = helper.make_tensor_value_info(
        requirements.output_tensor_id,
        TensorProto.FLOAT,
        list(requirements.output_shape),
    )
    parameters: Mapping[str, bool | int | str] = dict(requirements.parameters)
    node = helper.make_node(
        "MVAU_rtl",
        [requirements.activation_tensor_id, requirements.weight_tensor_id],
        [requirements.output_tensor_id],
        name=requirements.top_module_name,
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        PE=cast(int, parameters["PE"]),
        SIMD=cast(int, parameters["SIMD"]),
        MW=cast(int, parameters["MW"]),
        MH=cast(int, parameters["MH"]),
        TH=cast(int, parameters["TH"]),
        inputDataType=requirements.activation_datatype,
        weightDataType=requirements.weight_datatype,
        accDataType=requirements.accumulator_datatype,
        outputDataType=requirements.output_datatype,
        noActivation=1,
        binaryXnorMode=0,
        numInputVectors=list(requirements.activation_shape[:-1]) or [1],
        mem_mode=requirements.mem_mode,
        resType="dsp",
        ram_style=cast(str, parameters["RAM_STYLE"]),
        runtime_writeable_weights=int(cast(bool, parameters["RUNTIME_WRITABLE"])),
        pumpedMemory=int(cast(bool, parameters["PUMPED_MEMORY"])),
        pumpedCompute=int(cast(bool, parameters["PUMPED_COMPUTE"])),
    )
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [node],
                "requirements-backed-mvau",
                graph_inputs,
                [output],
            ),
            producer_name="finn-dataflow-artifact-requirements",
        )
    )
    model.set_tensor_datatype(
        requirements.activation_tensor_id, DataType[requirements.activation_datatype]
    )
    model.set_tensor_datatype(requirements.weight_tensor_id, DataType[requirements.weight_datatype])
    model.set_tensor_datatype(requirements.output_tensor_id, DataType[requirements.output_datatype])
    if (
        requirements.mem_mode == "internal_decoupled"
        and requirements.weight_initializer is not None
    ):
        model.set_initializer(
            requirements.weight_tensor_id, requirements.weight_initializer.as_array()
        )
    return model


def _stage_source_dependencies(
    requirements: MVAURTLArtifactRequirements, staging_root: Path
) -> tuple[str, ...]:
    expected_sources, expected_outputs = _artifact_file_requirements(
        Path(requirements.finn_root),
        requirements.top_module_name,
        cyclic=requirements.mem_mode == "internal_decoupled",
        initialized=requirements.weight_initializer is not None,
    )
    expected_source_shape = tuple(
        (item.id, item.source_path, item.relative_path)
        for item in sorted(expected_sources, key=lambda item: item.id)
    )
    actual_source_shape = tuple(
        (item.id, item.source_path, item.relative_path) for item in requirements.source_dependencies
    )
    expected_output_shape = tuple(
        (item.id, item.relative_path) for item in sorted(expected_outputs, key=lambda item: item.id)
    )
    actual_output_shape = tuple(
        (item.id, item.relative_path) for item in requirements.generated_outputs
    )
    if actual_source_shape != expected_source_shape or actual_output_shape != expected_output_shape:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-file-requirements-incomplete",
                    "source dependencies and generated outputs must match the covered builder",
                ),
            )
        )
    staged = []
    for dependency in requirements.source_dependencies:
        destination = staging_root / dependency.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dependency.source_path, destination)
        staged.append(str(destination))
    return tuple(staged)


@contextmanager
def _declared_finn_root(path: str) -> Iterator[None]:
    previous = os.environ.get("FINN_ROOT")
    os.environ["FINN_ROOT"] = path
    try:
        yield
    finally:
        if previous is None:
            del os.environ["FINN_ROOT"]
        else:
            os.environ["FINN_ROOT"] = previous


def build_mvau_rtl_artifact(
    requirements: MVAURTLArtifactRequirements,
    output_directory: str | Path,
    *,
    prepare_rtlsim: bool = False,
) -> MVAUBuiltRTLArtifact:
    """Drive the existing RTL generator from requirements, not an ambient graph."""
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged_root = output / "declared_sources"
    source_files = _stage_source_dependencies(requirements, staged_root)
    model = _materialize_model(requirements)
    node = model.graph.node[0]
    with _declared_finn_root(requirements.finn_root):
        operation = getCustomOp(node)
    with _declared_finn_root(str(staged_root)):
        operation.set_nodeattr("code_gen_dir_ipgen", str(output))
        operation.set_nodeattr("exec_mode", "rtlsim")
        operation.generate_hdl(
            model,
            requirements.target_fpga_part,
            requirements.clock_period_ns,
        )
        if prepare_rtlsim:
            operation.prepare_rtlsim(behav=True)
    generated_files = tuple(
        str(output / item.relative_path) for item in requirements.generated_outputs
    )
    missing = tuple(path for path in generated_files if not Path(path).is_file())
    if missing:
        raise MVAUArtifactError(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-artifact-generation-incomplete",
                    _ARTIFACT_PATH,
                    "the legacy generator did not produce the declared source manifest",
                    (("paths", missing),),
                ),
            )
        )
    return MVAUBuiltRTLArtifact(requirements, str(output), source_files, generated_files, model)


def _identity_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": value.value}
    if isinstance(value, QualifiedPath):
        return value.value
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
    raise TypeError(f"unsupported artifact identity value {type(value).__name__}")


def mvau_built_artifact_identity(artifact: MVAUBuiltRTLArtifact) -> str:
    """Fingerprint one generated artifact and the exact point that produced it."""
    requirements = artifact.requirements
    origin = requirements.elaboration.origin
    payload = {
        "assignments": [[path.value, _identity_value(value)] for path, value in origin.assignments],
        "kernel_ids": list(origin.kernel_ids),
        "provider_ids": list(origin.provider_ids),
        "clock_period_ns": requirements.clock_period_ns,
        "declaration_family_version": origin.declaration_family_version,
        "generated": [
            [Path(path).name, sha256(Path(path).read_bytes()).hexdigest()]
            for path in artifact.generated_files
        ],
        "problem_fingerprint": origin.problem_fingerprint,
        "source_dependencies": [
            [Path(path).name, sha256(Path(path).read_bytes()).hexdigest()]
            for path in artifact.source_files
        ],
        "source_scope_id": requirements.source_scope_id,
        "target_fpga_part": requirements.target_fpga_part,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def simulate_mvau_rtl_artifact(
    artifact: MVAUBuiltRTLArtifact,
    activation: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    mode: str = "rtlsim",
) -> np.ndarray:
    """Execute the requirements-backed legacy model in cppsim or RTL simulation."""
    if mode not in {"cppsim", "rtlsim"}:
        raise ValueError("mode must be 'cppsim' or 'rtlsim'")
    requirements = artifact.requirements
    if weights is None and requirements.weight_initializer is None:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-runtime-weights-missing",
                    "external delivery requires runtime weights when no initializer was captured",
                ),
            )
        )
    selected_weights = (
        cast(MVAUTensorData, requirements.weight_initializer).as_array()
        if weights is None
        else np.asarray(weights)
    )
    context = {
        requirements.activation_tensor_id: np.asarray(activation, dtype=np.float32),
        requirements.weight_tensor_id: np.asarray(selected_weights, dtype=np.float32),
        requirements.output_tensor_id: np.zeros(requirements.output_shape, dtype=np.float32),
    }
    with _declared_finn_root(requirements.finn_root):
        operation = getCustomOp(artifact.model.graph.node[0])
        operation.set_nodeattr("exec_mode", mode)
        operation.execute_node(context, artifact.model.graph)
    return cast(np.ndarray, context[requirements.output_tensor_id])


def observe_mvau_rtl_artifact(
    artifact: MVAUBuiltRTLArtifact,
    activation: np.ndarray,
    weights: np.ndarray | None = None,
) -> MVAURTLSimulationObservation:
    """Run XSIM and retain numerical output plus ordered output transactions."""
    requirements = artifact.requirements
    if weights is None and requirements.weight_initializer is None:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-runtime-weights-missing",
                    "an RTL observation requires concrete runtime weight values",
                ),
            )
        )
    selected_weights = (
        cast(MVAUTensorData, requirements.weight_initializer).as_array()
        if weights is None
        else np.asarray(weights, dtype=np.float32)
    )
    activation_array = np.asarray(activation, dtype=np.float32)
    output = simulate_mvau_rtl_artifact(
        artifact,
        activation_array,
        selected_weights,
        mode="rtlsim",
    )
    expected = np.matmul(activation_array, selected_weights).reshape(requirements.output_shape)
    output_requirement = next(
        item for item in requirements.interfaces if item.interface_name == "out0_V"
    )
    expected_matrix = expected.reshape(-1, requirements.output_shape[-1])
    folded_output_path = Path(artifact.output_directory) / "output.npy"
    if not folded_output_path.is_file():
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-output-transactions-missing",
                    "RTL simulation did not emit its folded output transaction file",
                ),
            )
        )
    folded_output = np.load(folded_output_path).reshape(
        -1, output_requirement.beat_sequence.elements_per_beat
    )
    output_transactions = tuple(tuple(float(value) for value in beat) for beat in folded_output)
    expected_transactions = tuple(
        tuple(float(expected_matrix[position]) for position in beat)
        for beat in output_requirement.beat_sequence.beats
    )
    return MVAURTLSimulationObservation(
        mvau_built_artifact_identity(artifact),
        "finn.xsi:MVAU_rtl",
        requirements.output_shape,
        tuple(float(value) for value in output.flat),
        tuple(float(value) for value in expected.flat),
        output_transactions,
        expected_transactions,
        mvau_rtlsim_cycles(artifact),
    )


def observe_mvau_cyclic_stitched_artifact(
    artifact: MVAUBuiltRTLArtifact,
    activation: np.ndarray,
    build_directory: str | Path,
) -> MVAUStitchedSimulationObservation:
    """Observe the stitched cyclic-delivery plus compute realization."""
    requirements = artifact.requirements
    if requirements.mem_mode != "internal_decoupled":
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-cyclic-topology-required",
                    "stitched cyclic simulation requires internal_decoupled delivery",
                ),
            )
        )
    if requirements.weight_initializer is None:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-stitched-runtime-write-required",
                    "stitched simulation needs a runtime AXI-lite write when no initializer exists",
                ),
            )
        )
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    staged_root = build_root / "declared_sources"
    _stage_source_dependencies(requirements, staged_root)
    model = _materialize_model(requirements)
    previous_build = os.environ.get("FINN_BUILD_DIR")
    os.environ["FINN_BUILD_DIR"] = str(build_root)
    try:
        with _declared_finn_root(requirements.finn_root):
            prepare_ip = getattr(
                importlib.import_module("finn.transformation.fpgadataflow.prepare_ip"),
                "PrepareIP",
            )
            hls_synth_ip = getattr(
                importlib.import_module("finn.transformation.fpgadataflow.hlssynth_ip"),
                "HLSSynthIP",
            )
            create_stitched_ip = getattr(
                importlib.import_module("finn.transformation.fpgadataflow.create_stitched_ip"),
                "CreateStitchedIP",
            )
            onnx_exec = importlib.import_module("finn.core.onnx_exec")
        with _declared_finn_root(str(staged_root)):
            model = model.transform(
                prepare_ip(requirements.target_fpga_part, requirements.clock_period_ns)
            )
            model = model.transform(hls_synth_ip())
            model = model.transform(
                create_stitched_ip(
                    requirements.target_fpga_part,
                    requirements.clock_period_ns,
                )
            )
            model.set_metadata_prop("exec_mode", "rtlsim")
            outputs = onnx_exec.execute_onnx(
                model,
                {requirements.activation_tensor_id: np.asarray(activation, dtype=np.float32)},
            )
    finally:
        if previous_build is None:
            del os.environ["FINN_BUILD_DIR"]
        else:
            os.environ["FINN_BUILD_DIR"] = previous_build
    output = cast(np.ndarray, outputs[requirements.output_tensor_id])
    weights = requirements.weight_initializer.as_array()
    expected = np.matmul(np.asarray(activation, dtype=np.float32), weights).reshape(
        requirements.output_shape
    )
    semantic_result = requirements.elaboration.semantic_result
    if not isinstance(semantic_result, NetworkRef):
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-cyclic-network-required",
                    "stitched cyclic observation requires a selected semantic network",
                ),
            )
        )
    return MVAUStitchedSimulationObservation(
        mvau_built_artifact_identity(artifact),
        "finn.xsi:stitched_mvau_memstream",
        tuple(edge.id for edge in semantic_result.network.edges),
        requirements.output_shape,
        tuple(float(value) for value in output.flat),
        tuple(float(value) for value in expected.flat),
    )


def simulate_mvau_cyclic_stitched_artifact(
    artifact: MVAUBuiltRTLArtifact,
    activation: np.ndarray,
    build_directory: str | Path,
) -> np.ndarray:
    """Return the numerical result from a stitched cyclic-network observation."""
    observation = observe_mvau_cyclic_stitched_artifact(
        artifact,
        activation,
        build_directory,
    )
    return np.asarray(observation.output_values, dtype=np.float32).reshape(observation.output_shape)


def mvau_rtlsim_cycles(artifact: MVAUBuiltRTLArtifact) -> int:
    """Return the cycle count recorded by the requirements-backed RTL simulation."""
    with _declared_finn_root(artifact.requirements.finn_root):
        operation = getCustomOp(artifact.model.graph.node[0])
        cycles = operation.get_nodeattr("cycles_rtlsim")
    if type(cycles) is not int or cycles <= 0:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-cycle-measurement-missing",
                    "RTL simulation has not recorded a positive cycle count",
                ),
            )
        )
    return cycles


__all__ = [
    "MVAUArtifactError",
    "MVAUBuiltRTLArtifact",
    "MVAURTLArtifactRequirements",
    "MVAURTLGeneratedOutput",
    "MVAURTLInterfaceRequirement",
    "MVAURTLSimulationObservation",
    "MVAURTLSourceRequirement",
    "MVAUStitchedSimulationObservation",
    "MVAUTensorData",
    "MVAUWeightPayloadKind",
    "build_mvau_rtl_artifact",
    "build_mvau_rtl_artifact_requirements",
    "mvau_built_artifact_identity",
    "mvau_rtlsim_cycles",
    "observe_mvau_cyclic_stitched_artifact",
    "observe_mvau_rtl_artifact",
    "simulate_mvau_cyclic_stitched_artifact",
    "simulate_mvau_rtl_artifact",
]
