# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed artifact requirements for the first MVAU RTL soft-vector slice."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import importlib
import os
from pathlib import Path
from typing import Iterator, cast

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.custom_op.registry import getCustomOp  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from finn.dataflow.design import Finding, FindingKind, QualifiedPath
from finn.dataflow.mvau.definition import MVAUComputeKernelPaths
from finn.dataflow.mvau.elaboration import (
    MVAUPhysicalElaboration,
    MVAUPhysicalNumericInterface,
)
from finn.dataflow.mvau.source import MVAUModelAccessor, MVAUResolvedDesign
from finn.dataflow.ops.mvau import NetworkRef, RegionRef
from finn.dataflow.parameters.cyclic.definition import CyclicParameterKernelPaths
from finn.dataflow.region import BeatSequence, NumericElementType

_ARTIFACT_PATH = QualifiedPath("artifact.mvau.rtl_softvec")


class MVAUSourceFileOrigin(str, Enum):
    GENERATED = "generated"
    FINN_RTL_LIBRARY = "finn_rtl_library"


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
    origin: MVAUSourceFileOrigin
    path: str


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
    weight_initializer: MVAUTensorData
    parameters: tuple[tuple[str, bool | int | str], ...]
    interfaces: tuple[MVAURTLInterfaceRequirement, ...]
    source_manifest: tuple[MVAURTLSourceRequirement, ...]
    elaboration: MVAUPhysicalElaboration

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))
        object.__setattr__(
            self, "interfaces", tuple(sorted(self.interfaces, key=lambda item: item.id))
        )
        object.__setattr__(
            self, "source_manifest", tuple(sorted(self.source_manifest, key=lambda item: item.id))
        )


@dataclass(frozen=True)
class MVAUBuiltRTLArtifact:
    """Generated artifact paths plus the requirements-backed legacy model."""

    requirements: MVAURTLArtifactRequirements
    output_directory: str
    source_files: tuple[str, ...]
    model: ModelWrapper


class MVAUArtifactError(ValueError):
    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = tuple(
            sorted(findings, key=lambda item: (item.path, item.kind.value, item.code))
        )
        super().__init__(f"MVAU artifact preparation failed with {len(self.findings)} finding(s)")


def _finding(code: str, message: str, path: QualifiedPath = _ARTIFACT_PATH) -> Finding:
    return Finding(FindingKind.REJECTION, code, path, message)


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


def _source_manifest(
    finn_root: Path, module_name: str, *, cyclic: bool
) -> tuple[MVAURTLSourceRequirement, ...]:
    sources = [
        MVAURTLSourceRequirement(
            "compute.wrapper",
            MVAUSourceFileOrigin.GENERATED,
            f"{module_name}_wrapper.v",
        ),
        *(
            MVAURTLSourceRequirement(
                f"compute.library.{index}",
                MVAUSourceFileOrigin.FINN_RTL_LIBRARY,
                str(finn_root / "finn-rtllib" / "mvu" / filename),
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
    if cyclic:
        sources.extend(
            (
                MVAURTLSourceRequirement(
                    "delivery.wrapper",
                    MVAUSourceFileOrigin.GENERATED,
                    f"{module_name}_memstream_wrapper.v",
                ),
                MVAURTLSourceRequirement(
                    "delivery.library.0",
                    MVAUSourceFileOrigin.FINN_RTL_LIBRARY,
                    str(finn_root / "finn-rtllib" / "memstream" / "hdl" / "memstream.sv"),
                ),
                MVAURTLSourceRequirement(
                    "delivery.library.1",
                    MVAUSourceFileOrigin.FINN_RTL_LIBRARY,
                    str(finn_root / "finn-rtllib" / "memstream" / "hdl" / "memstream_axi.sv"),
                ),
            )
        )
    missing = tuple(
        item.path
        for item in sources
        if item.origin is MVAUSourceFileOrigin.FINN_RTL_LIBRARY and not Path(item.path).is_file()
    )
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
    return tuple(sources)


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
    if elaboration.semantic_result != resolved.result:
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-elaboration-mismatch",
                    "physical elaboration does not belong to the selected semantic result",
                ),
            )
        )
    source = resolved.result.source_association
    nodes = tuple(node for node in model.graph.node if node.name == source.source_node_id)
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
    initializer = model.get_initializer(description.weight_operand_id)
    if initializer is None:
        raise MVAUArtifactError(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-artifact-weight-values-missing",
                    MVAUComputeKernelPaths.WEIGHT_INITIALIZER_AVAILABLE,
                    "the covered RTL build requires concrete weight values",
                ),
            )
        )
    weight_array = np.asarray(initializer, dtype=np.float32)
    matrix_width = cast(int, resolved.point.problem[MVAUComputeKernelPaths.MATRIX_WIDTH])
    matrix_height = cast(int, resolved.point.problem[MVAUComputeKernelPaths.MATRIX_HEIGHT])
    expected_weight_shape = (matrix_width, matrix_height)
    if tuple(weight_array.shape) != expected_weight_shape:
        raise MVAUArtifactError(
            (_finding("mvau-artifact-weight-shape-mismatch", "weight values do not match MW x MH"),)
        )
    activation_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUComputeKernelPaths.ACTIVATION_ELEMENT_TYPE],
    )
    weight_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE],
    )
    accumulator_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUComputeKernelPaths.ACCUMULATOR_ELEMENT_TYPE],
    )
    output_type = cast(
        NumericElementType,
        resolved.point.problem[MVAUComputeKernelPaths.OUTPUT_ELEMENT_TYPE],
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
    if isinstance(resolved.result, RegionRef):
        compute_region = resolved.result.region
        cyclic = False
        mem_mode = "external"
    else:
        compute_region = resolved.result.network.node("compute").region
        cyclic = True
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
        delivery_id = f"{source.source_node_id}.delivery.memstream"
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
    ram_style = resolved.point.assignments.get(CyclicParameterKernelPaths.RAM_STYLE)
    pumped_memory = resolved.point.assignments.get(CyclicParameterKernelPaths.PUMPED_MEMORY)
    runtime_writable = resolved.point.problem.get(
        CyclicParameterKernelPaths.RUNTIME_WRITABLE, False
    )
    parameters += (
        ("RAM_STYLE", "auto" if ram_style is None else cast(Enum, ram_style).value),
        ("RUNTIME_WRITABLE", cast(bool, runtime_writable)),
        ("PUMPED_MEMORY", False if pumped_memory is None else cast(bool, pumped_memory)),
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
        MVAUTensorData(expected_weight_shape, tuple(float(value) for value in weight_array.flat)),
        parameters,
        interfaces,
        _source_manifest(root, source.source_node_id, cyclic=cyclic),
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
    if requirements.mem_mode == "internal_decoupled":
        model.set_initializer(
            requirements.weight_tensor_id, requirements.weight_initializer.as_array()
        )
    return model


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
    model = _materialize_model(requirements)
    node = model.graph.node[0]
    with _declared_finn_root(requirements.finn_root):
        operation = getCustomOp(node)
        operation.set_nodeattr("code_gen_dir_ipgen", str(output))
        operation.set_nodeattr("exec_mode", "rtlsim")
        operation.generate_hdl(
            model,
            requirements.target_fpga_part,
            requirements.clock_period_ns,
        )
        if prepare_rtlsim:
            operation.prepare_rtlsim(behav=True)
    source_files = tuple(
        str(output / item.path) if item.origin is MVAUSourceFileOrigin.GENERATED else item.path
        for item in requirements.source_manifest
    )
    missing = tuple(path for path in source_files if not Path(path).is_file())
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
    return MVAUBuiltRTLArtifact(requirements, str(output), source_files, model)


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
    selected_weights = (
        requirements.weight_initializer.as_array() if weights is None else np.asarray(weights)
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


def simulate_mvau_cyclic_stitched_artifact(
    requirements: MVAURTLArtifactRequirements,
    activation: np.ndarray,
    build_directory: str | Path,
) -> np.ndarray:
    """Run the existing stitched-IP path including the selected cyclic memstream."""
    if requirements.mem_mode != "internal_decoupled":
        raise MVAUArtifactError(
            (
                _finding(
                    "mvau-artifact-cyclic-topology-required",
                    "stitched cyclic simulation requires internal_decoupled delivery",
                ),
            )
        )
    build_root = Path(build_directory).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
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
    return cast(np.ndarray, outputs[requirements.output_tensor_id])


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
    "MVAURTLInterfaceRequirement",
    "MVAURTLSourceRequirement",
    "MVAUSourceFileOrigin",
    "MVAUTensorData",
    "build_mvau_rtl_artifact",
    "build_mvau_rtl_artifact_requirements",
    "mvau_rtlsim_cycles",
    "simulate_mvau_cyclic_stitched_artifact",
    "simulate_mvau_rtl_artifact",
]
