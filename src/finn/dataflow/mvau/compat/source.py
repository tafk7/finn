# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit import of Provider-era MVAU specialization into the legacy space."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from onnx import NodeProto  # type: ignore[import-not-found]

from finn.dataflow.design import Decided, DesignPoint, Engine, Finding, FindingKind, QualifiedPath
from finn.dataflow.datatypes import is_qonnx_datatype
from finn.dataflow.kernels import NO_KERNEL
from finn.dataflow.mvau.compat.operation import (
    DataflowOpResult,
    MVAU_COMPUTE_SELECTION,
    MVAU_DATAFLOW_OP_SPEC,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAUDataflowOpPaths,
)
from finn.dataflow.mvau.associations import MVAUSourceAssociation
from finn.dataflow.mvau.compute_kernels import (
    BATCH_INTERLEAVED_PATHS,
    LEGACY_HLS_PATHS,
    PACKED_DSP_PATHS,
    SOFT_VECTOR_PATHS,
    MVAUComputeKernelId,
    MVAUComputeKernelPathSet,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.source import (
    MVAUModelAccessor,
    MVAUProjectionContext,
    MVAUSourceAdapterError,
    MVAUSourceProjection,
    _attribute_value,
    _dsp_block,
    _finding,
    _find_source_node,
    _initializer_excludes_minimum,
    project_mvau_source,
)
from finn.dataflow.mvau_problem import MVAUDspBlock, MVAUProblemPaths
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    CyclicRamStyle,
    MVAUWeightSupplyKernelId,
    WeightOrganization,
)
from finn.dataflow.region import NumericElementType, element_width
from finn.dataflow.resolution import ResolvedDataflowOp

_ADAPTER_PATH = QualifiedPath("compiler.mvau.compat.source_adapter")


@dataclass(frozen=True)
class MVAULegacyResolvedDesign(ResolvedDataflowOp):
    """Provider-era result envelope, including the retired Region alternative."""

    result: DataflowOpResult
    source_association: MVAUSourceAssociation
    projection: MVAUSourceProjection


class MVAULegacyImportMode(str, Enum):
    """The compatibility API accepts only explicit specialization import."""

    PRESERVE_SPECIALIZATION = "preserve_specialization"


def _effective_narrow_weights(
    excludes_minimum: bool | None,
    runtime_writable: bool,
    runtime_range_contract: bool | None,
) -> bool:
    if runtime_writable:
        return bool(runtime_range_contract)
    return bool(excludes_minimum)


def _legacy_compute_kernel(
    node: NodeProto,
    target: MVAUDspBlock | None,
    activation_type: NumericElementType,
    weight_type: NumericElementType,
    weights_narrow: bool,
    interleave: int,
    findings: list[Finding],
) -> MVAUComputeKernelId | None:
    if node.op_type == "MVAU_hls":
        return MVAUComputeKernelId.LEGACY_HLS
    if node.op_type != "MVAU_rtl":
        return None
    if interleave > 1:
        return MVAUComputeKernelId.BATCH_INTERLEAVED_DSP
    if target is None:
        findings.append(
            _finding(
                FindingKind.AUTHORING,
                "mvau-legacy-rtl-kernel-ambiguous",
                MVAU_COMPUTE_SELECTION.paths.kernel,
                "legacy RTL soft-vector versus packed selection requires target facts",
            )
        )
        return None
    if target is not MVAUDspBlock.DSP58:
        return MVAUComputeKernelId.SOFT_VECTOR
    weight_bits = element_width(weight_type)
    activation_bits = element_width(activation_type)
    lane_width = weight_bits + activation_bits - 1
    lanes = (
        1
        if weight_bits == 27
        else 1 + (27 - (0 if weights_narrow else 1) - weight_bits) // lane_width
    )
    packed = lanes <= 3 and weight_bits <= 8 and activation_bits <= 9
    return MVAUComputeKernelId.PACKED_DSP if packed else MVAUComputeKernelId.SOFT_VECTOR


def _legacy_hls_resource(node: NodeProto, findings: list[Finding]) -> MVAUHlsResource | None:
    resource = cast(str, _attribute_value(node, "resType", "auto"))
    if resource == "auto":
        findings.append(
            _finding(
                FindingKind.AUTHORING,
                "mvau-legacy-resource-ambiguous",
                LEGACY_HLS_PATHS.resource,
                "legacy HLS resType does not identify LUT versus DSP arithmetic mapping",
            )
        )
        return None
    mapped = {"lut": MVAUHlsResource.LUT, "dsp": MVAUHlsResource.DSP}.get(resource)
    if mapped is None:
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-legacy-resource-unsupported",
                LEGACY_HLS_PATHS.resource,
                "legacy HLS resType has no arithmetic resource mapping",
                value=resource,
            )
        )
    return mapped


_COMPUTE_KERNEL_PATHS: dict[MVAUComputeKernelId, MVAUComputeKernelPathSet] = {
    MVAUComputeKernelId.LEGACY_HLS: LEGACY_HLS_PATHS,
    MVAUComputeKernelId.SOFT_VECTOR: SOFT_VECTOR_PATHS,
    MVAUComputeKernelId.PACKED_DSP: PACKED_DSP_PATHS,
    MVAUComputeKernelId.BATCH_INTERLEAVED_DSP: BATCH_INTERLEAVED_PATHS,
}


def _legacy_assignments(
    node: NodeProto,
    mem_mode: str,
    target: MVAUDspBlock | None,
    activation_type: NumericElementType,
    weight_type: NumericElementType,
    weights_narrow: bool,
    findings: list[Finding],
) -> dict[QualifiedPath, object]:
    pe = cast(int, _attribute_value(node, "PE", 0))
    simd = cast(int, _attribute_value(node, "SIMD", 0))
    interleave = cast(int, _attribute_value(node, "TH", 1))
    assignments: dict[QualifiedPath, object] = {}
    kernel_id = _legacy_compute_kernel(
        node, target, activation_type, weight_type, weights_narrow, interleave, findings
    )
    if kernel_id is None:
        return assignments
    paths = _COMPUTE_KERNEL_PATHS[kernel_id]
    assignments[MVAU_COMPUTE_SELECTION.paths.kernel] = kernel_id.value
    if pe > 0:
        assignments[paths.pe] = pe
    if simd > 0:
        assignments[paths.simd] = simd
    if kernel_id is MVAUComputeKernelId.BATCH_INTERLEAVED_DSP:
        assignments[paths.interleave] = interleave
    if kernel_id is MVAUComputeKernelId.LEGACY_HLS:
        resource = _legacy_hls_resource(node, findings)
        if resource is not None:
            assignments[paths.resource] = resource
        assignments[paths.weight_source] = (
            MVAUWeightSource.EMBEDDED
            if mem_mode == "internal_embedded"
            else MVAUWeightSource.STREAMED
        )
    if kernel_id in {MVAUComputeKernelId.SOFT_VECTOR, MVAUComputeKernelId.PACKED_DSP}:
        assignments[paths.compute_pumping] = bool(_attribute_value(node, "pumpedCompute", 0))
    if mem_mode == "internal_decoupled":
        assignments[MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel] = (
            MVAUWeightSupplyKernelId.FINN_RTL_MEMSTREAM.value
        )
        assignments[FINN_RTL_MEMSTREAM_PATHS.organization] = WeightOrganization.AS_DEMANDED
        ram_style = cast(str, _attribute_value(node, "ram_style", "auto"))
        try:
            assignments[FINN_RTL_MEMSTREAM_PATHS.ram_style] = CyclicRamStyle(ram_style)
        except ValueError:
            findings.append(
                _finding(
                    FindingKind.LIMITATION,
                    "mvau-legacy-ram-style-unsupported",
                    FINN_RTL_MEMSTREAM_PATHS.ram_style,
                    "legacy RAM style has no cyclic-supply mapping",
                    value=ram_style,
                )
            )
        assignments[FINN_RTL_MEMSTREAM_PATHS.pumped_memory] = bool(
            _attribute_value(node, "pumpedMemory", 0)
        )
        assignments[MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel] = NO_KERNEL
    elif mem_mode == "external":
        assignments[MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel] = NO_KERNEL
    return assignments


def project_legacy_mvau_source(
    model: MVAUModelAccessor,
    source_node_id: str,
    context: MVAUProjectionContext,
    *,
    source_scope_id: str | None = None,
) -> MVAUSourceProjection:
    """Project facts and explicitly import Provider-era specialization choices."""

    projection = project_mvau_source(
        model,
        source_node_id,
        context,
        source_scope_id=source_scope_id,
    )
    node = _find_source_node(model, source_node_id)
    if node is None or projection.source_description is None:
        return projection
    problem = dict(projection.problem_data)
    findings = list(projection.findings)
    activation_type = problem.get(MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE)
    weight_type = problem.get(MVAUProblemPaths.WEIGHT_ELEMENT_TYPE)
    if not is_qonnx_datatype(activation_type):
        return projection
    if not is_qonnx_datatype(weight_type):
        return projection
    runtime_writable = bool(problem[MVAUProblemPaths.RUNTIME_WRITABLE])
    mem_mode = cast(str, _attribute_value(node, "mem_mode", "internal_decoupled"))
    target = _dsp_block(context.fpga_part) if context.fpga_part is not None else None
    excludes_minimum = _initializer_excludes_minimum(model, node.input[1], weight_type, mem_mode)
    assignments = _legacy_assignments(
        node,
        mem_mode,
        target,
        activation_type,
        weight_type,
        _effective_narrow_weights(
            excludes_minimum,
            runtime_writable,
            cast("bool | None", problem.get(MVAUProblemPaths.RUNTIME_WEIGHT_RANGE_CONTRACT)),
        ),
        findings,
    )
    return MVAUSourceProjection(
        projection.source_description,
        problem,
        assignments,
        tuple(findings),
    )


def start_legacy_mvau_projection(
    projection: MVAUSourceProjection,
    assignments: Mapping[QualifiedPath | str, object] | None = None,
) -> MVAULegacyResolvedDesign:
    """Resolve an explicitly imported legacy projection in the compatibility space."""

    if projection.blocking_findings:
        raise MVAUSourceAdapterError(projection.blocking_findings)
    commitments: dict[QualifiedPath | str, object] = {
        path: value for path, value in projection.imported_assignments.items()
    }
    if assignments is not None:
        commitments.update(assignments)
    engine = Engine()
    point = engine.start(engine.validate(MVAU_DATAFLOW_OP_SPEC), projection.problem_data)
    if commitments:
        committed = engine.commit_assignments(point, commitments)
        failures = tuple(
            finding
            for outcome in committed.outcomes
            if outcome.disposition not in {"committed", "unchanged"}
            for finding in outcome.findings
        )
        if failures or any(
            outcome.disposition not in {"committed", "unchanged"} for outcome in committed.outcomes
        ):
            raise MVAUSourceAdapterError(
                failures
                or (
                    _finding(
                        FindingKind.REJECTION,
                        "mvau-legacy-assignment-commit-failed",
                        _ADAPTER_PATH,
                        "one or more legacy assignments could not be committed",
                    ),
                )
            )
        point = committed.point
    return resolve_legacy_mvau_point(engine, point, projection)


def resolve_legacy_mvau_point(
    engine: Engine,
    point: DesignPoint,
    projection: MVAUSourceProjection,
    *,
    source_scope_id: str | None = None,
) -> MVAULegacyResolvedDesign:
    """Resolve a Provider-era Region-or-Network result inside compatibility."""

    result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    if not isinstance(result, Decided):
        raise MVAUSourceAdapterError(result.findings)
    selected = cast(DataflowOpResult, result.value)
    association = selected.source_association
    return MVAULegacyResolvedDesign(
        engine,
        point,
        selected,
        association,
        source_scope_id or association.source_node_id,
        projection,
    )


__all__ = [
    "MVAULegacyImportMode",
    "MVAULegacyResolvedDesign",
    "project_legacy_mvau_source",
    "resolve_legacy_mvau_point",
    "start_legacy_mvau_projection",
]
