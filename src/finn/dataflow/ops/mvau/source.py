# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiler-owned projection and persistence for real FINN MVAU nodes.

The mapping implemented here is deliberately explicit:

* graph and target observations become immutable problem fields;
* legacy specialization is imported only in the opt-in preservation mode;
* selected regions, networks, associations, and constraint results are recomputed; and
* persistence stores only committed decision paths and values.

This module is FINN integration code.  Kernel definitions remain independent of
ONNX, ``ModelWrapper``, and legacy node attributes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import prod
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np  # type: ignore[import-not-found]
from onnx import AttributeProto, GraphProto, NodeProto  # type: ignore[import-not-found]

from finn.dataflow.design import (
    Decided,
    DesignPoint,
    DesignSpace,
    Engine,
    Finding,
    FindingKind,
    QualifiedPath,
)
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.ops.mvau.assignments import MVAU_DECISION_NODEATTRS, local_mvau_assignment_path
from finn.dataflow.ops.mvau import (
    MVAU_DATAFLOW_OP_SPEC,
    MVAUDataflowOpPaths,
    NetworkRef,
    MVAUSourceAssociation,
    MVAUSourceDescription,
)
from finn.dataflow.resolution import ResolvedDataflowOp
from finn.dataflow.op_contracts import DataflowOpError, NodeAttributeType
from finn.dataflow.parameters.cyclic.definition import (
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblemPaths,
)
from finn.dataflow.datatypes import (
    DatatypeError,
    canonical_qonnx_datatype,
    resolve_qonnx_datatype_name,
    encode_datatype,
    is_qonnx_datatype,
)
from finn.dataflow.region import (
    BeatSequence,
    NumericElementType,
    element_width,
)

_ADAPTER_PATH = QualifiedPath("compiler.mvau.source_adapter")
_PERSISTENCE_PATH = QualifiedPath("compiler.mvau.selection")
_LEGACY_COMPUTE_SELECTION_PATH = QualifiedPath("mvau.compute.kernel")
_LEGACY_PARAMETER_TOPOLOGY_PATH = QualifiedPath("semantic.mvau.op.parameter_topology")
_ADAPTER_KEY = "finn.dataflow.mvau"
_FORMAT_VERSION = 1
#: v11 records the v6 DataflowDesign decision paths and Network-only result
#: model.  The v10 semantic-Kernel envelope is rejected explicitly; no old
#: selection is reinterpreted as a new design choice.
MVAU_DECLARATION_FAMILY_VERSION = "mvau-source-composition-v11"
MVAU_LOGICAL_SOURCE_NODEATTRS: Mapping[str, NodeAttributeType] = MappingProxyType(
    {
        "noActivation": ("i", False, 1, {0, 1}),
        "binaryXnorMode": ("i", False, 0, {0, 1}),
        "accDataType": ("s", False, "INT32", None),
        "ActVal": ("i", False, 0, None),
        # Durable provenance: every original source node a lowering consumed.
        "dataflow_source_nodes": ("s", False, "", None),
    }
)


class _MinimumLike(Protocol):
    def min(self) -> object: ...


class MVAUModelAccessor(Protocol):
    """Narrow ``ModelWrapper`` surface consumed by the source adapter."""

    @property
    def graph(self) -> GraphProto: ...

    def get_tensor_shape(self, tensor_name: str) -> list[int] | None: ...

    def get_tensor_datatype(self, tensor_name: str) -> object: ...

    def get_initializer(self, tensor_name: str) -> object | None: ...

    def get_metadata_prop(self, key: str) -> str | None: ...

    def set_metadata_prop(self, key: str, value: str) -> None: ...


@dataclass(frozen=True)
class MVAUProjectionContext:
    """Compiler context consumed by the first MVAU source projection."""

    accumulator_type_analysis_owner: str
    fpga_part: str | None = None
    clock_period_ns: float | None = None
    supports_initialized_uram: bool | None = None
    external_weight_sequence: BeatSequence | None = None
    runtime_writable_weights: bool | None = None

    def __post_init__(self) -> None:
        if not self.accumulator_type_analysis_owner:
            raise ValueError("accumulator_type_analysis_owner must not be empty")
        if self.clock_period_ns is not None and self.clock_period_ns <= 0:
            raise ValueError("clock_period_ns must be positive when supplied")


@dataclass(frozen=True)
class MVAUSourceMappingEntry:
    """One documented source-to-design-space ownership mapping."""

    observation: str
    destination: str
    classification: str


MVAU_SOURCE_MAPPING = (
    MVAUSourceMappingEntry("node/tensor identities and shapes", "problem.mvau.*", "problem"),
    MVAUSourceMappingEntry(
        "numeric and accumulator types", "problem.mvau.*_element_type", "problem"
    ),
    MVAUSourceMappingEntry(
        "accumulator analysis owner", "problem.mvau.accumulator_type_analysis_owner", "problem"
    ),
    MVAUSourceMappingEntry(
        "initializer availability, content fingerprints, and runtime-write requirements",
        "problem.mvau/problem.cyclic_parameter",
        "problem",
    ),
    MVAUSourceMappingEntry("target DSP and memory capabilities", "problem.target.*", "problem"),
    MVAUSourceMappingEntry("PE/SIMD/TH and explicit legacy modes", "decision paths", "assignment"),
    MVAUSourceMappingEntry(
        "selected ports, regions, networks, associations", "semantic.*", "derived"
    ),
    MVAUSourceMappingEntry("unknown or ambiguous legacy attributes", "Finding", "finding"),
)


def tensor_value_fingerprint(value: object) -> str:
    """Return a deterministic digest for a concrete source tensor value."""
    array = np.asarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(tuple(int(extent) for extent in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class MVAUSourceProjection:
    """Immutable projection product before an engine design point is started."""

    source_description: MVAUSourceDescription | None
    problem_data: Mapping[QualifiedPath, object]
    imported_assignments: Mapping[QualifiedPath, object]
    findings: tuple[Finding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_data", MappingProxyType(dict(self.problem_data)))
        object.__setattr__(
            self,
            "imported_assignments",
            MappingProxyType(dict(self.imported_assignments)),
        )
        object.__setattr__(
            self,
            "findings",
            tuple(sorted(self.findings, key=lambda item: (item.path, item.kind.value, item.code))),
        )

    @property
    def blocking_findings(self) -> tuple[Finding, ...]:
        """Return findings that prevent a truthful problem snapshot."""
        return tuple(
            item
            for item in self.findings
            if item.kind in {FindingKind.BLOCKER, FindingKind.REJECTION, FindingKind.LIMITATION}
        )


@dataclass(frozen=True)
class MVAUResolvedDesign(ResolvedDataflowOp):
    """A re-created design point and its recomputed selected semantics."""

    result: NetworkRef
    source_association: MVAUSourceAssociation
    projection: MVAUSourceProjection


@dataclass(frozen=True)
class MVAUSelectionEnvelope:
    """Compiler-owned portable envelope around a sparse choice map."""

    adapter_key: str
    format_version: int
    declaration_family_version: str
    source_scope_id: str
    problem_fingerprint: str
    assignments: tuple[tuple[str, object], ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "adapter_key": self.adapter_key,
                "assignments": [{"path": path, "value": value} for path, value in self.assignments],
                "declaration_family_version": self.declaration_family_version,
                "format_version": self.format_version,
                "problem_fingerprint": self.problem_fingerprint,
                "source_scope_id": self.source_scope_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class MVAUSourceAdapterError(DataflowOpError):
    """Raised when projection or reconstitution cannot preserve its contract."""

    def __init__(self, findings: tuple[Finding, ...]) -> None:
        ordered = tuple(sorted(findings, key=lambda item: (item.path, item.kind.value, item.code)))
        super().__init__(ordered)
        self.args = (f"MVAU source adaptation failed with {len(self.findings)} finding(s)",)


def _finding(
    kind: FindingKind,
    code: str,
    path: QualifiedPath,
    message: str,
    **values: object,
) -> Finding:
    return Finding(kind, code, path, message, tuple(values.items()))


def _find_source_node(model: MVAUModelAccessor, source_node_id: str) -> NodeProto | None:
    matches = tuple(node for node in model.graph.node if node.name == source_node_id)
    return matches[0] if len(matches) == 1 else None


def _attribute(node: NodeProto, name: str) -> tuple[object | None, bool]:
    attribute = next((item for item in node.attribute if item.name == name), None)
    if attribute is None:
        return None, False
    if attribute.type == AttributeProto.INT:
        return int(attribute.i), True
    if attribute.type == AttributeProto.STRING:
        return attribute.s.decode("utf-8"), True
    if attribute.type == AttributeProto.INTS:
        return tuple(int(value) for value in attribute.ints), True
    return None, True


def _attribute_value(node: NodeProto, name: str, default: object) -> object:
    value, present = _attribute(node, name)
    return value if present else default


def _logical_source_attributes(
    node: NodeProto,
    findings: list[Finding],
) -> Mapping[str, object]:
    values: dict[str, object] = {}
    for name, (storage_type, _required, default, allowed) in MVAU_LOGICAL_SOURCE_NODEATTRS.items():
        value, present = _attribute(node, name)
        value = value if present else default
        valid_type = (storage_type == "i" and type(value) is int) or (
            storage_type == "s" and type(value) is str
        )
        if not valid_type or (allowed is not None and value not in allowed):
            findings.append(
                _finding(
                    FindingKind.REJECTION,
                    "mvau-logical-source-attribute-invalid",
                    MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
                    "logical MVAU source attribute is malformed or outside its domain",
                    attribute=name,
                )
            )
            continue
        values[name] = value
    return MappingProxyType(values)


def _tensor_type(
    model: MVAUModelAccessor,
    tensor_id: str,
    path: QualifiedPath,
    role: str,
    findings: list[Finding],
) -> NumericElementType | None:
    """The tensor's datatype, carried through rather than classified.

    The adapter used to reduce the annotation to a local family-and-width pair
    here, and refuse anything that did not fit five known families.  That put
    two unrelated jobs in one place: reporting that a tensor is unusable *as a
    source*, and deciding which datatypes some hardware can build.

    Only the first is this function's.  A fixed-point or scaled-integer MatMul
    is a perfectly well-formed graph -- the reason FINN will not lower it is
    that no Kernel covers those operands, which is a coverage answer with a
    Kernel's name on it, not a projection failure.  So the datatype travels as
    itself, and the refusal happens where the reason lives.

    What remains here is genuine source trouble: an annotation QONNX cannot
    resolve, or a degenerate width that could not describe a beat.
    """

    try:
        element_type = canonical_qonnx_datatype(model.get_tensor_datatype(tensor_id))
    except (DatatypeError, KeyError, TypeError, ValueError, AttributeError):
        element_type = None
    if element_type is None or element_width(element_type) <= 0:
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-datatype-unknown",
                path,
                f"{role} tensor has no resolvable QONNX datatype of positive width",
                tensor=tensor_id,
            )
        )
        return None
    return element_type


def _tensor_shape(
    model: MVAUModelAccessor,
    tensor_id: str,
    path: QualifiedPath,
    role: str,
    findings: list[Finding],
) -> tuple[int, ...] | None:
    shape = model.get_tensor_shape(tensor_id)
    if shape is None or any(type(extent) is not int or extent <= 0 for extent in shape):
        findings.append(
            _finding(
                FindingKind.BLOCKER,
                "mvau-source-shape-missing",
                path,
                f"{role} tensor requires a concrete positive shape",
                tensor=tensor_id,
            )
        )
        return None
    return tuple(shape)


def _dsp_block(fpga_part: str) -> DspBlock | None:
    if len(fpga_part) < 4 or not fpga_part.startswith(("xc", "xq")):
        return None
    if fpga_part.startswith(("xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm", "xqrvc", "xcv80")):
        return DspBlock.DSP58
    if len(fpga_part) > 2 and fpga_part[2] == "7":
        return DspBlock.DSP48E1
    return DspBlock.DSP48E2


def classify_mvau_dsp_block(fpga_part: str) -> DspBlock | None:
    """Classify one FPGA part into the MVAU binding's supported DSP families."""

    return _dsp_block(fpga_part)


def _supports_initialized_uram(context: MVAUProjectionContext) -> bool | None:
    if context.supports_initialized_uram is not None:
        return context.supports_initialized_uram
    if context.fpga_part is None:
        return None
    return _dsp_block(context.fpga_part) is DspBlock.DSP58


def project_mvau_build_problem(
    context: MVAUProjectionContext,
    *,
    runtime_writable_weights: bool,
) -> Mapping[QualifiedPath, object]:
    """Project MVAU target and invocation facts from explicit build context."""

    problem: dict[QualifiedPath, object] = {
        MVAUProblemPaths.RUNTIME_WRITABLE: runtime_writable_weights,
    }
    target = _dsp_block(context.fpga_part) if context.fpga_part is not None else None
    if target is not None:
        problem[MVAUProblemPaths.TARGET_DSP_BLOCK] = target
    capabilities = _supports_initialized_uram(context)
    if capabilities is not None:
        problem[MVAUProblemPaths.TARGET_MEMORY_CAPABILITIES] = CyclicTargetMemoryCapabilities(
            capabilities
        )
    if context.external_weight_sequence is not None:
        problem[MVAUProblemPaths.EXTERNAL_WEIGHT_SEQUENCE] = context.external_weight_sequence
    if context.fpga_part is not None:
        problem[MVAUDataflowOpPaths.TARGET_FPGA_PART] = context.fpga_part
    if context.clock_period_ns is not None:
        problem[MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS] = context.clock_period_ns
    return MappingProxyType(problem)


def _initializer_excludes_minimum(
    model: MVAUModelAccessor,
    weight_id: str,
    weight_type: NumericElementType,
    mem_mode: str,
) -> bool | None:
    """Whether the governing initializer's values exclude the type minimum.

    This is a graph-analysis fact and nothing else.  Whether it is the fact
    that decides ``NARROW_WEIGHTS`` is the operation's question, answered by
    the ``effective_narrow_weights`` property, which also weighs the caller's
    runtime contract.  ``None`` means the graph cannot answer: there is no
    initializer, or an imported delivery mode makes it not the value authority.
    """

    initializer = model.get_initializer(weight_id)
    if initializer is None or mem_mode in {"external", "external_mem", "dynamic"}:
        return None
    try:
        minimum = float(cast(float, cast(_MinimumLike, initializer).min()))
    except (AttributeError, TypeError, ValueError):
        return None
    # The datatype's own minimum, rather than a width-and-family reconstruction
    # of it.  ``-(2 ** (w - 1)) if signed else 0`` is what ``min()`` computes,
    # and asking the datatype means special encodings answer for themselves
    # instead of being folded into whichever branch their label fell in.
    try:
        minimum_type_value = float(weight_type.min())
    except Exception:  # noqa: BLE001 - QONNX leaves min() undefined for some types
        return None
    return minimum != minimum_type_value


def project_mvau_graph_source(
    model: MVAUModelAccessor,
    source_node_id: str,
    *,
    source_scope_id: str | None = None,
) -> MVAUSourceProjection:
    """Project graph-owned facts for one real FINN MVAU source node."""
    findings: list[Finding] = []
    node = _find_source_node(model, source_node_id)
    if node is None:
        findings.append(
            _finding(
                FindingKind.BLOCKER,
                "mvau-source-node-not-unique",
                MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
                "source node identity must resolve to exactly one graph node",
                source_node_id=source_node_id,
            )
        )
        return MVAUSourceProjection(None, {}, {}, tuple(findings))
    logical_node = node.op_type == "MvauDataflowOp" and node.domain == "finn.custom_op.dataflow"
    if node.op_type not in {"MVAU", "MVAU_hls", "MVAU_rtl"} and not logical_node:
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-op-unsupported",
                MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
                "source node is not a supported FINN MVAU operation",
                op_type=node.op_type,
            )
        )
        return MVAUSourceProjection(None, {}, {}, tuple(findings))
    logical_attributes = _logical_source_attributes(node, findings) if logical_node else {}
    if logical_node and len(logical_attributes) != len(MVAU_LOGICAL_SOURCE_NODEATTRS):
        return MVAUSourceProjection(None, {}, {}, tuple(findings))

    def source_attribute(name: str, legacy_default: object) -> object:
        if logical_node:
            return logical_attributes[name]
        return _attribute_value(node, name, legacy_default)

    no_activation = bool(source_attribute("noActivation", 0))
    expected_inputs = 2 if no_activation else 3
    if len(node.input) != expected_inputs or len(node.output) != 1:
        findings.append(
            _finding(
                FindingKind.REJECTION,
                "mvau-source-arity-inconsistent",
                MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
                "MVAU source input/output arity is inconsistent with noActivation",
                actual_inputs=len(node.input),
                expected_inputs=expected_inputs,
            )
        )
        return MVAUSourceProjection(None, {}, {}, tuple(findings))

    activation_id, weight_id = node.input[:2]
    output_id = node.output[0]
    threshold_id = None if no_activation else node.input[2]
    activation_shape = _tensor_shape(
        model,
        activation_id,
        MVAUProblemPaths.MATRIX_WIDTH,
        "activation",
        findings,
    )
    weight_shape = _tensor_shape(
        model,
        weight_id,
        MVAUProblemPaths.MATRIX_HEIGHT,
        "weight",
        findings,
    )
    output_shape = _tensor_shape(
        model,
        output_id,
        MVAUProblemPaths.MATRIX_HEIGHT,
        "output",
        findings,
    )
    activation_type = _tensor_type(
        model,
        activation_id,
        MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE,
        "activation",
        findings,
    )
    weight_type = _tensor_type(
        model,
        weight_id,
        MVAUProblemPaths.WEIGHT_ELEMENT_TYPE,
        "weight",
        findings,
    )
    output_type = _tensor_type(
        model,
        output_id,
        MVAUProblemPaths.OUTPUT_ELEMENT_TYPE,
        "output",
        findings,
    )
    accumulator_name = source_attribute("accDataType", "INT32")
    # Resolved once, through the canonical name, and carried as that object --
    # the attribute is a name and this is the one place it becomes a value.
    try:
        # Permissive on purpose: ``accDataType`` is a legacy node attribute, so
        # it is whatever a human or an older FINN wrote, and adopting QONNX
        # means adopting its reading of that spelling.  Persistence is the
        # opposite case and uses the strict decoder.
        accumulator_type = resolve_qonnx_datatype_name(cast(str, accumulator_name))
    except DatatypeError:
        accumulator_type = None
    if accumulator_type is None or element_width(accumulator_type) <= 0:
        accumulator_type = None
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-accumulator-datatype-unknown",
                MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE,
                "accDataType does not identify a supported numeric datatype",
                value=str(accumulator_name),
            )
        )
    mw = (
        weight_shape[0]
        if logical_node and weight_shape is not None and len(weight_shape) == 2
        else cast(int, _attribute_value(node, "MW", 0))
    )
    mh = (
        weight_shape[1]
        if logical_node and weight_shape is not None and len(weight_shape) == 2
        else cast(int, _attribute_value(node, "MH", 0))
    )
    dimensions = [
        (mw, MVAUProblemPaths.MATRIX_WIDTH, "MW"),
        (mh, MVAUProblemPaths.MATRIX_HEIGHT, "MH"),
    ]
    if not logical_node:
        dimensions.extend(
            (
                (
                    cast(int, _attribute_value(node, "PE", 0)),
                    _LEGACY_COMPUTE_SELECTION_PATH,
                    "PE",
                ),
                (
                    cast(int, _attribute_value(node, "SIMD", 0)),
                    _LEGACY_COMPUTE_SELECTION_PATH,
                    "SIMD",
                ),
            )
        )
    for value, path, label in dimensions:
        if type(value) is not int or value <= 0:
            findings.append(
                _finding(
                    FindingKind.BLOCKER,
                    "mvau-source-positive-dimension-missing",
                    path,
                    f"legacy {label} must be a positive integer",
                    value=value,
                )
            )
    if activation_shape is not None and (not activation_shape or activation_shape[-1] != mw):
        findings.append(
            _finding(
                FindingKind.REJECTION,
                "mvau-source-activation-dimension-inconsistent",
                MVAUProblemPaths.MATRIX_WIDTH,
                "activation trailing dimension must equal MW",
                actual=activation_shape[-1] if activation_shape else 0,
                expected=mw,
            )
        )
    if weight_shape is not None and weight_shape != (mw, mh):
        findings.append(
            _finding(
                FindingKind.REJECTION,
                "mvau-source-weight-dimensions-inconsistent",
                MVAUProblemPaths.MATRIX_HEIGHT,
                "weight shape must equal (MW, MH)",
                actual=weight_shape,
                expected=(mw, mh),
            )
        )
    leading_shape = activation_shape[:-1] if activation_shape else ()
    if output_shape is not None and output_shape != (*leading_shape, mh):
        findings.append(
            _finding(
                FindingKind.REJECTION,
                "mvau-source-output-dimensions-inconsistent",
                MVAUProblemPaths.MATRIX_HEIGHT,
                "output shape must preserve activation leading dimensions and end in MH",
                actual=output_shape,
                expected=(*leading_shape, mh),
            )
        )

    threshold_shape: tuple[int, ...] | None = None
    threshold_type: NumericElementType | None = None
    threshold_initialized: bool | None = None
    threshold_initializer: object | None = None
    if threshold_id is not None:
        threshold_shape = _tensor_shape(
            model,
            threshold_id,
            MVAUProblemPaths.THRESHOLD_ELEMENT_TYPE,
            "threshold",
            findings,
        )
        threshold_type = _tensor_type(
            model,
            threshold_id,
            MVAUProblemPaths.THRESHOLD_ELEMENT_TYPE,
            "threshold",
            findings,
        )
        threshold_initializer = model.get_initializer(threshold_id)
        threshold_initialized = threshold_initializer is not None
        if threshold_shape is not None and (not threshold_shape or threshold_shape[0] != mh):
            findings.append(
                _finding(
                    FindingKind.REJECTION,
                    "mvau-source-threshold-dimensions-inconsistent",
                    MVAUProblemPaths.THRESHOLD_ELEMENT_TYPE,
                    "threshold leading dimension must equal MH",
                    actual=threshold_shape,
                    expected_first=mh,
                )
            )
        if not threshold_initialized:
            findings.append(
                _finding(
                    FindingKind.BLOCKER,
                    "mvau-source-threshold-initializer-missing",
                    MVAUProblemPaths.THRESHOLD_INITIALIZER_AVAILABLE,
                    "fused-threshold MVAU requires a threshold initializer",
                    tensor=threshold_id,
                )
            )

    mem_mode = (
        "logical"
        if logical_node
        else cast(str, _attribute_value(node, "mem_mode", "internal_decoupled"))
    )
    if not logical_node and mem_mode not in {"internal_embedded", "internal_decoupled", "external"}:
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-delivery-mode-unsupported",
                _LEGACY_PARAMETER_TOPOLOGY_PATH,
                "the current source adapter does not declare this delivery family",
                mem_mode=mem_mode,
            )
        )
    # Runtime writability is a build fact; the graph projection reads the
    # legacy attribute only in ``project_mvau_source``, where build context is
    # available to override it.
    weight_initialized = model.get_initializer(weight_id) is not None
    if not logical_node and mem_mode == "internal_embedded" and not weight_initialized:
        findings.append(
            _finding(
                FindingKind.BLOCKER,
                "mvau-source-weight-initializer-missing",
                MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE,
                "embedded MVAU weights require an initializer",
                tensor=weight_id,
            )
        )
    interleave = 1 if logical_node else cast(int, _attribute_value(node, "TH", 1))
    if type(interleave) is not int or interleave <= 0:
        findings.append(
            _finding(
                FindingKind.BLOCKER,
                "mvau-source-interleave-invalid",
                _LEGACY_COMPUTE_SELECTION_PATH,
                "legacy TH must be a positive integer",
                value=interleave,
            )
        )
    if (
        not logical_node
        and interleave > 1
        and (node.op_type != "MVAU_rtl" or mem_mode == "internal_embedded")
    ):
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-interleave-combination-unsupported",
                _LEGACY_COMPUTE_SELECTION_PATH,
                "TH > 1 requires an RTL streamed MVAU source combination",
                mem_mode=mem_mode,
                op_type=node.op_type,
            )
        )
    if not logical_node and not no_activation and node.op_type == "MVAU_rtl":
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-fused-rtl-unsupported",
                _LEGACY_COMPUTE_SELECTION_PATH,
                "legacy RTL MVAU does not implement the fused-threshold computation profile",
            )
        )

    computation = (
        MVAUComputationProfile.FUSED_THRESHOLD
        if not no_activation
        else MVAUComputationProfile.BIPOLAR_XNOR_ACCUMULATOR
        if bool(source_attribute("binaryXnorMode", 0))
        else MVAUComputationProfile.ACCUMULATOR_INTEGER
    )
    provenance = cast(str, source_attribute("dataflow_source_nodes", ""))
    description = MVAUSourceDescription(
        source_scope_id or source_node_id,
        activation_id,
        weight_id,
        output_id,
        leading_shape,
        threshold_id,
        tuple(item for item in provenance.split(",") if item),
        threshold_shape,
    )
    problem: dict[QualifiedPath, object] = {
        MVAUDataflowOpPaths.SOURCE_DESCRIPTION: description,
        MVAUProblemPaths.REPETITIONS: prod(leading_shape),
        MVAUProblemPaths.MATRIX_WIDTH: mw,
        MVAUProblemPaths.MATRIX_HEIGHT: mh,
        MVAUProblemPaths.COMPUTATION_PROFILE: computation,
        MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE: weight_initialized,
    }
    weight_initializer = model.get_initializer(weight_id)
    if weight_initializer is not None:
        problem[MVAUDataflowOpPaths.WEIGHT_INITIALIZER_FINGERPRINT] = tensor_value_fingerprint(
            weight_initializer
        )
    if threshold_initializer is not None:
        problem[MVAUDataflowOpPaths.THRESHOLD_INITIALIZER_FINGERPRINT] = tensor_value_fingerprint(
            threshold_initializer
        )
    if activation_type is not None:
        problem[MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE] = activation_type
    if weight_type is not None:
        problem[MVAUProblemPaths.WEIGHT_ELEMENT_TYPE] = weight_type
    if accumulator_type is not None:
        problem[MVAUProblemPaths.ACCUMULATOR_ELEMENT_TYPE] = accumulator_type
    if output_type is not None:
        problem[MVAUProblemPaths.OUTPUT_ELEMENT_TYPE] = output_type
    if threshold_type is not None:
        problem[MVAUProblemPaths.THRESHOLD_ELEMENT_TYPE] = threshold_type
    if threshold_initialized is not None:
        problem[MVAUProblemPaths.THRESHOLD_INITIALIZER_AVAILABLE] = threshold_initialized
    if activation_type is not None and weight_type is not None:
        excludes_minimum = _initializer_excludes_minimum(model, weight_id, weight_type, mem_mode)
        if excludes_minimum is not None:
            problem[MVAUProblemPaths.INITIALIZER_EXCLUDES_MINIMUM] = excludes_minimum
    return MVAUSourceProjection(description, problem, {}, tuple(findings))


def project_mvau_source(
    model: MVAUModelAccessor,
    source_node_id: str,
    context: MVAUProjectionContext,
    *,
    source_scope_id: str | None = None,
) -> MVAUSourceProjection:
    """Compose graph-owned and build-owned facts for one MVAU problem."""

    graph_projection = project_mvau_graph_source(
        model,
        source_node_id,
        source_scope_id=source_scope_id,
    )
    node = _find_source_node(model, source_node_id)
    if node is None or graph_projection.source_description is None:
        return graph_projection
    logical_node = node.op_type == "MvauDataflowOp" and node.domain == "finn.custom_op.dataflow"
    findings = list(graph_projection.findings)
    runtime_writable = (
        False
        if logical_node and context.runtime_writable_weights is None
        else bool(_attribute_value(node, "runtime_writeable_weights", 0))
        if context.runtime_writable_weights is None
        else context.runtime_writable_weights
    )
    weight_initialized = model.get_initializer(node.input[1]) is not None
    mem_mode = (
        "logical"
        if logical_node
        else cast(str, _attribute_value(node, "mem_mode", "internal_decoupled"))
    )
    if (
        not logical_node
        and mem_mode == "internal_decoupled"
        and not (weight_initialized or runtime_writable)
    ):
        findings.append(
            _finding(
                FindingKind.BLOCKER,
                "mvau-source-cyclic-state-unavailable",
                MVAUProblemPaths.WEIGHT_INITIALIZER_AVAILABLE,
                "cyclic delivery requires initialized or runtime-writable weights",
                tensor=node.input[1],
            )
        )

    target = _dsp_block(context.fpga_part) if context.fpga_part is not None else None
    if context.fpga_part is not None and target is None:
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-target-part-unknown",
                MVAUProblemPaths.TARGET_DSP_BLOCK,
                "target FPGA part cannot be classified into a supported DSP family",
                fpga_part=context.fpga_part,
            )
        )
    interleave = 1 if logical_node else cast(int, _attribute_value(node, "TH", 1))
    if not logical_node and interleave > 1 and target is not None and target is not DspBlock.DSP58:
        findings.append(
            _finding(
                FindingKind.LIMITATION,
                "mvau-source-interleave-target-unsupported",
                MVAUProblemPaths.TARGET_DSP_BLOCK,
                "batch-interleaved RTL MVAU requires a DSP58 target",
                target=target.value,
            )
        )

    problem = dict(graph_projection.problem_data)
    problem[MVAUDataflowOpPaths.ACCUMULATOR_TYPE_ANALYSIS_OWNER] = (
        context.accumulator_type_analysis_owner
    )
    problem.update(
        project_mvau_build_problem(
            context,
            runtime_writable_weights=runtime_writable,
        )
    )
    activation_type = problem.get(MVAUProblemPaths.ACTIVATION_ELEMENT_TYPE)
    weight_type = problem.get(MVAUProblemPaths.WEIGHT_ELEMENT_TYPE)
    excludes_minimum: bool | None = None
    if is_qonnx_datatype(activation_type) and is_qonnx_datatype(weight_type):
        excludes_minimum = _initializer_excludes_minimum(
            model, node.input[1], weight_type, mem_mode
        )
        if excludes_minimum is None:
            problem.pop(MVAUProblemPaths.INITIALIZER_EXCLUDES_MINIMUM, None)
        else:
            problem[MVAUProblemPaths.INITIALIZER_EXCLUDES_MINIMUM] = excludes_minimum

    return MVAUSourceProjection(
        graph_projection.source_description,
        problem,
        {},
        tuple(findings),
    )


def start_mvau_projection(
    projection: MVAUSourceProjection,
    assignments: Mapping[QualifiedPath | str, object] | None = None,
) -> MVAUResolvedDesign:
    """Start, commit, and resolve one projected MVAU design."""
    if projection.blocking_findings:
        raise MVAUSourceAdapterError(projection.blocking_findings)
    commitments: dict[QualifiedPath | str, object] = {
        path: value for path, value in projection.imported_assignments.items()
    }
    if assignments is not None:
        commitments.update(assignments)
    engine = Engine()
    space = engine.validate(MVAU_DATAFLOW_OP_SPEC)
    point = engine.start(space, projection.problem_data)
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
                        "mvau-assignment-commit-failed",
                        _ADAPTER_PATH,
                        "one or more projected assignments could not be committed",
                    ),
                )
            )
        point = committed.point
    return resolve_mvau_point(engine, point, projection)


def resolve_mvau_point(
    engine: Engine,
    point: DesignPoint,
    projection: MVAUSourceProjection,
    *,
    source_scope_id: str | None = None,
) -> MVAUResolvedDesign:
    """Resolve the MVAU result and source association for one hydrated point."""

    result = engine.query_property(point, MVAUDataflowOpPaths.RESULT)
    if not isinstance(result, Decided):
        raise MVAUSourceAdapterError(result.findings)
    selected = cast(NetworkRef, result.value)
    association = selected.source_association
    return MVAUResolvedDesign(
        engine,
        point,
        selected,
        association,
        source_scope_id or association.source_node_id,
        projection,
    )


def _canonical_problem_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if is_qonnx_datatype(value):
        return encode_datatype(value)
    if isinstance(value, MVAUSourceDescription):
        return {
            "source_description": {
                "activation_operand_id": value.activation_operand_id,
                "fused_source_node_ids": list(value.fused_source_node_ids),
                "leading_shape": list(value.leading_shape),
                "output_operand_id": value.output_operand_id,
                "source_node_id": value.source_node_id,
                "threshold_operand_id": value.threshold_operand_id,
                "threshold_shape": (
                    None if value.threshold_shape is None else list(value.threshold_shape)
                ),
                "weight_operand_id": value.weight_operand_id,
            }
        }
    if isinstance(value, CyclicTargetMemoryCapabilities):
        return {"supports_initialized_uram": value.supports_initialized_uram}
    if isinstance(value, BeatSequence):
        return {
            "beat_sequence": {
                "beats": [[list(position) for position in beat] for beat in value.beats],
                "elements_per_beat": value.elements_per_beat,
            }
        }
    raise TypeError(f"unsupported MVAU problem fingerprint value: {type(value).__name__}")


def mvau_problem_fingerprint(problem: Mapping[QualifiedPath, object]) -> str:
    """Return a deterministic digest of all projected problem facts."""
    payload = [
        [str(path), _canonical_problem_value(value)]
        for path, value in sorted(problem.items(), key=lambda item: item[0])
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(serialized).hexdigest()


def _encode_assignment(path: QualifiedPath, value: object) -> object:
    local = local_mvau_assignment_path(path)
    codec = None if local is None else MVAU_DECISION_NODEATTRS.get(local)
    if codec is not None:
        try:
            return codec.encode_json(value)
        except (KeyError, TypeError, ValueError):
            pass
    raise MVAUSourceAdapterError(
        (
            _finding(
                FindingKind.LIMITATION,
                "mvau-assignment-codec-missing",
                path,
                "no persistence codec is declared for this decision value",
            ),
        )
    )


def _decode_assignment(path: QualifiedPath, value: object) -> object:
    local = local_mvau_assignment_path(path)
    codec = None if local is None else MVAU_DECISION_NODEATTRS.get(local)
    if codec is not None:
        try:
            return codec.decode_json(value)
        except (TypeError, ValueError):
            pass
    raise MVAUSourceAdapterError(
        (
            _finding(
                FindingKind.REJECTION,
                "mvau-saved-assignment-unknown-or-obsolete",
                path,
                "saved assignment path or value is unknown, obsolete, or malformed",
            ),
        )
    )


def _metadata_key(source_node_id: str) -> str:
    return f"finn.dataflow.mvau.selection:{source_node_id}"


def make_mvau_selection_envelope(
    source_scope_id: str,
    point: DesignPoint,
) -> MVAUSelectionEnvelope:
    """Encode sparse MVAU choices, including qualified placed-Kernel paths."""
    assignments = tuple(
        (str(path), _encode_assignment(path, value))
        for path, value in sorted(point.assignments.items(), key=lambda item: item[0])
    )
    return MVAUSelectionEnvelope(
        _ADAPTER_KEY,
        _FORMAT_VERSION,
        MVAU_DECLARATION_FAMILY_VERSION,
        source_scope_id,
        mvau_problem_fingerprint(point.problem),
        assignments,
    )


def save_mvau_selection(
    model: MVAUModelAccessor,
    source_node_id: str,
    point: DesignPoint,
) -> MVAUSelectionEnvelope:
    """Store one resolved sparse assignment map on the graph."""
    result = Engine().query_property(point, MVAUDataflowOpPaths.RESULT)
    if not isinstance(result, Decided):
        raise MVAUSourceAdapterError(result.findings)
    description = point.problem.get(MVAUDataflowOpPaths.SOURCE_DESCRIPTION)
    if (
        not isinstance(description, MVAUSourceDescription)
        or description.source_node_id != source_node_id
    ):
        raise MVAUSourceAdapterError(
            (
                _finding(
                    FindingKind.REJECTION,
                    "mvau-selection-source-scope-mismatch",
                    _PERSISTENCE_PATH,
                    "design point source scope does not match the requested graph record",
                ),
            )
        )
    envelope = make_mvau_selection_envelope(source_node_id, point)
    model.set_metadata_prop(_metadata_key(source_node_id), envelope.to_json())
    return envelope


def parse_mvau_selection_envelope(raw: str) -> MVAUSelectionEnvelope:
    try:
        payload = json.loads(raw)
        required = {
            "adapter_key",
            "assignments",
            "declaration_family_version",
            "format_version",
            "problem_fingerprint",
            "source_scope_id",
        }
        if type(payload) is not dict or set(payload) != required:
            raise ValueError("envelope fields are not exact")
        raw_assignments = payload["assignments"]
        if not isinstance(raw_assignments, list):
            raise ValueError("assignments must be a list")
        assignments = tuple(
            (cast(str, item["path"]), item["value"])
            for item in raw_assignments
            if isinstance(item, dict) and set(item) == {"path", "value"}
        )
        if len(assignments) != len(raw_assignments) or any(
            not isinstance(path, str) for path, _value in assignments
        ):
            raise ValueError("assignment entries are malformed")
        return MVAUSelectionEnvelope(
            cast(str, payload["adapter_key"]),
            cast(int, payload["format_version"]),
            cast(str, payload["declaration_family_version"]),
            cast(str, payload["source_scope_id"]),
            cast(str, payload["problem_fingerprint"]),
            assignments,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MVAUSourceAdapterError(
            (
                _finding(
                    FindingKind.REJECTION,
                    "mvau-selection-envelope-malformed",
                    _PERSISTENCE_PATH,
                    "stored MVAU selection envelope is malformed",
                ),
            )
        ) from exc


def reconstitute_mvau_point(
    engine: Engine,
    design_space: DesignSpace,
    problem: Mapping[QualifiedPath, object],
    envelope: MVAUSelectionEnvelope,
    *,
    source_scope_id: str,
) -> DesignPoint:
    """Replay an envelope against an explicitly rebuilt MVAU-rooted design space."""
    if envelope.declaration_family_version != MVAU_DECLARATION_FAMILY_VERSION:
        raise MVAUSourceAdapterError(
            (
                _finding(
                    FindingKind.REJECTION,
                    "mvau-selection-family-version-incompatible",
                    _PERSISTENCE_PATH,
                    "stored selection uses an unsupported MVAU declaration family version",
                    actual_version=envelope.declaration_family_version,
                    expected_version=MVAU_DECLARATION_FAMILY_VERSION,
                ),
            )
        )
    expected_header = (
        _ADAPTER_KEY,
        _FORMAT_VERSION,
        source_scope_id,
    )
    actual_header = (
        envelope.adapter_key,
        envelope.format_version,
        envelope.source_scope_id,
    )
    if actual_header != expected_header:
        raise MVAUSourceAdapterError(
            (
                _finding(
                    FindingKind.REJECTION,
                    "mvau-selection-envelope-incompatible",
                    _PERSISTENCE_PATH,
                    "stored selection targets another adapter, format, family, or source scope",
                ),
            )
        )
    if mvau_problem_fingerprint(problem) != envelope.problem_fingerprint:
        raise MVAUSourceAdapterError(
            (
                _finding(
                    FindingKind.REJECTION,
                    "mvau-selection-problem-mismatch",
                    _PERSISTENCE_PATH,
                    "saved choices do not belong to the current source and target problem",
                ),
            )
        )
    decoded: dict[QualifiedPath | str, object] = {}
    for raw_path, value in envelope.assignments:
        try:
            path = QualifiedPath(raw_path)
        except (TypeError, ValueError) as exc:
            raise MVAUSourceAdapterError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "mvau-saved-assignment-path-malformed",
                        _PERSISTENCE_PATH,
                        "saved assignment contains a malformed path",
                    ),
                )
            ) from exc
        if path in decoded:
            raise MVAUSourceAdapterError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "mvau-saved-assignment-duplicate",
                        path,
                        "saved assignment path occurs more than once",
                    ),
                )
            )
        if path not in design_space.decisions:
            raise MVAUSourceAdapterError(
                (
                    _finding(
                        FindingKind.REJECTION,
                        "mvau-saved-assignment-unknown-or-obsolete",
                        path,
                        "saved assignment path is not declared in the rebuilt design space",
                    ),
                )
            )
        decoded[path] = _decode_assignment(path, value)
    point = engine.start(design_space, problem)
    committed = engine.commit_assignments(point, decoded)
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
                    "mvau-saved-assignment-rejected",
                    _PERSISTENCE_PATH,
                    "one or more saved assignments are incompatible with the rebuilt problem",
                ),
            )
        )
    return committed.point


def reconstitute_mvau_selection(
    model: MVAUModelAccessor,
    source_node_id: str,
    context: MVAUProjectionContext,
) -> MVAUResolvedDesign:
    """Re-project current facts and replay saved choices through the engine."""
    raw = model.get_metadata_prop(_metadata_key(source_node_id))
    if raw is None:
        raise MVAUSourceAdapterError(
            (
                _finding(
                    FindingKind.BLOCKER,
                    "mvau-selection-envelope-missing",
                    _PERSISTENCE_PATH,
                    "graph has no saved selection for the requested source scope",
                ),
            )
        )
    envelope = parse_mvau_selection_envelope(raw)
    projection = project_mvau_source(model, source_node_id, context)
    if projection.blocking_findings:
        raise MVAUSourceAdapterError(projection.blocking_findings)
    engine = Engine()
    space = engine.validate(MVAU_DATAFLOW_OP_SPEC)
    point = reconstitute_mvau_point(
        engine,
        space,
        projection.problem_data,
        envelope,
        source_scope_id=source_node_id,
    )
    return resolve_mvau_point(engine, point, projection)


__all__ = [
    "MVAU_DECLARATION_FAMILY_VERSION",
    "MVAU_LOGICAL_SOURCE_NODEATTRS",
    "MVAU_SOURCE_MAPPING",
    "MVAUModelAccessor",
    "MVAUProjectionContext",
    "MVAUResolvedDesign",
    "MVAUSelectionEnvelope",
    "MVAUSourceAdapterError",
    "MVAUSourceMappingEntry",
    "MVAUSourceProjection",
    "classify_mvau_dsp_block",
    "mvau_problem_fingerprint",
    "make_mvau_selection_envelope",
    "parse_mvau_selection_envelope",
    "project_mvau_graph_source",
    "project_mvau_source",
    "project_mvau_build_problem",
    "reconstitute_mvau_selection",
    "reconstitute_mvau_point",
    "resolve_mvau_point",
    "save_mvau_selection",
    "start_mvau_projection",
    "tensor_value_fingerprint",
]
