# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The canonical dataflow model: what a Region and a Network *are*.

Detached, immutable values and pure operations over them.  A ``DataflowRegion``
here is a finished statement about one logical schedule, its operands and its
boundary; a ``DataflowNetwork`` is a finished statement about how several of them
are wired.  Neither knows how it was authored, which design point produced it, or
what will be built from it.

```text
model/datatypes           QONNX datatype identity at the model boundary
model/region              operands, schedules, requirements, ports, inputs
model/region_profiles     construction profiles contributors build Regions from
model/region_validation   Region semantic validation
model/network             nodes, edges, boundaries, position maps
model/network_validation  Network semantic validation
model/refs                qualified RegionInputRef / RegionOutputRef resolution
model/presentation        exposure and edge/boundary/unpresented queries
```

The package imports the Python standard library, QONNX datatype identity through
``model.datatypes``, and its own siblings.  It imports no Space, no ``_engine``,
no Kernel, Design, operation, ONNX wrapper, physical value or artifact
capability -- ``test_package_boundaries`` enforces the direction, and
``space.dataflow_value_semantics`` is where the engine learns about these values,
one way.

Not to be confused with ``finn.dataflow.space``, which owns the generic
declaration language that *produces* these values.  ``RegionDeclaration`` is a
Kernel's recipe; ``DataflowRegion`` is what the recipe yields.
"""

from finn.dataflow.model.datatypes import (
    DATATYPE_PAYLOAD_KEY,
    QONNX_DATATYPE_TOKEN,
    DatatypeError,
    QONNXDataType,
    canonical_qonnx_datatype,
    decode_datatype,
    encode_datatype,
    is_qonnx_datatype,
    qonnx_datatype_width,
    resolve_qonnx_datatype_name,
)
from finn.dataflow.model.region import (
    BeatSequence,
    BeatType,
    DataflowRegion,
    InputInterface,
    InternalInput,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    RegionInput,
    RegionRefused,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    element_width,
    is_element_type,
)
from finn.dataflow.model.region_profiles import (
    CanonicalExtentProfile,
    ProfileCertificationError,
    ProfileCertificationIssue,
    direct_output_availability,
    explicit_beat_sequence,
    lexicographic_occurrence_to_field,
)
from finn.dataflow.model.region_validation import (
    RegionValidationIssue,
    RegionValidationReport,
    is_structurally_well_formed,
    validate_region,
)
from finn.dataflow.model.network import (
    BoundaryContract,
    ChannelSpec,
    DataflowNetwork,
    DirectConnection,
    Edge,
    EdgeTransport,
    FanoutMode,
    NetworkNode,
    OrderedChannel,
    PassCorrespondence,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.network_validation import (
    NetworkValidationIssue,
    NetworkValidationReport,
    is_network_structurally_well_formed,
    validate_network,
)
from finn.dataflow.model.refs import (
    DataflowOperandRef,
    NetworkOperandError,
    RegionInputRef,
    RegionOutputRef,
    resolve_input,
    resolve_output,
)
from finn.dataflow.model.presentation import (
    boundary_presented_positions,
    edge_presented_positions,
    exposing_boundaries,
    exposing_ports,
    unpresented_positions,
)

__all__ = [
    # the datatype boundary
    "DATATYPE_PAYLOAD_KEY",
    "QONNX_DATATYPE_TOKEN",
    "DatatypeError",
    "QONNXDataType",
    "canonical_qonnx_datatype",
    "decode_datatype",
    "encode_datatype",
    "is_qonnx_datatype",
    "qonnx_datatype_width",
    "resolve_qonnx_datatype_name",
    # the Region model
    "BeatSequence",
    "BeatType",
    "DataflowRegion",
    "InputInterface",
    "InternalInput",
    "LogicalSchedule",
    "NumericElementType",
    "Operand",
    "OutputInterface",
    "Port",
    "RegionInput",
    "RegionRefused",
    "ScheduleLevel",
    "ScheduledInputRequirements",
    "ScheduledOutputAvailability",
    "element_width",
    "is_element_type",
    # construction profiles
    "CanonicalExtentProfile",
    "ProfileCertificationError",
    "ProfileCertificationIssue",
    "direct_output_availability",
    "explicit_beat_sequence",
    "lexicographic_occurrence_to_field",
    # Region validation
    "RegionValidationIssue",
    "RegionValidationReport",
    "is_structurally_well_formed",
    "validate_region",
    # the Network model
    "BoundaryContract",
    "ChannelSpec",
    "DataflowNetwork",
    "DirectConnection",
    "Edge",
    "EdgeTransport",
    "FanoutMode",
    "NetworkNode",
    "OrderedChannel",
    "PassCorrespondence",
    "PositionMap",
    "RegionEndpoint",
    "SinkContract",
    # Network validation
    "NetworkValidationIssue",
    "NetworkValidationReport",
    "is_network_structurally_well_formed",
    "validate_network",
    # qualified references
    "DataflowOperandRef",
    "NetworkOperandError",
    "RegionInputRef",
    "RegionOutputRef",
    "resolve_input",
    "resolve_output",
    # exposure and presentation
    "boundary_presented_positions",
    "edge_presented_positions",
    "exposing_boundaries",
    "exposing_ports",
    "unpresented_positions",
]
