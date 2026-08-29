# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Design-engine value semantics for model-owned dataflow region values."""

from finn.dataflow._engine import ValueSemantics, as_object_semantics
from finn.dataflow.datatypes import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    canonical_qonnx_datatype,
    is_qonnx_datatype,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import NetworkValidationReport
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import RegionValidationReport

#: The one engine value domain for datatypes.
#:
#: Declared here rather than beside the datatype helpers because
#: ``finn.dataflow.region`` depends on those helpers, and the model layer must
#: stay importable without the engine -- a boundary
#: ``test_api_and_boundaries`` enforces.  Same reason
#: ``DATAFLOW_REGION_SEMANTICS`` lives here and not in ``region.py``.
#:
#: ``type_token`` must remain this single object: ``is_compatible_with``
#: compares tokens by identity, not by subtyping, so a second token -- deriving
#: one from the ``QONNXDataType`` protocol, say -- would silently partition the
#: domain and make two datatype fields report that they cannot be compared.
QONNX_DATATYPE_VALUE_SEMANTICS: ValueSemantics[QONNXDataType] = ValueSemantics(
    type_token=QONNX_DATATYPE_TOKEN,
    name="QONNXDataType",
    recognizes=is_qonnx_datatype,
    equal=lambda left, right: bool(left == right),
    snapshot=canonical_qonnx_datatype,
)

#: The same declaration widened for the engine, which stores values as
#: ``object``.  Two names for one object, never two objects.
QONNX_DATATYPE_SEMANTICS: ValueSemantics[object] = as_object_semantics(
    QONNX_DATATYPE_VALUE_SEMANTICS
)

DATAFLOW_REGION_SEMANTICS = ValueSemantics.immutable_nominal(
    DataflowRegion,
    name="DataflowRegion",
)
REGION_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    RegionValidationReport,
    name="RegionValidationReport",
)
DATAFLOW_NETWORK_SEMANTICS = ValueSemantics.immutable_nominal(
    DataflowNetwork,
    name="DataflowNetwork",
)
NETWORK_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    NetworkValidationReport,
    name="NetworkValidationReport",
)

__all__ = [
    "DATAFLOW_REGION_SEMANTICS",
    "DATAFLOW_NETWORK_SEMANTICS",
    "NETWORK_VALIDATION_REPORT_SEMANTICS",
    "QONNX_DATATYPE_SEMANTICS",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
]
