# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Space value semantics for the canonical logical values.

Regions, Networks, position maps, validation reports and logical results are
immutable values, so each is declared with the nominal ``finn.core.space``
policy: exact type recognition, structural equality and identity snapshots.
The one exception is ``LogicalResult``, a capability over two result classes,
which therefore carries its own token.

Scalar datatype semantics are not declared here; ``finn.kernels.datatypes``
owns them.
"""

from finn.core.space import ValueSemantics
from finn.parked.dataflow.logical_values.network import DataflowNetwork, PositionMap
from finn.parked.dataflow.logical_values.network_validation import NetworkValidationReport
from finn.parked.dataflow.logical_values.region import DataflowRegion
from finn.parked.dataflow.logical_values.region_validation import RegionValidationReport
from finn.parked.dataflow.logical_values.results import LogicalResult, NetworkResult, RegionResult

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
POSITION_MAP_SEMANTICS = ValueSemantics.immutable_nominal(
    PositionMap,
    name="PositionMap",
)
NETWORK_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    NetworkValidationReport,
    name="NetworkValidationReport",
)


class _LogicalResultToken:
    """Identity token for the Region-or-Network result capability."""


def _is_logical_result(value: object) -> bool:
    return isinstance(value, (RegionResult, NetworkResult))


def _logical_results_equal(left: LogicalResult, right: LogicalResult) -> bool:
    return bool(left == right)


def _logical_result_snapshot(value: LogicalResult) -> LogicalResult:
    return value


DATAFLOW_LOGICAL_RESULT_SEMANTICS: ValueSemantics[LogicalResult] = ValueSemantics(
    type_token=_LogicalResultToken,
    name="LogicalResult",
    recognizes=_is_logical_result,
    equal=_logical_results_equal,
    snapshot=_logical_result_snapshot,
)

__all__ = [
    "DATAFLOW_LOGICAL_RESULT_SEMANTICS",
    "DATAFLOW_NETWORK_SEMANTICS",
    "DATAFLOW_REGION_SEMANTICS",
    "NETWORK_VALIDATION_REPORT_SEMANTICS",
    "POSITION_MAP_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
]
