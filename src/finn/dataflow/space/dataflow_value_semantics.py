# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The bridge: engine value semantics for the canonical dataflow values.

One ``ValueSemantics`` per domain the Space language stores in a point, in one
place.  Declared here rather than beside each value because
``finn.dataflow.model.region`` and its siblings must stay importable without the
engine -- a boundary ``test_package_boundaries`` enforces -- and because a
second token for one domain would silently partition it.

This is the one module in ``finn.dataflow.space`` that names the dataflow model,
and it is deliberately explicit about it.  The generic core -- declarations,
compiler, occurrence, branching, domains, spec algebra -- stays domain-neutral
and speaks of no Region, Network, Kernel, Design or ONNX concept.  Everything
that knows both sides is here, and ``space.__init__`` does not import it, so
neither package pulls the other in implicitly.  The dependency runs one way:
``space`` may import ``model``; ``model`` never imports ``space`` or ``_engine``.
"""

from finn.dataflow._engine import ValueSemantics, as_object_semantics
from finn.dataflow.model.datatypes import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    canonical_qonnx_datatype,
    encode_datatype,
    is_qonnx_datatype,
)
from finn.dataflow.space.declarations import CanonicalValue, CanonicalValueCodec
from finn.dataflow.model.network import DataflowNetwork, PositionMap
from finn.dataflow.model.network_validation import NetworkValidationReport
from finn.dataflow.model.region import DataflowRegion
from finn.dataflow.model.region_validation import RegionValidationReport

#: The one engine value domain for datatypes.
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


#: How a ``Problem(QONNX_DATATYPE_VALUE_SEMANTICS)`` field is fingerprinted.
#:
#: Declaration-owned, because QONNX's ``BaseDataType`` is a class this project
#: does not own and must not teach to encode itself, and because the structural
#: default would refuse it outright.  The payload is the canonical name and
#: nothing else -- see ``encode_datatype`` for why a family-and-width pair
#: would give ``TERNARY`` and ``INT2`` the same fingerprint.
def _encode_qonnx_datatype(value: object) -> CanonicalValue:
    """``encode_datatype`` widened to the canonical result type.

    ``dict[str, str]`` is a canonical value and ``dict[str, object]`` is the
    declared one, but the first is not a subtype of the second because
    ``dict`` is invariant.  Widening here rather than in ``encode_datatype``
    keeps that module's own contract exact for its other callers.
    """

    return dict(encode_datatype(value))


QONNX_DATATYPE_CODEC: CanonicalValueCodec[object] = CanonicalValueCodec(
    "finn.dataflow.qonnx_datatype", 1, _encode_qonnx_datatype
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
POSITION_MAP_SEMANTICS = ValueSemantics.immutable_nominal(
    PositionMap,
    name="PositionMap",
)
NETWORK_VALIDATION_REPORT_SEMANTICS = ValueSemantics.immutable_nominal(
    NetworkValidationReport,
    name="NetworkValidationReport",
)

__all__ = [
    "DATAFLOW_REGION_SEMANTICS",
    "QONNX_DATATYPE_CODEC",
    "DATAFLOW_NETWORK_SEMANTICS",
    "NETWORK_VALIDATION_REPORT_SEMANTICS",
    "POSITION_MAP_SEMANTICS",
    "QONNX_DATATYPE_SEMANTICS",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
]
