# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""QONNX datatype value semantics and codec, independent of dataflow models."""

from finn.dataflow._engine import ValueSemantics, as_object_semantics
from finn.dataflow.model.logical.datatypes import (
    QONNX_DATATYPE_TOKEN,
    QONNXDataType,
    canonical_qonnx_datatype,
    encode_datatype,
    is_qonnx_datatype,
)
from finn.dataflow.space.declarations import CanonicalValue, CanonicalValueCodec

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

__all__ = ["QONNX_DATATYPE_CODEC", "QONNX_DATATYPE_SEMANTICS", "QONNX_DATATYPE_VALUE_SEMANTICS"]
