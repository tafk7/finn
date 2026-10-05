# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The QONNX-dataflow datatype boundary: one datatype identity, QONNX's own.

There is no intermediate representation. A local ``(type_id, bit_width)`` pair
would be a second authority for one fact, and a lossy one: ``TERNARY`` and
``INT2`` reduce to the same pair despite different value domains, and every
sixteen-bit floating format reduces to ``("float", 16)``.

What crosses it:

- **Datatypes** come from the model's QONNX annotations
  (``ModelWrapper.get_tensor_datatype``), read by the KernelOps
  (``finn.custom_op.kernels``) and passed to kernels as facts. They travel as the same
  QONNX datatype *value*; a kernel's result type goes back as an annotation.
- **Value information** is not a datatype. An initializer's values are
  admitted against its annotation from QONNX's value summary at the graph
  boundary; the range of values a stream carries lives on its element
  (``finn.dataflow.tensor.ScalarEncoding``), stated by the value owner and
  derived again on every build. Kernels read neither annotations nor
  summaries: they never read ONNX.

A datatype is a QONNX *value*: one immutable instance per canonical name
(qonnx's ``is_datatype`` recognizes exactly those), so what a tensor's element
holds is the caller's own instance and nothing can rename it in place. A
``BaseDataType`` subclass defined elsewhere is not one of those values and is
refused.

This module has **no FINN imports at all**, so ``finn.dataflow`` builds on it
without dragging the engine into its datatypes (the layer table,
``tests/layering.py``, holds the package directions). The engine-side ``ValueSemantics`` declaration
lives in ``finn.kernels.datatypes.semantics``, built from the helpers here.

Two rules it keeps, which qonnx leaves to its callers:

- **A datatype compares and hashes equal to its own canonical name.**
  ``DataType["INT8"] == "INT8"`` is true in both directions and the hashes
  agree, so a string and a datatype collide as mapping keys.  ``str`` is
  therefore not a member of this value domain, and the recognizer says so.
- **A zero-width integer type describes no value.** qonnx still resolves
  ``INT0`` and ``UINT0`` (with a ``DataTypeWarning``) until a later release
  refuses them; ``ordinary_integer_bounds`` and the element policies refuse
  them here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol, TypeGuard, cast

from qonnx.core.datatype import BaseDataType, DataType, is_datatype

__all__ = [
    "DatatypeError",
    "QONNXDataType",
    "QONNX_DATATYPE_TOKEN",
    "canonical_qonnx_datatype",
    "is_ordinary_integer",
    "is_qonnx_datatype",
    "ordinary_integer_bounds",
    "qonnx_datatype_width",
    "resolve_qonnx_datatype_name",
]


class QONNXDataType(Protocol):
    """The datatype surface kernels and dataflow models depend on.

    A typing aid only.  QONNX ships no ``py.typed``, and the gate deliberately
    hides it from mypy (see ``scripts/check-kernels.sh``), so annotating
    against ``BaseDataType`` would annotate against ``Any``.  This names what is
    actually used instead.

    It is **not** a second value representation, a registry, or an engine value
    token: runtime values are real ``BaseDataType`` instances and the engine
    token is ``BaseDataType`` itself.  Adding a method here is a claim that the
    dataflow stack needs it from every datatype -- and several QONNX methods
    are not total (see ``min`` below).
    """

    @property
    def name(self) -> str: ...

    def bitwidth(self) -> int: ...

    def signed(self) -> bool: ...

    def is_integer(self) -> bool: ...

    def is_fixed_point(self) -> bool: ...

    # PARTIAL.  Unlike everything above it, this is not defined for every QONNX
    # datatype: ``ScaledIntType`` raises, and wide types may answer in floating
    # point.  It is declared because a datatype's representable range is part
    # of the surface; an ordinary integer's exact bounds come from
    # ``ordinary_integer_bounds`` instead, which is what an element's range is
    # checked against.  Every caller must guard partial uses. Do not add further partial methods
    # without the same treatment; ``get_hls_datatype_str`` in particular raises
    # a bare ``AssertionError`` for arbitrary float formats, which no ``except``
    # clause should be catching.
    def min(self) -> int | float: ...

    def max(self) -> int | float: ...


class DatatypeError(ValueError):
    """A value was offered as a datatype and is not one this stack can hold."""


def is_ordinary_integer(value: QONNXDataType) -> bool:
    """Whether ``value`` is an ordinary INT/UINT encoding: ``INT<n>``, ``UINT<n>``, or
    ``BINARY`` (QONNX's canonical name for ``UINT1``)."""
    return value.name == "BINARY" or re.fullmatch(r"U?INT\d+", value.name) is not None


def ordinary_integer_bounds(value: QONNXDataType) -> tuple[int, int]:
    """Exact INT/UINT bounds; special integer-valued encodings are distinct.

    Compute from the ordinary encoding instead of third-party range methods,
    which may use floating point for very wide types. No datatype is converted.
    """
    if not is_ordinary_integer(value):
        raise DatatypeError(f"{value.name} is not an ordinary INT/UINT encoding")
    bits = qonnx_datatype_width(value)
    if bits < 1:
        raise DatatypeError("ordinary integer width must be positive")
    if value.name.startswith("INT"):
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def resolve_qonnx_datatype_name(name: str) -> QONNXDataType:
    """Resolve a datatype name **permissively**, as QONNX itself would.

    Accepts any spelling QONNX accepts, including the ones it normalizes into
    another name -- ``UINT1`` resolves to ``BINARY``, and ``FLOAT<5,10>`` to
    ``FLOAT<5,10,15>``. This is right for *source-facing* names: a legacy node
    attribute like ``accDataType`` is whatever a human or an older FINN wrote.
    """

    if not isinstance(name, str):
        raise DatatypeError(f"a datatype name must be a string, not {type(name)}")
    try:
        return cast(QONNXDataType, DataType[name])
    except KeyError as error:
        raise DatatypeError(f"no QONNX datatype is named {name!r}") from error


def canonical_qonnx_datatype(value: object) -> QONNXDataType:
    """``value`` itself, once recognized as a QONNX datatype value.

    Raises ``DatatypeError`` rather than returning ``None``: a caller that
    wants a predicate should use ``is_qonnx_datatype``, and a caller that
    reached here has already decided the value ought to be a datatype.
    """

    if isinstance(value, str):
        # Not merely unsupported -- actively dangerous.  A datatype compares and
        # hashes equal to its own canonical name, so admitting strings here
        # would make a name and the live value interchangeable and
        # let them collide as mapping keys without raising.
        raise DatatypeError(f"a canonical name is not a datatype value; resolve {value!r} first")
    if not is_datatype(value):
        raise DatatypeError(f"not a QONNX datatype value: a {type(value).__name__}")
    return cast(QONNXDataType, value)


def qonnx_datatype_width(value: object) -> int:
    """The bit width of a recognized datatype value."""

    return canonical_qonnx_datatype(value).bitwidth()


is_qonnx_datatype: Callable[[object], TypeGuard[QONNXDataType]] = is_datatype
"""Whether ``value`` is a QONNX datatype value: qonnx's ``is_datatype``, total
(it runs no method of the value), and False for a ``str``."""


#: The one engine value domain for datatypes.
#:
#: ``type_token`` is ``BaseDataType`` and must stay that single object:
#: ``ValueSemantics.is_compatible_with`` compares tokens by identity, not by
#: subtyping, so a second token -- the protocol above, say -- would silently
#: partition the domain and make two datatype fields report that they cannot be
#: compared.
#: The single object every datatype value-semantics declaration must use as its
#: ``type_token``.  Exported so that the declaration -- which lives in
#: ``finn.kernels.datatypes.semantics`` to keep this module engine-independent --
#: names the same object this module recognizes against.
QONNX_DATATYPE_TOKEN = BaseDataType
