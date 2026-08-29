# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A structural projection that survives the datatype representation change.

The migration baseline pins ``repr`` digests.  That works while a change is
supposed to alter nothing, and stops working the moment a change alters one
field everywhere: adopting QONNX datatypes moves every digest that contains an
operand type, which is most of them, and a digest that has moved discriminates
nothing.  Re-blessing them by hand would record that the values changed, not
that they changed *only where they were entitled to*.

So this builds a second view of the same objects in which **the whole structure
is preserved and only the datatype representation is normalized**:

    schedules, requirements, availability, beat sequences, shapes, ids,
    topology, boundary contracts        -- carried through verbatim
    element types                       -- replaced by canonical QONNX names

A list of datatype names would have proven that datatypes stayed equal while
proving nothing about the schedules and topology sharing those digests, which is
the majority of what the digests were protecting and exactly what goes unguarded
while they move.  Normalizing inside the full structure keeps that protection:
a schedule or topology regression during the migration still fails here.

Recorded under the *current* representation, then held across the change.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

__all__ = ["canonical_datatype_name", "normalized"]


_LEGACY_SPELLINGS = {
    ("bipolar", 1): "BIPOLAR",
    ("binary", 1): "BINARY",
}


def canonical_datatype_name(value: Any) -> str:
    """The canonical QONNX name, whichever datatype representation this is.

    Two branches on purpose, so one function spans the migration.

    The QONNX branch is exact: ``name`` *is* ``get_canonical_name()``.  The
    legacy ``NumericElementType`` branch reconstructs a name from a reduced
    family and width, and reconstruction is precisely the lossy step this
    migration exists to remove -- ``("int", 2)`` is spelled ``INT2`` here
    because that is what the current stack means by it, even though a `TERNARY`
    source also reduces to that pair.

    That is a real limitation and it is bounded: no pinned configuration uses a
    special encoding, so the two branches agree over everything this projection
    is recorded against.  A configuration where they disagree *should* move the
    projection, because the value it stands for really did change.  The legacy
    branch goes away with ``NumericElementType`` itself.

    Anything unrecognized raises rather than normalizing to a shared string --
    a projection that silently mapped two datatypes to one name would be a
    weaker oracle than the digests it is replacing.
    """

    name = getattr(value, "name", None)
    if isinstance(name, str) and callable(getattr(value, "bitwidth", None)):
        return name  # a QONNX BaseDataType: its name is already canonical

    type_id = getattr(value, "type_id", None)
    bit_width = getattr(value, "bit_width", None)
    if not isinstance(type_id, str) or not isinstance(bit_width, int):
        raise TypeError(f"not a datatype this projection recognizes: {value!r}")
    spelled = _LEGACY_SPELLINGS.get((type_id, bit_width))
    if spelled is not None:
        return spelled
    if type_id in {"int", "uint", "float"}:
        return f"{type_id.upper()}{bit_width}"
    raise TypeError(f"no canonical QONNX spelling for {value!r}")


def _is_datatype(value: Any) -> bool:
    if callable(getattr(value, "bitwidth", None)):
        return True
    return hasattr(value, "type_id") and hasattr(value, "bit_width")


def normalized(value: Any) -> Any:
    """Recursively rebuild ``value``, substituting datatypes for their names.

    The result is plain nested tuples of primitives, so it is hashable,
    ``repr``-stable, and comparable across the change.  Class and field names
    are kept: a field disappearing or being renamed is a structural difference
    and must not be normalized away.
    """

    if _is_datatype(value):
        return ("datatype", canonical_datatype_name(value))
    if isinstance(value, Enum):
        return ("enum", type(value).__name__, value.name)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple((item.name, normalized(getattr(value, item.name))) for item in fields(value)),
        )
    if isinstance(value, (str, bytes, bool, int, float)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return tuple(normalized(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((normalized(item) for item in value), key=repr))
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((normalized(key), normalized(item)) for key, item in value.items()),
                key=repr,
            )
        )
    raise TypeError(f"normalized() does not know how to project {type(value).__name__}")
