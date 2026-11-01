# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How a declaration's values are read (``finn.platform.declarations``, and a shell's facts,
``finn.platform.shells``): each value checked for its type as it is read, a value not of
it raised as ``Invalid`` with where it is and why, which its reader refuses as
``declaration-invalid``, owned by the declaration's file and table."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

_CODE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*")
_KINDS: dict[type, str] = {
    str: "a string",
    int: "a whole number",
    float: "a number",
    bool: "true or false",
    list: "a list",
    dict: "a table",
}


class Invalid(ValueError):
    """A value of a declaration not of its type, raised with where it is and why."""


def table(value: object, where: str) -> dict[str, Any]:
    """``value``, a table."""
    if not isinstance(value, dict):
        raise Invalid(f"{where} is a table, not {value!r}")
    return value


def keys(stated: Mapping[str, Any], where: str, required: set[str], optional: set[str]) -> None:
    """Check that ``stated`` states each of ``required`` and nothing beyond ``optional``."""
    missing, unknown = required - set(stated), set(stated) - required - optional
    if missing:
        raise Invalid(f"{where} states no {', '.join(sorted(missing))}")
    if unknown:
        raise Invalid(
            f"{where} states {', '.join(sorted(unknown))}, which it does not have "
            f"(it has {', '.join(sorted(required | optional)) or 'nothing'})"
        )


def expect(value: object, where: str, required: set[str], optional: set[str]) -> dict[str, Any]:
    """``value``, a table stating each of ``required`` and nothing beyond ``optional``."""
    found = table(value, where)
    keys(found, where, required, optional)
    return found


def typed(value: object, kind: type | tuple[type, ...], where: str) -> Any:
    """``value``, of ``kind``; a count is never true or false."""
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if (isinstance(value, bool) and bool not in kinds) or not isinstance(value, kinds):
        said = " or ".join(_KINDS.get(each, each.__name__) for each in kinds)
        raise Invalid(f"{where} is {said}, not {value!r}")
    return value


def count(value: object, where: str, *, least: int = 0) -> int:
    """``value``, a whole number of at least ``least``."""
    found: int = typed(value, int, where)
    if found < least:
        raise Invalid(f"{where} is a count of at least {least}, not {found}")
    return found


def positive(value: object, where: str) -> float:
    """``value``, a finite number above zero."""
    found = float(typed(value, (int, float), where))
    if not (math.isfinite(found) and found > 0):
        raise Invalid(f"{where} is a finite number above zero, not {value!r}")
    return found


def text(value: object, where: str) -> str:
    """``value``, a non-empty string."""
    if not isinstance(value, str) or not value:
        raise Invalid(f"{where} is a non-empty string, not {value!r}")
    return value


def texts(value: object, where: str) -> tuple[str, ...]:
    """``value``, a list of non-empty strings."""
    if not isinstance(value, list):
        raise Invalid(f"{where} is a list of strings, not {value!r}")
    return tuple(text(each, f"{where}[{index}]") for index, each in enumerate(value))


def code(value: object, where: str) -> str:
    """``value``, code named as ``module:attr``."""
    found = text(value, where)
    if not _CODE.fullmatch(found):
        raise Invalid(f"{where} names code as module:attr, not {found!r}")
    return found


def whole_bytes(value: object, where: str) -> int:
    """``value``, a width in bits of whole bytes, at least one."""
    found = count(value, where, least=8)
    if found % 8:
        raise Invalid(f"{where}: {found} bits is no whole bytes")
    return found
