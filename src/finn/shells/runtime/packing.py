# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An edge's values as the bytes its mover moves (``pack``), and back (``unpack``): each
beat's lanes a bit vector, lane 0 in its least significant bits, padded to whole bytes,
the bytes least significant first. That is the layout of FINN's
``finnpy_to_packed_bytearray`` with both its inner and its byte order reversed, the one the
movers read and write.

An element is a FINN data type by its name (``Element``): ``BINARY``, ``BIPOLAR``,
``TERNARY``, ``INT<n>`` and ``UINT<n>`` (to 64 bits), ``FIXED<w,i>`` and the IEEE floats
``FLOAT16``, ``FLOAT32`` and ``FLOAT64``. Any other is refused (``element-unsupported``), as
is a value the element cannot hold (``value-outside``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import Refused

_INTEGER = re.compile(r"(U?)INT([1-9][0-9]*)")
_FIXED = re.compile(r"FIXED<([1-9][0-9]*),([1-9][0-9]*)>")
_FLOATS = {"FLOAT16": 16, "FLOAT32": 32, "FLOAT64": 64}


@dataclass(frozen=True)
class Element:
    """A FINN data type by its ``name``: its ``bits``, whether its code is ``signed``, the
    ``scale`` of a code's unit (a fixed point's ``2**-fraction``) and whether it is an IEEE
    ``float``."""

    name: str
    bits: int
    signed: bool
    scale: float = 1.0
    float: bool = False

    @staticmethod
    def named(name: str) -> Element:
        """The element named ``name``; one it does not know is refused
        (``element-unsupported``)."""
        if name in ("BINARY", "BIPOLAR"):
            return Element(name, 1, False)
        if name == "TERNARY":
            return Element(name, 2, True)
        if name in _FLOATS:
            return Element(name, _FLOATS[name], False, float=True)
        integer = _INTEGER.fullmatch(name)
        if integer is not None and int(integer.group(2)) <= 64:
            return Element(name, int(integer.group(2)), not integer.group(1))
        fixed = _FIXED.fullmatch(name)
        if fixed is not None and int(fixed.group(2)) <= int(fixed.group(1)) <= 64:
            width, whole = int(fixed.group(1)), int(fixed.group(2))
            return Element(name, width, True, scale=2.0 ** -(width - whole))
        raise Refused(
            "element-unsupported",
            f"{name}: no element the driver packs (BINARY, BIPOLAR, TERNARY, INT<n>, UINT<n>, "
            "FIXED<w,i>, FLOAT16, FLOAT32, FLOAT64)",
        )

    def codes(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint64]]:
        """Each value's code, ``bits`` wide: its bit pattern. A value the element cannot
        hold is refused (``value-outside``)."""
        if self.float:
            floats = np.asarray(values, dtype=f"<f{self.bits // 8}")
            return floats.view(f"<u{self.bits // 8}").astype(np.uint64)
        array = np.asarray(values, dtype=np.float64)
        if self.name == "BIPOLAR":
            allowed = np.isin(array, (-1.0, 1.0))
            array = (array + 1) / 2
        else:
            array = array / self.scale
            low = -(2 ** (self.bits - 1)) if self.signed else 0
            high = 2 ** (self.bits - 1) - 1 if self.signed else 2**self.bits - 1
            if self.name == "TERNARY":
                low, high = -1, 1
            allowed = (array == np.round(array)) & (array >= low) & (array <= high)
        if not np.all(allowed):
            outside = np.asarray(values).reshape(-1)[~np.asarray(allowed).reshape(-1)][:4]
            raise Refused(
                "value-outside", f"{self.name} holds no {outside.tolist()}: a value of another type"
            )
        mask = np.uint64((1 << self.bits) - 1) if self.bits < 64 else np.uint64(2**64 - 1)
        return array.astype(np.int64).astype(np.uint64) & mask

    def values(self, codes: np.ndarray[Any, np.dtype[np.uint64]]) -> np.ndarray[Any, Any]:
        """The values of ``codes``, as float32 (as FINN's unpacking gives them)."""
        if self.float:
            width = f"<u{self.bits // 8}"
            return codes.astype(width).view(f"<f{self.bits // 8}").astype(np.float32)
        if self.name == "BIPOLAR":
            return (codes.astype(np.int64) * 2 - 1).astype(np.float32)
        signed = codes.astype(np.int64)
        if self.signed and self.bits < 64:
            top = np.int64(1 << (self.bits - 1))
            signed = np.where(signed >= top, signed - (top << np.int64(1)), signed)
        return (signed * self.scale).astype(np.float32)


def beat_bytes(element: Element, lanes: int) -> int:
    """The bytes a beat of ``lanes`` values of ``element`` takes."""
    return -(-lanes * element.bits // 8)


def pack(values: np.ndarray[Any, Any], element: Element) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """``values`` packed along their last axis, the lanes of a beat (the module docstring):
    the same shape, the last axis its bytes."""
    codes = element.codes(values)
    lanes = codes.shape[-1]
    shifts = np.arange(element.bits, dtype=np.uint64)
    bits = ((codes[..., None] >> shifts) & np.uint64(1)).astype(np.uint8)
    bits = bits.reshape(*codes.shape[:-1], lanes * element.bits)
    pad = beat_bytes(element, lanes) * 8 - lanes * element.bits
    if pad:
        bits = np.concatenate([bits, np.zeros((*bits.shape[:-1], pad), np.uint8)], axis=-1)
    return np.packbits(bits, axis=-1, bitorder="little")


def unpack(
    packed: np.ndarray[Any, np.dtype[np.uint8]], element: Element, lanes: int
) -> np.ndarray[Any, Any]:
    """The values of ``packed``, ``pack``'s bytes, ``lanes`` a beat along its last axis, as
    float32."""
    bits = np.unpackbits(packed, axis=-1, bitorder="little")[..., : lanes * element.bits]
    bits = bits.reshape(*packed.shape[:-1], lanes, element.bits).astype(np.uint64)
    shifts = np.arange(element.bits, dtype=np.uint64)
    codes = np.bitwise_or.reduce(bits << shifts, axis=-1)
    return element.values(codes)


__all__ = ["Element", "beat_bytes", "pack", "unpack"]
