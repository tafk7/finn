# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.authoring import PortableCodec
from finn.dataflow.datatypes import QONNX_DATATYPE_TOKEN


class Mode(str, Enum):
    __dataflow_identity_token__ = "test.portable.mode"

    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True)
class StructuredChoice:
    __dataflow_identity_token__ = "test.portable.structured-choice"

    enabled: bool
    count: int
    ratio: float
    label: str
    mode: Mode
    optional: int | None
    sequence: tuple[int, ...]
    mapping: dict[str, int]


def test_portable_codec_round_trips_every_required_value_category() -> None:
    value = StructuredChoice(
        True,
        7,
        0.125,
        "choice",
        Mode.SECOND,
        None,
        (3, 1, 2),
        {"z": 2, "a": 1},
    )
    codec = PortableCodec(StructuredChoice)
    encoded = codec.dumps(value)
    restored = codec.loads(encoded)

    assert restored == value
    assert '"float":"0x1.0000000000000p-3"' in encoded
    assert encoded.index('"a"') < encoded.index('"z"')


def test_portable_qonnx_codec_is_strict_about_canonical_names() -> None:
    codec = PortableCodec(QONNX_DATATYPE_TOKEN)
    assert codec.loads(codec.dumps(DataType["BINARY"])) == DataType["BINARY"]
    with pytest.raises(ValueError, match="not canonical"):
        codec.loads('{"qonnx_datatype":"UINT1"}')


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_portable_codec_refuses_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        PortableCodec(float).dumps(value)
