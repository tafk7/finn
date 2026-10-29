# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed data-only timing records, separate from durable logical descriptors.

The logical record is embedded verbatim and independently validated. Tags name
registered class identities, never import paths, callables or native contexts.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from hashlib import sha256
from typing import cast

from finn.dataflow.descriptors import dump as logical_dump
from finn.dataflow.descriptors import load as logical_load
from finn.dataflow.logical import LogicalPlan
from finn.dataflow.profiles import (
    Attachment,
    Capacity,
    ChannelProfile,
    ChannelRules,
    Clock,
    Evidence,
    Launch,
    Profile,
    TimedPlan,
)

SCHEMA = "optional-timing-1"
MAX_RECORD_BYTES = 16 * 1024 * 1024
_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Attachment,
        Capacity,
        ChannelProfile,
        ChannelRules,
        Clock,
        Evidence,
        Launch,
        Profile,
        TimedPlan,
    )
}


def _types() -> dict[str, type]:
    # The executor consumes this codec; import its closed data schema only at
    # encode/decode time, after both modules have initialized.
    from finn.dataflow.execution import (  # noqa: PLC0415
        Capabilities,
        Criterion,
        Execution,
        Flags,
        IndexStream,
        Program,
        Reference,
        Request,
        ResultData,
        RunData,
        TraceEvent,
    )

    return {
        **_TYPES,
        **{
            cls.__name__: cls
            for cls in (
                Capabilities,
                Criterion,
                Execution,
                Flags,
                IndexStream,
                Program,
                Reference,
                Request,
                ResultData,
                RunData,
                TraceEvent,
            )
        },
    }


def _encode(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is tuple:
        return [_encode(item) for item in value]
    if type(value) is LogicalPlan:
        return {"type": "LogicalPlan", "record": logical_dump(value).decode("utf-8")}
    if (
        is_dataclass(value)
        and not isinstance(value, type)
        and _types().get(type(value).__name__) is type(value)
    ):
        return {
            "type": type(value).__name__,
            **{f.name: _encode(getattr(value, f.name)) for f in fields(value)},
        }
    raise TypeError(f"not a timing record: {type(value).__name__}")


def _decode(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is list:
        return tuple(_decode(item) for item in value)
    if type(value) is not dict:
        raise ValueError("timing records contain only integral JSON data")
    record = cast(dict[str, object], value)
    tag = record.get("type")
    if tag == "LogicalPlan":
        if set(record) != {"type", "record"} or type(record["record"]) is not str:
            raise ValueError("invalid logical record")
        plan = logical_load(record["record"].encode("utf-8"))
        if type(plan) is not LogicalPlan:
            raise ValueError("bundle must contain a LogicalPlan")
        return plan
    types = _types()
    if type(tag) is not str or tag not in types:
        raise ValueError("unknown timing record tag")
    cls = types[tag]
    if set(record) != {"type", *(f.name for f in fields(cls))}:
        raise ValueError("timing record fields mismatch")
    return cls(**{f.name: _decode(record[f.name]) for f in fields(cls)})


def dump(value: object) -> bytes:
    """Encode data, never host callbacks or engine objects."""
    raw = json.dumps(
        {"schema": SCHEMA, "value": _encode(value)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("timing record exceeds byte budget")
    return raw


def load(raw: bytes) -> object:
    """Reject unknown fields, duplicate keys and nonintegral JSON before construction."""
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("timing record exceeds byte budget")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate JSON keys")
        return result

    def forbidden(value: str) -> object:
        raise ValueError(f"nonintegral JSON value: {value}")

    value = json.loads(
        raw, object_pairs_hook=pairs, parse_float=forbidden, parse_constant=forbidden
    )
    if type(value) is not dict or set(value) != {"schema", "value"} or value["schema"] != SCHEMA:
        raise ValueError("timing schema mismatch")
    return _decode(value["value"])


def digest(value: object) -> str:
    return sha256(dump(value)).hexdigest()
