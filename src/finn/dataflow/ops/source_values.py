# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable source identity and graph interface references, independent of effects."""

from dataclasses import dataclass
from enum import Enum


class InterfaceDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class SourceDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class QualifiedInterfaceRef:
    node_id: str
    direction: InterfaceDirection
    operand_id: str
    interface_id: str | None


@dataclass(frozen=True, slots=True)
class SourceOperandKey:
    operand_id: str
    direction: SourceDirection
    index: int


@dataclass(frozen=True, slots=True)
class SourceValueRef:
    key: SourceOperandKey
    shape: tuple[int, ...]
    carrier_dtype: int
    logical_datatype: str
    initializer_content_digest: str | None

    @property
    def initializer_present(self) -> bool:
        return self.initializer_content_digest is not None


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    schema_version: int
    problem_fingerprint: str
    scope_id: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("native schema origin must be a positive integer")
        if type(self.problem_fingerprint) is not str or not self.problem_fingerprint:
            raise ValueError("origin problem fingerprint must be a non-empty string")
        if self.scope_id is not None and (type(self.scope_id) is not str or not self.scope_id):
            raise ValueError("origin scope_id must be None or a non-empty string")
