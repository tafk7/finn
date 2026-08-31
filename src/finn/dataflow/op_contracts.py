# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-light operation records shared by source adapters and custom ops."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from finn.dataflow.design import Finding

NodeAttributeType = tuple[str, bool, int | float | str | bool, set[object] | None]
NodeAttrKind = Literal["integer", "boolean", "string", "enum"]


@dataclass(frozen=True)
class NodeAttrCodec:
    """Stable encoding for one committed decision in an ONNX node attribute."""

    attribute_name: str
    kind: NodeAttrKind
    value_type: type[object]
    enum_tokens: tuple[tuple[str, Enum], ...] = ()

    def __post_init__(self) -> None:
        if not self.attribute_name:
            raise ValueError("node attribute name must not be empty")
        if self.kind == "integer" and self.value_type is not int:
            raise ValueError("integer codecs require int value semantics")
        if self.kind == "boolean" and self.value_type is not bool:
            raise ValueError("boolean codecs require bool value semantics")
        if self.kind == "string" and self.value_type is not str:
            raise ValueError("string codecs require str value semantics")
        if self.kind == "enum":
            if not issubclass(self.value_type, Enum) or not self.enum_tokens:
                raise ValueError("enum codecs require an Enum type and explicit tokens")
            tokens = tuple(token for token, _value in self.enum_tokens)
            values = tuple(value for _token, value in self.enum_tokens)
            if len(set(tokens)) != len(tokens) or len(set(values)) != len(values):
                raise ValueError("enum codec tokens and values must be unique")
            if any(type(value) is not self.value_type for value in values):
                raise ValueError("enum codec values must have the declared enum type")
        elif self.enum_tokens:
            raise ValueError("only enum codecs may declare enum tokens")

    @classmethod
    def integer(cls, attribute_name: str) -> NodeAttrCodec:
        return cls(attribute_name, "integer", int)

    @classmethod
    def boolean(cls, attribute_name: str) -> NodeAttrCodec:
        return cls(attribute_name, "boolean", bool)

    @classmethod
    def string(cls, attribute_name: str) -> NodeAttrCodec:
        return cls(attribute_name, "string", str)

    @classmethod
    def finite_enum(
        cls,
        attribute_name: str,
        enum_type: type[Enum],
        tokens: Mapping[str, Enum],
    ) -> NodeAttrCodec:
        return cls(attribute_name, "enum", enum_type, tuple(tokens.items()))

    @property
    def nodeattr_definition(self) -> NodeAttributeType:
        if self.kind == "integer":
            return ("i", False, 0, None)
        if self.kind == "boolean":
            return ("i", False, 0, {0, 1})
        if self.kind == "string":
            return ("s", False, "", None)
        return ("s", False, "", {token for token, _value in self.enum_tokens})

    def encode(self, value: object) -> int | str:
        if self.kind == "integer" and type(value) is int:
            return value
        if self.kind == "boolean" and type(value) is bool:
            return int(value)
        if self.kind == "string" and type(value) is str:
            return value
        if self.kind == "enum" and type(value) is self.value_type:
            by_value = {enum_value: token for token, enum_value in self.enum_tokens}
            return by_value[cast(Enum, value)]
        raise TypeError(
            f"{self.attribute_name} cannot encode {type(value).__name__} as {self.kind}"
        )

    def decode(self, value: object) -> object:
        if self.kind == "integer" and type(value) is int:
            return value
        if self.kind == "boolean" and type(value) is int and value in {0, 1}:
            return bool(value)
        if self.kind == "string" and type(value) is str:
            return value
        if self.kind == "enum" and type(value) is str:
            by_token = dict(self.enum_tokens)
            if value in by_token:
                return by_token[value]
        raise ValueError(f"{self.attribute_name} does not contain a valid {self.kind} encoding")

    def encode_json(self, value: object) -> object:
        """Encode a decision value for a portable JSON-compatible envelope."""

        if self.kind == "boolean" and type(value) is bool:
            return value
        return self.encode(value)

    def decode_json(self, value: object) -> object:
        """Decode the portable representation without ONNX integer coercion."""

        if self.kind == "boolean" and type(value) is bool:
            return value
        return self.decode(value)


class DataflowOpError(ValueError):
    """Failure to project, hydrate, persist, or resolve a dataflow operation."""

    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = tuple(sorted(findings, key=lambda item: item.path))
        super().__init__(f"dataflow operation failed with {len(self.findings)} finding(s)")


__all__ = ["DataflowOpError", "NodeAttrCodec", "NodeAttrKind", "NodeAttributeType"]
