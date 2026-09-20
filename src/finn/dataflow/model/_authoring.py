# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared bookkeeping for authored and generated Kernel declarations."""

from __future__ import annotations

from typing import cast

from finn.dataflow.space.declarations import AuthoringError

GENERATED_MEMBERS = "_kernel_generated_members"


def member_owner(kernel_type: type[object], name: str) -> type[object] | None:
    return next((base for base in kernel_type.__mro__ if name in base.__dict__), None)


def authored_member(kernel_type: type[object], name: str) -> object | None:
    owner = member_owner(kernel_type, name)
    if owner is None:
        return None
    generated = cast("frozenset[str]", owner.__dict__.get(GENERATED_MEMBERS, frozenset()))
    return None if name in generated else owner.__dict__[name]


def generated_member(
    kernel_type: type[object], name: str, value: object, generated: set[str]
) -> object:
    setattr(kernel_type, name, value)
    generated.add(name)
    return value


def use_authored_or_generated(
    kernel_type: type[object],
    name: str,
    generated_value: object,
    expected: type | tuple[type, ...],
    generated: set[str],
) -> object:
    authored = authored_member(kernel_type, name)
    if authored is not None:
        if not isinstance(authored, expected):
            raise AuthoringError(
                f"{kernel_type.__name__}.{name} is {type(authored).__name__}; "
                f"expected {getattr(expected, '__name__', 'the standard declaration type')}"
            )
        return authored
    return generated_member(kernel_type, name, generated_value, generated)


__all__ = [
    "GENERATED_MEMBERS",
    "authored_member",
    "generated_member",
    "member_owner",
    "use_authored_or_generated",
]
