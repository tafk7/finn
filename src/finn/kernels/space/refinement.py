# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Advanced typed monotone commitment operations."""

from __future__ import annotations

from typing import Protocol, TypeVar, cast

from .declarations import Decision, DecisionRef, Space
from .edits import Change, ChangeRequest, CommitmentReport
from .occurrence import change as _change
from .occurrence import commit as _commit

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound=Space)


class _EditableReference(Protocol[T_co]):
    def _choice_type(self) -> T_co: ...


def change(instance: Space, reference: _EditableReference[T], value: object) -> Change[T]:
    result: Change[object] = _change(
        instance, cast(Decision[object] | DecisionRef[object], reference), value
    )
    return cast(Change[T], result)


def commit(instance: S, *changes: ChangeRequest) -> CommitmentReport[S]:
    return _commit(instance, *changes)


__all__ = ["change", "commit"]
