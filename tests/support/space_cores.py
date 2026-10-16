# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Two cores sharing a ``width``, and the helpers that commit choices on a point and read
its forced ones."""

from __future__ import annotations

from typing import Any

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    constraint,
    divisors_of,
    inspection,
    reject,
)


class Packed(Space):
    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed core takes at most 64 lanes")
        return True

    admission = ConstraintGroup(fits)


class Stub(Space):
    width: int = Param()
    rows: int = Decision(values=(1, 2, 4))

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub core takes an even width")
        return True

    admission = ConstraintGroup(even)


CORES: dict[str, type[Space] | Space] = {"packed": Packed, "stub": Stub}


def commit(point: Any, choices: dict[str, object]) -> Any:
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    return point.with_choices({owned[key]: value for key, value in choices.items()})


def forced(point: Any) -> dict[str, object]:
    return {item.key: item.value for item in inspection.forced(point)}
