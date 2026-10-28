# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared inclusion declarations for Group behavior and whole-space reference tests."""

from __future__ import annotations

from finn.core.space import Decision, Group, Param, Space, constraint, derived, view


class Bundle(Group):
    extent: int = Param()
    lanes: int = Decision(domain=(1, 2, 4))

    @derived
    def doubled(self) -> int:
        return 2 * self.extent

    @constraint
    def fits(self) -> bool:
        return self.lanes <= self.extent

    @view
    def shown(self) -> int:
        return self.doubled // self.lanes


class Host(Space):
    extent: int = Param(default=3)
    first = Bundle(extent=extent)
    second = Bundle(extent=first.doubled)
