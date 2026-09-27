# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel identity and optional view discovery over Space."""

from __future__ import annotations

from typing import ClassVar

from finn.core.space import Space
from finn.core.space.errors import DefinitionError
from finn.core.space.inspection import NodeInfo, members


class Kernel(Space):
    """A named family with explicitly declared, independently typed views.

    A kernel may describe plain pins, a stream boundary, source requirements,
    or other values. Identity does not imply any particular interface or view.
    """

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("id", "version"):
            value = getattr(cls, name)
            if type(value) is not str or not value:
                raise DefinitionError(f"{cls.__qualname__} must declare a nonempty string {name}")

    def capabilities(self) -> tuple[NodeInfo, ...]:
        """Inspect authored views in this scope and its children without evaluating."""

        return tuple(member for member in members(self) if member.kind == "view")


__all__ = ["Kernel"]
