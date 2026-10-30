# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Reusable primitive declarations included without another validity boundary."""

from __future__ import annotations

from ._configuration import Space, _Namespace
from .errors import DefinitionError


class Group(_Namespace):
    """A typed declaration namespace included in its nearest enclosing Space.

    Calling the class declares a fresh inclusion with frozen symbolic Param bindings.
    Members resolve independently through the enclosing design space's evaluator and
    configuration. A Group has no validity, presence condition or root configuration;
    included real children and Selectors retain their ordinary semantics. Edit included
    choices through the enclosing Space using their typed references.

    Class access yields declarations, placement access yields symbolic references,
    and bound namespace access computes nothing. Ordinary value access, query, inspect
    and field follow the same rules as Space members.

    ``inspection.groups(host)`` describes inclusions; ``inspection.members(host)``
    includes their primitives. ``Members(Child.member)`` discovers real children
    through included namespaces and stops at real Space boundaries. A Members inside
    a Group ranges over the children declared in that namespace. ``Members(Group.member)``
    is unsupported. Query members, not the namespace itself; ``present(group)`` is
    undefined because a Group introduces no presence scope.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if issubclass(cls, Space):
            raise DefinitionError.broken(
                "group-space-mixed",
                "space.groups",
                "Group and Space are distinct declaration kinds; include or place one in the other",
            )
