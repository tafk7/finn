# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel identity over Space; each kernel declares the questions it can answer."""

from __future__ import annotations

from typing import ClassVar

from finn.kernels.space.compiler import _CompiledSpace
from finn.kernels.space.declarations import AuthoringError, Problem, Space, declared_members


class Kernel(Space):
    """A named kernel design space with explicitly assessed declarations."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    _implicit_exports: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._configure_views()

    @classmethod
    def _configure_views(cls) -> None:
        """Extension point for the conventional authoring adapter."""

    @classmethod
    def _finalize_compilation(cls, compiled: object) -> object:
        if not isinstance(compiled, _CompiledSpace):
            raise AuthoringError(f"{cls.__name__} received an invalid Space compilation")
        if not cls.id:
            raise AuthoringError(f"{cls.__name__} must declare a non-empty id")
        if not cls.version:
            raise AuthoringError(f"{cls.__name__} must declare a non-empty version")
        if any(isinstance(item, Problem) for _, item in declared_members(cls)):
            raise AuthoringError(
                f"{cls.__name__} must consume external facts through Input declarations"
            )
        return compiled


__all__ = ["Kernel"]
