# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every question FINN asks ``finn.dataflow`` while it derives a design, recorded.

``Recorder`` wraps each public function, method, class method and property of
``finn.dataflow``'s modules (``datatypes`` aside: both packages share it), in the module
and class that declares it and in every loaded module that imported it by name, so a
call made anywhere in FINN goes through the wrapper. A call is recorded only when FINN
makes it: a call the package makes of itself, inside one recorded, is the package's
own working and is not. Each distinct call (by its arguments' values) is kept once,
with its outcome (the value returned, or the exception raised, by type and message) and
the sources it was met in, by the label the caller sets (``Recorder.source``). Entering a
recorder drops the composed classes FINN keeps (``finn.kernels.composer.clear_classes``): a
class keeps the engine's node cache, so a structure met before in the process would answer
from it and ask ``finn.dataflow`` nothing.

The recorded calls are what ``replay`` asks the revised package. Recording changes no
answer: the wrapper returns what the function returns and raises what it raises, and a
generator's items are collected and handed on as an iterator over them.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import numpy as np
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from finn.kernels.composer import clear_classes

MODULES = (
    "finn.dataflow.gemm",
    "finn.dataflow.schedule",
    "finn.dataflow.steps",
    "finn.dataflow.streams",
    "finn.dataflow.tensor",
    "finn.dataflow.traversal",
)
"""The line's modules whose answers are compared: all but the shared ``datatypes``."""


@dataclass(frozen=True)
class Raised:
    """An exception as an outcome: its class's module and name, and its message."""

    kind: str
    message: str

    @classmethod
    def of(cls, error: BaseException) -> Raised:
        return cls(f"{type(error).__module__}.{type(error).__qualname__}", str(error))


@dataclass(frozen=True)
class Target:
    """What was called: ``kind`` is ``function``, ``method``, ``classmethod`` or
    ``property``; ``module`` the declaring module; ``owner`` the class (empty for a
    function); ``name`` the attribute."""

    kind: str
    module: str
    owner: str
    name: str

    @property
    def label(self) -> str:
        return ".".join(
            part for part in (self.module.rsplit(".", 1)[-1], self.owner, self.name) if part
        )


@dataclass
class Call:
    """One distinct call: its target, arguments and outcome, and where it was met."""

    target: Target
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    outcome: Any
    sources: set[str] = field(default_factory=set)
    count: int = 0


def frozen(value: Any) -> Any:
    """A hashable key for ``value``: containers item by item, an array by its bytes, a
    value that will not hash by its ``repr``."""
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(frozen(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple((frozen(key), frozen(item)) for key, item in value.items()))
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return value


class Recorder:
    """A context that records FINN's calls of ``finn.dataflow`` (the module docstring)."""

    def __init__(self) -> None:
        self.calls: dict[Any, Call] = {}
        self.source = ""
        self._depth = 0
        self._undo: list[Callable[[], None]] = []

    def __enter__(self) -> Recorder:
        # A composed class keeps its compiled model and the engine's node cache with it, by
        # the structure it was made for: a second run of a structure in this process would
        # read the first run's answers from the cache and ask ``finn.dataflow`` nothing.
        # Each recording starts from no class, so it sees every question its source asks.
        clear_classes()
        for name in MODULES:
            self._wrap_module(importlib.import_module(name))
        return self

    def __exit__(self, *exc: object) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    # -- wrapping -------------------------------------------------------------------------

    def _set(self, owner: Any, name: str, value: Any) -> None:
        old = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
        setattr(owner, name, value)
        self._undo.append(lambda: setattr(owner, name, old))

    def _wrap_module(self, module: types.ModuleType) -> None:
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if isinstance(value, type) and value.__module__ == module.__name__:
                self._wrap_class(value)
            elif inspect.isfunction(value) and value.__module__ == module.__name__:
                wrapped = self._wrapper(Target("function", module.__name__, "", name), value)
                self._rebind(value, wrapped)

    def _rebind(self, original: Any, wrapped: Any) -> None:
        """Every loaded module's name for ``original`` (its own module's included) bound to
        ``wrapped``."""
        for module in list(sys.modules.values()):
            names = [
                key for key, item in getattr(module, "__dict__", {}).items() if item is original
            ]
            for name in names:
                self._set(module, name, wrapped)

    def _wrap_class(self, cls: type) -> None:
        for name, member in list(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            target_of = functools.partial(
                Target, module=cls.__module__, owner=cls.__qualname__, name=name
            )
            if isinstance(member, classmethod):
                wrapped = self._wrapper(target_of(kind="classmethod"), member.__func__)
                self._set(cls, name, classmethod(wrapped))
            elif isinstance(member, property) and member.fget is not None:
                wrapped = self._wrapper(target_of(kind="property"), member.fget)
                self._set(cls, name, property(wrapped))
            elif inspect.isfunction(member):
                self._set(cls, name, self._wrapper(target_of(kind="method"), member))

    def _wrapper(self, target: Target, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def recorded(*args: Any, **kwargs: Any) -> Any:
            if self._depth:
                return function(*args, **kwargs)
            self._depth += 1
            try:
                result = function(*args, **kwargs)
                if isinstance(result, types.GeneratorType):
                    result = tuple(result)
                    outcome, handed = result, iter(result)
                else:
                    outcome = handed = result
            except Exception as error:
                self._note(target, args, kwargs, Raised.of(error))
                raise
            finally:
                self._depth -= 1
            self._note(target, args, kwargs, outcome)
            return handed

        return recorded

    def _note(
        self, target: Target, args: tuple[Any, ...], kwargs: dict[str, Any], outcome: Any
    ) -> None:
        key = (target, frozen(args), frozen(kwargs))
        call = self.calls.get(key)
        if call is None:
            call = self.calls[key] = Call(target, args, dict(kwargs), outcome)
        call.sources.add(self.source)
        call.count += 1

    def recorded(self) -> Iterator[Call]:
        return iter(self.calls.values())
