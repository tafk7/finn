# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The value types: classes whose ``==`` and hash are their values' semantics.

A value type's owner registers it (``value_type``), and states by that:

- two of its instances are equal exactly when they are the same value, and equal ones
  hash alike;
- equal instances cannot be told apart: every public attribute and the ``repr`` agree;
- an instance never changes: setting an attribute is refused.

The owner registers its types in one place, and checks them there in a test, on samples
of its own: the engine cannot read a class's ``==``.

What the engine does with it: its node cache keys a registered class's instances by type
and ``==``, wherever it meets one: a choice's or fact's value, an item of a tuple, a field
of a value class (a ``Tensor``'s datatype). A class derived from a registered one is keyed
as it when it is defined in the same module and keeps its ``==`` and hash: the owner's
statement covers the classes of its own module, and none defined elsewhere.
"""

from __future__ import annotations

import weakref
from typing import TypeVar

T = TypeVar("T", bound=type)

_REGISTER: dict[type, None] = {}
"""The registered classes, in the order they were registered."""

_KEYED: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()
"""``is_value_type``'s answer by class, made once (made again after a registration)."""


def value_type(cls: T) -> T:
    """Register ``cls`` as a value type: its owner states that its ``==`` and hash are its
    values' semantics, that equal instances cannot be told apart, and that an instance
    never changes (see the module docstring). Returns ``cls``, so it serves as a class
    decorator too. A class that is not hashable is refused."""
    if not isinstance(cls, type):
        raise TypeError(f"a value type is a class, not {cls!r}")
    if getattr(cls, "__hash__", None) is None:
        raise TypeError(f"a value type is hashable; {cls.__qualname__} is not")
    _REGISTER[cls] = None
    _KEYED.clear()
    return cls


def registered() -> tuple[type, ...]:
    """The registered classes, in the order they were registered."""
    return tuple(_REGISTER)


def is_value_type(kind: type) -> bool:
    """Whether ``kind``'s instances are keyed by type and ``==``: a registered class, or
    one derived from it in the same module that keeps its ``__eq__`` and ``__hash__``."""
    if kind in _REGISTER:
        return True
    found = _KEYED.get(kind)
    if found is None:
        found = _KEYED[kind] = _derived(kind)
    return found


def _derived(kind: type) -> bool:
    for base in kind.__mro__[1:]:
        if base in _REGISTER:
            return (
                kind.__module__ == base.__module__
                and kind.__eq__ is base.__eq__
                and kind.__hash__ is base.__hash__
            )
    return False


__all__ = ["is_value_type", "registered", "value_type"]
