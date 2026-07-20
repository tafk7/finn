############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``make_registry`` — the op-agnostic implementation-registry factory.

Each op package calls :func:`make_registry` once to get a fresh, isolated
``(register, build_pool, registered_names)`` triple bound to its own private dict.
The mechanism makes "add a backend = add one ``impl_*.py`` that self-registers, edit
nothing else" literally true, per op, with no shared global state between ops.

A bundle module registers itself at import time via the ``register`` decorator on a
zero-arg factory returning an :class:`Implementation`. The op package ``__init__``
imports its built-in bundle modules for that side effect; assembling the schema just
reads whatever has registered. Registration order is preserved, so the first-
registered bundle is the pool default (the root axis default).
"""

from __future__ import annotations

from collections.abc import Callable

from finn.kernels.space import Implementation


def make_registry(op_name: str):
    """Build a fresh registry for one op. Returns ``(register, build_pool,
    registered_names, unregister)`` closed over a private ordered dict."""
    registry: dict[str, Callable[[], Implementation]] = {}

    def register(factory: Callable[[], Implementation]) -> Callable[[], Implementation]:
        """Decorator: register a zero-arg ``Implementation`` factory by its bundle
        name. The factory is called once here to read ``.name`` (bundles are cheap to
        build); it is re-invoked per :func:`build_pool` so each schema gets fresh
        axis/derived objects. Duplicate names are a registration error."""
        name = factory().name
        if name in registry:
            raise ValueError(f"duplicate {op_name} implementation registered: {name!r}")
        registry[name] = factory
        return factory

    def build_pool() -> tuple[Implementation, ...]:
        """Instantiate every registered bundle (fresh objects), registration order."""
        return tuple(factory() for factory in registry.values())

    def registered_names() -> tuple[str, ...]:
        return tuple(registry)

    def unregister(name: str) -> None:
        """Remove a registered bundle by name (no-op if absent). Mainly for tests
        that register a transient stub and must not leak it into other tests."""
        registry.pop(name, None)

    return register, build_pool, registered_names, unregister
