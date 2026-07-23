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
bundle factory returning a :class:`Backend`. The op package ``__init__`` imports its
built-in bundle modules for that side effect; assembling the schema just reads whatever
has registered. Registration order is preserved, so the first-registered bundle is the
pool default (the root axis default).

**Factory arity.** A factory may be zero-arg (``() -> Backend``, the common case) OR take
a single argument threaded through by ``build_pool`` (``(arg) -> Backend``). The
``parameters`` pool uses the latter to build each storage topology for a specific
parameter INTERFACE (its axis/derived point keys are interface-namespaced). The bundle
IDENTITY (``.name``) must not depend on that argument, so ``register`` reads ``.name`` by
probing the factory with the registry's ``probe_arg`` — ``None`` for a zero-arg pool.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from finn.kernels.space import Backend


def make_registry(op_name: str, *, probe_arg=None):
    """Build a fresh registry for one op. Returns ``(register, build_pool,
    registered_names, unregister)`` closed over a private ordered dict.

    ``probe_arg`` is the value passed to a factory (a) once at registration to read its
    interface-independent ``.name``, and (b) as the default when ``build_pool`` is called
    with no argument. Leave it ``None`` for a plain zero-arg pool (the factory is then
    called with no arguments)."""
    registry: dict[str, Callable[..., Backend]] = {}

    def _invoke(factory, arg):
        # Support both zero-arg factories and single-arg (interface-threaded) factories.
        return factory() if _is_zero_arg(factory) else factory(arg)

    def register(factory: Callable[..., Backend]) -> Callable[..., Backend]:
        """Decorator: register a ``Backend`` factory by its bundle name. The factory is
        called once here (probed with ``probe_arg``) to read ``.name`` (bundles are cheap
        to build); it is re-invoked per :func:`build_pool` so each schema gets fresh
        axis/derived objects. Duplicate names are a registration error."""
        name = _invoke(factory, probe_arg).name
        if name in registry:
            raise ValueError(f"duplicate {op_name} implementation registered: {name!r}")
        registry[name] = factory
        return factory

    def build_pool(arg=probe_arg) -> tuple[Backend, ...]:
        """Instantiate every registered bundle (fresh objects), registration order. ``arg``
        is threaded to each single-arg factory (e.g. the parameter interface name);
        defaults to the registry's ``probe_arg``."""
        return tuple(_invoke(factory, arg) for factory in registry.values())

    def registered_names() -> tuple[str, ...]:
        return tuple(registry)

    def unregister(name: str) -> None:
        """Remove a registered bundle by name (no-op if absent). Mainly for tests
        that register a transient stub and must not leak it into other tests."""
        registry.pop(name, None)

    return register, build_pool, registered_names, unregister


def _is_zero_arg(factory) -> bool:
    """True if ``factory`` takes no required positional parameters (a plain ``() ->
    Backend`` bundle); False if it expects one (an interface-threaded bundle)."""
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return True
    return not any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        for p in sig.parameters.values()
    )
