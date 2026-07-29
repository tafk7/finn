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

**Factory arity is DECLARED, not sniffed.** A registry is either zero-arg (``() ->
Backend`` factories, the common compute pool) or INTERFACE-THREADED (``(iface) -> Backend``
factories) — every factory in one registry is the same kind. The kind is declared once at
construction by passing ``probe_arg``: leave it ``None`` for a zero-arg pool, or pass the
default interface (e.g. ``WEIGHTS``) for an interface-threaded pool. There is NO per-factory
signature inspection — the registry knows its own arity from that single declaration. The
``parameters`` pool is interface-threaded (each storage topology is built for a specific
parameter INTERFACE, its point keys interface-namespaced); the bundle IDENTITY (``.name``)
is interface-independent, so ``register`` probes with ``probe_arg`` to read it.
"""

from __future__ import annotations

from collections.abc import Callable

from .backend import Backend


def make_registry(op_name: str, *, probe_arg=None):
    """Build a fresh registry for one op. Returns ``(register, build_pool,
    registered_names, unregister)`` closed over a private ordered dict.

    ``probe_arg`` DECLARES the registry's arity: ``None`` (default) → a zero-arg pool whose
    factories are called ``factory()``; a non-``None`` value → an interface-threaded pool
    whose factories are called ``factory(arg)`` (``arg`` defaults to ``probe_arg``). It is
    also the value passed once at registration to read a bundle's interface-independent
    ``.name``."""
    registry: dict[str, Callable[..., Backend]] = {}
    interface_threaded = probe_arg is not None

    def _invoke(factory, arg):
        # Arity is a registry-level fact (declared via probe_arg), not a per-factory sniff.
        return factory(arg) if interface_threaded else factory()

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
        is threaded to each factory on an interface-threaded pool (e.g. the parameter
        interface name); defaults to the registry's ``probe_arg``."""
        return tuple(_invoke(factory, arg) for factory in registry.values())

    def registered_names() -> tuple[str, ...]:
        return tuple(registry)

    def unregister(name: str) -> None:
        """Remove a registered bundle by name (no-op if absent). Mainly for tests
        that register a transient stub and must not leak it into other tests."""
        registry.pop(name, None)

    return register, build_pool, registered_names, unregister
