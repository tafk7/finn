############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""The MVAU implementation registry — the mechanism that makes "add a backend = add
one file, edit nothing" literally true.

An implementation bundle lives in its own module and registers itself at import time
via the :func:`register` decorator on a zero-arg factory returning an
:class:`Implementation`. The package ``__init__`` imports the built-in bundle modules
(and could discover third-party ones); assembling the schema just reads whatever has
registered. Adding a backend never touches this file, ``shared.py``, or a sibling
bundle — the additive property the composability thesis claims, enforced by structure
rather than asserted in a comment.

Ordering: registration order is preserved, so the first-registered bundle is the pool
default (the root axis default). Built-ins register HLS first to match prior behaviour.
"""

from __future__ import annotations

from collections.abc import Callable

from finn.design_space.space import Implementation

# Ordered name -> factory. Ordered so the first registrant is the pool default.
_REGISTRY: dict[str, Callable[[], Implementation]] = {}


def register(factory: Callable[[], Implementation]) -> Callable[[], Implementation]:
    """Decorator: register a zero-arg ``Implementation`` factory by its bundle name.

    The factory is called once here to read ``.name`` (bundles are cheap to build);
    it is re-invoked per :func:`build_pool` so each schema gets fresh axis/derived
    objects. Duplicate names are a registration error.
    """
    name = factory().name
    if name in _REGISTRY:
        raise ValueError(f"duplicate MVAU implementation registered: {name!r}")
    _REGISTRY[name] = factory
    return factory


def build_pool() -> tuple[Implementation, ...]:
    """Instantiate every registered bundle (fresh objects), in registration order."""
    return tuple(factory() for factory in _REGISTRY.values())


def registered_names() -> tuple[str, ...]:
    return tuple(_REGISTRY)
