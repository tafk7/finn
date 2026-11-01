# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's name, two ways.

``live_name`` is the name a build gives the module now: the snapshot of the files it stages
read from FinnLib as FINN resolves it (``finn.resources.finnlib_root``,
``FINN_RESOURCES_FINNLIB`` overriding it), so it moves with the FinnLib checked out.
``fixed_name`` is the module's name over fixed stand-in bytes (``fixed_snapshot``), so it
moves with the module's value and the fingerprint's format only: a digest oracle states that
one, whichever FinnLib is checked out."""

from __future__ import annotations

from finn.kernels.artifacts.build import capture
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.module import Composed, Leaf, Module, Snapshot, module_name
from finn.resources import finnlib_root


def live_name(module: Module) -> str:
    """``module``'s name as a build names it from FinnLib now (the root
    ``finn.transformation.kernels.package.source_roots`` gives); no HLS leaf (nothing here
    synthesizes)."""
    return module_name(module, capture(module, roots={"finnlib": finnlib_root()}))


def fixed_snapshot(module: Module) -> Snapshot:
    """A snapshot of every file ``module`` stages, each copied source's bytes its root's name
    and path (``b"finnlib:rtl/shape/vpc.sv\\n"``), each HLS product its top alone, ``<top>.v``
    holding its name: no file is read."""
    leaves: tuple[Leaf, ...] = (module,) if isinstance(module, Leaf) else ()
    if isinstance(module, Composed):
        leaves = tuple(leaf for _, leaf in module.fragment.instances)
    copied: dict[tuple[str, str, bytes], None] = {}
    products: dict[str, tuple[tuple[str, bytes], ...]] = {}
    for leaf in leaves:
        for source in leaf.sources:
            if isinstance(source, HlsSource):
                top = source.function
                products[top] = ((f"{top}.v", top.encode()),)
            else:
                stand_in = f"{source.root}:{source.path}\n".encode()
                copied[(source.root, source.path, stand_in)] = None
    return Snapshot(tuple(copied), tuple(products.items()))


def fixed_name(module: Module) -> str:
    """``module``'s name over ``fixed_snapshot``."""
    return module_name(module, fixed_snapshot(module))
