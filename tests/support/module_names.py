# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's name, two ways.

``live_name`` is the name a build gives the module now: the snapshot of the files it stages
read from FinnLib as FINN resolves it (``finn.transformation.kernels.package.source_roots``,
``FINN_RESOURCES_FINNLIB`` overriding it), so it moves with the FinnLib checked out.
``fixed_name`` is the module's name over fixed stand-in bytes (``fixed_snapshot``), so it
moves with the module's value and the fingerprint's format only: a digest oracle states that
one, whichever FinnLib is checked out."""

from __future__ import annotations

from finn.kernels.artifacts.build import capture
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.module import Composed, Module, Snapshot, module_name
from finn.transformation.kernels.package import source_roots


def live_name(module: Module) -> str:
    """``module``'s name as a build names it from FinnLib now; no HLS leaf (nothing here
    synthesizes)."""
    return module_name(module, capture(module, roots=source_roots()))


def fixed_snapshot(module: Module) -> Snapshot:
    """A snapshot of every file ``module`` stages, each copied source's bytes its root's name
    and path (``b"finnlib:rtl/shape/vpc.sv\\n"``), each HLS product its top alone, ``<top>.v``
    holding its name: no file is read."""
    leaves = (
        (module,)
        if not isinstance(module, Composed)
        else (leaf for _, leaf in module.fragment.instances)
    )
    copied: dict[tuple[str, str], bytes] = {}
    products: dict[str, dict[str, bytes]] = {}
    for leaf in leaves:
        for source in leaf.sources:
            if isinstance(source, HlsSource):
                products[source.function] = {f"{source.function}.v": source.function.encode()}
            else:
                copied[(source.root, source.path)] = f"{source.root}:{source.path}\n".encode()
    return Snapshot(copied, products)


def fixed_name(module: Module) -> str:
    """``module``'s name over ``fixed_snapshot``."""
    return module_name(module, fixed_snapshot(module))
