# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's HLS products: every HLS request among its leaves, synthesized for a part.

``built_hls`` lists the module's HLS requests (``hls_requests``: each leaf whose
source is an ``HlsSource``, each request once), stages each
(``finn.kernels.artifacts.hls.stage_hls``, FinnLib as FINN resolves it), and takes
its product from the HLS cache, synthesizing what is missing
(``finn.util.hls.synthesize``), the requests in parallel, each its own tool
process. Each product's exported top is then checked against the leaf's declared
pins (``finn.kernels.artifacts.rtl.check_abi``): the declaration stays the ABI, and a
product that contradicts it is refused (``HlsAbiRefused``), as is one the checker
cannot read. What it returns is what ``emit_module`` takes as ``built``: each
request's exported Verilog by its top's name.

Every caller that emits a module calls it first: ``PackagePartition`` and
``ElaboratePartition`` with the partition's part, the harness with the part it
simulates (``finn.core.executors.xsim.rtl``). A module without HLS leaves runs nothing.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from finn import resources
from finn.kernels.artifacts.contributions import HlsSource
from finn.kernels.artifacts.hls import stage_hls
from finn.kernels.artifacts.module import Composed, Leaf, Module
from finn.kernels.artifacts.rtl import Declined, check_abi
from finn.util.hls import HlsProduct, synthesize
from finn.util.toolchain import Toolchain, machine_toolchain


class HlsAbiRefused(Exception):
    """An HLS product's top contradicts its leaf's declared pins, or cannot be read."""


def hls_requests(module: Module) -> tuple[tuple[Leaf, HlsSource], ...]:
    """Each HLS leaf of ``module`` with its request, each request once, in netlist order."""
    leaves: tuple[Leaf, ...] = (module,) if isinstance(module, Leaf) else ()
    if isinstance(module, Composed):
        leaves = tuple(leaf for _, leaf in module.fragment.instances)
    found: dict[str, tuple[Leaf, HlsSource]] = {}
    for leaf in leaves:
        for item in leaf.sources:
            if isinstance(item, HlsSource):
                found.setdefault(item.function, (leaf, item))
    return tuple(found.values())


def _verified(leaf: Leaf, product: HlsProduct) -> Path:
    files = sorted(path for path in product.verilog.iterdir() if path.suffix == ".v")
    checked = check_abi(leaf.abi.pins, files, leaf.name)
    if isinstance(checked, Declined):
        raise HlsAbiRefused(f"{leaf.name}: the RTL checker cannot read the product: {checked}")
    if checked:
        raise HlsAbiRefused(f"{leaf.name} refuses its declared pins: " + "; ".join(checked))
    return product.verilog


def built_hls(
    module: Module,
    part: str,
    *,
    toolchain: Toolchain | None = None,
    cache: Path | None = None,
    roots: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Each HLS request's exported Verilog in ``module``, by its top's name, synthesized for
    ``part`` unless cached (``cache``, ``$FINN_HOME/hls`` by default) and checked against
    its leaf's pins. ``roots`` resolves the headers' roots (FinnLib as FINN resolves it by
    default); ``toolchain`` runs HLS (the machine's by default)."""
    requests = hls_requests(module)
    if not requests:
        return {}
    toolchain = toolchain or machine_toolchain()
    roots = {"finnlib": resources.finnlib_root()} if roots is None else roots
    scratch: Path = resources.scratch()  # type: ignore[no-untyped-call]
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="hls_requests_") as staging:

        def build(item: tuple[Leaf, HlsSource]) -> tuple[str, Path]:
            leaf, request = item
            staged = stage_hls(request, Path(staging) / request.function, roots=roots)
            product = synthesize(
                staged.directory, part, name=request.function, toolchain=toolchain, cache=cache
            )
            return request.function, _verified(leaf, product)

        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            return dict(pool.map(build, requests))


__all__ = ["HlsAbiRefused", "built_hls", "hls_requests"]
