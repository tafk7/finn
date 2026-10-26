# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle's TFC target (``tfc``): TFC's partition (``support.tfc``) after
the cut, and after each of seven strategy lists explored on the cut's body
(``SEQUENCES``), each its partition's class as the graph layer binds it in its shell
(``in_shell``), configured with its stored choices; and the oracle's tool with every target,
TFC's too:

    python -m support.space_oracle_tfc dump tfc OUT.json [--implementation M:A]
    python -m support.space_oracle_tfc walk tfc --seeds A-B --steps N --against M:A

(``diff`` and the other targets as ``python -m support.space_oracle`` runs them.) It takes
no ``--permuted``: this module imports FINN's kernels and flow before its tool could set the
numbering, which is read when a model is made.

Outside the ``space_oracle`` package and the layer table because it reads FINN's builder
(``support.tfc``): the XSim groups are keyed by the code their tests import, every module
the layer table names included (``scripts/emitted_text.py``), and a kernel group must not be
keyed to the builder. Like ``support.tfc``, it is checked by the gates' ruff and mypy only.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper

from finn.core.search import Hold, Ranked
from finn.custom_op.kernels.shell import in_shell
from finn.kernels.explore import (
    Defaults,
    KernelExplorer,
    MaxThroughput,
    SizeFifos,
    TargetThroughput,
)
from finn.transformation.kernels import explore_kernel_choices
from support.space_oracle.targets import Station, _middle, _partition
from support.tfc import LANES, cut, streamlined

PINS: Mapping[str, Mapping[str, object]] = {
    "MatMul_0": {"pe": 4, "simd": 49},
    "MatMul_2": {"pe": 2},
}
"""The holds of ``pinned_then_target``, by node and key."""

SEQUENCES: Mapping[str, Callable[[], list[KernelExplorer]]] = {
    "ranked": lambda: [Ranked(LANES)],
    "target_throughput": lambda: [TargetThroughput(1_000_000)],
    "target_throughput_then_size_fifos": lambda: [TargetThroughput(1_000_000), SizeFifos()],
    "max_throughput": lambda: [MaxThroughput({"lut": 0.2})],
    "max_throughput_then_size_fifos": lambda: [MaxThroughput({"lut": 0.2}), SizeFifos()],
    "pinned_then_target": lambda: [Hold(PINS), TargetThroughput(3_000_000)],
    "defaults": lambda: [TargetThroughput(1_000_000), Defaults()],
}
"""TFC's seven strategy sequences, with the lanes, throughput targets, budget and pins an
earlier probe explored TFC with to compare what two checkouts choose, each explored on the
cut's body, one list on one seam; ``defaults`` is that probe's ``rebaseline``,
whose strategy is now ``Defaults``."""


def _tfc_partition(label: str, body: ModelWrapper) -> Station:
    nodes = list(body.graph.node)
    matmuls = [node.name for node in nodes if "MatMul" in node.op_type]
    bound = in_shell(body, nodes)
    return _partition(f"tfc-{label}", bound, len(nodes), _middle(matmuls))


def tfc(work: Path | None = None) -> list[Station]:
    """TFC's partition after the cut, and after each strategy sequence, built in ``work``
    (a new directory under ``FINN_BUILD_DIR`` by default)."""
    if work is None:
        work = Path(
            tempfile.mkdtemp(prefix="space-oracle-tfc-", dir=os.environ.get("FINN_BUILD_DIR"))
        )
    source = streamlined(work / "tfc")
    _, body, _ = cut(source, work / "cut")
    saved = str(work / "body.onnx")
    body.save(saved)
    stations = [_tfc_partition("cut", ModelWrapper(saved))]
    for name, sequence in SEQUENCES.items():
        model = ModelWrapper(saved)
        explore_kernel_choices(model, sequence())
        stations.append(_tfc_partition(name, model))
    return stations


__all__ = ["PINS", "SEQUENCES", "tfc"]


if __name__ == "__main__":
    import sys

    from support.space_oracle.targets import TARGETS
    from support.space_oracle.tool import main

    if "--permuted" in sys.argv[1:]:
        sys.exit("support.space_oracle_tfc takes no --permuted (see its docstring)")
    sys.exit(main(targets={**TARGETS, "tfc": tfc}))
