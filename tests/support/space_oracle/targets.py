# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The design spaces the oracle compares, by target name (``TARGETS``), each a list of
stations: a design space, opened by today's API, and the change made on it.

- ``toys``: FINN's own toy Spaces (``toys.TOYS``: the house, the pipeline, a chain of
  stages, the boundary of a formal read from outside its scope, and the three shapes of
  an optional fact left out), each opened and changed once;
- ``chain-8``, ``chain-64``: stage 7's chain of 8 or 64 kernels (``support.one_space``),
  configured with what conversion and the cut stored (``commit``), then the middle
  MatMul's ``compute.packed.pe`` set to 2;
- ``tfc``: TFC's partition (``support.tfc``) after the cut, and after each of seven
  strategy sequences explored on the cut's body (``SEQUENCES``), each as one Space
  (``one_space``) configured with its stored choices.

Possible values are asked of every choice up to 8 kernels, and of the middle MatMul's
choices beyond. A target is made in the process that compares it: set
``FINN_SPACE_PERMUTE=1`` before it is made to compare the permuted numbering.

This module reads FINN's kernels and flow; the rest of the package is engine-only.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper

from finn.core.space import Space, design_space, inspection
from finn.kernels.configure import commit
from finn.kernels.explore import (
    Defaults,
    Explorer,
    Hold,
    MaxThroughput,
    Ranked,
    SizeFifos,
    TargetThroughput,
)
from finn.transformation.kernels import ExploreKernelChoices
from support.one_space import chain_space, one_space
from support.space_oracle import Implementation, forms
from support.space_oracle.toys import TOYS
from support.tfc import LANES, cut, streamlined

WIDEST = 8
"""Up to this many kernels, every choice's possible values are asked; beyond, the middle
MatMul's."""


@dataclass(frozen=True)
class Station:
    """One design space compared (``make`` opens it) and the change made on it (``change``,
    by key, or ``None``). ``viable``: the choices whose possible values are asked (every
    one: ``forms.ALL``); ``facts``: values a walk may give each root fact, by key;
    ``stored``: the choices it was configured with, by key."""

    label: str
    make: Callable[[], Space]
    change: Mapping[str, object] | None = None
    viable: Sequence[str] | None = forms.ALL
    facts: Mapping[str, Sequence[object]] = field(default_factory=dict)
    stored: Mapping[str, object] = field(default_factory=dict)


def toys() -> list[Station]:
    """FINN's own toy Spaces (``toys.TOYS``), each opened with nothing stored and changed
    once."""
    return [
        Station(name, toy.make, dict(toy.change), facts=toy.facts) for name, toy in TOYS.items()
    ]


def _middle(names: Sequence[str]) -> str:
    """The middle of the MatMuls, in graph order."""
    return names[len(names) // 2]


def chain(kernels: int) -> list[Station]:
    """Stage 7's chain of ``kernels`` kernels, configured as conversion and the cut store
    it, then the middle MatMul's ``compute.packed.pe`` set to 2."""
    whole, stored = chain_space(kernels // 2)
    middle = _middle([f"mm{layer}" for layer in range(kernels // 2)])
    return [_partition(f"chain-{kernels}", whole, stored, kernels, middle, change=True)]


def _partition(
    label: str,
    whole: type[Space],
    stored: Mapping[str, object],
    kernels: int,
    middle: str,
    *,
    change: bool = False,
) -> Station:
    def make() -> Space:
        return commit(design_space(whole()), stored)

    viable: Sequence[str] | None = forms.ALL
    if kernels > WIDEST:
        prefix = f"{middle}."
        viable = [item.key for item in inspection.decisions(make()) if item.key.startswith(prefix)]
    return Station(
        label,
        make,
        {f"{middle}.compute.packed.pe": 2} if change else None,
        viable,
        stored=dict(stored),
    )


PINS: Mapping[str, Mapping[str, object]] = {
    "MatMul_0": {"compute.packed.pe": 4, "compute.packed.simd": 49},
    "MatMul_2": {"compute.packed.pe": 2},
}
"""The holds of ``pinned_then_target``, by node and key."""

SEQUENCES: Mapping[str, Callable[[], list[Explorer]]] = {
    "ranked": lambda: [Ranked(LANES)],
    "target_throughput": lambda: [TargetThroughput(1_000_000)],
    "target_throughput_then_size_fifos": lambda: [TargetThroughput(1_000_000), SizeFifos()],
    "max_throughput": lambda: [MaxThroughput({"lut": 0.2})],
    "max_throughput_then_size_fifos": lambda: [MaxThroughput({"lut": 0.2}), SizeFifos()],
    "pinned_then_target": lambda: [Hold(PINS), TargetThroughput(3_000_000)],
    "defaults": lambda: [TargetThroughput(1_000_000), Defaults()],
}
"""TFC's seven strategy sequences (the identity probes', corpus
``records/bounded-change-2026-10-16/probes/identity/strategies2.py``), each explored on the
cut's body, one strategy after another; ``defaults`` is the probes' ``rebaseline``,
whose strategy is now ``Defaults``."""


def _tfc_partition(label: str, body: ModelWrapper) -> Station:
    nodes = list(body.graph.node)
    whole, stored = one_space(body, f"tfc_{label}")
    matmuls = [node.name for node in nodes if "MatMul" in node.op_type]
    return _partition(f"tfc-{label}", whole, stored, len(nodes), _middle(matmuls))


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
        for strategy in sequence():
            model = model.transform(ExploreKernelChoices(strategy))
        stations.append(_tfc_partition(name, model))
    return stations


TARGETS: Mapping[str, Callable[[], list[Station]]] = {
    "toys": toys,
    "chain-8": lambda: chain(8),
    "chain-64": lambda: chain(64),
    "tfc": tfc,
}
"""Every target, by name: each makes its stations when called."""


def dumped(station: Station, implementation: Implementation) -> forms.Form:
    """``station`` as ``implementation`` answers it, in its canonical form: the stored
    choices it was configured with, the design space, and its change (the request, the
    report and the successor)."""
    answers = implementation.open(station.make())
    found: dict[str, forms.Form] = {
        "stored": {key: forms.value(value) for key, value in sorted(station.stored.items())},
        "space": forms.space(answers, station.viable),
    }
    if station.change is not None:
        changed = answers.change(station.change, ())
        found["change"] = {
            "request": {key: forms.value(value) for key, value in station.change.items()},
            "report": forms.report(changed.report),
            "after": None if changed.after is None else forms.space(changed.after, station.viable),
        }
    return found


def stations(name: str) -> list[Station]:
    """The stations of target ``name``."""
    if name not in TARGETS:
        raise ValueError(f"{name!r}: a target is one of {', '.join(TARGETS)}")
    return TARGETS[name]()


__all__ = [
    "PINS",
    "SEQUENCES",
    "TARGETS",
    "WIDEST",
    "Station",
    "chain",
    "dumped",
    "stations",
    "tfc",
    "toys",
]
