# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Exploring a model's open kernel choices through the DSE seam, and saving them.

``ExploreKernelChoices(strategies)`` builds the partition root of the model's
KernelOps once (``partition_root``: their kernels and the channels between them,
the nodes' saved choices replayed, a stale one dropped with why), runs the
strategies in order through one ``Seam`` (``finn.kernels.explore``), each from the
point the one before returned, and then:

- refuses an open Decision no strategy chose, by name (one known by its domain's
  membership only, a FIFO's depth, flagged), and a point a member refuses
  (``Seam.refusals``);
- persists the point's commitments on the nodes that own them, each node's whole
  (``persist``): a saved choice is never changed (an explorer fills open choices
  only), and a stale one is cleared;
- keeps what it found (``explored``): the configured point, its cost and the
  report (the strategies, each with the choices it committed, attempts and time;
  every committed choice by its owner, with the strategy that made it, ``saved``
  for one the model held before; whether FIFOs were sized; the dropped choices with
  why, per member cycles and buffering, the bottleneck), so an outer search can
  compare.

``fresh`` clears the nodes' choices before the root is built, so the strategies
explore from scratch; otherwise a saved choice is pinned and an exploration
resumes from what was saved.

A strategy is written as a spec, ``{"strategy": name, **parameters}``
(``strategy(spec)``, the names ``KERNEL_STRATEGIES``), as a build configuration
lists them: ``[{"strategy": "target_throughput", "fps": 1000000}, {"strategy":
"size_fifos"}, {"strategy": "placeholder"}]``. A list runs as written: nothing is
appended, and nothing is read from anywhere else.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.base import KernelOpError, kernel_op, read_target
from finn.custom_op.kernels.partition import partition_root, persist
from finn.kernels.explore import (
    Bottleneck,
    Cost,
    Explorer,
    Pinned,
    Placeholder,
    Seam,
    SizeFifos,
    TargetThroughput,
)
from finn.transformation.fpgadataflow.kernel_partitions import KERNEL_OPS_DOMAIN

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

KERNEL_STRATEGIES: Mapping[str, Callable[..., Explorer]] = {
    "pinned": Pinned,
    "target_throughput": TargetThroughput,
    "size_fifos": SizeFifos,
    "placeholder": Placeholder,
}
"""The strategies a spec names, each made from the spec's other keys: ``pinned``
(``path``), ``target_throughput`` (``fps``, ``relax``), ``size_fifos`` (``method``,
``margin``, ``ram_style``, ``frames``), ``placeholder`` (``lanes``)."""


def strategy(spec: Mapping[str, Any]) -> Explorer:
    """The strategy ``spec`` writes: ``{"strategy": name, **parameters}``."""
    parameters = dict(spec)
    name = parameters.pop("strategy", None)
    if name not in KERNEL_STRATEGIES:
        raise ValueError(
            f"{dict(spec)} names no kernel strategy (one of {sorted(KERNEL_STRATEGIES)})"
        )
    try:
        return KERNEL_STRATEGIES[name](**parameters)
    except TypeError as error:
        raise ValueError(f"{dict(spec)}: {error}") from error


@dataclass(frozen=True)
class Explored:
    """An exploration's result: the configured point of the partition root, its cost,
    and the report (JSON values)."""

    point: Any
    cost: Cost
    report: Mapping[str, Any]


def _cost_report(seam: Seam, cost: Cost) -> dict[str, object]:
    bottleneck = cost.bottleneck
    return {
        "members": {
            name: {"cycles": cost.cycles.get(name), "buffering": cost.buffering.get(name)}
            for name in seam.members
        },
        "bottleneck": None
        if bottleneck is None
        else {"members": list(bottleneck.members), "cycles": bottleneck.cycles},
        "buffering": sum(cost.buffering.values()),
    }


def _choices_by_owner(seam: Seam, made_by: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Each committed choice, by the node and attribute that persist it (as
    ``kernel_choices.json`` names it), with who made it."""
    found: dict[str, dict[str, str]] = {}
    for key, strategy_name in made_by.items():
        node, attribute = seam.owner(key) or ("", key)
        found.setdefault(node, {})[attribute] = strategy_name
    return found


def _fifos(strategies: Sequence[Explorer], explorers: Sequence[Mapping[str, Any]]) -> str:
    """Whether FIFOs were sized, said so that a chain without sizing does not read as
    sized: by ``size_fifos`` (the channels it sized), or not, and why."""
    sized = [
        report for explorer, report in zip(strategies, explorers) if isinstance(explorer, SizeFifos)
    ]
    if not sized:
        return "not sized (no size_fifos in the chain)"
    channels = sum(len(report["channels"]) for report in sized)
    if not channels:
        return "not sized: size_fifos found no open transport (each was saved before)"
    return f"sized by size_fifos: {channels} channels"


def explore_kernel_choices(
    model: ModelWrapper, strategies: Sequence[Explorer], *, fresh: bool = False
) -> Explored:
    """The model's KernelOps explored by ``strategies`` and their choices persisted; see
    the module docstring."""
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    if not nodes:
        raise KernelOpError("no KernelOp to explore")
    if fresh:
        for node in nodes:
            op = kernel_op(model, node)
            op.save(dict.fromkeys(op.choices()))
    started = time.perf_counter()
    root = partition_root(model, nodes)
    seam = Seam(root.members, root.owners, read_target(model).platform)
    point = root.point
    # Who made each choice: the model before the strategies, or the strategy that
    # committed it (the point it returned commits the choice, the one before did not).
    made_by = dict.fromkeys(seam.chosen(point), "saved")
    explorers: list[dict[str, object]] = []
    for explorer in strategies:
        attempts, began = seam.attempts, time.perf_counter()
        point = explorer.explore(seam, point)
        report = explorer.report()
        committed = [key for key in seam.chosen(point) if key not in made_by]
        made_by.update(dict.fromkeys(committed, str(report["strategy"])))
        explorers.append(
            {
                **report,
                "committed": len(committed),
                "attempts": seam.attempts - attempts,
                "seconds": round(time.perf_counter() - began, 3),
            }
        )
    left = seam.choices(point)
    if left:
        named = [
            choice.key + (" (known by membership only)" if choice.cases is None else "")
            for choice in left
        ]
        raise KernelOpError(f"open choices no strategy chose: {', '.join(named)}")
    refused = seam.refusals(point)
    if refused:
        raise KernelOpError(
            "the explored point is refused: "
            + "; ".join(f"{name}: {why}" for name, why in refused.items())
        )
    persist(model, root, point)
    cost = seam.cost(point)
    report = {
        "strategies": explorers,
        "choices": _choices_by_owner(seam, made_by),
        "fifos": _fifos(strategies, explorers),
        "fresh": fresh,
        "dropped": dict(root.dropped),
        **_cost_report(seam, cost),
        "seconds": round(time.perf_counter() - started, 3),
    }
    return Explored(point, cost, report)


def partition_bottleneck(model: ModelWrapper) -> Bottleneck | None:
    """The slowest members of a partition model of KernelOps and their cycles a frame,
    its saved choices replayed (None while a member's cycles wait on an open choice)."""
    root = partition_root(model, model.graph.node)
    seam = Seam(root.members, root.owners, read_target(model).platform)
    return seam.cost(root.point).bottleneck


class ExploreKernelChoices(Transformation):
    """Every open choice of the model's KernelOps explored by ``strategies``, in order,
    and saved; ``explored`` holds the result (``explore_kernel_choices``)."""

    def __init__(self, strategies: Sequence[Explorer], *, fresh: bool = False) -> None:
        super().__init__()
        self.strategies = tuple(strategies)
        self.fresh = fresh
        self.explored: Explored | None = None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.explored = explore_kernel_choices(model, self.strategies, fresh=self.fresh)
        return model, False


__all__ = [
    "KERNEL_STRATEGIES",
    "ExploreKernelChoices",
    "Explored",
    "explore_kernel_choices",
    "partition_bottleneck",
    "strategy",
]
