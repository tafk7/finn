# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Committing a model's open kernel choices by a policy: the seam where DSE belongs.

``CommitKernelChoices(policy)`` builds the partition root of the model's
KernelOps (``partition_root``: their kernels and the streams between them) and
commits every Decision the configuration leaves open, the kernels' (folding,
memories, compute cores), the streams' (a ``source`` when several are viable,
its memories, adapter memories), by asking the policy. It persists them on the
nodes that own them (``save_partition_choices``, D8), so the model replays
them.

The policy only ranks. The engine decides what is viable
(``inspection.viable``): a forced Decision (one viable case) is never offered
and never stored, it is derived again on every read; a case the configuration
refuses (``ultra`` without UltraRAM) is never offered. The policy returns the
viable cases in its order of preference; the first the configuration accepts
is committed. A Decision with no viable case, or one whose cases the engine
cannot enumerate, is refused, named.

A preference between a Decision's cases is the policy's: the order of a
kernel's domain states none. ``PlaceholderPolicy`` is the only policy, and a
placeholder: it stands where a design space exploration will rank by cost
(gate A, G4), and nothing should come to rely on its choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

from qonnx.transformation.base import Transformation

from finn.core.space import inspection
from finn.custom_op.kernels.base import KernelOpError
from finn.custom_op.kernels.partition import partition_root, save_partition_choices
from finn.kernels.configure import commit, undecided

DOMAIN = "finn.custom_op.kernels"


class KernelChoicePolicy(Protocol):
    """Ranks an open Decision's viable cases, most preferred first."""

    def rank(self, choice: inspection.Viable) -> Sequence[object]: ...


class PlaceholderPolicy:
    """The DSE seam's placeholder (gate A, G4): deterministic, and not a design.

    It folds every PE and SIMD to ``lanes`` where that is viable, otherwise to the
    largest viable factor, ranks the cases it has a preference for first
    (``PREFERRED``), and otherwise takes the first viable case in its domain's
    order (``auto`` memories, nothing pumped, no AXI-Lite, a direct transport),
    which states no preference and is only deterministic. A design space
    exploration replaces it; it ranks no cost.
    """

    FOLDING = ("pe", "simd")
    #: Cases preferred, most first, by a Decision's key within its node, each with its reason.
    PREFERRED: Mapping[str, tuple[object, ...]] = MappingProxyType(
        {
            # The adder tree over the compressor: it meets timing with the larger margin,
            # and one synthesis of a packed dotp takes about 3 GB of memory rather than
            # about 37 GB. The compressor saves LUTs and registers at wide SIMD.
            "compute.packed.reducer": ("tree",),
        }
    )

    def __init__(self, lanes: int = 16) -> None:
        self.lanes = lanes

    def rank(self, choice: inspection.Viable) -> Sequence[object]:
        if choice.key.rsplit(".", 1)[-1] in self.FOLDING:
            factors = sorted((case for case in choice.cases if isinstance(case, int)), reverse=True)
            return sorted(factors, key=lambda factor: factor != self.lanes)
        preferred = next(
            (
                cases
                for key, cases in self.PREFERRED.items()
                if choice.key == key or choice.key.endswith(f".{key}")
            ),
            (),
        )
        first = [case for case in preferred if case in choice.cases]
        return [*first, *(case for case in choice.cases if case not in first)]


class CommitKernelChoices(Transformation):  # type: ignore[misc]
    """Every open choice of the model's KernelOps committed by ``policy``, and saved."""

    def __init__(self, policy: KernelChoicePolicy) -> None:
        super().__init__()
        self.policy = policy

    def apply(self, model: Any) -> tuple[Any, bool]:
        nodes = [node for node in model.graph.node if node.domain == DOMAIN]
        if not nodes:
            return model, False
        root = partition_root(model, nodes)
        point, chosen = root.point, dict[str, object]()
        while True:
            open_ = inspection.viable(point)
            if not open_:
                break
            choice = open_[0]
            point, chosen[choice.key] = self._commit(point, choice)
        left = undecided(point, "*")
        if left:
            raise KernelOpError(f"open choices no policy can rank (not enumerable): {left}")
        save_partition_choices(model, root, chosen)
        return model, False

    def _commit(self, point: Any, choice: inspection.Viable) -> tuple[Any, object]:
        if not choice.cases:
            detail = "; ".join(f"{case}: {why}" for case, why in choice.refused.items())
            raise KernelOpError(f"{choice.key}: no case is viable: {detail}")
        ranked = list(self.policy.rank(choice))
        offered = [case for case in ranked if case not in choice.cases]
        if offered or not ranked:
            raise KernelOpError(
                f"{choice.key}: the policy ranked {ranked}, "
                f"not among the viable {list(choice.cases)}"
            )
        refusals = []
        for case in ranked:
            try:
                return commit(point, {choice.key: case}), case
            except ValueError as error:
                refusals.append(f"{case!r}: {error}")
        raise KernelOpError(f"{choice.key}: no ranked case is accepted: " + "; ".join(refusals))


__all__ = ["CommitKernelChoices", "KernelChoicePolicy", "PlaceholderPolicy"]
