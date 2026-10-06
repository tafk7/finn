# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""CommitKernelChoices: every open choice of a model's KernelOps, committed by a policy
that only ranks what the engine says is viable, persisted on the owning nodes.

On the Chain (``kernels.chain``) as a model (MatMul, Thresholding, MatMul) for Ultra96 without
a shell: no UltraRAM, a doubled clock (nothing states it away).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from kernels.helpers import FULL_DSP48E2
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.space import inspection
from finn.custom_op.kernels.base import KernelOpError, kernel_op, write_target
from finn.custom_op.kernels.partition import partition_root
from finn.kernels.configure import undecided
from finn.kernels.target import Target
from finn.transformation.kernels import (
    CommitKernelChoices,
    InferKernelTensors,
    PlaceholderPolicy,
    ToKernelOps,
)
from kernel_ops.models import TARGET, chain_source

URAM = Target("a part with UltraRAM it initializes", FULL_DSP48E2)


def kernel_model(target: Target = TARGET) -> ModelWrapper:
    model = chain_source().transform(InferShapes()).transform(ToKernelOps(TARGET))
    write_target(model, target)
    return model.transform(InferKernelTensors())


def choices(model: ModelWrapper) -> dict[str, dict[str, object]]:
    return {
        node.name: kernel_op(model, node).choices()
        for node in model.graph.node
        if node.domain == "finn.custom_op.kernels"
    }


class Recording:
    """Ranks as ``inner`` does, keeping every choice it was offered."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.offered: list[inspection.Viable] = []

    def rank(self, choice: inspection.Viable) -> Sequence[object]:
        self.offered.append(choice)
        return self.inner.rank(choice)  # type: ignore[attr-defined, no-any-return]


class Last:
    """Prefers each Decision's last viable case (of memories only, with ``memories``)."""

    def __init__(self, memories: bool = False) -> None:
        self.memories = memories

    def rank(self, choice: inspection.Viable) -> Sequence[object]:
        if self.memories and not choice.key.endswith("ram_style"):
            return choice.cases
        return tuple(reversed(choice.cases))


def test_the_placeholder_folds_to_its_lanes_or_the_largest_factor() -> None:
    policy = PlaceholderPolicy(lanes=4)
    fold = inspection.Viable("first.compute.packed.pe", (1, 2, 4, 8), {})
    assert list(policy.rank(fold)) == [4, 8, 2, 1]
    assert list(policy.rank(inspection.Viable("x.simd", (1, 3), {}))) == [3, 1]
    memory = inspection.Viable("w.source.memstream.ram_style", ("auto", "block"), {})
    assert list(policy.rank(memory)) == ["auto", "block"]


def test_the_placeholder_states_its_preference_by_key_not_by_domain_order() -> None:
    policy = PlaceholderPolicy()
    reducer = inspection.Viable("first.compute.packed.reducer", ("compressor", "tree"), {})
    assert list(policy.rank(reducer)) == ["tree", "compressor"]
    # A preferred case that is not viable is not offered; the rest keep their order.
    assert list(
        policy.rank(inspection.Viable("x.compute.packed.reducer", ("compressor",), {}))
    ) == ["compressor"]
    assert list(policy.rank(inspection.Viable("first.reducer", ("compressor", "tree"), {}))) == [
        "compressor",
        "tree",
    ]


def test_the_packed_reducer_is_offered_open_and_the_placeholder_commits_the_tree() -> None:
    policy = Recording(PlaceholderPolicy())
    model = kernel_model().transform(CommitKernelChoices(policy))
    offered = {choice.key: choice.cases for choice in policy.offered}
    reducers = {
        key: cases for key, cases in offered.items() if key.endswith("compute.packed.reducer")
    }
    assert reducers and all(set(cases) == {"compressor", "tree"} for cases in reducers.values())
    saved = choices(model)
    assert saved["first"]["compute.packed.reducer"] == "tree"


def test_every_open_choice_is_committed_and_the_model_replays_it(tmp_path: Path) -> None:
    model = kernel_model().transform(CommitKernelChoices(PlaceholderPolicy(lanes=2)))
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 2
    assert saved["activate"]["pe"] == 2
    model.save(tmp_path / "chosen.onnx")
    again = ModelWrapper(str(tmp_path / "chosen.onnx"))
    root = partition_root(again, again.graph.node)
    assert undecided(root.point, "*") == [] and root.dropped == ()
    assert inspection.viable(root.point) == ()
    # Nothing is left to choose, so a second pass commits nothing.
    assert choices(again.transform(CommitKernelChoices(PlaceholderPolicy()))) == saved


def test_the_policy_is_offered_viable_cases_only_and_no_forced_decision() -> None:
    policy = Recording(PlaceholderPolicy())
    model = kernel_model().transform(CommitKernelChoices(policy))
    offered = {choice.key: choice.cases for choice in policy.offered}
    assert offered and all(len(cases) > 1 for cases in offered.values())
    # Ultra96 has no UltraRAM: ultra is never offered; DSP48E2 forces the packed core.
    memories = [cases for key, cases in offered.items() if key.endswith("memstream.ram_style")]
    assert memories and all("ultra" not in cases for cases in memories)
    assert not any(key.endswith(".compute") for key in offered)
    # Forced Decisions are derived on every read, never stored.
    stored = {key for node in choices(model).values() for key in node}
    assert "compute" not in stored and "w.source" not in stored
    assert "ultra_stages" not in stored  # one viable count without UltraRAM


def test_a_refused_case_is_never_picked_even_when_preferred() -> None:
    model = kernel_model().transform(CommitKernelChoices(Last(memories=True)))
    assert choices(model)["first"]["w.source.memstream.ram_style"] not in ("ultra", "auto")
    uram = kernel_model(URAM).transform(CommitKernelChoices(Last(memories=True)))
    assert choices(uram)["first"]["w.source.memstream.ram_style"] == "ultra"


def test_a_choice_the_engine_cannot_enumerate_is_refused_by_name() -> None:
    # Preferring the FIFO transport opens its depth, a domain known by membership only.
    with pytest.raises(KernelOpError, match="not enumerable.*transport.fifo.buffer.depth"):
        kernel_model().transform(CommitKernelChoices(Last()))


def test_a_policy_ranking_a_case_that_is_not_viable_is_refused() -> None:
    class Ultra:
        def rank(self, choice: inspection.Viable) -> Sequence[object]:
            if choice.key.endswith("memstream.ram_style"):
                return ("ultra",)
            return choice.cases

    with pytest.raises(KernelOpError, match="ram_style: the policy ranked \\['ultra'\\]"):
        kernel_model().transform(CommitKernelChoices(Ultra()))


class PreferUltra:
    """The placeholder's ranking, with ``ultra`` first wherever it is viable."""

    def rank(self, choice: inspection.Viable) -> Sequence[object]:
        return sorted(PlaceholderPolicy().rank(choice), key=lambda case: case != "ultra")


def adapter_memories(model: ModelWrapper) -> dict[str, object]:
    return {
        f"{node}.{key}": value
        for node, saved in choices(model).items()
        for key, value in saved.items()
        if ".adapter." in key and key.endswith(".ram_style")
    }


def test_an_adapter_memory_is_never_ultra_without_ultraram() -> None:
    # Ultra96 has no UltraRAM: an adapter's input_gen is never offered ultra, so a
    # policy that prefers it does not get it.
    policy = Recording(PreferUltra())
    ultra96 = kernel_model().transform(CommitKernelChoices(policy))
    offered = {choice.key: choice.cases for choice in policy.offered}
    adapters = [cases for key, cases in offered.items() if ".adapter." in key]
    assert adapters and all("ultra" not in cases for cases in adapters)
    memories = adapter_memories(ultra96)
    assert memories and "ultra" not in memories.values()
    # With UltraRAM it is viable, and the policy gets what it prefers.
    uram = kernel_model(URAM).transform(CommitKernelChoices(PreferUltra()))
    assert set(adapter_memories(uram).values()) == {"ultra"}
