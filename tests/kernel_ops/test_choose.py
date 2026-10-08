# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ExploreKernelChoices with a rank-style policy (``Ranked``): every open choice of a
model's KernelOps, committed by a policy that only ranks what the engine says is viable,
persisted on the owning nodes.

On the Chain (``kernels.chain``) as a model (MatMul, Thresholding, MatMul) for Ultra96 without
a shell: no UltraRAM, a doubled clock (nothing states it away).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from kernels.helpers import FULL_DSP48E2, Lanes
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.space import inspection
from finn.custom_op.kernels.base import KernelOpError, kernel_op, write_target
from finn.custom_op.kernels.shell import shell_root
from finn.kernels.configure import undecided
from finn.kernels.explore import Choice, ExploreError, Ranked, RankPolicy
from finn.kernels.target import Target
from finn.transformation.kernels import (
    ExploreKernelChoices,
    InferKernelTensors,
    ToKernelOps,
    completion,
)
from finn.transformation.kernels.package import configured_root
from kernel_ops.models import TARGET, chain_source

URAM = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP48E2, shell="ip")


def kernel_model(target: Target = TARGET) -> ModelWrapper:
    model = chain_source().transform(InferShapes()).transform(ToKernelOps(TARGET))
    write_target(model, target)
    return model.transform(InferKernelTensors())


def ranked(policy: RankPolicy) -> ExploreKernelChoices:
    return ExploreKernelChoices([Ranked(policy)])


def offer(key: str, cases: tuple[object, ...], refused: dict[str, str]) -> Choice:
    """An open Decision with ``cases``, as the seam offers it."""
    return Choice(key, None, None, cases, False, refused)


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
        self.offered: list[Choice] = []

    def rank(self, choice: Choice) -> Sequence[object]:
        self.offered.append(choice)
        return self.inner.rank(choice)  # type: ignore[attr-defined, no-any-return]


class Last:
    """Prefers each Decision's last viable case (of memories only, with ``memories``)."""

    def __init__(self, memories: bool = False) -> None:
        self.memories = memories

    def rank(self, choice: Choice) -> Sequence[object]:
        if self.memories and not choice.key.endswith("ram_style"):
            return choice.cases or ()
        return tuple(reversed(choice.cases or ()))


def test_the_test_policy_folds_to_its_lanes_or_the_largest_factor() -> None:
    policy = Lanes(4)
    fold = Choice("first.compute.packed.pe", None, None, (1, 2, 4, 8), True)
    assert list(policy.rank(fold)) == [4, 8, 2, 1]
    odd = Choice("x.simd", None, None, (1, 3), True)
    assert list(policy.rank(odd)) == [3, 1] and list(Lanes().rank(odd)) == [1, 3]
    memory = offer("w.source.memstream.ram_style", ("auto", "block"), {})
    assert list(policy.rank(memory)) == ["auto", "block"]


def test_the_packed_reducer_is_offered_open_its_adder_tree_first() -> None:
    policy = Recording(Lanes())
    model = kernel_model().transform(ranked(policy))
    offered = {choice.key: choice.cases or () for choice in policy.offered}
    reducers = {
        key: cases for key, cases in offered.items() if key.endswith("compute.packed.reducer")
    }
    assert reducers and all(cases == ("tree", "compressor") for cases in reducers.values())
    saved = choices(model)
    assert saved["first"]["compute.packed.reducer"] == "tree"


def test_every_open_choice_is_committed_and_the_model_replays_it(tmp_path: Path) -> None:
    model = kernel_model().transform(ranked(Lanes(2)))
    saved = choices(model)
    assert saved["first"]["compute.packed.pe"] == 2
    assert saved["activate"]["pe"] == 2
    model.save(tmp_path / "chosen.onnx")
    again = ModelWrapper(str(tmp_path / "chosen.onnx"))
    root = shell_root(again, again.graph.node)
    assert undecided(root.point, "*") == [] and not root.dropped
    assert inspection.viable(root.point) == ()
    # Nothing is left to choose, so a second pass commits nothing.
    assert choices(again.transform(ranked(Lanes()))) == saved


def test_the_policy_is_offered_viable_cases_only_and_no_forced_decision() -> None:
    policy = Recording(Lanes())
    model = kernel_model().transform(ranked(policy))
    offered = {choice.key: choice.cases or () for choice in policy.offered}
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
    model = kernel_model().transform(ranked(Last(memories=True)))
    assert choices(model)["first"]["w.source.memstream.ram_style"] not in ("ultra", "auto")
    uram = kernel_model(URAM).transform(ranked(Last(memories=True)))
    assert choices(uram)["first"]["w.source.memstream.ram_style"] == "ultra"


def test_a_required_choice_left_open_is_refused_by_name_where_it_is_built() -> None:
    # Preferring the FIFO transport opens its depth, a domain known by membership only
    # and required: no completion takes a case of it. The exploration leaves it open
    # and reports it; hardware generation refuses it, named.
    explore = ExploreKernelChoices([Ranked(Last())])
    model = kernel_model().transform(explore)
    assert explore.explored is not None
    left = explore.explored.report["completion"]["open"]
    assert left and all(name.endswith("transport.fifo.buffer.depth (required)") for name in left)
    with pytest.raises(KernelOpError, match=r"to choose before packaging: .*depth \(required\)"):
        configured_root(model, "the chain")
    # The debug placeholder takes the first case of a required choice, and refuses one
    # known by membership only, named.
    with pytest.raises(KernelOpError, match=r"no case to take .*transport.fifo.buffer.depth"):
        configured_root(model, "the chain", completion("placeholder"))


def test_a_policy_ranking_a_case_that_is_not_viable_is_refused() -> None:
    class Ultra:
        def rank(self, choice: Choice) -> Sequence[object]:
            if choice.key.endswith("memstream.ram_style"):
                return ("ultra",)
            return choice.cases or ()

    with pytest.raises(ExploreError, match="ram_style: the policy ranked \\['ultra'\\]"):
        kernel_model().transform(ranked(Ultra()))


class PreferUltra:
    """Two lanes, with ``ultra`` first wherever it is viable. Two lanes leave each
    MatMul's reduction two beats, which its activation channel's adapter frames (a
    frame of one beat needs none: its marker is tied high)."""

    def rank(self, choice: Choice) -> Sequence[object]:
        return sorted(Lanes(2).rank(choice), key=lambda case: case != "ultra")


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
    ultra96 = kernel_model().transform(ranked(policy))
    offered = {choice.key: choice.cases or () for choice in policy.offered}
    adapters = [cases for key, cases in offered.items() if ".adapter." in key]
    assert adapters and all("ultra" not in cases for cases in adapters)
    memories = adapter_memories(ultra96)
    assert memories and "ultra" not in memories.values()
    # With UltraRAM it is viable, and the policy gets what it prefers.
    uram = kernel_model(URAM).transform(ranked(PreferUltra()))
    assert set(adapter_memories(uram).values()) == {"ultra"}
