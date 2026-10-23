# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ExploreKernelChoices with a rank-style policy (``Ranked``): every open choice of a
model's KernelOps, one possible value or several, committed by a policy that only ranks
what the engine says is possible, persisted on their owners: a kernel's on its node, a
channel's on its tensor.

On the Chain (``support.chain``) as a model (MatMul, Thresholding, MatMul) for Ultra96 without
a shell: no UltraRAM, a doubled clock (nothing states it away).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from support.explore_models import choices, kernel_model
from support.helpers import FULL_DSP48E2, Lanes
from support.refusals import refused

from finn.core.graph.errors import KernelOpError
from finn.custom_op.kernels.network import network
from finn.custom_op.kernels.shell import configured_network
from finn.kernels.explore import (
    Choice,
    DebugPlaceholder,
    ExploreError,
    Ranked,
    RankPolicy,
    dotted,
)
from finn.platform.target import Target
from finn.transformation.kernels import (
    ExploreKernelChoices,
)

URAM = Target(part="a part with UltraRAM it initializes", platform=FULL_DSP48E2, shell="ip")


def ranked(policy: RankPolicy) -> ExploreKernelChoices:
    return ExploreKernelChoices(Ranked(policy))


def offer(key: str, cases: tuple[object, ...]) -> Choice:
    """An open choice with ``cases``, as the seam offers it (``key``: its owner and the
    key there, joined)."""
    owner, _, below = key.partition(".")
    return Choice((owner, below), None, cases, False)


class Recording:
    """Ranks as ``inner`` does, keeping every choice it was offered."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.offered: list[Choice] = []

    def rank(self, choice: Choice) -> Sequence[object]:
        self.offered.append(choice)
        return self.inner.rank(choice)  # type: ignore[attr-defined, no-any-return]


class Last:
    """Prefers each Decision's last viable case; with ``memories``, only a memory's, and
    its last explicit style (UltraRAM where viable), ``auto`` aside."""

    def __init__(self, memories: bool = False) -> None:
        self.memories = memories

    def rank(self, choice: Choice) -> Sequence[object]:
        cases = choice.cases or ()
        if not self.memories:
            return tuple(reversed(cases))
        if not choice.key[1].endswith("ram_style"):
            return cases
        return tuple(reversed([case for case in cases if case != "auto"]))


def test_the_test_policy_folds_to_its_lanes_or_the_largest_factor() -> None:
    policy = Lanes(4)
    fold = Choice(("first", "pe"), None, (1, 2, 4, 8), True)
    assert list(policy.rank(fold)) == [4, 8, 2, 1]
    odd = Choice(("x", "simd"), None, (1, 3), True)
    assert list(policy.rank(odd)) == [3, 1] and list(Lanes().rank(odd)) == [1, 3]
    memory = offer("w.source.ram_style", ("block", "auto"))
    assert list(policy.rank(memory)) == ["block", "auto"]


def test_the_packed_reducer_is_offered_open_its_adder_tree_first() -> None:
    policy = Recording(Lanes())
    model = kernel_model().transform(ranked(policy))
    offered = {dotted(choice.key): choice.cases or () for choice in policy.offered}
    reducers = {
        key: cases for key, cases in offered.items() if key.endswith("compute.packed.reducer")
    }
    assert reducers and all(cases == ("tree", "compressor") for cases in reducers.values())
    saved = choices(model)
    assert saved["first"]["compute.packed.reducer"] == "tree"


def test_every_open_choice_is_committed_and_the_model_replays_it(tmp_path: Path) -> None:
    model = kernel_model().transform(ranked(Lanes(2)))
    saved = choices(model)
    assert saved["first"]["pe"] == 2
    assert saved["activate"]["pe"] == 2
    model.save(tmp_path / "chosen.onnx")
    again = ModelWrapper(str(tmp_path / "chosen.onnx"))
    bound = network(again, again.graph.node)
    assert bound.undecided() == () and not bound.stale
    # A second pass reopens every choice and ranks it again: the same policy makes the
    # same choices, another replaces them.
    assert choices(again.transform(ranked(Lanes(2)))) == saved
    assert choices(again.transform(ranked(Lanes())))["first"]["pe"] == 1


def test_the_policy_is_offered_every_open_choice_its_possible_values_only() -> None:
    """Every open choice is offered, one possible value or several: nothing is set because
    one value is possible. Each is stored."""
    policy = Recording(Lanes())
    model = kernel_model().transform(ranked(policy))
    offered = {dotted(choice.key): choice.cases or () for choice in policy.offered}
    assert offered and all(cases for cases in offered.values())
    # Ultra96 has no UltraRAM: ultra is never offered; on DSP48E2 the packed core is the
    # one possible, and offered so.
    memories = [cases for key, cases in offered.items() if key.endswith("source.ram_style")]
    assert memories and all("ultra" not in cases for cases in memories)
    assert {key: cases for key, cases in offered.items() if key.endswith(".compute")} == {
        "first.compute": ("packed",),
        "second.compute": ("packed",),
    }
    # A weight's source is placed, no choice: never offered.
    assert "w1.source" not in offered
    # Stored, each of them, as any choice.
    stored = {key for node in choices(model).values() for key in node}
    assert {"compute", "source.ram_style", "adapter_ram_style", "ultra_stages"} <= stored


def test_a_refused_case_is_never_picked_even_when_preferred() -> None:
    model = kernel_model().transform(ranked(Last(memories=True)))
    assert choices(model)["w1"]["source.ram_style"] not in ("ultra", "auto")
    uram = kernel_model(URAM).transform(ranked(Last(memories=True)))
    assert choices(uram)["w1"]["source.ram_style"] == "ultra"


def test_a_required_choice_left_open_is_refused_by_name_where_it_is_built() -> None:
    # Preferring the FIFO transport opens its depth, a domain known by membership only
    # and required: no baseline takes a case of it. The exploration leaves it open
    # and reports it; emission completes nothing and refuses it, named, with what sets it.
    explore = ExploreKernelChoices(Ranked(Last()))
    model = kernel_model().transform(explore)
    assert explore.explored is not None
    left = explore.explored.report["open"]
    assert left and all(name.endswith("transport.fifo.depth (required)") for name in left)
    with refused("choices-open", error=KernelOpError, node="the chain") as open_choices:
        configured_network(model, "the chain")
    assert isinstance(open_choices.error, KernelOpError)
    assert all(key.endswith("transport.fifo.depth") for key in open_choices.error.keys)
    required = "(required: run step_size_fifos or step_placeholder_choices)"
    assert str(open_choices.error).count(required) == len(left)
    # The debug placeholder strategy takes the first case of a required choice, and
    # refuses one known by membership only, named.
    with refused("placeholder-no-case", error=ExploreError, message=r"transport\.fifo\.depth"):
        model.transform(ExploreKernelChoices(DebugPlaceholder()))


def test_a_policy_ranking_a_case_that_is_not_viable_is_refused() -> None:
    class Ultra:
        def rank(self, choice: Choice) -> Sequence[object]:
            if choice.key[1].endswith("source.ram_style"):
                return ("ultra",)
            return choice.cases or ()

    with refused(
        "policy-case-not-viable",
        error=ExploreError,
        message=r"ram_style: the policy ranked \['ultra'\]",
    ):
        kernel_model().transform(ranked(Ultra()))


class PreferUltra:
    """Two lanes, with ``ultra`` first wherever it is viable. Two lanes leave each
    MatMul's reduction two beats, which its activation channel's adapter frames (a
    frame of one beat needs none: its marker is tied high)."""

    def rank(self, choice: Choice) -> Sequence[object]:
        return sorted(Lanes(2).rank(choice), key=lambda case: case != "ultra")


def adapter_memories(model: ModelWrapper) -> dict[str, object]:
    return {
        f"{owner}.{key}": value
        for owner, saved in choices(model).items()
        for key, value in saved.items()
        if key == "adapter_ram_style"
    }


def test_an_adapter_memory_is_never_ultra_without_ultraram() -> None:
    # Ultra96 has no UltraRAM: an adapter's memory (its channel's one style for its
    # input_gen stages) is never offered ultra, so a policy that prefers it does not get it.
    policy = Recording(PreferUltra())
    ultra96 = kernel_model().transform(ranked(policy))
    offered = {dotted(choice.key): choice.cases or () for choice in policy.offered}
    adapters = [cases for key, cases in offered.items() if key.endswith(".adapter_ram_style")]
    assert adapters and all("ultra" not in cases for cases in adapters)
    memories = adapter_memories(ultra96)
    assert memories and "ultra" not in memories.values()
    # With UltraRAM it is viable, and the policy gets what it prefers.
    uram = kernel_model(URAM).transform(ranked(PreferUltra()))
    assert set(adapter_memories(uram).values()) == {"ultra"}
