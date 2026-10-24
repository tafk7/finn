# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The seam's fold rule (``finn.kernels.explore.Seam.set``) on a partition's one design
space: a change refused where a stored channel choice stands in its way is made again
with it cleared, as one change, and the seam then sets each channel choice the change
cleared or brought into play by the channel's policy, the strategy's; a held one standing in
the way refuses the change (``held``); a kernel's stored choice standing in the way
refuses it, naming it, where clearing the channel's choices does not make it accepted.

No graph reaches the case: a fold's rules on a channel's stored choices (FIFO sizing's
depths are never refused by one) are met by neither target, so the case is composed by
hand (``finn.kernels.composer``), as a graph's partition is, one design space of its
class: the Chain's first MatMul and its thresholding (``support.chain``), the hidden
channel between them on the platform a ``Board`` kernel states, with UltraRAM or
without, by its own choice. Taking the UltraRAM away is the change that reaches the
hidden channel's FIFO, as a fold reaches a channel through its kernel's stream. A
channel's choice a fold brings into play is shown on the Chain's graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from support import chain
from support.chain import LEVELS, THRESHOLD_DTYPE, THRESHOLDS, W1, X_TENSOR, W
from support.configure import configured
from support.helpers import FULL_DSP48E2, keyed, with_defaults
from support.models import kernel_model
from support.refusals import refused

from finn.core.graph.annotations.platform import read_target
from finn.core.space import Decision, Space, derived, inspection
from finn.custom_op.kernels.network import persist
from finn.custom_op.kernels.shell import in_shell
from finn.kernels.composer import Facts, PartitionDeclaration, Port, Stated, channel
from finn.kernels.explore import Accepted, ExploreError, Hold, Refused, Seam, TargetCycles
from finn.kernels.matmul import KnownWeights, MatMulKernel
from finn.kernels.partition import Partition, partition
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.platform.capabilities import Platform
from finn.transformation.kernels import explore_kernel_choices


class Board(Space):
    """A member that states the hidden channel's platform: with UltraRAM or without, by
    its own choice."""

    uram: bool = Decision(values=(True, False))

    @derived
    def platform(self) -> Platform:
        return replace(FULL_DSP48E2, uram=self.uram, uram_init=self.uram)


def declared(*, adaptable: bool) -> PartitionDeclaration:
    """The Board, the Chain's first MatMul and its thresholding; the hidden channel on
    the Board's platform (admitting no adapter unless ``adaptable``)."""
    platform = FULL_DSP48E2
    return PartitionDeclaration(
        {
            "board": Facts(Board, {}),
            "first": Facts(
                MatMulKernel,
                {
                    "platform": platform,
                    "x_tensor": X_TENSOR,
                    "weights": KnownWeights(IntegerTensorValue.of(W1), W),
                },
            ),
            "activate": Facts(
                ThresholdingAxiKernel,
                {
                    "threshold_dtype": THRESHOLD_DTYPE,
                    "thresholds": IntegerTensorValue.of(THRESHOLDS),
                    "bias": 0,
                    "input_tensor": chain.HIDDEN_TENSOR,
                    "platform": platform,
                },
            ),
        },
        {
            "x": channel(
                Side.boundary("s_axis_0"), Port("first", "x"), tensor=X_TENSOR, platform=platform
            ),
            "w1": channel(
                Side.source(),
                Port("first", "w"),
                tensor=Stated("first", "carried_weights"),
                contents=Stated("first", "stored_weights"),
                platform=platform,
            ),
            "hidden": channel(
                Port("first", "y"),
                Port("activate", "input"),
                tensor=chain.HIDDEN_TENSOR,
                platform=Stated("board", "platform"),
                adaptable=adaptable,
            ),
            "levels": channel(
                Port("activate", "output"),
                Side.boundary("m_axis_0"),
                tensor=LEVELS,
                platform=platform,
            ),
        },
    )


def boarded(
    hidden: Mapping[str, object] = {}, *, folding: Mapping[str, object] = {}, adaptable: bool = True
) -> Partition:
    """The partition with UltraRAM, the thresholding at the Chain's PE and ``folding``,
    every other choice by its policy, as conversion and the cut leave it; then the hidden
    channel's ``hidden`` choices."""
    empty = partition(declared(adaptable=adaptable), name="board")
    chosen = empty.with_choices(keyed({"board.uram": True, "activate.pe": chain.PE, **folding}))
    return with_defaults(chosen).with_choices(keyed({f"hidden.{k}": v for k, v in hidden.items()}))


#: The hidden channel's FIFO in UltraRAM.
ULTRA = {
    "transport": "fifo",
    "transport.fifo.depth": 16,
    "transport.fifo.buffer.ram_style": "ultra",
}
#: A FIFO depth an UltraRAM FIFO holds and a block RAM one cannot address: ``auto`` takes
#: UltraRAM where the platform has it, block RAM where not (``fifo-capacity``).
DEEP = 3 * 2**30 + 10

NO_URAM = keyed({"board.uram": False})
STYLE = ("hidden", "transport.fifo.buffer.ram_style")
DEPTH = ("hidden", "transport.fifo.depth")


def test_a_change_a_stored_channel_choice_stands_in_the_way_of_is_made_with_it_cleared() -> None:
    """Taking the UltraRAM away, asked as it is, makes the FIFO's rule that UltraRAM needs
    the platform's newly false, its stored style standing in the way: the change is
    refused naming it. Through the seam, it is made again with the style cleared, and the
    channel's policy sets the style again (first valid: ``auto``), each named as the
    strategy's (``Accepted.cleared``, ``settled``); the FIFO's depth, not in the way,
    stays stored. A strategy's run reports the same (``Run.cleared``, ``settled``)."""
    network = boarded(ULTRA)
    asked = network.trial(NO_URAM)
    rule = ("hidden", "transport.fifo.buffer.uram_present")
    assert not asked.accepted and list(asked.refused) == [rule]
    assert [f.code for f in asked.refused[rule]] == ["uram-absent"]
    assert asked.standing == {rule: (STYLE,)}
    seam = Seam(FULL_DSP48E2)
    done = seam.set(network, NO_URAM)
    assert isinstance(done, Accepted)
    assert {key: [f.code for f in why] for key, why in done.cleared.items()} == {
        STYLE: ["uram-absent"]
    }
    assert done.settled == {STYLE: "auto"} and done.unsettled == {}
    states = dict(done.network.states(["hidden"]))
    assert states[STYLE] == inspection.Set("auto")
    assert states[DEPTH] == inspection.Set(16)
    assert done.network.chosen()[("board", "uram")] is False
    # As a strategy's change: its run names what the fold rule cleared and set.
    explorer = Seam(FULL_DSP48E2)
    held = Hold({"board": {"uram": False}}).explore(explorer, network)
    (run,) = explorer.runs
    assert list(run.cleared) == [STYLE] and run.settled == {STYLE: "auto"}
    assert explorer.chosen(held) == explorer.chosen(done.network)


def test_a_held_channel_choice_standing_in_the_way_refuses_the_change() -> None:
    """The style held, the fold rule never clears it: the change is refused by the hold,
    the rule it stands in the way of its cause, and a strategy making it is refused the
    same way."""
    network = boarded(ULTRA)
    seam = Seam(FULL_DSP48E2)
    seam.held[STYLE] = "ultra"
    done = seam.set(network, NO_URAM)
    assert isinstance(done, Refused) and list(done.why) == [STYLE]
    (found,) = done.why[STYLE]
    assert (found.code, found.owner) == ("held", "hidden.transport.fifo.buffer.ram_style")
    assert found.message.startswith("held to 'ultra', standing in the way of the change: ")
    assert [cause.code for cause in found.causes] == ["uram-absent"]
    with refused("pin-refused", error=ExploreError, message="standing in the way of the change"):
        Hold({"board": {"uram": False}}).explore(seam, network)


def test_a_kernel_s_stored_choice_standing_in_the_way_refuses_the_change_naming_it() -> None:
    """With a hidden channel that admits no adapter, a fold of the MatMul to other lanes
    than the thresholding's needs a width conversion: refused, the neighbour's stored PE
    (and the MatMul's own other choices) standing in the way, none of them a channel's, so
    the seam clears nothing. Asked with the UltraRAM taken away too, the seam clears the
    FIFO's style, which stands in the way of one rule, and makes the change again: still
    refused by the other, naming the neighbour's PE, so the change is refused with that
    second refusal's findings, and nothing is cleared."""
    folding = {"first.pe": 2, "first.simd": 2}
    network = boarded(folding=folding, adaptable=False)
    seam = Seam(FULL_DSP48E2)
    fold = keyed({"first.pe": 4})
    done = seam.set(network, fold)
    assert isinstance(done, Refused)
    rules = {("hidden", "realizable"), ("hidden", "compatible")}
    assert set(done.why) == rules
    assert [f.code for f in done.why[("hidden", "realizable")]] == ["channel-plan"]
    assert all(("activate", "pe") in done.standing[rule] for rule in rules)
    assert not any(key[0] in network.channels for keys in done.standing.values() for key in keys)
    both = boarded(ULTRA, folding=folding, adaptable=False)
    asked = both.trial({**fold, **NO_URAM})
    assert set(asked.refused) == {*rules, ("hidden", "transport.fifo.buffer.uram_present")}
    done = seam.set(both, {**fold, **NO_URAM})
    assert isinstance(done, Refused) and set(done.why) == rules
    assert all(("activate", "pe") in done.standing[rule] for rule in rules)
    assert dict(both.states(["hidden"]))[STYLE] == inspection.Set("ultra")


def test_a_kernel_choice_beside_a_cleared_channel_choice_refuses_nothing() -> None:
    """A FIFO deeper than block RAM addresses, in ``auto`` (UltraRAM here): taking the
    UltraRAM away makes its storage's rule false, its depth and style standing in the way
    beside the kernels' choices its word reads. The channel's cleared, the rule reads an
    open choice, open, not false: the change is accepted, the style set again by the channel's
    policy and the depth, required, left open for ``SizeFifos``."""
    network = boarded(
        {**ULTRA, "transport.fifo.depth": DEEP, "transport.fifo.buffer.ram_style": "auto"}
    )
    asked = network.trial(NO_URAM)
    rule = ("hidden", "transport.fifo.buffer.planned_storage")
    assert list(asked.refused) == [rule]
    assert [f.code for f in asked.refused[rule]] == ["fifo-capacity"]
    assert {STYLE, DEPTH, ("activate", "pe")} <= set(asked.standing[rule])
    done = Seam(FULL_DSP48E2).set(network, NO_URAM)
    assert isinstance(done, Accepted)
    assert set(done.cleared) == {STYLE, DEPTH}
    assert done.settled == {STYLE: "auto"}
    ((key, (found,)),) = done.unsettled.items()
    assert key == DEPTH and found.code == "choice-no-valid-value" and "required" in found.message
    assert dict(done.network.states(["hidden"]))[DEPTH] == inspection.Open(())


def test_a_choice_cleared_twice_in_a_run_is_reported_by_its_first_clearing() -> None:
    """In one run, the FIFO's style in ``ultra`` is cleared by taking the UltraRAM away and
    set again (``auto``); the UltraRAM given back and the FIFO made deeper than block RAM
    addresses, taking the UltraRAM away again clears the style a second time, beside the
    depth, and sets it ``auto`` again. The run reports the style cleared, by its first
    clearing (``ultra`` before it, ``uram-absent``): it ends in another value than it held
    before the run's first clearing."""
    network = boarded(ULTRA)
    seam = Seam(FULL_DSP48E2)
    changes = (NO_URAM, keyed({"board.uram": True}), {DEPTH: DEEP}, NO_URAM)

    def search(opened: Partition, _: Mapping[tuple[str, str], object]) -> Partition:
        for change in changes:
            done = seam.set(opened, change)
            assert isinstance(done, Accepted), done
            if STYLE in done.cleared:
                assert done.settled[STYLE] == "auto"
            opened = done.network
        return opened

    found = seam.run(network, (), search, "twice", restore=False)
    assert dict(found.states(["hidden"]))[STYLE] == inspection.Set("auto")
    (run,) = seam.runs
    assert [f.code for f in run.cleared[STYLE]] == ["uram-absent"]
    assert [f.code for f in run.cleared[DEPTH]] == ["fifo-capacity"]
    assert run.settled == {STYLE: "auto"}


#: A strategy's own order of the FIFO's styles, other than the channel's (``auto`` first).
PREFERRED = ("distributed", "block", "auto", "ultra")


def test_a_cleared_choice_the_running_strategy_sets_as_it_finds_it_is_left_to_its_loop() -> None:
    """A strategy whose search sets every open choice as it finds it, by its own order
    (``Seam.run``'s ``decides``, as ``Ranked``'s), sets the FIFO's style itself
    (``ultra``); a later change of its search takes the UltraRAM away, and the style,
    standing in its way, is cleared. The fold rule leaves it open for the strategy's loop
    (cleared, not settled), which sets it again by the strategy's own order
    (``distributed``), not by the channel's (``auto``).

    ``Ranked`` itself does not reach this on FINN's channels: its loop sets every kernel
    choice before any channel's (a partition's members are ranked kernels first, and no
    kernel choice comes into play after a channel's is set: a kernel reads its facts only),
    and no channel's choice it sets later in its seam order makes a rule false that an
    earlier channel choice stands in the way of (none found on the Chain's channels)."""
    network = boarded(ULTRA)
    seam = Seam(FULL_DSP48E2)

    def accepted(done: Accepted[Partition] | Refused) -> Accepted[Partition]:
        assert isinstance(done, Accepted), done
        return done

    def search(opened: Partition, _: Mapping[tuple[str, str], object]) -> Partition:
        network = accepted(seam.set(opened, {STYLE: "ultra"})).network
        done = accepted(seam.set(network, NO_URAM))
        assert list(done.cleared) == [STYLE] and done.settled == {}
        assert dict(done.network.states(["hidden"]))[STYLE] == inspection.Open(())
        network = done.network
        while STYLE in network.undecided():
            (choice,) = seam.choices(network, keys=(STYLE,))
            assert choice.cases is not None
            value = next(each for each in PREFERRED if each in choice.cases)
            network = accepted(seam.set(network, {STYLE: value})).network
        return network

    found = seam.run(network, (STYLE,), search, "preferring", restore=False, decides=lambda _: True)
    assert dict(found.states(["hidden"]))[STYLE] == inspection.Set("distributed")
    assert found.chosen()[("board", "uram")] is False


def test_a_choice_the_restore_brings_into_play_is_set_by_the_fold_rule_not_left_to_a_loop() -> None:
    """A strategy that decides every open choice as it finds it (``decides``, as
    ``Ranked``'s) leaves a choice a change brings into play to its loop only while its
    search runs. The restore after it has no loop left: the folding the search left open
    is given back (one lane a side), which brings the input's adapter memory into play,
    and the fold rule sets it by the channel's policy (``distributed``) as the restore's change
    is accepted, as for a strategy that decides nothing. (Neither of FINN's strategies
    reaches this: ``Defaults`` restores nothing, and ``Ranked``'s restore touches only
    what its search left open, the choices only a membership rule restricts.)"""
    four = boarded(folding={"first.pe": 4, "first.simd": 4})
    style = ("x", "adapter_ram_style")
    assert dict(four.states(["x"]))[style] == inspection.Absent()
    folded = four.with_choices(keyed({"first.simd": 1}))
    assert dict(folded.states(["x"]))[style] == inspection.Open(())
    seam = Seam(FULL_DSP48E2)
    simd = ("first", "simd")

    def search(opened: Partition, prior: Mapping[tuple[str, str], object]) -> Partition:
        assert prior == {simd: 1}
        assert style not in opened.undecided()
        return opened

    found = seam.run(folded, (simd,), search, "idle", decides=lambda _: True)
    assert found.chosen()[simd] == 1
    assert dict(found.states(["x"]))[style] == inspection.Set("distributed")
    (run,) = seam.runs
    assert run.settled == {style: "distributed"} and run.cleared == {}


def test_a_fifo_newly_in_play_leaves_its_depth_open_and_its_style_set_by_policy() -> None:
    """A change that places the hidden channel's FIFO brings its choices into play: the
    seam sets its style by the channel's policy (``auto``) and leaves its depth, required,
    open for ``SizeFifos``; nothing is cleared."""
    network = boarded()
    done = Seam(FULL_DSP48E2).set(network, keyed({"hidden.transport": "fifo"}))
    assert isinstance(done, Accepted)
    assert done.cleared == {} and done.settled == {STYLE: "auto"}
    assert list(done.unsettled) == [DEPTH]
    states = dict(done.network.states(["hidden"]))
    assert states[DEPTH] == inspection.Open(()) and states[STYLE] == inspection.Set("auto")


def test_a_channel_choice_a_fold_brings_into_play_is_set_by_the_fold_rule_the_strategy_s() -> None:
    """On the Chain's graph, the first MatMul at four lanes a side needs no adapter on its
    input, so the input's adapter memory is not in play (nor stored). A target of 48
    cycles folds it to one lane: the input starts adapting, and its adapter's memory
    comes into play. The fold rule sets it by the channel's policy as the fold is accepted;
    the strategy's run names it (``Run.settled``), and the exploration's report names it
    (``settled``) and gives it to the strategy (``by``)."""
    model = kernel_model()
    configured(model, {"first": {"pe": 4, "simd": 4}})
    style = ("x", "adapter_ram_style")
    stored = in_shell(model, model.graph.node)
    persist(model, stored.trial({}, [style]).network)
    bound = in_shell(model, model.graph.node)
    assert dict(bound.states(["x"]))[style] == inspection.Absent()
    seam = Seam(read_target(model).platform)
    seam.held[("activate", "pe")] = chain.PE
    folded = TargetCycles(48).explore(seam, bound)
    assert folded.chosen(["first"])[("first", "simd")] == 1
    assert dict(folded.states(["x"]))[style] == inspection.Set("distributed")
    (run,) = seam.runs
    assert (run.cleared, run.settled, run.by_policy) == ({}, {style: "distributed"}, {})
    report = explore_kernel_choices(model, TargetCycles(48)).report
    assert (report["strategy"]["cleared"], report["strategy"]["settled"]) == (
        {},
        {"x.adapter_ram_style": "distributed"},
    )
    assert report["choices"]["x"]["adapter_ram_style"] == {
        "value": "distributed",
        "by": "target_cycles",
    }
