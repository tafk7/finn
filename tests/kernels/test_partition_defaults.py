# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A partition's policies applied once (``Partition.defaulted``), on the Chain
(``support.chain``): one pass over the partition's design space in its flow order, each
choice's policy read from its member's owner (a kernel's its KernelOp, a channel's the
cut's); a member refused on its own skipped whole; a choice no value can be set for left
open, reported, the pass going on; a pass over some of the members only.

What the order of the pass decides, and a member's refusal read when the pass reaches
the member's first choice, are pinned on toy Spaces where it shows
(``tests/kernels/test_policies.py``): the Chain's choices read nothing a later choice of
another member sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from support import chain
from support.helpers import FULL_DSP48E2, keyed

from finn.kernels.composer import ChannelDeclaration, Facts, PartitionDeclaration, Port, channel
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.partition import Partition, partition
from finn.kernels.policies import Fixed, Policy, stated


def owner_of(policies: Mapping[object, Policy]) -> Any:
    """An owner stating ``policies`` for a MatMul's kernel, checked as a KernelOp's are."""
    checked = stated("Owner", MatMulKernel, policies)

    class Owner:
        policies: ClassVar[Mapping[object, Policy]] = checked

    return Owner


KERNELS = ("first", "activate", "second")


def chained(
    owners: Mapping[str, Any] | None = None,
    stored: Mapping[str, object] | None = None,
    *,
    adaptable: bool = True,
) -> Partition:
    """The Chain, its thresholding's PE stored, ``stored`` besides (``member.key``), each
    kernel under its owner in ``owners`` (by name: none stated where missing), its hidden
    channel admitting an adapter unless not ``adaptable``."""
    members = chain.members()
    if not adaptable:
        members["hidden"] = channel(
            Port("first", "y"),
            Port("activate", "input"),
            tensor=chain.HIDDEN_TENSOR,
            platform=FULL_DSP48E2,
            adaptable=False,
        )
    return partition(
        PartitionDeclaration(
            {name: each for name, each in members.items() if isinstance(each, Facts)},
            {name: each for name, each in members.items() if isinstance(each, ChannelDeclaration)},
        ),
        keyed({"activate.pe": chain.PE, **(stored or {})}),
        order=tuple(members),
        owners=owners,
    )


def test_the_pass_sets_every_member_by_its_owner_s_policy_once() -> None:
    """Nothing open is left but what is required or out of play: each kernel's choices
    (first valid in its flow order: the core, then its own), each channel's (its transport
    direct); no finding."""
    held = chained()
    done = held.defaulted()
    assert done.unset == () and done.refused == {} and done.fallbacks == ()
    after = held.at(done.point)
    assert after.undecided() == ()
    chosen = after.chosen()
    assert chosen[("first", "compute")] == "packed"
    assert {chosen[(name, "transport")] for name in ("x", "w1", "hidden", "levels")} == {"direct"}
    assert chosen[("activate", "pe")] == chain.PE  # a stored choice is kept


def test_a_kernel_class_bound_by_two_owners_takes_each_owner_s_policies_at_its_own_node() -> None:
    """The Chain's two MatMuls are one kernel class: the first's owner fixes the packed
    core's reducer to the compressor, the second's states none (first valid: the tree)."""
    fixing = owner_of({PackedDotpKernel.reducer: Fixed("compressor")})
    held = chained({"first": fixing, "second": owner_of({})})
    chosen = held.at(held.defaulted().point).chosen()
    assert chosen[("first", "compute.packed.reducer")] == "compressor"
    assert chosen[("second", "compute.packed.reducer")] == "tree"


def test_a_member_refused_on_its_own_is_skipped_and_the_others_are_set() -> None:
    """Stored folding a channel admitting no adapter cannot take (the first MatMul at four
    lanes, the thresholding at two): the channel between them is refused on its own, none of
    its choices set (its transport stays open), reported beside the point; every other
    member is set."""
    held = chained(
        stored={"first.compute": "packed", "first.pe": 4, "first.simd": 2}, adaptable=False
    )
    assert held.validity("hidden").status == "refused"
    done = held.defaulted()
    assert list(done.refused) == ["hidden"] and done.unset == ()
    after = held.at(done.point)
    assert ("hidden", "transport") in after.undecided()
    assert after.chosen()[("x", "transport")] == "direct"
    assert after.chosen()[("second", "compute")] == "packed"


def test_a_choice_no_value_can_be_set_for_is_left_open_and_the_pass_goes_on() -> None:
    """The first MatMul's owner fixes its PE to four, which the hidden channel (admitting no
    adapter) refuses beside the thresholding's two: that choice is left open and reported
    (once, owned by its key), the MatMul's reducer and every other member's choices are
    set."""
    held = chained(
        {"first": owner_of({MatMulKernel.pe: Fixed(4)})},
        {
            "first.compute": "packed",
            "first.simd": 2,
            "first.compute.packed.compute_pumping": False,
        },
        adaptable=False,
    )
    done = held.defaulted()
    assert done.refused == {}
    (finding,) = done.unset
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "first.pe")
    assert dict(finding.details) == {"policy": "fixed"}
    after = held.at(done.point)
    assert after.undecided() == (("first", "pe"),)
    assert after.chosen()[("first", "compute.packed.reducer")] == "tree"
    assert after.chosen()[("hidden", "transport")] == "direct"


def test_the_pass_over_some_members_leaves_the_others_open() -> None:
    """``members`` restricts the pass to the choices of those members (the test helpers'
    pass over the channels): the kernels' stay open, every channel's is set."""
    held = chained()
    after = held.at(held.defaulted(members=held.channels).point)
    assert {key[0] for key in after.undecided()} == set(KERNELS)
    assert after.chosen()[("hidden", "transport")] == "direct"
