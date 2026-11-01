# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A core's policies applied once (``CoreComposition.defaulted``), on the Chain
(``support.chain``): one pass over the partition's design space in its flow order, each
choice's policy read from its member's owner (a kernel's its KernelOp, a channel's its
channel class's, ``Channel.policies``); a member refused on its own skipped whole; a
choice no value can be set for left open, reported, the pass going on; a pass over some of
the members only.

What the order of the pass decides, and a member's refusal read when the pass reaches
the member's first choice, are pinned on toy Spaces where it shows
(``tests/kernels/test_policies.py``): the Chain's choices read nothing a later choice of
another member sets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from support import chain
from support.helpers import FULL_DSP48E2, TEST_CORE, keyed

from finn.kernels.channels import Channel
from finn.kernels.composer import (
    ChannelDeclaration,
    DataflowCoreDeclaration,
    Facts,
    Port,
    channel,
    compose,
)
from finn.kernels.dataflow_core import bound
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.explore import CoreComposition
from finn.kernels.matmul import MatMulKernel
from finn.kernels.policies import Fixed, Policy, stated


def owner_of(policies: Mapping[object, Policy]) -> Any:
    """An owner stating ``policies`` for a MatMul's kernel, checked as a KernelOp's are."""
    checked = stated("Owner", MatMulKernel, policies)

    class Owner:
        policies: ClassVar[Mapping[object, Policy]] = checked

    return Owner


KERNELS = ("MatMul_0", "Thresholding_0", "MatMul_1")


def chained(
    owners: Mapping[str, Any] | None = None,
    stored: Mapping[str, object] | None = None,
    *,
    adaptable: bool = True,
) -> CoreComposition:
    """The Chain, its thresholding's PE stored, ``stored`` besides (``member.key``), each
    kernel under its owner in ``owners`` (by name: none stated where missing), its hidden
    channel admitting an adapter unless not ``adaptable``."""
    members = chain.members()
    if not adaptable:
        members["MatMul_0_out0"] = channel(
            Port("MatMul_0", "y"),
            Port("Thresholding_0", "input"),
            tensor=chain.HIDDEN_TENSOR,
            platform=FULL_DSP48E2,
            adaptable=False,
        )
    declared = DataflowCoreDeclaration(
        {name: each for name, each in members.items() if isinstance(each, Facts)},
        {name: each for name, each in members.items() if isinstance(each, ChannelDeclaration)},
        name=TEST_CORE,
        order=tuple(members),
        owners=owners,
    )
    return CoreComposition(
        bound(compose(declared), keyed({"Thresholding_0.pe": chain.PE, **(stored or {})})).point
    )


def test_the_pass_sets_every_member_by_its_owner_s_policy_once() -> None:
    """Nothing open is left but what is required or out of play: each kernel's choices
    (first valid in its flow order: the core, then its own), each channel's (its transport
    direct); no finding."""
    held = chained()
    done = held.defaulted()
    assert done.unset == () and done.refused == {} and done.fallbacks == ()
    after = held.made(done.point)
    assert after.undecided() == ()
    chosen = after.chosen()
    assert chosen[("MatMul_0", "compute")] == "packed"
    assert {
        chosen[(name, "transport")]
        for name in ("x", "MatMul_0_param0", "MatMul_0_out0", "Thresholding_0_out0")
    } == {"direct"}
    assert chosen[("Thresholding_0", "pe")] == chain.PE  # a stored choice is kept


def test_a_channel_s_owner_is_the_channel_class() -> None:
    """A channel's policies are its class's (``Channel.policies``): whichever the partition,
    its owner is ``Channel``, where a kernel's is the owner it was given."""
    fixing = owner_of({})
    held = chained({"MatMul_0": fixing})
    assert {held.owner(name) for name in held.channels} == {Channel}
    assert held.owner("MatMul_0") is fixing


def test_a_kernel_class_bound_by_two_owners_takes_each_owner_s_policies_at_its_own_node() -> None:
    """The Chain's two MatMuls are one kernel class: the first's owner fixes the packed
    core's reducer to the compressor, the second's states none (first valid: the tree)."""
    fixing = owner_of({PackedDotpKernel.reducer: Fixed("compressor")})
    held = chained({"MatMul_0": fixing, "MatMul_1": owner_of({})})
    chosen = held.made(held.defaulted().point).chosen()
    assert chosen[("MatMul_0", "compute.packed.reducer")] == "compressor"
    assert chosen[("MatMul_1", "compute.packed.reducer")] == "tree"


def test_a_member_refused_on_its_own_is_skipped_and_the_others_are_set() -> None:
    """Stored folding a channel admitting no adapter cannot take (the first MatMul at four
    lanes, the thresholding at two): the channel between them is refused on its own, none of
    its choices set (its transport stays open), reported beside the point; every other
    member is set."""
    held = chained(
        stored={"MatMul_0.compute": "packed", "MatMul_0.pe": 4, "MatMul_0.simd": 2}, adaptable=False
    )
    assert held.validity("MatMul_0_out0").status == "refused"
    done = held.defaulted()
    assert list(done.refused) == ["MatMul_0_out0"] and done.unset == ()
    after = held.made(done.point)
    assert ("MatMul_0_out0", "transport") in after.undecided()
    assert after.chosen()[("x", "transport")] == "direct"
    assert after.chosen()[("MatMul_1", "compute")] == "packed"


def test_a_choice_no_value_can_be_set_for_is_left_open_and_the_pass_goes_on() -> None:
    """The first MatMul's owner fixes its PE to four, which the hidden channel (admitting no
    adapter) refuses beside the thresholding's two: that choice is left open and reported
    (once, owned by its key), the MatMul's reducer and every other member's choices are
    set."""
    held = chained(
        {"MatMul_0": owner_of({MatMulKernel.pe: Fixed(4)})},
        {
            "MatMul_0.compute": "packed",
            "MatMul_0.simd": 2,
            "MatMul_0.compute.packed.compute_pumping": False,
        },
        adaptable=False,
    )
    done = held.defaulted()
    assert done.refused == {}
    (finding,) = done.unset
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "MatMul_0.pe")
    assert dict(finding.details) == {"policy": "fixed"}
    after = held.made(done.point)
    assert after.undecided() == (("MatMul_0", "pe"),)
    assert after.chosen()[("MatMul_0", "compute.packed.reducer")] == "tree"
    assert after.chosen()[("MatMul_0_out0", "transport")] == "direct"


def test_the_pass_over_some_members_leaves_the_others_open() -> None:
    """``members`` restricts the pass to the choices of those members (the test helpers'
    pass over the channels): the kernels' stay open, every channel's is set."""
    held = chained()
    after = held.made(held.defaulted(members=held.channels).point)
    assert {key[0] for key in after.undecided()} == set(KERNELS)
    assert after.chosen()[("MatMul_0_out0", "transport")] == "direct"
