# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A choice's policy (``finn.kernels.policies``): fixed, first valid (in listed order or in
a ranking view's), or required; stated by the owner of a design space, never by the engine
or the kernel, keyed by class-level Decision references, checked where it is stated
(``stated``), read through one reader (``policy``) and applied once, in the flow order
(``defaulted``). The cut's own (``finn.kernels.cut``) are a channel's."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import pytest
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Param,
    Rejected,
    Space,
    constraint,
    design_space,
    domain,
    inspection,
    reject,
    view,
)
from finn.kernels.channels import Channel, ChannelFifo
from finn.kernels.cut import CHANNEL_POLICIES, Cut
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.policies import (
    FIRST_VALID,
    Defaulted,
    FirstValid,
    Fixed,
    Policy,
    Required,
    Unstated,
    defaulted,
    policy,
    ranked,
    stated,
)
from finn.kernels.transpose import TransposeKernel
from finn.kernels.utilization import BLOCK_FIRST_BITS, memory_preference, memory_styles


class Narrow(Space):
    width: int = Param()
    lanes: int = Decision(values=(4, 2, 1))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.lanes > self.width:
            return reject("toy-lanes", f"{self.lanes} lanes are wider than {self.width}")
        return True


class Wide(Space):
    width: int = Param()
    lanes: int = Decision(values=(8, 16))


class Toy(Space):
    width: int = Param()
    core: Narrow | Wide = Decision({"narrow": Narrow, "wide": Wide}, width=width)
    depth: int = Decision(domain=domain(accepts=lambda *, candidate: candidate > 0))
    style: str = Decision(values=("small", "large"))

    @constraint
    def style_fits(self) -> bool | Rejected:
        if self.style == "small" and self.width > 4:
            return reject("toy-style", f"small holds 4, not {self.width}")
        return True


def owner(root: type[Space], policies: Mapping[object, Policy]) -> Any:
    """An owner stating ``policies`` for ``root``, checked as a KernelOp's are."""
    checked = stated("Owner", root, policies)

    class Owner:
        policies: ClassVar[Mapping[object, Policy]] = checked

    return Owner


TOY = owner(Toy, {Toy.depth: Required()})
"""Toy's owner: its depth, known by membership only, required; every other choice first
valid."""


def states(point: Any) -> dict[str, object]:
    return dict(inspection.states(point))


def done(found: Defaulted[Any] | tuple[Any, ...]) -> Any:
    assert isinstance(found, Defaulted), found
    assert found.fallbacks == ()
    return found.point


def test_each_choice_in_play_takes_its_first_valid_value_in_the_flow_order() -> None:
    """The selector's first case first, then the choices of that case alone; each value
    the first its rules do not refuse; the required choice left open, the other case's
    choices absent."""
    point = done(defaulted(design_space(Toy(width=2)), TOY))
    assert states(point) == {
        "core": inspection.Set("narrow"),
        "core.narrow.lanes": inspection.Set(2),  # 4 is wider than 2
        "core.wide.lanes": inspection.Absent(),
        "depth": inspection.Open(),
        "style": inspection.Set("small"),
    }
    assert states(done(defaulted(design_space(Toy(width=8)), TOY)))["style"] == (
        inspection.Set("large")
    )


class Memory(Space):
    bits: int = Param()
    ram_style: str = Decision(domain=memory_styles())

    @view
    def ram_preference(self) -> tuple[str, ...]:
        found: int = self.bits
        return memory_preference(found)


def test_first_valid_is_in_listed_order_unless_its_policy_names_a_ranking() -> None:
    """A memory's style lists LUTRAM first (``inspection.viable``). Unlisted, first valid
    takes it whatever the size; with a ranking view named, block RAM from
    ``BLOCK_FIRST_BITS``, LUTRAM below."""
    large = design_space(Memory(bits=BLOCK_FIRST_BITS))
    (listed,) = inspection.viable(large, [Memory.ram_style])
    assert listed.cases[:2] == ("distributed", "block")
    assert states(done(defaulted(large, Unstated)))["ram_style"] == inspection.Set("distributed")
    ranking = owner(Memory, {Memory.ram_style: FirstValid(order=Memory.ram_preference)})
    assert states(done(defaulted(large, ranking)))["ram_style"] == inspection.Set("block")
    small = done(defaulted(design_space(Memory(bits=BLOCK_FIRST_BITS - 1)), ranking))
    assert states(small)["ram_style"] == inspection.Set("distributed")


class Refusing(Space):
    """A memory whose ranking view refuses while the memory is refused."""

    bits: int = Param()
    ram_style: str = Decision(domain=memory_styles())

    @constraint
    def holds(self) -> bool | Rejected:
        return reject("toy-memory", "refused") if self.bits < 0 else True

    @view
    def ram_preference(self) -> tuple[str, ...]:
        found: int = self.bits
        return memory_preference(found)


def test_a_ranking_that_cannot_be_read_falls_back_to_listed_order_named() -> None:
    """A ranking view refuses while its Space is refused: first valid takes the listed
    order, and the fallback is named (``choice-order-unknown``): the choice, the view and
    why, the view's findings its causes."""
    ranking = owner(Refusing, {Refusing.ram_style: FirstValid(order=Refusing.ram_preference)})
    point = design_space(Refusing(bits=-1))
    (item,) = (each for each in inspection.decisions(point) if each.key == "ram_style")
    (listed,) = inspection.viable(point, [item.reference])
    values, fallback = ranked(point, ranking, item, listed.values or ())
    assert values == listed.values
    assert fallback is not None
    assert (fallback.code, fallback.owner) == ("choice-order-unknown", "ram_style")
    assert fallback.message == (
        "first valid in listed order: its order is not known "
        "(Refusing.ram_preference: it is refused)"
    )
    assert dict(fallback.details) == {
        "policy": "first valid",
        "order": "Refusing.ram_preference",
    }
    assert {cause.code for cause in fallback.causes} == {"toy-memory"}
    readable = design_space(Refusing(bits=BLOCK_FIRST_BITS))
    values, fallback = ranked(readable, ranking, item, listed.values or ())
    assert fallback is None and values[0].value == "block"


def test_a_choice_already_set_is_kept() -> None:
    held = design_space(Toy(width=2)).with_choices({Toy.style: "large"})
    assert states(done(defaulted(held, TOY)))["style"] == inspection.Set("large")


def test_a_choice_named_kept_is_left_open() -> None:
    """``kept`` names open choices the caller leaves open (the cut's policies keep a choice
    a change cleared): every other one takes its policy."""
    point = done(defaulted(design_space(Toy(width=2)), TOY, kept={"style"}))
    assert states(point)["style"] == inspection.Open()
    assert states(point)["core"] == inspection.Set("narrow")


def test_a_choice_no_value_can_be_set_for_names_its_policy_each_value_and_each_rule() -> None:
    """No lane count fits a width of 0: one finding, owned by the choice's key, its policy
    and each value's rule named, those rules' findings its causes."""
    found = defaulted(design_space(Toy(width=0)), TOY)
    assert isinstance(found, tuple)
    (finding,) = found
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "core.narrow.lanes")
    assert dict(finding.details) == {"policy": "first valid"}
    assert "4 breaks core.narrow.fits; 2 breaks core.narrow.fits" in finding.message
    assert {cause.code for cause in finding.causes} == {"toy-lanes"}


SMALL = owner(Toy, {Toy.depth: Required(), Toy.style: Fixed("small"), Toy.core: Fixed("wide")})
ONE_LANE = owner(Toy, {Toy.depth: Fixed(4), Narrow.lanes: Fixed(1)})


def test_a_fixed_value_is_set_in_place_of_the_first_valid_one() -> None:
    """A selector's case and a value alike; a nested class's Decision wherever the class
    is placed; a choice known by membership only, fixed."""
    point = done(defaulted(design_space(Toy(width=2)), SMALL))
    assert states(point) == {
        "core": inspection.Set("wide"),
        "core.narrow.lanes": inspection.Absent(),
        "core.wide.lanes": inspection.Set(8),
        "depth": inspection.Open(),
        "style": inspection.Set("small"),
    }
    point = done(defaulted(design_space(Toy(width=4)), ONE_LANE))
    assert states(point)["core.narrow.lanes"] == inspection.Set(1)
    assert states(point)["depth"] == inspection.Set(4)


def test_a_fixed_value_its_rules_refuse_names_the_value_and_the_rule() -> None:
    found = defaulted(design_space(Toy(width=8)), SMALL)
    assert isinstance(found, tuple)
    (finding,) = found
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "style")
    assert dict(finding.details) == {"policy": "fixed"}
    assert (
        finding.message == "no value can be set (fixed): 'small' is refused: it breaks style_fits"
    )
    assert {cause.code for cause in finding.causes} == {"toy-style"}


def test_one_reader_gives_each_decision_its_owner_s_policy_or_first_valid() -> None:
    """``policy``: the policy stated for the Decision's class-level reference, wherever it
    is placed; first valid, in listed order, where none is."""
    found = {item.key: policy(ONE_LANE, item) for item in inspection.decisions(Toy)}
    assert found == {
        "core": FIRST_VALID,
        "core.narrow.lanes": Fixed(1),
        "core.wide.lanes": FIRST_VALID,
        "depth": Fixed(4),
        "style": FIRST_VALID,
    }
    assert all(policy(Unstated, item) == FIRST_VALID for item in inspection.decisions(Toy))


class Other(Space):
    lanes: int = Decision(values=(1, 2))

    @view
    def ranking(self) -> tuple[int, ...]:
        return (2, 1)


@pytest.mark.parametrize(
    ("policies", "code", "message"),
    [
        (
            {Toy.depth: Required(), Other.lanes: Fixed(1)},
            "default-policy-undeclared",
            "Other.lanes",
        ),
        ({Toy.depth: Required(), Toy.style: "small"}, "default-policy-invalid", "Fixed"),
        ({Toy.depth: Required(), Toy.style: Fixed("medium")}, "default-policy-value", "domain"),
        ({Toy.depth: Required(), Toy.style: Fixed(3)}, "default-policy-value", "type"),
        ({Toy.depth: Required(), Toy.core: Fixed("huge")}, "default-policy-value", "cases"),
        ({Toy.depth: Fixed(0)}, "default-policy-value", "domain"),
        (
            {Toy.depth: Required(), Narrow.lanes: FirstValid(order=Other.ranking)},
            "default-policy-order",
            "Other.ranking",
        ),
        ({}, "default-policy-unlisted", "Toy.depth is known by membership only"),
        ({Toy.depth: FirstValid()}, "default-policy-unlisted", "Required()"),
    ],
)
def test_policies_are_checked_where_they_are_stated(
    policies: Mapping[object, object], code: str, message: str
) -> None:
    """Each key a Decision of a class the root reaches; each value a policy; a fixed
    value one the Decision can take, as far as is known without facts; an order a view of
    the Decision's class; a choice known by membership only required or fixed."""
    with refused(code, error=DefinitionError, message=message):
        stated("Owner", Toy, policies)


def test_the_cut_states_a_channel_s_policies() -> None:
    """A FIFO's depth required, a memory's style first valid in its kernel's ranking, and
    every other channel choice first valid, in listed order; checked against ``Channel``
    when the module is imported."""
    assert dict(CHANNEL_POLICIES) == {
        ChannelFifo.depth: Required(),
        MemStreamKernel.ram_style: FirstValid(order=MemStreamKernel.ram_preference),
        InputGeneratorKernel.ram_style: FirstValid(order=InputGeneratorKernel.ram_preference),
    }
    found = {item.key: policy(Cut, item) for item in inspection.decisions(Channel)}
    assert {key for key, each in found.items() if each == Required()} == {"transport.fifo.depth"}
    ranked_keys = {
        key
        for key, each in found.items()
        if isinstance(each, FirstValid) and each.order is not None
    }
    assert ranked_keys == {
        "source.memstream.ram_style",
        "adapter.input_gen.input_gen.ram_style",
        "adapter.input_gen_vpc.input_gen.ram_style",
        "adapter.input_gen_vpc_input_gen.input_gen.ram_style",
        "adapter.input_gen_vpc_input_gen.input_gen_1.ram_style",
    }


def test_no_kernel_states_a_policy_but_a_ranking_it_reads_out() -> None:
    """A kernel holds hardware knowledge only: the three memories' rankings are views, no
    domain ranks a value, and a kernel has no policies of its own."""
    for kernel in (MemStreamKernel, InputGeneratorKernel, TransposeKernel):
        assert not hasattr(kernel, "defaults") and not hasattr(kernel, "policies")
        declared: Any = kernel.ram_style
        assert declared.domain.order is None
        assert callable(cast(Any, kernel.ram_preference).function)
