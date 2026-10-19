# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A choice's default policy (``finn.kernels.defaults``): fixed, first valid, or none
(required), stated on the class that declares the choice and applied once, in the flow
order, on the kernel alone (``defaulted``)."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import pytest
from support.refusals import refused

import finn.kernels
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
)
from finn.kernels.base import ComposedKernel, Kernel, LeafKernel
from finn.kernels.defaults import FIRST_VALID, Fixed, defaulted, policies
from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.utilization import BLOCK_FIRST_BITS, memory_styles


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
    depth: int = Decision(domain=domain(accepts=lambda *, candidate: candidate > 0), required=True)
    style: str = Decision(values=("small", "large"))

    @constraint
    def style_fits(self) -> bool | Rejected:
        if self.style == "small" and self.width > 4:
            return reject("toy-style", f"small holds 4, not {self.width}")
        return True


def states(point: Any) -> dict[str, object]:
    return dict(inspection.states(point))


def test_each_choice_in_play_takes_its_first_valid_value_in_the_flow_order() -> None:
    """The selector's first case first, then the choices of that case alone; each value
    the first its rules do not refuse; the required choice left open, the other case's
    choices absent."""
    point = defaulted(design_space(Toy(width=2)))
    assert states(point) == {
        "core": inspection.Set("narrow"),
        "core.narrow.lanes": inspection.Set(2),  # 4 is wider than 2
        "core.wide.lanes": inspection.Absent(),
        "depth": inspection.Open(required=True),
        "style": inspection.Set("small"),
    }
    assert states(defaulted(design_space(Toy(width=8))))["style"] == inspection.Set("large")


class Memory(Space):
    bits: int = Param()
    ram_style: str = Decision(domain=memory_styles(bits))


def test_first_valid_is_the_first_in_the_domain_s_order_not_its_list_s() -> None:
    """A memory's style lists its values in one order (``inspection.viable``: LUTRAM
    first) and ranks them by the bits it holds (``memory_styles``): first valid is the
    first possible value in the ranking, block RAM from ``BLOCK_FIRST_BITS``, LUTRAM
    below."""
    large = design_space(Memory(bits=BLOCK_FIRST_BITS))
    (listed,) = inspection.viable(large, [Memory.ram_style])
    assert listed.cases[:2] == ("distributed", "block")
    assert states(defaulted(large))["ram_style"] == inspection.Set("block")
    small = defaulted(design_space(Memory(bits=BLOCK_FIRST_BITS - 1)))
    assert states(small)["ram_style"] == inspection.Set("distributed")


def test_a_choice_already_set_is_kept() -> None:
    held = design_space(Toy(width=2)).with_choices({Toy.style: "large"})
    assert states(defaulted(held))["style"] == inspection.Set("large")


def test_a_choice_named_kept_is_left_open() -> None:
    """``kept`` names open choices the caller leaves open (the interim channel rule keeps
    a choice a change cleared): every other one takes its policy."""
    point = defaulted(design_space(Toy(width=2)), kept={"style"})
    assert states(point)["style"] == inspection.Open()
    assert states(point)["core"] == inspection.Set("narrow")


def test_a_choice_no_value_can_be_set_for_names_its_policy_each_value_and_each_rule() -> None:
    """No lane count fits a width of 0: one finding, owned by the choice's key, its policy
    and each value's rule named, those rules' findings its causes."""
    found = defaulted(design_space(Toy(width=0)))
    assert isinstance(found, tuple)
    (finding,) = found
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "core.narrow.lanes")
    assert dict(finding.details) == {"policy": FIRST_VALID}
    assert "4 breaks core.narrow.fits; 2 breaks core.narrow.fits" in finding.message
    assert {cause.code for cause in finding.causes} == {"toy-lanes"}


class SmallToy(Toy):
    defaults = {"style": Fixed("small"), "core": Fixed("wide")}


class OneLane(Narrow):
    defaults = {"lanes": Fixed(1)}


class NarrowToy(Toy):
    core: OneLane | Wide = Decision({"narrow": OneLane, "wide": Wide}, width=Toy.width)


def test_a_fixed_value_is_set_in_place_of_the_first_valid_one() -> None:
    """A selector's case and a value alike; a nested class's own key, where it is placed."""
    point = defaulted(design_space(SmallToy(width=2)))
    assert states(point) == {
        "core": inspection.Set("wide"),
        "core.narrow.lanes": inspection.Absent(),
        "core.wide.lanes": inspection.Set(8),
        "depth": inspection.Open(required=True),
        "style": inspection.Set("small"),
    }
    assert policies(NarrowToy)["core.narrow.lanes"] == Fixed(1)
    assert states(defaulted(design_space(NarrowToy(width=4))))["core.narrow.lanes"] == (
        inspection.Set(1)
    )


def test_a_fixed_value_its_rules_refuse_names_the_value_and_the_rule() -> None:
    found = defaulted(design_space(SmallToy(width=8)))
    assert isinstance(found, tuple)
    (finding,) = found
    assert (finding.code, finding.owner) == ("choice-no-valid-value", "style")
    assert dict(finding.details) == {"policy": "fixed"}
    assert (
        finding.message == "no value can be set (fixed): 'small' is refused: it breaks style_fits"
    )
    assert {cause.code for cause in finding.causes} == {"toy-style"}


def test_a_policy_names_its_class_s_own_choices_and_never_a_required_one() -> None:
    class Undeclared(Toy):
        defaults = {"lanes": Fixed(1)}  # Narrow's, not Toy's

    class FixedDepth(Toy):
        defaults = {"depth": Fixed(4)}

    with refused("default-policy-undeclared", error=DefinitionError, message=r"\['lanes'\]"):
        policies(Undeclared)
    with refused("default-policy-required", error=DefinitionError, message="FixedDepth.depth"):
        policies(FixedDepth)


def test_a_choice_known_by_membership_only_is_required_or_fixed() -> None:
    """A domain that lists no value has no first valid one: say so (required), explicitly."""

    class Sized(Space):
        size: int = Decision(domain=domain(accepts=lambda *, candidate: candidate > 0))

    class FixedSize(Sized):
        defaults = {"size": Fixed(16)}

    with refused("default-policy-unlisted", error=DefinitionError, message="mark it required"):
        policies(Sized)
    assert policies(FixedSize) == {"size": Fixed(16)}
    assert states(defaulted(design_space(FixedSize()))) == {"size": inspection.Set(16)}


def test_a_kernel_class_states_its_policies_as_fixed_values_by_name() -> None:
    """Checked where the class is made: a mapping from names to ``Fixed``."""
    with refused("default-policy-invalid", error=DefinitionError, message="Fixed"):

        class Unfixed(PackedDotpKernel):
            defaults = {"reducer": "compressor"}  # type: ignore[dict-item]

    class Compressing(PackedDotpKernel):
        defaults = {"reducer": Fixed("compressor")}

    assert policies(Compressing)["reducer"] == Fixed("compressor")
    assert policies(PackedDotpKernel)["reducer"] == FIRST_VALID


def _kernels() -> list[type[Kernel]]:
    for module in pkgutil.iter_modules(finn.kernels.__path__):
        importlib.import_module(f"finn.kernels.{module.name}")
    found: list[type[Kernel]] = []
    pending: list[type[Kernel]] = [Kernel]
    while pending:
        for each in pending.pop().__subclasses__():
            pending.append(each)
            if each.__module__.startswith("finn.kernels.") and each not in found:
                found.append(each)
    return [each for each in found if each not in (LeafKernel, ComposedKernel)]


@pytest.mark.parametrize("kernel", _kernels(), ids=lambda kernel: kernel.__qualname__)
def test_every_kernel_states_its_policies_and_each_choice_but_a_fifo_s_depth_is_first_valid(
    kernel: type[Kernel],
) -> None:
    """Every kernel's policies are well formed. None fixes a value: each choice's first
    valid value is today's; the FIFO's depth, required, has none."""
    found = policies(kernel)
    assert {key for key, policy in found.items() if policy is None} <= {
        "buffer.depth",
        "transport.fifo.buffer.depth",
    }
    assert all(policy in (FIRST_VALID, None) for policy in found.values())
