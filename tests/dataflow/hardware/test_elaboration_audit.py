# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 6b: "elaboration makes no design choice", as a check rather than a claim.

Item 9 of the migration plan's evidence list has been asserted since Phase 1
and never tested.  The mechanism it rests on is real -- a ``KernelBinding``
holds the covered Regions, the Kernel's own committed choices and its resolved
parameters, and nothing else is reachable from it -- but unreachable is not the
same as unused.  ``elaborate`` is an ordinary classmethod, and nothing stopped
its body from computing ``LANES * 2`` and putting the result in a component.
Such a value has no owner in the design point, appears in no key, and reaches
generated HDL anyway.

**What this module establishes.**  Every parameter value leaving an elaboration
is one the design point already answered: an undeclared name is refused, and a
declared name carrying anything but its resolved value is refused.

**What it does not, and cannot.**  That elaboration makes no choice *at all*.
A passing test catches specific violations; it cannot quantify over the ones
nobody thought of.  And elaboration legitimately chooses things that are not
design-space coordinates -- instance names, hierarchy, how many components to
emit -- so a check that refused those would be wrong rather than stricter.  The
audit is the falsifiable part of item 9, and the plan's §5 note that item 9 is
unfalsifiable as written is the reason this docstring says so out loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataflow.hardware.test_forcing_cases import (
    SingleComponentKernel,
    _compute_role,
    _place,
)
from finn.dataflow.design import Decided
from finn.dataflow.hardware import (
    KernelBinding,
    PhysicalComponent,
    audit_elaboration,
    scalar_parameters,
)
from finn.dataflow.spec_algebra import SpecAuthoringError

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "finn" / "dataflow"


def _binding() -> KernelBinding:
    placed = _place(hardware_kernel="single")
    bound = placed.selection.bind(placed.engine, placed.point, _compute_role(placed))
    assert isinstance(bound, Decided)
    return bound.value


# -- the audit fires -----------------------------------------------------------


def test_a_declared_elaboration_passes_unchanged() -> None:
    """The baseline, so the refusals below are not just "everything fails"."""

    binding = _binding()
    components = binding.components()
    assert components
    for component in components:
        for name, _value in component.parameters:
            assert name in dict(binding.parameters)


def test_a_kernel_that_invents_a_parameter_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adversarial direction, which nothing tested before.

    ``STAGES`` is not in the parameter table, so no route -- projected fact,
    committed decision, derived property, documented constant -- owns its
    value.  It would reach the generated HDL and appear in no artifact key,
    which means two builds differing only in it would collide.
    """

    def inventing(cls: type, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        values = dict(binding.parameters)
        values["STAGES"] = 3
        return (PhysicalComponent("single.core", "example.single", scalar_parameters(values)),)

    monkeypatch.setattr(SingleComponentKernel, "elaborate", classmethod(inventing))
    with pytest.raises(SpecAuthoringError) as refusal:
        _binding().components()
    codes = {issue.code for issue in refusal.value.issues}
    assert "hardware-elaboration-parameter-undeclared" in codes


def test_a_kernel_that_recomputes_a_declared_parameter_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subtler half: the right name carrying a value nobody resolved.

    Harder to see than an invented name and worse in effect -- the parameter
    *is* in the key, with the value the point answered, while the HDL gets a
    different one.  The key would then claim two unequal artifacts are one.
    """

    def doubling(cls: type, binding: KernelBinding) -> tuple[PhysicalComponent, ...]:
        values = dict(binding.parameters)
        values["LANES"] = int(str(values["LANES"])) * 2
        return (PhysicalComponent("single.core", "example.single", scalar_parameters(values)),)

    monkeypatch.setattr(SingleComponentKernel, "elaborate", classmethod(doubling))
    with pytest.raises(SpecAuthoringError) as refusal:
        _binding().components()
    codes = {issue.code for issue in refusal.value.issues}
    assert "hardware-elaboration-parameter-recomputed" in codes


def test_omitting_a_declared_parameter_is_allowed() -> None:
    """Deliberately permitted, and worth saying why.

    A Kernel elaborating several components gives each the subset it takes, so
    "every declared parameter appears in every component" would be false for
    the multi-component forcing case.  That a declared parameter *resolves* is
    already checked at binding; this audit is about values arriving from
    nowhere, not about coverage of the table.
    """

    binding = _binding()
    assert audit_elaboration(binding, (PhysicalComponent("bare", "example.single"),))


def test_the_audit_says_nothing_about_names_hierarchy_or_component_count() -> None:
    """The stated limit, asserted so it cannot quietly become a rule.

    These are elaboration's to choose and none of them is a design-space
    coordinate.  A future check that refused them would be a different claim.
    """

    binding = _binding()
    values = scalar_parameters(dict(binding.parameters))
    assert audit_elaboration(
        binding,
        (
            PhysicalComponent("wherever.shell", "example.single"),
            PhysicalComponent("wherever.shell.core", "example.single", values, "wherever.shell"),
        ),
    )


# -- and cannot be walked around -----------------------------------------------


def test_nothing_in_the_source_tree_calls_the_unchecked_entry_point() -> None:
    """A check one call site can bypass is not a check.

    ``KernelBinding.components`` is the audited entry point; ``elaborate`` is
    the raw classmethod behind it.  The MVAU composition called the raw one,
    which is precisely the assembly that matters, so this scans for a
    regression rather than trusting the convention.
    """

    offenders = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if ".elaborate(" not in line or "def elaborate" in line:
                continue
            if path.name == "kernel.py" and "audit_elaboration(" in line:
                continue  # the audited entry point itself
            offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{number}")
    assert offenders == [], offenders
