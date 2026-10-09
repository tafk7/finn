# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each kernel's spec (``kernels.specs``), checked offline against its design space.

- its lean covering points (``finn.harness.points.covering``, decision KT10) reach
  every case of each of its Decisions and each ordered domain's extremes, and each
  is a point its kernel admits whose module builds (its adapters' memories chosen);
- the cases its design space refuses where they are drawn from are exactly those the
  spec states, each with its finding codes; and every probe is refused with its codes;
- what it gathers is there: its conformance cases (one spec each), the sweeps' case
  lists it names, its unit tests; its reference is the oracle or test-side.

The simulations are elsewhere: its conformance cases in ``test_conformance.py``,
the KernelOps' parity in ``tests/kernel_ops/test_parity.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any

import pytest
from layering import ROOT

from finn.core.space import Available, Space
from finn.harness.points import covering, refusal, refused_cases, rejected
from finn.kernels.configure import commit, undecided
from kernels.conformance import KERNEL, unchosen
from kernels.helpers import with_adapter_memories
from kernels.specs import SPECS, conformance_cases
from kernels.specs.base import ORACLE, TEST_SIDE, KernelSpec, Probe

BY_NAME = {spec.name: spec for spec in SPECS}
PATTERN = f"{KERNEL}.*"


def _base(spec: KernelSpec) -> Any:
    return unchosen(**spec.cases[spec.space]())


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_the_covering_points_reach_every_case_and_are_admitted(name: str) -> None:
    spec = BY_NAME[name]
    base = _base(spec)
    found = covering(base, PATTERN)
    assert not found.missed, f"{name}: no covering point takes {found.missed}"
    assert {key for key, _ in found.targets} == {key for point in found.points for key in point}, (
        f"{name}: a Decision no point commits"
    )
    for chosen in found.points or ({},):
        point = with_adapter_memories(commit(base, chosen))
        kernel: Space = getattr(point, KERNEL)
        assert not rejected(kernel), f"{name} refuses its covering point {chosen}"
        assert not undecided(point, "*"), f"{name} at {chosen}"
        module = point.query(type(point).module)
        assert isinstance(module, Available), f"{name} at {chosen}: {module}"


def test_an_ordered_domain_is_covered_at_its_extremes_and_any_other_whole() -> None:
    """Thresholding's PE (the divisors of 6) at 1 and 6, never 2 or 3; its pipeline, a
    bool, both ways."""
    found = covering(_base(BY_NAME["ThresholdingAxiKernel"]), PATTERN)
    assert sorted({point["kernel.pe"] for point in found.points}) == [1, 6]  # type: ignore[type-var]
    assert {point["kernel.deep_pipeline"] for point in found.points} == {False, True}


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_the_refused_side_is_what_the_spec_states(name: str) -> None:
    spec = BY_NAME[name]
    assert set(refused_cases(_base(spec), PATTERN)) == set(spec.refuses)


PROBES = [(spec.name, probe) for spec in SPECS for probe in spec.probes]


@pytest.mark.parametrize(
    ("name", "probe"), PROBES, ids=[f"{name}: {probe.label}" for name, probe in PROBES]
)
def test_a_probe_is_refused_with_its_codes(name: str, probe: Probe) -> None:
    point = probe.build()
    found = refusal(point, probe.choices) if probe.choices else rejected(point)
    assert found == probe.codes, f"{name}: {probe.label}"


def test_a_spec_names_each_conformance_case_once_and_draws_from_one_of_them() -> None:
    cases = conformance_cases()
    assert len(cases) == sum(len(spec.cases) for spec in SPECS)
    for spec in SPECS:
        assert spec.space in spec.cases, spec.name
        assert all(case()["space_type"] for case in spec.cases.values())


def _sweep_jobs() -> set[str]:
    """The numeric sweeps' job names (``scripts/emitted_text.py``'s ``SWEEPS``)."""
    found = importlib.util.spec_from_file_location("emitted_text", ROOT / "scripts/emitted_text.py")
    assert found is not None and found.loader is not None
    tool = sys.modules.get(found.name) or importlib.util.module_from_spec(found)
    if found.name not in sys.modules:
        sys.modules[found.name] = tool  # its dataclasses resolve their module
        found.loader.exec_module(tool)
    return {name for name, _, _ in tool.SWEEPS}


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_what_a_spec_gathers_is_there(name: str) -> None:
    spec = BY_NAME[name]
    jobs = _sweep_jobs()
    for sweep in spec.sweeps:
        assert set(sweep.jobs) <= jobs, f"{name}: no sweep job {set(sweep.jobs) - jobs}"
        module = importlib.import_module(sweep.module)
        for listed in sweep.lists:
            assert getattr(module, listed), f"{sweep.module}.{listed} is empty"
    for path in spec.unit:
        assert (ROOT / path).is_file(), path
    assert spec.reference == TEST_SIDE or spec.reference.startswith(ORACLE.split("{")[0])
