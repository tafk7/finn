# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each kernel's spec (``support.kernel_specs``), checked offline against its design space.

- its lean covering points (``finn.harness.points.covering``) reach
  every case of each of its Decisions and each ordered domain's extremes, and each
  is a point its kernel finds valid whose module builds (its adapters' memories chosen);
- the cases its design space refuses where they are drawn from are exactly those the
  spec states, each with its finding codes; and every probe is refused with its codes;
- what it gathers is there: its conformance cases (one spec each), the sweeps' case
  lists it names, its unit tests; its reference is the oracle or test-side;
- every kernel of ``finn.kernels`` that binds a library module is a spec's, so its
  covering points build its module, checking that its sources provide it;
- every stream port of every member of every conformance case, at each of the case's
  covering points, states its contract's label (``x.label``) and the side it gives a
  channel (``x.side``: its stream, that label and, an output's, its word width), so the
  side a channel is bound to and the contract its network reads cannot drift.

The simulations are elsewhere: its conformance cases in ``test_conformance.py``,
the KernelOps' parity in ``tests/kernel_ops/test_parity.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from dataclasses import replace
from typing import Any, cast

import pytest
from support.conformance import (
    KERNEL,
    configured,
    design_member,
    design_module,
    kernel_space,
    unchosen,
)
from support.kernel_specs import SPECS, conformance_cases
from support.kernel_specs.base import ORACLE, TEST_SIDE, KernelSpec, Probe
from support.layering import ROOT

import finn.kernels
from finn.core.space import Available, Inapplicable, Space
from finn.core.space.declarations import unmet_required
from finn.dataflow.streams import Stream
from finn.harness.points import Covering, covering, refusal, refused_cases, rejected
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel, PortContract
from finn.kernels.configure import undecided
from finn.kernels.laws import Advance
from finn.kernels.library import ModuleSpec
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.partition import Partition
from finn.kernels.port import port_contract, port_pace, port_side
from finn.kernels.port_model import ServedPort, StreamPort, stream_ports
from finn.kernels.sides import Side

BY_NAME = {spec.name: spec for spec in SPECS}


def _base(spec: KernelSpec) -> Any:
    return unchosen(**spec.cases[spec.space]())


def _covering(base: Any) -> Covering:
    """The covering points of a design's kernel's Decisions, each keyed ``kernel.<key>``."""
    space, pattern, prefix = kernel_space(base)
    found = covering(space, pattern)
    return replace(
        found,
        points=tuple(
            {prefix + key: value for key, value in point.items()} for point in found.points
        ),
        targets=tuple((prefix + key, value) for key, value in found.targets),
        missed=tuple((prefix + key, value) for key, value in found.missed),
    )


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_the_covering_points_reach_every_case_and_are_valid(name: str) -> None:
    spec = BY_NAME[name]
    base = _base(spec)
    found = _covering(base)
    assert not found.missed, f"{name}: no covering point takes {found.missed}"
    assert {key for key, _ in found.targets} == {key for point in found.points for key in point}, (
        f"{name}: a Decision no point commits"
    )
    for chosen in found.points or ({},):
        point = configured(base, chosen)
        kernel: Space = design_member(point, KERNEL)
        assert not rejected(kernel), f"{name} refuses its covering point {chosen}"
        open_ = point.undecided() if isinstance(point, Partition) else undecided(point, "*")
        assert not open_, f"{name} at {chosen}"
        module = design_module(point, spec.kernel)
        assert isinstance(module, Available), f"{name} at {chosen}: {module}"


def test_an_ordered_domain_is_covered_at_its_extremes_and_any_other_whole() -> None:
    """Thresholding's PE (the divisors of 6) at 1 and 6, never 2 or 3; its pipeline, a
    bool, both ways."""
    found = _covering(_base(BY_NAME["ThresholdingAxiKernel"]))
    assert sorted({point["kernel.pe"] for point in found.points}) == [1, 6]  # type: ignore[type-var]
    assert {point["kernel.deep_pipeline"] for point in found.points} == {False, True}


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_the_refused_side_is_what_the_spec_states(name: str) -> None:
    spec = BY_NAME[name]
    space, pattern, prefix = kernel_space(_base(spec))
    found = {replace(each, key=prefix + each.key) for each in refused_cases(space, pattern)}
    assert found == set(spec.refuses)


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


def _binding_a_module() -> set[type[Kernel]]:
    """Every kernel class of ``finn.kernels`` that binds a library module and can be
    placed (no ``required()`` member left unmet)."""
    for module in pkgutil.iter_modules(finn.kernels.__path__):
        importlib.import_module(f"finn.kernels.{module.name}")
    found: set[type[Kernel]] = set()
    pending: list[type[Kernel]] = [Kernel]
    while pending:
        for each in pending.pop().__subclasses__():
            pending.append(each)
            effective: dict[str, object] = {}
            for base in reversed(each.__mro__):
                effective.update(vars(base))
            native = any(isinstance(item, ModuleSpec) for item in effective.values())
            if each.__module__.startswith("finn.kernels.") and native and not unmet_required(each):
                found.add(each)
    return found


def test_every_kernel_binding_a_module_is_a_spec_s_so_its_module_is_built() -> None:
    kernels = _binding_a_module()
    assert {
        BY_NAME[name].kernel for name in ("FifoKernel", "InputGeneratorKernel", "VpcKernel")
    } <= kernels
    assert len(kernels) >= 11
    assert kernels <= {spec.kernel for spec in SPECS}, kernels - {spec.kernel for spec in SPECS}


def test_the_base_assembles_every_library_module_s_netlist() -> None:
    # Each kernel binding a module states the binding and its ports; the base's netlist,
    # read through the module's parts, is the one assembly (no kernel restates it).
    for kernel in _binding_a_module():
        declaring = next(each for each in kernel.__mro__ if "netlist" in vars(each))
        assert declaring is Kernel, kernel


def _ports(member: Space) -> tuple[str, ...]:
    """The stream ports ``member``'s class declares, by name: its module's, and those it
    serves for a child; none for a member that is no kernel."""
    if not isinstance(member, Kernel):
        return ()
    effective: dict[str, object] = {}
    for base in reversed(type(member).__mro__):
        effective.update(vars(base))
    served = tuple(name for name, value in effective.items() if isinstance(value, ServedPort))
    return (*stream_ports(type(member)), *served)


def _ports_checked(point: Partition) -> int:
    """Each placed stream port of each member of ``point``, checked: its label is its
    contract's, and its side states its stream, that label and, on an output port, its
    word width; a served port's, its child's port's, labelled as its contract. How many it
    checked."""
    checked = 0
    for name in point.members:
        member = point.member(name)
        for port in _ports(member):
            offered = member.query(port_contract(type(member), port))
            if not isinstance(offered, Available):
                continue
            contract: PortContract = offered.value
            given = member.query(port_side(type(member), port))
            where = f"{name}.{port}"
            assert isinstance(given, Available), where
            side: Side = given.value
            frame: object = type(member)
            for part in port.split("."):
                frame = getattr(frame, part)
            if isinstance(frame, StreamPort):
                label: str = member.field(frame.label).get()
                stream: Stream = member.field(frame.stream).get()
            else:
                label, stream = contract.label, cast(Stream, side.stream)
            assert label == contract.label, where
            transport = contract.contract.transport
            output = transport.endpoint is Endpoint.INITIATOR
            width = transport.data_width if output else None
            assert side == Side.kernel(stream, label, width), where
            # Its pace (W6): a scheduled port's, over its kernel's one schedule; a port of
            # another rule has none; a served port's is its child's port's.
            paced = member.query(port_pace(type(member), port))
            if isinstance(frame, StreamPort):
                scheduled = member.field(frame.advance).get() is Advance.SCHEDULED
                assert isinstance(paced, Available) is scheduled, where
                if scheduled:
                    schedule = member.query(Kernel.schedule)
                    assert isinstance(schedule, Available), where
                    assert paced.value.schedule == schedule.value, where
            else:
                assert isinstance(paced, (Available, Inapplicable)), where
            checked += 1
    return checked


def _rate_checked(point: Partition) -> int:
    """Each member of ``point`` binding a module, its cycles checked against the rate law
    (CONTRACT.md L6, W6): its schedule's iterations where it states one, else the most
    beats a frame a placed port of it carries. How many it checked."""
    checked = 0
    for name in point.members:
        member = point.member(name)
        if not isinstance(member, Kernel) or type(member).module_spec is None:
            continue
        cycles = member.query(Kernel.cycles)
        assert isinstance(cycles, Available), name
        schedule = member.query(Kernel.schedule)
        if isinstance(schedule, Available):
            assert cycles.value == schedule.value.iterations, name
        else:
            assert isinstance(schedule, Inapplicable), name
            beats = [
                member.field(port.stream).get().sequence.form.beats
                for port in stream_ports(type(member)).values()
                if member.field(port.placed).get()
            ]
            assert cycles.value == max(beats), name
        checked += 1
    return checked


@pytest.mark.parametrize("case", sorted(conformance_cases()))
def test_every_module_kernel_s_cycles_follow_the_rate_law(case: str) -> None:
    base = unchosen(**conformance_cases()[case]())
    for chosen in _covering(base).points or ({},):
        checked = _rate_checked(configured(base, chosen))
        # MatMul binds no module: its cycles are its core's, a dotp, checked in dotp's cases.
        assert checked or case == "matmul", f"{case} at {chosen}"


def test_no_library_kernel_states_its_own_cycles() -> None:
    """The base derives every module kernel's cycles by the rate law (W6); a kernel states
    none of its own. MemStream restates the base's so that its channel counts it by kind."""
    found: set[type] = set()
    for module in pkgutil.iter_modules(finn.kernels.__path__, "finn.kernels."):
        for value in vars(importlib.import_module(module.name)).values():
            if isinstance(value, type) and issubclass(value, Kernel) and value.module_spec:
                found.add(value)
    assert len(found) >= 11, found
    for kind in found:
        for cls in kind.__mro__[: kind.__mro__.index(Kernel)]:
            if "cycles" in vars(cls):
                assert cls is MemStreamKernel, cls


@pytest.mark.parametrize("case", sorted(conformance_cases()))
def test_every_port_states_its_contracts_label_and_its_side(case: str) -> None:
    base = unchosen(**conformance_cases()[case]())
    for chosen in _covering(base).points or ({},):
        assert _ports_checked(configured(base, chosen)), f"{case} at {chosen}"
