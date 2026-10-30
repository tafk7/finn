# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's intake (CONTRACT.md § 5): how far it takes input ahead of delivering output.

Its terms are the laws module's (``PortIntake``, ``Intake``, ``Standing``): one entry an
input port, in beats of that port, read from the RTL (structural) or, for an HLS kernel,
measured. The base answers a kernel with a module that states none ``intake-unstated``, by
name, and its design point stays valid; a kernel with children derives its own from its
members', a port it serves by its name, any other below the member.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.conformance import KERNEL, configured, kernel_space, unchosen
from support.helpers import FULL_DSP48E2, codes
from support.kernel_specs import conformance_cases

from finn.core.space import (
    Available,
    Param,
    Rejected,
    Select,
    derived,
    design_space,
    inspection,
    view,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.harness.points import covering
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.rtl import Declined, evaluate
from finn.kernels.base import Kernel
from finn.kernels.laws import (
    INTAKE_UNSTATED,
    NATIVE,
    NATIVE_CLOCKING,
    Intake,
    PortIntake,
    Standing,
)
from finn.kernels.library import ModuleSpec
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.pool import PoolKernel
from finn.kernels.port_model import StreamPort, served, stream_ports
from finn.kernels.values.semantics import IntegerTensorValue
from finn.kernels.vpc import VpcKernel
from finn.platform.capabilities import Resources

TENSOR = Tensor((4, 1), ScalarEncoding(DataType["INT4"]))
ARRIVING = BeatSequence(vector_major((4, 1), 1))
PROBE = CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",))


def _intake(port: str, before_stop: int = 2, most: int = 3) -> PortIntake:
    return PortIntake(port, before_stop, most, "stops when full", Standing.STRUCTURAL, "probe.sv:1")


# -- the terms --------------------------------------------------------------------------------


def test_an_intake_counts_beats_and_states_its_rule_standing_and_source() -> None:
    entry = _intake("input")
    assert entry.under("adapter.input_gen") == _intake("adapter.input_gen.input")
    for wrong in (
        {"port": ""},
        {"before_stop": -1},
        {"most": 1.5},
        {"rule": ""},
        {"source": ""},
        {"standing": "structural"},
    ):
        given = {
            "port": "input",
            "before_stop": 2,
            "most": 3,
            "rule": "stops when full",
            "standing": Standing.STRUCTURAL,
            "source": "probe.sv:1",
        }
        with pytest.raises(ValueError):
            PortIntake(**(given | wrong))  # type: ignore[arg-type]


def test_an_intake_states_each_port_once() -> None:
    found = Intake((_intake("x"), _intake("w", 4, 4)))
    assert found.names == ("x", "w")
    assert found.of("w").before_stop == 4
    with pytest.raises(KeyError):
        found.of("y")
    with pytest.raises(ValueError, match=r"\['x'\]"):
        Intake((_intake("x"), _intake("x")))


# -- the base: unstated, stated, derived ------------------------------------------------------


class Stage(Kernel):
    """A model-only stage on a library module, one input port and one output port, which
    states no intake."""

    id = "test.intake.stage"
    clock_domain = NATIVE_CLOCKING

    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    input = StreamPort(transport=NATIVE, tensor=input_tensor, sequence=arriving)
    output = StreamPort(
        endpoint=Endpoint.INITIATOR, transport=NATIVE, tensor=input_tensor, sequence=arriving
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {"WIDTH": 4}

    @view
    def resource_use(self) -> Resources:
        return Resources()

    module_spec = ModuleSpec("probe", (PROBE,))


class StatedStage(Stage):
    """The stage, stating its intake."""

    id = "test.intake.stated"

    @derived
    def intake(self) -> Intake:
        return Intake((_intake("input"),))


class OtherStage(StatedStage):
    id = "test.intake.other"

    @derived
    def intake(self) -> Intake:
        return Intake((_intake("input", 5, 7),))


class Chain(Kernel):
    """Two stated stages, the first's input served as its own ``x``."""

    id = "test.intake.chain"
    core = Select(one=StatedStage, other=OtherStage, input_tensor=TENSOR, arriving=ARRIVING)
    kept = StatedStage(input_tensor=TENSOR, arriving=ARRIVING)
    x = served(core, "input")


class Holding(Kernel):
    """A stage that states none, beside one that does."""

    id = "test.intake.holding"
    first = StatedStage(input_tensor=TENSOR, arriving=ARRIVING)
    second = Stage(input_tensor=TENSOR, arriving=ARRIVING)


def test_a_kernel_stating_no_intake_answers_it_unstated_by_name_and_stays_valid() -> None:
    point = design_space(Stage(input_tensor=TENSOR, arriving=ARRIVING))
    found = point.query(Kernel.intake)
    assert isinstance(found, Rejected) and codes(found) == {INTAKE_UNSTATED}
    assert found.findings[0].message == "Stage states no intake"
    # No rule reads intake: the design point is valid.
    assert inspection.validity(point).status == "valid"


def test_a_kernel_with_children_derives_its_intake_from_its_members() -> None:
    point = design_space(Chain()).with_choices({Chain.core: "other"})
    # The served port by its name, the member's own below the member.
    assert point.intake == Intake((_intake("x", 5, 7), _intake("kept.input")))
    # A member that states none leaves its parent's unstated, by its name.
    holding = design_space(Holding()).query(Kernel.intake)
    assert isinstance(holding, Rejected) and codes(holding) == {INTAKE_UNSTATED}


def test_a_kernel_with_no_module_and_no_member_takes_nothing_ahead() -> None:
    class Wires(Kernel):
        id = "test.intake.wires"

    assert design_space(Wires()).query(Kernel.intake) == Available(Intake())


# -- the library's kernels: each states its own, from its RTL ---------------------------------


def _points(case: str) -> list[tuple[str, Any]]:
    """A conformance case's design at each covering point of its kernel's Decisions."""
    base = unchosen(**conformance_cases()[case]())
    space, pattern, prefix = kernel_space(base)
    points = covering(space, pattern).points or ({},)
    return [
        (f"{case} at {chosen}", configured(base, {prefix + key: v for key, v in chosen.items()}))
        for chosen in points
    ]


def _targets(member: Kernel) -> tuple[str, ...]:
    """The input ports a module kernel places, in declaration order."""
    return tuple(
        name
        for name, port in stream_ports(type(member)).items()
        if member.field(port.endpoint).get() is Endpoint.TARGET and member.field(port.placed).get()
    )


@pytest.mark.parametrize("case", sorted(conformance_cases()))
def test_every_module_kernel_states_its_intake_at_each_input_port_it_places(case: str) -> None:
    """At each covering point of each conformance case, every member binding a module
    states its intake (none unstated), one entry for each input port it places, in its
    declaration order: structural, read from the RTL, but an HLS kernel's, measured."""
    for where, point in _points(case):
        for name in point.members:
            member = point.member(name)
            if not isinstance(member, Kernel):
                continue
            found = member.query(Kernel.intake)
            assert isinstance(found, Available), f"{where}: {name}: {found}"
            if type(member).module_spec is None:
                continue
            assert found.value.names == _targets(member), f"{where}: {name}"
            hls = isinstance(member, PoolKernel)
            standing = Standing.MEASURED if hls else Standing.STRUCTURAL
            assert {each.standing for each in found.value.ports} <= {standing}, where


def _rtl(member: Kernel, top: str, names: tuple[str, ...], **binding: object) -> Any:
    """Constants ``names`` of ``top``, as slang elaborates it over ``member``'s module's
    sources at ``binding`` (its module's parameters, unless given)."""
    spec = type(member).module_spec
    assert spec is not None and spec.constant_sources is not None
    files = tuple(
        finnlib_root() / source.path
        for source in spec.constant_sources
        if isinstance(source, CopiedSource) and source.path.endswith(".sv")
    )
    given = binding or _parameters(member)
    found = evaluate(files, top, tuple((key, str(value)) for key, value in given.items()), names)
    assert not isinstance(found, Declined), found
    return found


def _parameters(member: Kernel) -> dict[str, int | str]:
    found = member.query(Kernel.parameters)
    assert isinstance(found, Available), found
    return found.value


def _kernel(case: str) -> list[tuple[str, Kernel, Intake]]:
    found = []
    for where, point in _points(case):
        member = point.member(KERNEL)
        intake = member.query(Kernel.intake)
        assert isinstance(intake, Available), where
        found.append((where, member, intake.value))
    return found


def test_the_credit_kernels_state_their_rtl_s_credit_and_sidestep() -> None:
    """eltwise and pwpolyf take ``CREDIT + 1`` beats a port, ``CREDIT`` as slang reads it."""
    for case, ports in (("eltwise", ("lhs", "rhs")), ("pwpolyf_gelu", ("input",))):
        for where, member, intake in _kernel(case):
            top = "eltwise" if case == "eltwise" else "pwpolyf"
            credit = _rtl(member, top, ("CREDIT",))["CREDIT"]
            for port in ports:
                assert intake.of(port).before_stop == intake.of(port).most == credit + 1, where


def test_thresholding_states_its_rtl_s_pending_cap() -> None:
    """``MAX_PENDING``, its RTL's own text: slang declines to elaborate ``thresholding``
    (its ``THRESHOLDS`` default pattern), so its formula is pinned here as written."""
    text = (finnlib_root() / "rtl/nonlin/thresholding.sv").read_text()
    assert "localparam int unsigned  M = $clog2(N+1);" in text
    assert "localparam int unsigned  MAX_PENDING = (DEEP_PIPELINE+1)*M + 3;" in text
    for case in ("thresholding", "thresholding-written-rows"):
        for where, member, intake in _kernel(case):
            given = _parameters(member)
            m = int(given["N"]).bit_length()  # $clog2(N+1)
            entry = intake.of("input")
            pending = (int(given["DEEP_PIPELINE"]) + 1) * m + 3
            assert entry.before_stop == entry.most == pending, where


def test_transpose_states_both_its_rtl_s_pages_and_its_skids() -> None:
    """Both pages before it stops; at most both and the ten output beats its two skids hold
    (an XSim trace at a page of four beats held eighteen, rtl-intake on the shelf)."""
    for where, member, intake in _kernel("transpose"):
        page = _rtl(member, "inner_shuffle", ("PAGE_OFFSET",))["PAGE_OFFSET"]
        entry = intake.of("input")
        assert (entry.before_stop, entry.most) == (2 * page, 2 * page + 10), where


def test_vpc_states_its_intake_by_the_branch_its_rtl_takes() -> None:
    names = ("GCD", "PI0", "PO0", "N0")
    for lanes_in, lanes_out, counts in (
        (3, 3, (0, 0)),  # the wire
        (6, 2, (1, 2)),  # the serializer
        (2, 6, (4, 4)),  # the deserializer
        (2, 3, (2, 3)),  # the general branch: 3 // 2 + 1, 1 + ceil(3 / 2)
        (3, 2, (1, 2)),
        (4, 6, (2, 3)),  # normalized by the GCD: 2 and 3
    ):
        tensor = Tensor((12,), ScalarEncoding(DataType["INT4"]))
        vpc = design_space(
            VpcKernel(
                input_tensor=tensor,
                arriving=BeatSequence(vector_major((12,), lanes_in)),
                leaving=BeatSequence(vector_major((12,), lanes_out)),
            )
        )
        found = _rtl(vpc, "vpc", names)
        pi0, po0 = found["PI0"], found["PO0"]
        assert found["N0"] == pi0 * po0  # whole vectors: no padded beat
        entry = vpc.intake.of("input")
        assert (entry.before_stop, entry.most) == counts, (lanes_in, lanes_out)


def test_memstream_states_its_set_port_s_intake_by_its_credit() -> None:
    """With one set its set port is idle and it states none; with several, its set port's,
    by FULL_CREDIT as slang reads it."""
    contents = tuple(tuple(row) for row in ((1, 2, 3), (4, 5, 6)))
    for depth, pumped, counts in (
        (1, False, (9, 9)),
        (2, False, (5, 6)),
        (3, False, (3, 5)),
        (8, False, (2, 3)),
        (9, False, (1, 3)),
        (1, True, (8, 9)),
        (3, True, (4, 5)),
        (6, True, (3, 4)),
        (7, True, (2, 4)),
    ):
        facts: dict[str, Any] = {
            "dtype": DataType["INT4"],
            "form": vector_major((depth, 3), 3),
            "contents": IntegerTensorValue.of(tuple(contents[0] for _ in range(depth))),
            "platform": FULL_DSP48E2,
        }
        one = design_space(MemStreamKernel(**facts)).with_choices(
            ram_style="distributed", pumped_memory=pumped
        )
        assert one.intake == Intake()
        sets = IntegerTensorValue.of(tuple(tuple(contents[0] for _ in range(depth)) for _ in "ab"))
        several = design_space(
            MemStreamKernel(
                **{**facts, "contents": sets},
                sets=2,
                set_tensor=Tensor((4,), ScalarEncoding(DataType["UINT1"])),
            )
        ).with_choices(ram_style="distributed", pumped_memory=pumped)
        entry = several.intake.of("set")
        assert (entry.before_stop, entry.most) == counts, (depth, pumped)
        credit = _rtl(several, "memstream", ("FULL_CREDIT",), SETS=2, DEPTH=depth, WIDTH=12)
        assert credit["FULL_CREDIT"] == 8


def test_pool_states_its_measured_intake_by_its_window_and_period() -> None:
    """Measured, in windows of ``K = KH * KW`` beats: ``4K + 5`` before it stops, ``5K + 5``
    at most at 5 and 10 ns, one more each at 2.5 ns (the shelf's pool-intake traces)."""
    for where, member, intake in _kernel("pool"):
        window = member.query(PoolKernel.window)
        assert isinstance(window, Available), where
        rows, columns = window.value
        entry, k = intake.of("input"), rows * columns
        assert isinstance(member, PoolKernel)
        period = member.platform.period_ns
        expected = (4 * k + 5 + (period <= 2.5), 5 * k + 5 + (period < 5.0))
        assert (entry.before_stop, entry.most) == expected, where
        assert entry.standing is Standing.MEASURED and "Vitis HLS" in entry.source, where
