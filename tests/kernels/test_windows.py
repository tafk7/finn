# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A sliding window is a channel's adapter: one ``input_gen`` reading the image's frame.

A consumer presenting a window of its image (``oh + kh``) is fed by a channel
whose plan is a reorder (``finn.dataflow.traversal``), so its adapter is the
``input_gen`` candidate, with ``FM_SIZE`` the image, ``DIMS`` the window's loops
and ``COEFS`` their steps through the image. The review's probe, a flat 3 x 3
window over a 4 x 4 map, is that module. Under XSim the realized module carries
every case's sequence over two frames with a stalled output, among them frames
whose beats a stride passes (dropped), and holds far less than a frame:
``input_gen`` frees what no later window reads (a line buffer).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.rtl import simulate
from finn.core.space import design_space
from finn.dataflow.plan import Step, plan
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Traversal, vector_major
from finn.harness.toolchain import finnlib_root
from finn.kernels.adapters import ADAPTERS, Stage
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.artifacts.module import Leaf
from finn.kernels.configure import commit
from finn.kernels.input_generator import InputGeneratorKernel
from kernels.helpers import FULL_DSP48E2
from kernels.xsim import requires_xsim

ELEMENT = ScalarEncoding(DataType["INT8"])
oh, ow, kh, kw, c = (Index(name) for name in ("oh", "ow", "kh", "kw", "c"))


def three_taps() -> tuple[Traversal, Traversal]:
    """A 1-D window: three taps at stride 1 over six inputs."""
    return vector_major((6,), 1), Schedule({oh: 4, kh: 3}).present((6,), (oh + kh,))


def three_by_three(
    h: int, w: int, ch: int, simd: int, stride: int = 1, dilation: int = 1
) -> tuple[Traversal, Traversal]:
    """A 3 x 3 window over an (h, w, ch) image, ``simd`` channels a beat."""
    span = 2 * dilation + 1
    schedule = Schedule(
        {oh: (h - span) // stride + 1, ow: (w - span) // stride + 1, kh: 3, kw: 3, c: ch},
        factors={c: simd},
        order=(oh, ow, kh, kw, c),
    )
    access = (oh * stride + kh * dilation, ow * stride + kw * dilation, c)
    window = schedule.present((h, w, ch), access, lanes=(c,))
    return vector_major((h, w, ch), simd), window


def first_of_two() -> tuple[Traversal, Traversal]:
    """A sink reading the first beat of every two: the frame's second beat is dropped."""
    return vector_major((4,), 2), Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))


CASES: dict[str, Callable[[], tuple[Traversal, Traversal]]] = {
    "three_taps": three_taps,
    "three_by_three": lambda: three_by_three(4, 4, 2, 1),
    # CNV's first layer: 32 x 32 x 3 at SIMD 3.
    "cnv_conv0": lambda: three_by_three(32, 32, 3, 3),
    # Stride 2, dilation 2 over 8 x 8: only rows and columns 0, 2, 4 and 6 are read.
    "strided_dilated": lambda: three_by_three(8, 8, 2, 1, stride=2, dilation=2),
    "first_of_two": first_of_two,
}


def adapter(source: Traversal, sink: Traversal) -> Stage:
    """The channel's adapter for ``source`` to ``sink``: its one stage."""
    found = plan(BeatSequence(source), BeatSequence(sink))
    assert found.steps == (Step.REORDER,)
    candidate = design_space(
        ADAPTERS["input_gen"](
            tensor=Tensor(source.shape, ELEMENT), plan=found, platform=FULL_DSP48E2
        )
    )
    (stage,) = commit(candidate, {"input_gen.ram_style": "auto"}).stages
    return stage


def parameter(stage: Stage, name: str) -> int:
    value = dict(stage.module.parameters)[name]
    assert isinstance(value, int)
    return value


def nest(stage: Stage) -> tuple[int, str, str]:
    parameters = dict(stage.module.parameters)
    dims, coefs = parameters["DIMS"], parameters["COEFS"]
    assert isinstance(dims, str) and isinstance(coefs, str)
    return parameter(stage, "FM_SIZE"), dims, coefs


def test_the_three_tap_window_is_an_input_gen_over_the_frame() -> None:
    assert nest(adapter(*three_taps())) == (6, "'{4, 3}", "'{1, 1}")


def test_a_three_by_three_window_steps_rows_pixels_and_channel_beats() -> None:
    # Two channels, one a beat: a window row is three pixels' six beats.
    assert nest(adapter(*three_by_three(4, 4, 2, 1))) == (32, "'{2, 2, 3, 6}", "'{8, 2, 8, 1}")
    assert nest(adapter(*three_by_three(32, 32, 3, 3))) == (
        1024,
        "'{30, 30, 3, 3}",
        "'{32, 1, 32, 1}",
    )


def test_the_reviews_flat_probe_is_the_adapters_module() -> None:
    # A flat 3 x 3 window over a 4 x 4 map, as the kernel-system review built it by hand.
    flat = design_space(
        InputGeneratorKernel(
            word_bits=8,
            frame_words=16,
            dims=(2, 2, 3, 3),
            strides=(4, 1, 4, 1),
            platform=FULL_DSP48E2,
        )
    ).with_choices(ram_style="auto")
    stage = adapter(*three_by_three(4, 4, 1, 1))
    module = flat.module
    assert isinstance(module, Leaf)
    assert dict(module.parameters) == dict(stage.module.parameters)


# -- XSim ---------------------------------------------------------------------------------

PASSES = 2


def expected_words(source: Traversal, sink: Traversal) -> list[int]:
    """Each sink beat as the index of the source beat that presents it, pass after pass."""
    index = {beat: number for number, beat in enumerate(source.positions())}
    words = [index[beat] for beat in sink.positions()]
    return [source.beats * done + word for done in range(PASSES) for word in words]


def bench(stage: Stage, source: Traversal, expected: list[int]) -> str:
    """Words carry their source beat index; the output stalls, then 3 cycles in 7 accept."""
    width, levels = parameter(stage, "DATA_WIDTH"), parameter(stage, "D")
    parameters = ", ".join(f".{key}({raw})" for key, raw in stage.module.abi.parameters)
    sent_total = PASSES * source.beats
    table = ",".join(f"{width}'h{word:x}" for word in expected)
    return f"""module check;
    logic clk=0; always #5 clk=~clk;
    logic rst=1;
    logic [{width - 1}:0] idat;
    logic ivld=0, ordy=0;
    wire irdy, ovld;
    wire [{width - 1}:0] odat;
    wire [{levels - 1}:0] olst;
    {stage.module.name} #({parameters})
        dut(.clk, .rst, .idat, .ivld, .irdy, .odat, .ovld, .olst, .ordy);
    logic [{width - 1}:0] expected[{len(expected)}] = '{{{table}}};
    integer sent=0, received=0;
    logic held=0; logic [{width - 1}:0] held_data;
    initial begin
        repeat(25) @(negedge clk);
        rst=0;
        for(integer cycle=0; received<{len(expected)}; cycle=cycle+1) begin
            if(cycle > {8 * len(expected) + 4 * sent_total + 1000})
                $fatal(1,"timeout at %0d", received);
            @(negedge clk);
            ivld=sent<{sent_total};
            idat=sent;
            // First the output stalls until the input stops: what is accepted is the buffer.
            if(cycle=={sent_total + 64})
                $display("CAPACITY accepted=%0d frame={source.beats}", sent);
            ordy=cycle>{sent_total + 64} && (cycle%7)>=4;
            @(posedge clk);
            if(held && (!ovld || odat !== held_data)) $fatal(1,"output changed while stalled");
            held=ovld && !ordy; held_data=odat;
            if(ivld && irdy) sent=sent+1;
            if(ovld && ordy) begin
                if(odat !== expected[received])
                    $fatal(1,"output %0d: got %h expected %h",received,odat,expected[received]);
                received=received+1;
            end
        end
        if(sent!={sent_total}) $fatal(1,"input left: %0d of {sent_total}", sent);
        $display("WINDOW_PASS"); $finish;
    end
endmodule
"""


@requires_xsim
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_window_adapter_carries_the_window_in_xsim(case: str, tmp_path: Path) -> None:
    source, sink = CASES[case]()
    stage = adapter(source, sink)
    root = finnlib_root()
    sources = []
    for contribution in stage.module.sources:
        assert isinstance(contribution, CopiedSource)
        sources.append(root / contribution.path)
    log = simulate(sources, bench(stage, source, expected_words(source, sink)), tmp_path)
    found = re.search(r"CAPACITY accepted=(\d+)", log)
    assert found is not None, log
    accepted = int(found.group(1))
    print(f"{case}: {nest(stage)}; accepted while stalled: {accepted} of {source.beats}")
    if case == "cnv_conv0":
        # A line buffer, not the frame: 1024 beats in, at most a quarter held.
        assert accepted <= source.beats // 4
