# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ROM's cyclic stream: complete initialization, identity, and RTL transport."""

import pytest

import shutil
import subprocess
from pathlib import Path

from qonnx.core.datatype import DataType

from finn.core.space import Rejected, design_space
from finn.dataflow.traversal import vector_major
from finn.kernels.artifacts.build import (
    materialize_module_sources,
    module_build_fingerprint,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
    prepared_module_fingerprint,
)
from finn.kernels.artifacts.rtl import Declined, check_abi
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.resources import resource_root
from finn.kernels.rom import CYCLIC_ROM_STYLES, RomKernel

ROOT = Path(__file__).resolve().parents[2]
ROOTS = {"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"}


def rom(bits, image, rom_style="auto"):
    """A ROM of one-lane unsigned words: its image is its contents."""
    point = design_space(
        RomKernel(
            dtype=DataType[f"UINT{bits}"], form=vector_major((len(image),), 1), contents=image
        )
    )
    return point.with_choices(rom_style=rom_style)


@pytest.mark.parametrize("bits,image", [(1, (1,)), (13, (0, 8191, 37)), (65, (1 << 64, 7))])
def test_cyclic_image_is_complete_and_buildable(tmp_path, bits, image):
    requirements = rom(bits, image).build_requirements
    parameters = dict(requirements.parameters)
    literal_width, packed_hex = parameters["INIT_DATA"].split("'h")
    assert int(literal_width) == bits * len(image)
    packed = int(packed_hex, 16)
    decoded = tuple((packed >> (index * bits)) & ((1 << bits) - 1) for index in range(len(image)))
    assert decoded == image
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(requirements, roots=ROOTS, template_roots=(), blobs=store)
    assert not prepared.slots
    source = materialize_module_sources(prepared, store)
    component = portable_module_component(prepared, source)
    files = [Path(source.directory) / path for path, _ in component.files]
    result = check_abi(component.abi, files, component.abi.entry_point, component.abi.parameters)
    assert not isinstance(result, Declined), result
    assert result == ()


def test_cyclic_image_changes_concrete_identity_without_changing_reusable_source(tmp_path):
    first = rom(8, (1, 2, 3)).build_requirements
    second = rom(8, (1, 2, 4)).build_requirements
    assert module_build_fingerprint(first) != module_build_fingerprint(second)
    store = ArtifactStore(tmp_path / "store")
    prepared = [
        prepare_module_build(item, roots=ROOTS, template_roots=(), blobs=store)
        for item in (first, second)
    ]
    assert prepared_module_fingerprint(prepared[0]) != prepared_module_fingerprint(prepared[1])
    assert module_source_derivation(prepared[0]) == module_source_derivation(prepared[1])


@pytest.mark.parametrize(
    "image,code",
    [
        ((256,), "cyclic-values"),  # a word wider than its element
        ((-1,), "cyclic-values"),  # below an unsigned element
    ],
)
def test_cyclic_refuses_contents_outside_its_element(image, code):
    point = design_space(
        RomKernel(dtype=DataType["UINT8"], form=vector_major((1,), 1), contents=image)
    ).with_choices(rom_style="auto")
    answer = point.query(RomKernel.build_requirements)
    assert isinstance(answer, Rejected)
    assert code in {finding.code for finding in answer.findings}


def test_ultraram_is_no_rom_style():
    assert "ultra" not in CYCLIC_ROM_STYLES
    with pytest.raises(Exception, match="refused"):
        rom(8, (0,), rom_style="ultra")


_CYCLIC_TESTBENCH = r"""
module cyclic_check #(
    parameter int W = 13,
    parameter int DEPTH = 3,
    parameter logic [W*DEPTH-1:0] IMAGE = 39'h127ffe123,
    parameter ROM_STYLE = "auto"
)(output logic done = 0);
    logic clk = 0, rst = 1, ready = 0;
    wire [W-1:0] data;
    wire valid;
    always #5 clk = !clk;
    cyclic_stream #(.W(W), .DEPTH(DEPTH), .INIT_DATA(IMAGE), .ROM_STYLE(ROM_STYLE)) dut (
        .clk, .rst, .odat(data), .ovld(valid), .ordy(ready)
    );
    task automatic collect(input int count);
        int received = 0;
        int cycles = 0;
        bit blocked = 0;
        logic [W-1:0] held;
        while (received < count && cycles < count*10+20) begin
            @(negedge clk);
            ready = cycles % 7 >= 3;
            @(posedge clk);
            if (blocked && (!valid || data !== held))
                $fatal(1, "cyclic W=%0d DEPTH=%0d changed while stalled", W, DEPTH);
            if (valid && ready) begin
                if (data !== IMAGE[(received % DEPTH)*W +: W])
                    $fatal(1, "cyclic W=%0d DEPTH=%0d wrong word %0d", W, DEPTH, received);
                received++;
            end
            blocked = valid && !ready;
            held = data;
            cycles++;
        end
        if (received != count) $fatal(1, "cyclic timed out");
    endtask
    initial begin
        repeat (2) @(negedge clk);
        rst = 0;
        collect(DEPTH*3+1);
        @(negedge clk);
        ready = 0;
        repeat (3) @(negedge clk);
        // Reset a stalled valid word at a nonzero sequence position.
        rst = 1;
        @(posedge clk);
        #1;
        if (valid) $fatal(1, "cyclic valid did not clear on reset");
        @(negedge clk);
        rst = 0;
        collect(DEPTH*3+2);
        @(negedge clk);
        ready = 0;
        done = 1;
    end
endmodule

module stream_test;
    wire [2:0] done;
    // Each ROM_STYLE is one synthesis attribute over the same behavior.
    cyclic_check #(.W(1), .DEPTH(1), .IMAGE(1'b1), .ROM_STYLE("distributed")) c1(done[0]);
    cyclic_check c3(done[1]);
    cyclic_check #(
        .W(65), .DEPTH(4), .IMAGE(260'h1ffffffffffffffffa123456789abcdef0), .ROM_STYLE("block")
    ) c4(done[2]);
    initial begin
        wait (&done);
        $display("STREAM_COMPONENTS_PASS");
        $finish;
    end
    initial begin
        #20000;
        $fatal(1, "stream component watchdog");
    end
endmodule
"""


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("xvlog", "xelab", "xsim")),
    reason="Vivado simulator tools are unavailable",
)
def test_the_cyclic_stream_preserves_words_stalls_and_reset(tmp_path):
    testbench = tmp_path / "stream_test.sv"
    testbench.write_text("`timescale 1ns/1ps\n" + _CYCLIC_TESTBENCH)
    sources = [
        resource_root() / "cyclic_stream.sv",
        testbench,
    ]
    commands = (
        ["xvlog", "--sv", *(str(source) for source in sources)],
        [
            "xelab",
            "work.stream_test",
            "--mt",
            "2",
            "--snapshot",
            "stream_test",
            "--timescale",
            "1ns/1ps",
        ],
        ["xsim", "stream_test", "--runall"],
    )
    for command in commands:
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert "STREAM_COMPONENTS_PASS" in result.stdout, result.stdout + result.stderr
