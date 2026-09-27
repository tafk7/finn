# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical stream requirements, complete initialization, and RTL transport."""

import pytest

import shutil
import subprocess
from pathlib import Path

from finn.kernels.artifacts.abi import Bus, ComponentABI, Reset, Signal
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
from finn.kernels.streaming import (
    cyclic_stream_requirements,
    replay_buffer_requirements,
    replay_buffer_interfaces,
)
from finn.kernels.resources import resource_root

ROOT = Path(__file__).resolve().parents[2]
ROOTS = {"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"}


@pytest.mark.parametrize("bits,length,repeats", [(1, 1, 1), (13, 3, 2), (65, 5, 3)])
def test_replay_requirements_match_native_rtl(bits, length, repeats):
    requirements = replay_buffer_requirements(
        word_bits=bits, sequence_length=length, replay_count=repeats
    )
    assert dict(requirements.parameters) == {"W": bits, "LEN": length, "REP": repeats}
    abi = ComponentABI(
        requirements.abi.entry_point.value, requirements.abi.ports, requirements.abi.parameters
    )
    assert not any(isinstance(port, Bus) for port in abi.ports)
    streams = replay_buffer_interfaces(word_bits=bits)
    assert streams[0].data_width == streams[1].data_width == bits
    assert [(marker.signal, marker.kind.value) for marker in streams[1].markers] == [
        ("olast", "last"),
        ("ofin", "replay_end"),
    ]
    reset = next(port for port in abi.ports if isinstance(port, Signal) and port.name == "rst")
    assert reset.role == Reset(False, True, ("clk",))
    result = check_abi(
        abi, [ROOTS["finnlib"] / "rtl/replay_buffer.sv"], abi.entry_point, abi.parameters
    )
    assert not isinstance(result, Declined), result
    assert result == ()


@pytest.mark.parametrize("name", ["word_bits", "sequence_length", "replay_count"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_replay_rejects_invalid_dimensions(name, value):
    arguments = {"word_bits": 8, "sequence_length": 3, "replay_count": 2, name: value}
    with pytest.raises(ValueError, match=name):
        replay_buffer_requirements(**arguments)


@pytest.mark.parametrize("bits,image", [(1, (1,)), (13, (0, 8191, 37)), (65, (1 << 64, 7))])
def test_cyclic_image_is_complete_and_buildable(tmp_path, bits, image):
    requirements = cyclic_stream_requirements(
        word_bits=bits, depth=len(image), image=image, rom_style="auto"
    )
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
    first = cyclic_stream_requirements(word_bits=8, depth=3, image=(1, 2, 3), rom_style="auto")
    second = cyclic_stream_requirements(word_bits=8, depth=3, image=(1, 2, 4), rom_style="auto")
    assert module_build_fingerprint(first) != module_build_fingerprint(second)
    store = ArtifactStore(tmp_path / "store")
    prepared = [
        prepare_module_build(item, roots=ROOTS, template_roots=(), blobs=store)
        for item in (first, second)
    ]
    assert prepared_module_fingerprint(prepared[0]) != prepared_module_fingerprint(prepared[1])
    assert module_source_derivation(prepared[0]) == module_source_derivation(prepared[1])


@pytest.mark.parametrize(
    "arguments,match",
    [
        ({"word_bits": 0, "depth": 1, "image": (0,)}, "word_bits"),
        ({"word_bits": 8, "depth": 0, "image": ()}, "depth"),
        ({"word_bits": 8, "depth": True, "image": (0,)}, "depth"),
        ({"word_bits": 8, "depth": 3, "image": (1, 2)}, "exactly depth"),
        ({"word_bits": 8, "depth": 1, "image": (256,)}, "fitting word_bits"),
        ({"word_bits": 8, "depth": 1, "image": (-1,)}, "unsigned integers"),
        ({"word_bits": 8, "depth": 1, "image": (True,)}, "unsigned integers"),
        ({"word_bits": 8, "depth": 1, "image": (0,), "rom_style": "ultra"}, "rom_style"),
    ],
)
def test_cyclic_rejects_incomplete_or_invalid_initialization(arguments, match):
    with pytest.raises(ValueError, match=match):
        cyclic_stream_requirements(**{"rom_style": "auto", **arguments})


def test_cyclic_snapshots_the_image():
    image = [0, 255]
    requirements = cyclic_stream_requirements(word_bits=8, depth=2, image=image, rom_style="auto")
    image[0] = 11
    assert dict(requirements.parameters)["INIT_DATA"] == "16'hff00"


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

module replay_check #(
    parameter int LEN = 3,
    parameter int REP = 3,
    parameter int W = 13
)(output logic done = 0);
    logic clk = 0, rst = 1, ivld = 0, ordy = 0;
    logic [W-1:0] idat;
    wire [W-1:0] odat;
    wire irdy, ovld, olast, ofin;
    always #5 clk = !clk;
    replay_buffer #(.W(W), .LEN(LEN), .REP(REP)) dut (.*);
    function automatic logic [W-1:0] word(input int index);
        return W'((index+1)*71);
    endfunction
    initial begin
        int sent = 0;
        int received = 0;
        int cycles = 0;
        bit pending = 0;
        bit blocked = 0;
        logic [W+1:0] held;
        repeat (2) @(negedge clk);
        rst = 0;
        while (received < 3*LEN*REP && cycles < 1000) begin
            @(negedge clk);
            if (!pending) begin
                ivld = sent < 3*LEN && cycles % 5 != 0;
                idat = word(sent);
            end
            ordy = cycles % 7 >= 3;
            @(posedge clk);
            if (blocked && (!ovld || {odat, olast, ofin} !== held))
                $fatal(1, "replay LEN=%0d REP=%0d changed while stalled", LEN, REP);
            if (ovld && ordy) begin
                if (odat !== word((received / (LEN*REP))*LEN + received % LEN))
                    $fatal(1, "replay LEN=%0d REP=%0d wrong word %0d", LEN, REP, received);
                if (olast !== (received % LEN == LEN-1)) $fatal(1, "replay wrong last");
                if (ofin !== (received % (LEN*REP) == LEN*REP-1))
                    $fatal(1, "replay wrong final repetition flag");
                received++;
            end
            if (ivld && irdy) sent++;
            pending = ivld && !irdy;
            blocked = ovld && !ordy;
            held = {odat, olast, ofin};
            cycles++;
        end
        if (received != 3*LEN*REP || sent != 3*LEN) $fatal(1, "replay timed out");
        @(negedge clk);
        ordy = 0;
        ivld = 0;
        done = 1;
    end
endmodule

module stream_test;
    wire [7:0] done;
    // Each ROM_STYLE is one synthesis attribute over the same behavior.
    cyclic_check #(.W(1), .DEPTH(1), .IMAGE(1'b1), .ROM_STYLE("distributed")) c1(done[0]);
    cyclic_check c3(done[1]);
    cyclic_check #(
        .W(65), .DEPTH(4), .IMAGE(260'h1ffffffffffffffffa123456789abcdef0), .ROM_STYLE("block")
    ) c4(done[2]);
    replay_check #(.LEN(1), .REP(1)) r11(done[3]);
    replay_check #(.LEN(1), .REP(3)) r13(done[4]);
    replay_check #(.LEN(3), .REP(1)) r31(done[5]);
    replay_check r33(done[6]);
    replay_check #(.LEN(4), .REP(2)) r42(done[7]);
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
def test_rtl_streams_preserve_words_framing_stalls_and_reset(tmp_path):
    testbench = tmp_path / "stream_test.sv"
    testbench.write_text("`timescale 1ns/1ps\n" + _CYCLIC_TESTBENCH)
    sources = [
        ROOTS["finnlib"] / "rtl/replay_buffer.sv",
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
