# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stream contracts, checked composition, and the reusable cyclic delivery kernel."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, Unresolved
from finn.kernels.artifacts.abi import Clock, Direction, Endpoint, Reset, Signal
from finn.kernels.artifacts.build import (
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    RenderedSourceRequirement,
    SELF_CONTAINED_JINJA_RENDERER,
    materialize_module_sources,
    prepare_module_build,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.datatypes.scalar import ScalarEncoding
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.composition import Composition, StreamEnd
from finn.kernels.physical.contract import StreamContract, StreamMismatch, compatibility
from finn.kernels.physical.forms import Batch, Every, Fold, Repeat, Repetition, Tile, pack
from finn.kernels.physical.lowering import lower_module_structure
from finn.kernels.physical.stream import MarkerKind, ReadyValidStream, StreamMarker
from finn.kernels.resources import resource_root, template_root
from kernels.test_migrated_simple import eltwise

ROOT = Path(__file__).resolve().parents[2]
INT3 = ScalarEncoding(DataType["INT3"])
INT4 = ScalarEncoding(DataType["INT4"])


def native(name, bits, endpoint, clock="clk", reset="rst", markers=()):
    return ReadyValidStream(
        name, bits, endpoint, f"{name}_d", f"{name}_v", f"{name}_r", clock, reset, markers
    )


def contract(form, endpoint, *, element=INT3, repetition=Repetition.ONCE, width=None, **kw):
    bits = width or form.lanes * element.bits
    return StreamContract(native("s", bits, endpoint), element, form, repetition, **kw)


def codes(mismatches):
    return {item.code for item in mismatches}


# -- forms -------------------------------------------------------------------------------


def test_tile_form_is_the_mvau_weight_order_and_packs_the_known_image():
    tile = Tile(4, 4, 2, 2)
    assert (tile.lanes, tile.beats, tile.shape) == (4, 4, (4, 4))
    first, second = list(tile.positions())[:2]
    assert first == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert second == ((0, 2), (0, 3), (1, 2), (1, 3))
    weights = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
    # Hand-packed INT3 fields: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert pack(tile, weights, 3) == (0x22C, 0x6BE, 0xDD3, 0x941)


def test_composite_forms_enumerate_repetition_and_batches():
    vector = Fold(4, 2)
    assert list(Repeat(vector, 2).positions()) == [((0,), (1,)), ((2,), (3,))] * 2
    assert list(Batch(vector, 2).positions())[2] == ((1, 0), (1, 1))
    assert Batch(Repeat(vector, 3), 2).beats == 12
    assert Repeat(Tile(4, 4, 2, 2), 3).shape == (4, 4)
    with pytest.raises(ValueError, match="divide"):
        Fold(5, 2)
    with pytest.raises(ValueError, match="shape"):
        pack(vector, (1, 2, 3), 4)
    assert Every(3).asserted(2) and not Every(3).asserted(3)


# -- compatibility -----------------------------------------------------------------------


def test_equal_contracts_connect_and_cyclic_sources_repeat_into_consumer_passes():
    tile = Tile(4, 4, 2, 2)
    source = contract(tile, Endpoint.INITIATOR, repetition=Repetition.CYCLIC)
    for sink_form in (tile, Repeat(tile, 5)):
        sink = contract(sink_form, Endpoint.TARGET)
        assert compatibility(source, sink, source_is_top=False, sink_is_top=False) == ()


@pytest.mark.parametrize(
    "source,sink,code",
    [
        # Same lanes and widths, different positions: the principal logical gap.
        (Tile(4, 4, 1, 4), Tile(4, 4, 4, 1), "stream-form"),
        (Fold(4, 2), Fold(4, 4), "stream-lanes"),
        (Fold(8, 2), Repeat(Fold(4, 2), 2), "stream-form"),
    ],
)
def test_logical_mismatches_are_refused(source, sink, code):
    found = compatibility(
        contract(source, Endpoint.INITIATOR),
        contract(sink, Endpoint.TARGET),
        source_is_top=False,
        sink_is_top=False,
    )
    assert code in codes(found)


def test_element_repetition_direction_and_marker_rules_are_checked():
    fold = Fold(4, 2)
    ok = contract(fold, Endpoint.INITIATOR)
    assert "stream-element" in codes(
        compatibility(
            ok,
            contract(fold, Endpoint.TARGET, element=INT4, width=8),
            source_is_top=False,
            sink_is_top=False,
        )
    )
    assert "stream-repetition" in codes(
        compatibility(
            ok,
            contract(fold, Endpoint.TARGET, repetition=Repetition.CYCLIC),
            source_is_top=False,
            sink_is_top=False,
        )
    )
    assert "stream-direction" in codes(
        compatibility(
            ok, contract(fold, Endpoint.INITIATOR), source_is_top=False, sink_is_top=False
        )
    )
    last = (StreamMarker("s_m", MarkerKind.LAST),)
    produced = StreamContract(
        native("s", 6, Endpoint.INITIATOR, markers=last), INT3, fold, markers={"s_m": Every(2)}
    )
    required = StreamContract(
        native("s", 6, Endpoint.TARGET, markers=last), INT3, fold, markers={"s_m": Every(3)}
    )
    assert "stream-marker" in codes(
        compatibility(produced, required, source_is_top=False, sink_is_top=False)
    )


def test_contracts_reject_lanes_wider_than_the_word_and_unknown_marker_rules():
    with pytest.raises(ValueError, match="exceed"):
        contract(Fold(4, 4), Endpoint.TARGET, width=8)
    with pytest.raises(ValueError, match="marker"):
        contract(Fold(4, 2), Endpoint.TARGET, markers={"missing": Every(2)})


# -- the delivery kernel -----------------------------------------------------------------


def delivery(form=Fold(4, 2), values=(1, -2, 7, -8), **choices):
    base = CyclicDelivery(dtype=DataType["INT4"], form=form, values=values)
    return base.with_choices(**choices) if choices else base


def test_delivery_publishes_a_cyclic_contract_and_waits_only_for_its_own_choice():
    base = delivery()
    output = base.output()
    assert output.repetition is Repetition.CYCLIC and output.form == Fold(4, 2)
    assert output.payload_bits == output.transport.data_width == 8
    assert base.image == (0xE1, 0x87)
    assert isinstance(base.build_requirements.query(), Unresolved)
    requirements = delivery(rom_style="block").build_requirements()
    assert dict(requirements.parameters)["ROM_STYLE"] == '"block"'


@pytest.mark.parametrize(
    "values,dtype,message",
    [
        ((1, 2, 3), "INT4", "shape"),
        ((1, 2, 3, 8), "INT4", "admitted"),
        ((1, 2, 3, 4), "FLOAT32", None),
    ],
)
def test_delivery_refuses_values_outside_the_operand_contract(values, dtype, message):
    point = CyclicDelivery(dtype=DataType[dtype], form=Fold(4, 2), values=values)
    answer = point.with_choices(rom_style="auto").build_requirements.query()
    assert isinstance(answer, Rejected)
    if message:
        assert any(message in finding.message for finding in answer.findings)


# -- a second consumer: cyclic channel parameters into eltwise ---------------------------

CLOCKING = (
    Signal("ap_clk", Direction.IN, 1, Clock()),
    Signal(
        "ap_rst_n",
        Direction.IN,
        1,
        Reset(active_low=True, synchronous=True, synchronous_to=("ap_clk",)),
    ),
)
CHANNELS, PE, PIXELS = 4, 2, 3
PARAMETERS = (1, -2, 7, -8)


def eltwise_with_constant(*, form=Fold(CHANNELS, PE), values=PARAMETERS):
    """Eltwise ADD whose rhs is a cyclic channel vector, composed through contracts."""
    compute = eltwise(operation="ADD", pe=PE, lhs="INT4", rhs="INT4")
    source = delivery(form, values, rom_style="distributed")
    lhs, rhs, result = compute.interfaces()
    pixels = Batch(Fold(CHANNELS, PE), PIXELS)
    int5 = ScalarEncoding(DataType["INT5"])

    def top(name, element, endpoint):
        stream = AxiStream(name, element.dtype, PE, endpoint=endpoint)
        return StreamContract(stream.native(clock="ap_clk", reset="ap_rst_n"), element, pixels)

    x_top, y_top = top("in0_V", INT4, Endpoint.TARGET), top("out0_V", int5, Endpoint.INITIATOR)
    top_abi = ModuleABIRequirements(
        GeneratedModuleName("eltwise_constant"),
        (*CLOCKING, x_top.transport.axis_bus(), y_top.transport.axis_bus()),
        (),
    )
    composition = Composition(top_abi)
    composition.add("u_rhs", source.build_requirements())
    composition.add("u_eltwise", compute.build_requirements())
    for owner in ("u_rhs", "u_eltwise"):
        composition.drive(owner, "clk", "ap_clk")
        composition.drive(owner, "rst", "ap_rst_n")
    composition.connect(
        StreamEnd(None, x_top), StreamEnd("u_eltwise", StreamContract(lhs, INT4, pixels))
    )
    composition.connect(
        StreamEnd("u_rhs", source.output()),
        StreamEnd("u_eltwise", StreamContract(rhs, INT4, Repeat(Fold(CHANNELS, PE), PIXELS))),
    )
    composition.connect(
        StreamEnd("u_eltwise", StreamContract(result, int5, pixels)), StreamEnd(None, y_top)
    )
    structure = composition.finish()
    wrapper = RenderedSourceRequirement(
        EntryPointSourceName(),
        "decomposed_wrapper.sv.j2",
        ("PORT_DECLARATIONS", "NET_DECLARATIONS", "ASSIGNMENTS", "INSTANCES"),
        SELF_CONTAINED_JINJA_RENDERER,
        requires=tuple(
            "module:" + instance.requirements.abi.entry_point.value
            for instance in structure.instances
            if isinstance(instance.requirements.abi.entry_point, FixedModuleName)
        ),
        provides_entry_point=True,
    )
    return lower_module_structure(
        structure, producer=ProducerIdentity("test.eltwise_constant", "1"), wrapper_template=wrapper
    ), structure


def test_the_delivery_kernel_serves_a_second_consumer_through_the_same_contract():
    _, structure = eltwise_with_constant()
    wires = {(w.destination.pin.instance_id, w.destination.pin.signal_id) for w in structure.wires}
    assert ("u_eltwise", "bdat") in wires and ("u_rhs", "ordy") in wires
    # A delivery whose form does not match the consumer's broadcast is refused.
    # A delivery in another form (one batched vector) is refused, though its
    # lanes, element and words per pass are identical.
    with pytest.raises(StreamMismatch, match="stream-form"):
        eltwise_with_constant(form=Batch(Fold(CHANNELS, PE), 1), values=(PARAMETERS,))


def test_clock_domains_must_be_attached_and_equal():
    stream = native("o", 8, Endpoint.INITIATOR, clock="clk")
    sink = native("i", 8, Endpoint.TARGET, clock="clk")
    top_abi = ModuleABIRequirements(GeneratedModuleName("t"), CLOCKING, ())
    composition = Composition(top_abi)
    composition.add("a", delivery(rom_style="auto").build_requirements())
    composition.add("b", delivery(rom_style="auto").build_requirements())
    found = composition.check(
        StreamEnd("a", StreamContract(stream, INT4, Fold(4, 2))),
        StreamEnd("b", StreamContract(sink, INT4, Fold(4, 2))),
    )
    assert "stream-clock" in codes(found)


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("xvlog", "xelab", "xsim")),
    reason="Vivado simulator tools are unavailable",
)
def test_eltwise_with_cyclic_constant_computes_the_broadcast_sum(tmp_path):
    requirements, _ = eltwise_with_constant()
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    sources = [str(Path(materialized.directory) / path) for path in materialized.files]
    inputs = [(-8 + 3 * index) % 16 - 8 for index in range(CHANNELS * PIXELS)]
    expected = [value + PARAMETERS[index % CHANNELS] for index, value in enumerate(inputs)]

    def pack_words(values, bits):
        return [
            sum(
                (value & ((1 << bits) - 1)) << (lane * bits)
                for lane, value in enumerate(values[i : i + PE])
            )
            for i in range(0, len(values), PE)
        ]

    words_in, words_out = pack_words(inputs, 4), pack_words(expected, 5)
    count = len(words_in)
    testbench = tmp_path / "check.sv"
    testbench.write_text(f"""`timescale 1ns/1ps
module check;
    logic ap_clk = 0, ap_rst_n = 0;
    logic [7:0] in0_V_tdata; logic in0_V_tvalid = 0; wire in0_V_tready;
    wire [15:0] out0_V_tdata; wire out0_V_tvalid; logic out0_V_tready = 0;
    logic [7:0] words_in [{count}] = '{{{", ".join(f"8'h{w:02x}" for w in words_in)}}};
    logic [9:0] words_out [{count}] = '{{{", ".join(f"10'h{w:03x}" for w in words_out)}}};
    always #5 ap_clk = !ap_clk;
    {prepared.abi.entry_point} dut (.*);
    int sent = 0, received = 0, cycle = 0;
    always @(posedge ap_clk) begin
        cycle <= cycle + 1;
        if (ap_rst_n) begin
            if (in0_V_tvalid && in0_V_tready) sent <= sent + 1;
            if (out0_V_tvalid && out0_V_tready) begin
                if (out0_V_tdata[9:0] !== words_out[received])
                    $fatal(1, "word %0d: %h != %h", received, out0_V_tdata, words_out[received]);
                received <= received + 1;
            end
        end
    end
    always @(negedge ap_clk) begin
        in0_V_tvalid = ap_rst_n && sent < {count} && cycle % 3 != 0;
        in0_V_tdata = words_in[sent < {count} ? sent : 0];
        out0_V_tready = cycle % 4 != 1;
    end
    initial begin
        repeat (4) @(posedge ap_clk);
        ap_rst_n = 1;
        wait (received == {count});
        $display("ELTWISE_CONSTANT_PASS");
        $finish;
    end
    initial begin #50000; $fatal(1, "timeout"); end
endmodule
""")
    vivado = Path(os.environ.get("XILINX_VIVADO", str(Path(shutil.which("xelab")).parent.parent)))
    commands = (
        ["xvlog", "--sv", *sources, str(vivado / "data/verilog/src/glbl.v"), str(testbench)],
        [
            "xelab",
            "work.check",
            "work.glbl",
            "--mt",
            "2",
            "-L",
            "unisims_ver",
            "--snapshot",
            "check",
            "--timescale",
            "1ns/1ps",
        ],
        ["xsim", "check", "--runall"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "ELTWISE_CONSTANT_PASS" in result.stdout, result.stdout
