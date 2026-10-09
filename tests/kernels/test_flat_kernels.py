# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authoring examples assessed against native modules, not copied width formulas."""

import struct
from dataclasses import replace

import pyslang
import pytest
from pyslang import ast, syntax
from qonnx.core.datatype import DataType

from finn.core.executors.xsim.rtl import simulate
from finn.core.space import (
    DefinitionError,
    Rejected,
    design_space,
)
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.rtl import TOLERATED_DIAGNOSTICS
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.input_generator import InputGeneratorKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import FULL_DSP48E2, FULL_DSP58, controlled, point_for
from kernels.xsim import requires_xsim

FINNLIB = finnlib_root()
SOURCE_ROOTS = {"finnlib": FINNLIB}


def fifo(**changes):
    facts = dict(word_bits=13, depth=8, platform=FULL_DSP48E2)
    facts.update(changes)
    return point_for(FifoKernel, facts, ram_style="auto")


def generator(**changes):
    facts = dict(word_bits=13, frame_words=6, dims=(3, 6), strides=(0, 1), platform=FULL_DSP48E2)
    facts.update(changes)
    return point_for(InputGeneratorKernel, facts, ram_style="auto")


@pytest.mark.parametrize(
    "space_type,facts",
    (
        (FifoKernel, dict(word_bits=13, depth=8)),
        (InputGeneratorKernel, dict(word_bits=13, frame_words=6, dims=(3, 6), strides=(0, 1))),
    ),
)
def test_an_ultra_memory_needs_the_platforms_ultraram(space_type, facts):
    without = design_space(space_type(**facts, platform=replace(FULL_DSP48E2, uram=False)))
    report = without.try_with_choices(ram_style="ultra")
    assert not report.accepted
    (outcome,) = report.outcomes
    assert {finding.code for finding in outcome.result.findings} == {"uram-absent"}
    # The memory starts empty: UltraRAM that takes no initial contents is enough.
    platform = replace(FULL_DSP48E2, uram_init=False)
    assert (
        design_space(space_type(**facts, platform=platform))
        .try_with_choices(ram_style="ultra")
        .accepted
    )


def eltwise(pe=2, **changes):
    """Flat: no channel binds its extent, so its folding factor is any the RTL takes, committed
    as a choice."""
    facts = dict(
        operation="ADD",
        lhs_dtype=DataType["INT3"],
        rhs_dtype=DataType["INT3"],
        b_scale=1.0,
        platform=FULL_DSP58,
    )
    facts.update(changes)
    return point_for(EltwiseKernel, facts, pe=pe)


def threshold(*, use_axilite=False, deep_pipeline=False, pe=1, **changes):
    """Flat, its PE is any the RTL takes with its table's rows; ``pe=None`` leaves it open
    with the memories (a table without rows has no PE to commit, and one without
    thresholds no stages)."""
    facts = dict(
        input_dtype=DataType["INT8"],
        threshold_dtype=DataType["INT5"],
        thresholds=(((-2, 0, 3), (-1, 1, 4)),),
        bias=-1,
        platform=FULL_DSP48E2,
    )
    facts.update(changes)
    factors = {} if pe is None else {"pe": pe, "ram_style": "auto", "ultra_stages": 0}
    # Runtime-writable thresholds present their bus through a control node.
    place = controlled if use_axilite else point_for
    return place(
        ThresholdingAxiKernel,
        facts,
        use_axilite=use_axilite,
        deep_pipeline=deep_pipeline,
        **factors,
    )


def native_ports(requirements, tmp_path):
    """Elaborate a real child so string/array parameters remain native SV values."""
    options = ast.CompilationOptions()
    options.topModules = {"probe"}
    options.flags = ast.CompilationFlags.IgnoreUnknownModules
    compilation = ast.Compilation(pyslang.Bag([options]))
    for contribution in requirements.sources:
        compilation.addSyntaxTree(
            syntax.SyntaxTree.fromFile(str(SOURCE_ROOTS[contribution.root] / contribution.path))
        )
    parameters = ", ".join(f".{name}({raw})" for name, raw in requirements.abi.parameters)
    wrapper = tmp_path / "probe.sv"
    wrapper.write_text(f"module probe; {requirements.name} #({parameters}) native(); endmodule\n")
    compilation.addSyntaxTree(syntax.SyntaxTree.fromFile(str(wrapper)))
    errors = [
        d
        for d in compilation.getAllDiagnostics()
        if d.isError() and str(d.code) not in TOLERATED_DIAGNOSTICS
    ]
    if errors:
        engine = pyslang.DiagnosticEngine(compilation.sourceManager)
        client = pyslang.TextDiagnosticClient()
        engine.addClient(client)
        for diagnostic in errors:
            engine.issue(diagnostic)
        pytest.fail(client.getString())
    probe = compilation.getRoot().topInstances[0]
    native = next(member for member in probe.body if member.name == "native")
    return {
        port.name: (
            {"In": "input", "Out": "output", "InOut": "inout"}[str(port.direction).split(".")[-1]],
            port.type.bitstreamWidth,
        )
        for port in native.body.portList
    }


@pytest.mark.parametrize(
    "factory",
    [
        fifo,
        lambda: fifo(depth=64, word_bits=17),
        generator,
        lambda: generator(frame_words=56, dims=(3, 4, 2, 3), strides=(16, 1, 16, 2)),
        eltwise,
        lambda: eltwise(operation="SUB", lhs_dtype=DataType["UINT7"], rhs_dtype=DataType["UINT7"]),
        lambda: eltwise(operation="MUL", lhs_dtype=DataType["FLOAT32"]),
        lambda: eltwise(lhs_dtype=DataType["FLOAT32"], rhs_dtype=DataType["FLOAT32"], b_scale=0.25),
        threshold,
        lambda: threshold(use_axilite=True, deep_pipeline=True),
        lambda: threshold(pe=2),
        lambda: threshold(pe=4),  # PE a multiple of C: rows carried in the lanes
        # One row shared by every channel (C = 1), its configuration address from N alone,
        # and none at all for one threshold.
        lambda: threshold(use_axilite=True, pe=4, thresholds=(((-2, 0, 3),),)),
        lambda: threshold(use_axilite=True, pe=4, thresholds=(((0,),),)),
        lambda: threshold(thresholds=(((-2, 0, 3), (-1, 1, 4)), ((-3, 0, 5), (-2, 0, 6)))),
    ],
)
def test_native_rtl_pin_names_directions_and_widths(factory, tmp_path):
    point = factory()
    requirements = point.module
    observed = native_ports(requirements, tmp_path)
    declared = {}
    for port in requirements.abi.pins:
        if isinstance(port, Bus):
            directions = dict(port.member_directions())
            declared.update(
                {
                    member.physical: (directions[member.physical].value, member.width)
                    for member in port.signals
                }
            )
        else:
            declared[port.name] = (port.direction.value, port.width)
    assert observed == declared
    emitted = emit_module(requirements, tmp_path / "module", roots=SOURCE_ROOTS)
    assert set(emitted.sources) == {source.path for source in requirements.sources}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: fifo(word_bits=0),
        lambda: fifo(depth=1),
        lambda: generator(dims=()),
        lambda: generator(strides=(1,)),
        lambda: generator(dims=(2, 6), strides=(1, 1)),
        lambda: generator(strides=(-1, 1)),
        lambda: eltwise(operation="DIV"),
        lambda: eltwise(lhs_dtype=DataType["INT4"]),
        lambda: eltwise(lhs_dtype=DataType["BIPOLAR"]),
        lambda: eltwise(b_scale=1e100),
        lambda: eltwise(b_scale=0.5),
        lambda: eltwise(operation="MUL", lhs_dtype=DataType["FLOAT32"], b_scale=0.5),
        lambda: eltwise(lhs_dtype=DataType["FLOAT32"], platform=FULL_DSP48E2),
        lambda: threshold(thresholds=(), pe=None),
        lambda: threshold(thresholds=(((2, 1),),)),
        lambda: threshold(thresholds=(((0, 20),),)),
        lambda: threshold(thresholds=(((0,), (0, 1)),)),
        lambda: threshold(threshold_dtype=DataType["UINT5"]),
        lambda: threshold(bias=1 << 31),
        lambda: threshold(bias=-10),
        lambda: threshold(use_axilite=True, thresholds=(((0, 1),), ((0, 1),))),
    ],
)
def test_unsupported_cases_are_refused_without_constructing_invalid_interfaces(factory):
    point = factory()
    assessment = point.inspect(type(point).module)
    assert isinstance(assessment.accepted_result, Rejected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: eltwise(pe=0),
        lambda: eltwise(pe=1 << 32),  # beyond the RTL's 32-bit PE
        lambda: threshold(pe=0),
        # Neither of the table's two rows and PE 3 is a multiple of the other.
        lambda: threshold(pe=3),
    ],
)
def test_a_folding_factor_outside_its_domain_is_refused_where_it_is_committed(factory):
    with pytest.raises(ValueError, match="domain-membership"):
        factory()


def test_typed_integer_vectors_and_tables_reject_mutable_or_mistyped_payloads():
    invalid_scale = eltwise(b_scale=float("nan"))
    assert isinstance(invalid_scale.query(EltwiseKernel.native_scale), Rejected)
    assert isinstance(invalid_scale.inspect(EltwiseKernel.module).accepted_result, Rejected)
    # A mistyped formal is refused at the node call.
    for bad in ([3, 6], (3, True), (3, [6])):
        with pytest.raises(DefinitionError):
            generator(dims=bad)
    with pytest.raises(DefinitionError):
        threshold(thresholds=(([-2, 0, 3],),))


@pytest.mark.parametrize(
    "operation,a,b,result",
    [
        ("ADD", "INT3", "INT3", "INT4"),
        ("ADD", "UINT3", "UINT3", "UINT4"),
        ("SUB", "UINT3", "UINT3", "INT4"),
        ("SBR", "UINT3", "UINT3", "INT4"),
        ("MUL", "UINT3", "UINT3", "UINT6"),
        ("MUL", "INT3", "INT3", "INT6"),
        ("ADD", "INT9", "FLOAT32", "FLOAT32"),
    ],
)
def test_elementwise_output_encoding_follows_operation_and_operand_types(operation, a, b, result):
    point = eltwise(operation=operation, lhs_dtype=DataType[a], rhs_dtype=DataType[b])
    _ = point.module
    assert point.result_dtype == DataType[result]


def test_rounding_of_scale_is_explicit_and_precedes_native_support_checks():
    point = eltwise(b_scale=1.0 + 2**-30)
    assert point.native_scale == 1.0
    assert dict(point.module.parameters)["B_SCALE"] == "1.0"


def test_threshold_initialization_is_owned_and_changes_the_module():
    a = threshold().module
    b = threshold(thresholds=(((-2, 0, 2), (-1, 1, 4)),)).module
    assert a.data[0].data == b"1e\n00\n03\n00\n1f\n01\n04\n00\n"
    assert dict(a.parameters)["THRESHOLDS_FILE"] != dict(b.parameters)["THRESHOLDS_FILE"]
    assert a != b
    assert threshold().result_dtype == DataType["INT3"]
    assert threshold(bias=0).result_dtype == DataType["UINT2"]


def test_required_root_bindings_and_explicit_optional_inputs_preserve_partial_queries():
    for kernel in (
        FifoKernel,
        InputGeneratorKernel,
        EltwiseKernel,
        ThresholdingAxiKernel,
    ):
        # A missing required formal is refused when design_space() prepares the root.
        with pytest.raises(DefinitionError, match="is not supplied"):
            point_for(kernel, {})

    assert all(not isinstance(port, Bus) for port in fifo().module.abi.pins)


def run(requirements, body, tmp_path):
    parameters = ", ".join(f".{key}({raw})" for key, raw in requirements.abi.parameters)
    dut = requirements.name + " #(" + parameters + ")"
    sources = [SOURCE_ROOTS[source.root] / source.path for source in requirements.sources]
    # The RTL reads its generated data (a THRESHOLDS_FILE) by name, where xsim runs.
    for item in requirements.data:
        (tmp_path / item.path).write_bytes(item.data)
    simulate(sources, body.replace("@DUT@", dut), tmp_path)


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def flow_case(case):
    """Independent numeric/sequence expectations, using each native interface."""
    if case == "fifo":
        point = fifo()
        inputs = [0, 1, 8191, 42, 500, 127, 4096, 19]
        return (
            point,
            13,
            1,
            13,
            inputs,
            [],
            inputs,
            "",
            ".clk, .rst, .idat(adat), .ivld(avld), .irdy(ardy), .odat, .ovld, .ordy",
        )
    if case == "generator":
        point = generator(frame_words=4, dims=(2, 4), strides=(0, 1))
        inputs = [10, 20, 30, 40, 50, 60, 70, 80]
        outputs = []
        for frame in (inputs[:4], inputs[4:]):
            for i, item in enumerate(frame * 2):
                last = (2 if i % 4 == 3 else 0) | (1 if i == 7 else 0)
                outputs.append(item | (last << 13))
        extra = (
            "wire [12:0] native_data; wire [1:0] native_last; "
            "assign odat={native_last,native_data};"
        )
        return (
            point,
            13,
            1,
            15,
            inputs,
            [],
            outputs,
            extra,
            ".clk, .rst, .idat(adat), .ivld(avld), .irdy(ardy), "
            ".odat(native_data), .olst(native_last), .ovld, .ordy",
        )
    if case == "threshold":
        point = threshold()
        numbers = [-100, -1, 0, 4, 100, 0, -2, -100]
        rows = ((-2, 0, 3), (-1, 1, 4))
        outputs = [
            (sum(t <= max(-16, min(15, item)) for t in rows[i % 2]) - 1) & 7
            for i, item in enumerate(numbers)
        ]
        connections = (
            ".ap_clk(clk), .ap_rst_n(!rst), .s_axis_tdata(adat), .s_axis_tvalid(avld), "
            ".s_axis_tready(ardy), .m_axis_tdata(odat), .m_axis_tvalid(ovld), "
            ".m_axis_tready(ordy), .s_axis_set_tdata('0), .s_axis_set_tvalid(1'b0), "
            ".s_axilite_AWVALID(1'b0), .s_axilite_WVALID(1'b0), .s_axilite_BREADY(1'b0), "
            ".s_axilite_ARVALID(1'b0), .s_axilite_RREADY(1'b0), .s_axilite_AWADDR('0), "
            ".s_axilite_ARADDR('0), .s_axilite_WDATA('0), .s_axilite_WSTRB('0)"
        )
        return point, 8, 1, 8, [item & 255 for item in numbers], [], outputs, "", connections
    connections = ".clk, .rst, .adat, .avld, .ardy, .bdat, .bvld, .brdy, .odat, .ovld, .ordy"
    if case == "integer":
        point = eltwise(operation="SUB", lhs_dtype=DataType["UINT3"], rhs_dtype=DataType["UINT3"])
        a, b = [(0, 7), (3, 1), (7, 7), (2, 5)], [(7, 0), (1, 7), (7, 7), (6, 2)]

        def pack3(row):
            return row[0] | (row[1] << 3)

        outputs = [((x[0] - y[0]) & 15) | (((x[1] - y[1]) & 15) << 4) for x, y in zip(a, b)]
        return point, 6, 6, 8, list(map(pack3, a)), list(map(pack3, b)), outputs, "", connections
    point = eltwise(pe=1, rhs_dtype=DataType["FLOAT32"])
    a, b = [-4, 3, -1, 0], [1.5, -2.5, 0.5, -0.25]
    return (
        point,
        3,
        32,
        32,
        [item & 7 for item in a],
        list(map(float_bits, b)),
        [float_bits(x + y) for x, y in zip(a, b)],
        "",
        connections,
    )


@requires_xsim
@pytest.mark.parametrize("case", ("fifo", "generator", "threshold", "integer", "float"))
def test_generated_rtl_preserves_values_sequences_and_backpressure(case, tmp_path):
    point, a_width, b_width, o_width, a, b, expected, extra, connections = flow_case(case)
    requirements = point.module

    def array(values, width):
        return "'{" + ",".join(f"{width}'h{item:x}" for item in values) + "}"

    body = f"""module check;
    logic clk=0; always #5 clk=~clk;
    logic rst=1;
    logic [{a_width - 1}:0] adat;
    logic [{b_width - 1}:0] bdat;
    logic avld=0, bvld=0, ordy=0;
    wire ardy, brdy, ovld;
    wire [{o_width - 1}:0] odat;
    {extra}
    @DUT@ dut({connections});
    logic [{a_width - 1}:0] inputs_a[{len(a)}] = {array(a, a_width)};
    logic [{b_width - 1}:0] inputs_b[{max(1, len(b))}] = {array(b or [0], b_width)};
    logic [{o_width - 1}:0] expected[{len(expected)}] = {array(expected, o_width)};
    integer sent_a=0, sent_b=0, received=0;
    logic held=0; logic [{o_width - 1}:0] held_data;
    initial begin
        repeat(25) @(negedge clk);
        rst=0;
        for(integer cycle=0;cycle<500;cycle=cycle+1) begin
            @(negedge clk);
            avld=sent_a<{len(a)};
            bvld=sent_b<{len(b)};
            adat=avld ? inputs_a[sent_a] : '0;
            bdat=bvld ? inputs_b[sent_b] : '0;
            ordy=(cycle%7)>=3;
            @(posedge clk);
            if(held && (!ovld || odat !== held_data)) $fatal(1,"output changed while stalled");
            held=ovld && !ordy; held_data=odat;
            if(avld && ardy) sent_a=sent_a+1;
            if(bvld && brdy) sent_b=sent_b+1;
            if(ovld && ordy) begin
                if(received>={len(expected)}) $fatal(1,"extra output");
                if(odat !== expected[received])
                    $fatal(1,"output %0d: got %h expected %h",received,odat,expected[received]);
                received=received+1;
            end
        end
        if(sent_a!={len(a)} || sent_b!={len(b)} || received!={len(expected)})
            $fatal(1,"missing transfers %0d %0d %0d",sent_a,sent_b,received);
        $display("FLAT_KERNEL_PASS"); $finish;
    end
endmodule
"""
    run(requirements, body, tmp_path)
