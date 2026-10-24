# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib memstream as a delivery kernel, and as the weight channel's source.

The memory image is packed in the consumer's order and shipped as a generated
INIT_FILE named by its contents. A pumped memory stores each word as two
half-words, low first. The memory states the range of its contents on its
output; its AXI-Lite port is tied off. Several weight sets are selected per row
through in2_V.
"""

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.ranges import TensorRange
from support.adapted import CHANNELS, ROWS, adapted
from support.helpers import (
    FULL_DSP48E2,
    WeightDelivery,
    built,
    labels,
    matmul_assembly,
    pin_names,
    placed,
)

from finn.core.space import Available, Rejected, design_space
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import BeatSequence, Repetition, tile, vector_major
from finn.harness.toolchain import finnlib_root
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.contributions import ContributionError, GeneratedData
from finn.kernels.channels import OUTPUT_ADAPTER_LABEL, SOURCE_LABEL
from finn.kernels.explore import dotted
from finn.kernels.library import library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.values.semantics import IntegerTensorValue

WEIGHTS = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
# MatMul stores its weights (k, n): WEIGHTS read by output.
MATMUL = dict(
    m=3,
    k=4,
    n=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    pe=2,
    simd=2,
    platform=FULL_DSP48E2,
    weight_delivery=WeightDelivery.MEMSTREAM,
    weights=tuple(zip(*WEIGHTS)),
)


def memory(**changes):
    facts = {
        "dtype": DataType["INT3"],
        "form": tile(4, 4, 2, 2),
        "contents": WEIGHTS,
        "platform": FULL_DSP48E2,
        **changes,
    }
    facts["contents"] = IntegerTensorValue.of(facts["contents"])
    return design_space(MemStreamKernel(**facts)).with_choices(
        ram_style="block", pumped_memory=False
    )


def test_the_image_is_the_consumers_order_in_a_content_named_init_file():
    point = memory()
    # The same image as the cyclic ROM: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert point.image == (0x22C, 0x6BE, 0xDD3, 0x941)
    init = point.init_file
    assert init.data == b"22c\n6be\ndd3\n941\n"
    assert init.path.startswith("memstream_") and init.path.endswith(".dat")
    requirements = library_module(point)
    parameters = dict(requirements.parameters)
    assert parameters["INIT_FILE"] == f'"{init.path}"'
    assert (parameters["DEPTH"], parameters["WIDTH"], parameters["SETS"]) == (4, 12, 1)
    assert init in requirements.data
    # Other contents, another file and another identity.
    other = memory(contents=tuple(tuple(-value - 1 for value in row) for row in WEIGHTS))
    assert other.init_file.path != init.path


def test_a_pumped_memory_stores_half_words_low_first():
    point = design_space(
        MemStreamKernel(
            platform=FULL_DSP48E2,
            dtype=DataType["INT3"],
            form=tile(4, 4, 2, 2),
            contents=IntegerTensorValue.of(WEIGHTS),
        )
    ).with_choices(ram_style="auto", pumped_memory=True)
    # 12-bit words as 6-bit halves: 0x22C -> 0x2C, 0x08.
    assert point.init_file.data.split(b"\n")[:4] == [b"2c", b"08", b"3e", b"1a"]
    ports = {port.name: port for port in library_module(point).abi.pins}
    assert "clk2x" in ports and library_module(point).abi.clock_alignments


def test_idle_interfaces_are_tied_off():
    tieoffs = library_module(memory()).held
    tied = dict(tieoffs.inputs)
    assert tied["awvalid"] == 0 and tied["s_axis_0_tvalid"] == 0 and tied["clk2x"] == 0
    assert "awready" in tieoffs.unused and "s_axis_0_tready" in tieoffs.unused


def test_the_output_element_carries_the_range_of_every_set():
    assert memory().element == ScalarEncoding(DataType["INT3"])  # -4 to 3: the datatype's own
    narrow = tuple(tuple(max(value, -3) for value in row) for row in WEIGHTS)
    assert str(memory(contents=narrow).element) == "INT3 over [-3, 3]"
    # Several sets: the range spans them all.
    sets = (narrow, tuple(tuple(min(value, 1) for value in row) for row in narrow))
    several = memory(contents=sets, sets=2)
    assert several.element == ScalarEncoding(DataType["INT3"], TensorRange(-3, 3))
    refused = memory(contents=((4,) * 4,) * 4).query(MemStreamKernel.element)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"memstream-values"}


def test_generated_data_is_a_relative_name_with_bytes():
    for path, data in (("/abs.dat", b""), ("../up.dat", b""), ("x.dat", "text")):
        with pytest.raises(ContributionError):
            GeneratedData(path, data)  # type: ignore[arg-type]


def test_matmul_memstream_delivery_materializes_its_image(tmp_path):
    built = matmul_assembly(**MATMUL)
    # The memory is the weight channel's source, below the channel declared before MatMul.
    assert labels(built.module) == [
        "x.adapter.input_gen.input_gen",
        "w.source.memstream",
        "matmul.compute.packed",
    ]
    assert built.initializer == (0x22C, 0x6BE, 0xDD3, 0x941)
    assert "in1_V" not in pin_names(built.module)
    emitted = emit_module(built.module, tmp_path, roots={"finnlib": finnlib_root()})
    (image,) = emitted.data
    assert (emitted.directory / image).read_bytes() == b"22c\n6be\ndd3\n941\n"


def test_several_weight_sets_take_a_set_index_per_row():
    sets = (WEIGHTS, tuple(tuple(-value - 1 for value in row) for row in WEIGHTS))
    built = matmul_assembly(**{**MATMUL, "weights": sets}, weight_sets=2)
    assert "in2_V" in pin_names(built.module)
    memstream = dict(placed(built.module, "w.source.memstream").parameters)
    assert memstream["SETS"] == 2
    assert len(built.initializer) == 8  # both sets, set after set
    # Its cycles are the beats its output carries a frame: one pass a row's index, where a
    # memory of one set streams one pass.
    source = built.network.member("w").source
    assert source.output_sequence.form.beats == 3 * source.form.beats
    assert source.cycles == 3 * source.form.beats
    assert memory().cycles == memory().form.beats
    # The memory's choices are the weight channel's, keyed below it.
    keys = {dotted(key) for key in built.network.decisions()}
    assert {"w.source.ram_style", "w.source.pumped_memory"} <= keys


def test_several_sets_without_their_indices_refuse_the_sequences_that_read_them():
    # Its rule refuses such a memory (``selected``); its set port's order and its output's
    # read the indices, and refuse the same way rather than read what is not given.
    space = design_space(
        MemStreamKernel(
            dtype=DataType["INT3"],
            form=tile(4, 4, 2, 2),
            contents=IntegerTensorValue.of((WEIGHTS, WEIGHTS)),
            sets=2,
            platform=FULL_DSP48E2,
        )
    )
    for member, owner in (
        (MemStreamKernel.set_sequence, "set_sequence"),
        (MemStreamKernel.output_sequence, "output_sequence"),
    ):
        answer = space.query(member)
        assert isinstance(answer, Rejected)
        assert [(finding.code, finding.owner) for finding in answer.findings] == [
            ("memstream-set-channel", owner)
        ]


def test_a_single_set_refuses_its_idle_set_port_s_order():
    # One set takes no set channel: its set port is idle, and the port's order, which
    # would read the indices, is refused with the kernel's set-channel code. Open with the
    # human (SZ124 (1)): refused, or inapplicable.
    space = design_space(
        MemStreamKernel(
            dtype=DataType["INT3"],
            form=tile(4, 4, 2, 2),
            contents=IntegerTensorValue.of(WEIGHTS),
            platform=FULL_DSP48E2,
        )
    )
    answer = space.query(MemStreamKernel.set_sequence)
    assert isinstance(answer, Rejected)
    assert [(finding.code, finding.owner, finding.message) for finding in answer.findings] == [
        ("memstream-set-channel", "set_sequence", "a single set takes no set channel")
    ]
    assert space.query(MemStreamKernel.output_sequence) == Available(
        BeatSequence(tile(4, 4, 2, 2), Repetition.CYCLIC)
    )


def test_a_channel_s_placements_keep_the_labels_it_states_in_the_emitted_netlist():
    """A channel's source and output adapter are placements at its nodes ``source`` and
    ``output_adapter``, but its netlist places each at the label the channel states, which
    every name emitted from it reads: the source's instance, the hop into its set port, and
    the output adapter's stage."""
    sets = (WEIGHTS, tuple(tuple(-value - 1 for value in row) for row in WEIGHTS))
    assembly = matmul_assembly(**{**MATMUL, "weights": sets}, weight_sets=2)
    w = assembly.network.member("w")
    instance = f"w.{SOURCE_LABEL}"
    links = assembly.module.fragment.links
    # The source: its producer's owner, its instance, and the hop out of it.
    assert w.source_label == SOURCE_LABEL == "source.memstream"
    assert instance in labels(assembly.module)
    assert [link.source.data for link in links if link.source.instance == instance] == [
        "m_axis_0_tdata"
    ]
    # Its set port: labelled below it, and the index channel's hop ends at it.
    assert w.set_contract.label == f"{SOURCE_LABEL}.set"
    assert [link.sink.data for link in links if link.sink.instance == instance] == [
        "s_axis_0_tdata"
    ]
    # The output adapter: its stage below the stated label, emitted there.
    point = adapted(vector_major((ROWS, CHANNELS), 2), 4)
    x = point.member("x")
    (stage,) = x.stages
    (streams,) = x.staged
    assert stage.label == streams.label == f"{OUTPUT_ADAPTER_LABEL}.vpc"
    assert stage.label == "output_adapter.vpc.vpc"
    assert f"x.{stage.label}" in labels(built(point, "adapted"))
