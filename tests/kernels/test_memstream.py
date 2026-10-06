# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib memstream as a delivery kernel, and MatMulKernel's memstream candidate.

The memory image is packed in the consumer's order and shipped as a generated
INIT_FILE named by its contents. A pumped memory stores each word as two
half-words, low first. The memory states the range of its contents on its
output; its AXI-Lite port is tied off. Several weight sets are selected per row
through in2_V.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, design_space, inspection
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import tile
from finn.kernels.artifacts.build import emit_module
from finn.kernels.artifacts.contributions import ContributionError, GeneratedData
from finn.kernels.memstream import MemStreamKernel
from kernels.helpers import (
    FULL_DSP48E2,
    WeightDelivery,
    finnlib_root,
    labels,
    matmul_assembly,
    pin_names,
    placed,
)

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
    requirements = point.module
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
            platform=FULL_DSP48E2, dtype=DataType["INT3"], form=tile(4, 4, 2, 2), contents=WEIGHTS
        )
    ).with_choices(ram_style="auto", pumped_memory=True)
    # 12-bit words as 6-bit halves: 0x22C -> 0x2C, 0x08.
    assert point.init_file.data.split(b"\n")[:4] == [b"2c", b"08", b"3e", b"1a"]
    ports = {port.name: port for port in point.module.abi.pins}
    assert "clk2x" in ports and point.module.abi.clock_alignments


def test_idle_interfaces_are_tied_off():
    tieoffs = memory().module.held
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
    assert several.element == ScalarEncoding(DataType["INT3"], (-3, 3))
    refused = memory(contents=((4,) * 4,) * 4).query(MemStreamKernel.element)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"memstream-values"}


def test_generated_data_is_a_relative_name_with_bytes():
    for path, data in (("/abs.dat", b""), ("../up.dat", b""), ("x.dat", "text")):
        with pytest.raises(ContributionError):
            GeneratedData(path, data)  # type: ignore[arg-type]


def test_matmul_memstream_delivery_materializes_its_image(tmp_path):
    built = matmul_assembly(**MATMUL)
    # The memory is the weight stream's source, below the stream declared before MatMul.
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
    # The memory's choices are the weight stream's, keyed below it.
    keys = {item.key for item in inspection.decisions(built.point)}
    assert {"w.source.memstream.ram_style", "w.source.memstream.pumped_memory"} <= keys
