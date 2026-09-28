# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FinnLib memstream as a delivery kernel, and MatMulKernel's memstream candidate.

The memory image is packed in the consumer's order and shipped as a generated
INIT_FILE named by its contents. A pumped memory stores each word as two
half-words, low first. Runtime-writable weights export the AXI-Lite port at
the module boundary; several weight sets are selected per row through in2_V.
"""

from pathlib import Path

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected, design_space, inspection
from finn.kernels.artifacts.abi import Bus
from finn.kernels.artifacts.build import materialize_module_sources, prepare_module_build
from finn.kernels.artifacts.contributions import ContributionError, GeneratedData
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.matmul import MatMulKernel, WeightDelivery, matmul_assembly
from finn.kernels.memstream import MemStreamKernel
from finn.dataflow.traversal import tile
from finn.kernels.resources import resource_root, template_root
from finn.kernels.target import DspBlock

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
MATMUL = dict(
    rows=3,
    reduction=4,
    outputs=4,
    activation_dtype=DataType["INT3"],
    weights_dtype=DataType["INT3"],
    pe=2,
    simd=2,
    target_dsp=DspBlock.DSP48E2,
    weight_delivery=WeightDelivery.MEMSTREAM,
    weights=WEIGHTS,
)


def memory(**changes):
    facts = {"dtype": DataType["INT3"], "form": tile(4, 4, 2, 2), "values": WEIGHTS, **changes}
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
    requirements = point.build_requirements
    parameters = dict(requirements.parameters)
    assert parameters["INIT_FILE"] == f'"{init.path}"'
    assert (parameters["DEPTH"], parameters["WIDTH"], parameters["SETS"]) == (4, 12, 1)
    assert init in requirements.contributions
    # Other contents, another file and another identity.
    other = memory(values=tuple(tuple(-value - 1 for value in row) for row in WEIGHTS))
    assert other.init_file.path != init.path


def test_a_pumped_memory_stores_half_words_low_first():
    point = design_space(
        MemStreamKernel(dtype=DataType["INT3"], form=tile(4, 4, 2, 2), values=WEIGHTS)
    ).with_choices(ram_style="auto", pumped_memory=True)
    # 12-bit words as 6-bit halves: 0x22C -> 0x2C, 0x08.
    assert point.init_file.data.split(b"\n")[:4] == [b"2c", b"08", b"3e", b"1a"]
    ports = {port.name: port for port in point.build_requirements.abi.ports}
    assert "clk2x" in ports and point.build_requirements.abi.clock_alignments


def test_idle_interfaces_are_tied_off():
    tieoffs = memory().tieoffs
    tied = dict(tieoffs.inputs)
    assert tied["awvalid"] == 0 and tied["s_axis_0_tvalid"] == 0 and tied["clk2x"] == 0
    assert "awready" in tieoffs.unused and "s_axis_0_tready" in tieoffs.unused
    refused = memory(writable=True).query(MemStreamKernel.tieoffs)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"memstream-control"}


def test_generated_data_is_a_relative_name_with_bytes():
    for path, data in (("/abs.dat", b""), ("../up.dat", b""), ("x.dat", "text")):
        with pytest.raises(ContributionError):
            GeneratedData(path, data)  # type: ignore[arg-type]


def test_matmul_memstream_delivery_materializes_its_image(tmp_path):
    built = matmul_assembly(**MATMUL)
    assert [item.instance_id for item in built.structure.instances] == [
        "u_compute_packed",
        "u_delivery_memstream",
        "u_activations_input_gen",
    ]
    assert built.initializer == (0x22C, 0x6BE, 0xDD3, 0x941)
    assert "in1_V" not in {port.name for port in built.structure.top_abi.ports}
    store = ArtifactStore(tmp_path / "store")
    prepared = prepare_module_build(
        built.requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    (image,) = [name for name in materialized.files if name.endswith(".dat")]
    assert (Path(materialized.directory) / image).read_bytes() == b"22c\n6be\ndd3\n941\n"


def test_writable_weights_export_axilite_and_need_the_memstream():
    built = matmul_assembly(**MATMUL, writable_weights=True)
    (bus,) = [port for port in built.structure.top_abi.ports if isinstance(port, Bus)][-1:]
    assert bus.name == "s_axilite"
    assert {member.physical for member in bus.signals} >= {"s_axilite_AWADDR", "s_axilite_WDATA"}
    with pytest.raises(ValueError, match="matmul-writable"):
        matmul_assembly(
            **{**MATMUL, "weight_delivery": WeightDelivery.CYCLIC}, writable_weights=True
        )


def test_several_weight_sets_take_a_set_index_per_row():
    sets = (WEIGHTS, tuple(tuple(-value - 1 for value in row) for row in WEIGHTS))
    built = matmul_assembly(**{**MATMUL, "weights": sets}, weight_sets=2)
    ports = {port.name: port for port in built.structure.top_abi.ports}
    assert "in2_V" in ports
    (memstream,) = (
        dict(item.requirements.parameters)
        for item in built.structure.instances
        if item.instance_id == "u_delivery_memstream"
    )
    assert memstream["SETS"] == 2
    assert len(built.initializer) == 8  # both sets, set after set
    with pytest.raises(ValueError, match="matmul-sets"):
        matmul_assembly(
            **{**MATMUL, "weights": sets, "weight_delivery": WeightDelivery.CYCLIC}, weight_sets=2
        )
    facts = {name: MATMUL[name] for name in ("rows", "reduction", "outputs", "target_dsp")}
    base = design_space(
        MatMulKernel(
            **facts,
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["INT3"],
            target_period_ns=5.0,
        )
    )
    keys = {item.key for item in inspection.decisions(base)}
    assert {"delivery.memstream.ram_style", "delivery.memstream.pumped_memory"} <= keys
