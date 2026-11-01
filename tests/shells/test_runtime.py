# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shared host driver (``finn.shells.runtime``) against a device double
(``support.devices.FakeDevice``): TFC's runtime file, recorded (``recorded/``), read as a
runtime reads it, its core run, every call to the device against its recorded sequence; the
parent graph run, its host nodes by ONNX execution; a node's run-time weights; what the
driver refuses; and its packing against FINN's own, both ways, in the order the movers
move lanes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor
from support.devices import BASE, FakeDevice
from support.tfc import partition

from finn.shells.integration import MAIN, Linked, describe, runtime_file
from finn.shells.pynq.linker import linked_row, unit_bases
from finn.shells.runtime import Driver, Refused, read
from finn.shells.runtime.movers import DONE, IDLE
from finn.shells.runtime.packing import Element, pack, unpack
from finn.util import data_packing

RECORDED = Path(__file__).parent / "recorded"

#: The main clock as TFC's Ultra96 builds deliver it (report/delivered_clock.json).
DELIVERED = {MAIN: 187.512}


@pytest.fixture(scope="module")
def tfc_parent(tfc_streamlined: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """TFC's parent graph (Reshape, its core, TopK), its choices committed and its body
    verified, saved."""
    directory = tmp_path_factory.mktemp("tfc")
    parent, _ = partition(ModelWrapper(str(tfc_streamlined)), directory)
    path = directory / "parent.onnx"
    parent.save(str(path))
    return path


def recorded() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((RECORDED / "tfc-integration.json").read_text())
    return found


def the_runtime_file(parent: ModelWrapper) -> str:
    """The runtime file of ``parent``'s description, linked by the ``pynq`` linker's
    placement, its main clock delivered as TFC's builds deliver it, as written."""
    described = describe(parent)
    linked = Linked(
        image=Path("finn-accel.bit"),
        placement=unit_bases(described, linked_row(described)),
        clocks_mhz=DELIVERED,
        reports={},
    )
    return json.dumps(runtime_file(described, linked), indent=2) + "\n"


def device_for(found: dict[str, Any], results: dict[str, bytes] | None = None) -> FakeDevice:
    """A device double whose movers' block controls each read idle and done."""
    controls = {
        (each["unit"], each["offset"] + each["registers"]["control"]["offset"]): IDLE | DONE
        for each in found["endpoints"]
        if each["registers"]
    }
    return FakeDevice(answers=controls, results=results or {})


def finn_packed(values: np.ndarray[Any, Any], element: BaseDataType) -> np.ndarray[Any, Any]:
    """``values`` as FINN packs them in the movers' order: inner and bytes reversed."""
    packed: np.ndarray[Any, Any] = data_packing.finnpy_to_packed_bytearray(  # type: ignore[no-untyped-call]
        values, element, reverse_inner=True, reverse_endian=True
    )
    return packed


def finn_unpacked(
    packed: np.ndarray[Any, Any], element: BaseDataType, shape: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    """``packed`` as FINN unpacks it in the movers' order into ``shape``."""
    values: np.ndarray[Any, Any] = data_packing.packed_bytearray_to_finnpy(  # type: ignore[no-untyped-call]
        np.ascontiguousarray(packed), element, shape, reverse_inner=True, reverse_endian=True
    )
    return values


def test_tfc_s_description_writes_the_recorded_runtime_file(tfc_parent: Path) -> None:
    """The build half: TFC's description and its placement write the recorded file, which
    the runtime half's tests read."""
    assert (
        the_runtime_file(ModelWrapper(str(tfc_parent)))
        == (RECORDED / "tfc-integration.json").read_text()
    )


def test_tfc_s_core_runs_as_the_recorded_operations() -> None:
    """TFC's core on the device: its input packed into its mover's buffer, the output mover
    found idle and started, then the input mover, the output waited on and unpacked; every
    call against the recorded sequence."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    (taken,) = core["outputs"]
    out_bytes = taken["beats"] * taken["bits"] // 8
    device = device_for(found, {taken["unit"]: bytes(range(out_bytes))})
    driver = Driver(device, read(found))
    driver.load(Path("bitfile") / "finn-accel.bit")
    values = (np.arange(np.prod(given["shape"])) % 256).reshape(given["shape"])
    outputs = driver.run_core(core["name"], {given["tensor"]: values})
    assert "\n".join(device.calls) + "\n" == (RECORDED / "tfc-ops.txt").read_text()
    # The input mover's buffer holds the input as FINN packs it; the output is the output
    # buffer's bytes as FINN unpacks them.
    element = DataType[given["element"]]
    folded = values.reshape(1, given["beats"], given["lanes"])
    packed = finn_packed(folded, element)
    in_buffer = [line for line in device.calls if line.startswith(f"buffer {given['unit']}")]
    assert in_buffer == [f"buffer {given['unit']} m_axi_gmem {packed.size} HP0 -> {BASE:#x}"]
    expected = finn_unpacked(
        np.arange(out_bytes, dtype=np.uint8).reshape(1, taken["beats"], -1),
        DataType[taken["element"]],
        (1, taken["beats"], taken["lanes"]),
    )
    assert np.array_equal(outputs[taken["tensor"]], expected.reshape(taken["shape"]))


def test_the_parent_graph_runs_its_host_nodes_around_the_core(tfc_parent: Path) -> None:
    """The parent graph: the Reshape before the core and the TopK after it by ONNX
    execution, the core on the device by its name, as ``run_core`` runs it."""
    found = recorded()
    (core,) = found["cores"]
    (taken,) = core["outputs"]
    out_bytes = taken["beats"] * taken["bits"] // 8
    results = {taken["unit"]: bytes(range(out_bytes))}
    model = ModelWrapper(str(tfc_parent))
    assert [node.op_type for node in model.graph.node] == ["Reshape", "DataflowCore", "TopK"]
    shape = model.get_tensor_shape(model.graph.input[0].name)
    assert shape is not None
    image = (np.arange(np.prod(shape)) % 256).reshape(shape).astype(np.float32)
    device = device_for(found, results)
    outputs = Driver(device, read(found)).run_graph(tfc_parent, {model.graph.input[0].name: image})
    alone = device_for(found, results)
    run = Driver(alone, read(found)).run_core(
        core["name"], {core["inputs"][0]["tensor"]: image.reshape(core["inputs"][0]["shape"])}
    )
    assert device.calls == alone.calls
    (output,) = model.graph.output
    top = np.argsort(-run[taken["tensor"]], axis=-1, kind="stable")[..., :1]
    assert np.array_equal(outputs[output.name].reshape(-1), top.reshape(-1))


def test_a_node_s_weights_are_written_at_its_endpoint_and_read_back() -> None:
    """A node's run-time weights: words from its endpoint's offset in its unit, each read
    back; more words than its endpoint holds refused."""
    found = recorded()
    found["endpoints"].append(
        {
            "unit": "DataflowCore_0",
            "bus": "Thresholding_0_s_axilite",
            "node": "Thresholding_0",
            "offset": 64,
            "range": 16,
            "registers": {},
        }
    )
    device = FakeDevice()
    driver = Driver(device, read(found))
    driver.write_weights("Thresholding_0", [1, 2, 3, 4])
    assert device.calls[:4] == [
        f"write32 DataflowCore_0 {offset:#04x} {word:#x}"
        for offset, word in zip((64, 68, 72, 76), (1, 2, 3, 4))
    ]
    assert [line.split(" -> ")[1] for line in device.calls[4:]] == ["0x1", "0x2", "0x3", "0x4"]
    with pytest.raises(Refused, match="weights-outside: Thresholding_0: 5 words"):
        driver.write_weights("Thresholding_0", [0] * 5)
    with pytest.raises(Refused, match="node-unknown: MatMul_0"):
        driver.write_weights("MatMul_0", [0])


def edited(change: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    found = recorded()
    change(found)
    return found


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda found: found.update(schema=0), "schema-unsupported"),
        (lambda found: found.pop("schema"), "schema-unsupported"),
        (lambda found: found["endpoints"].pop(0), "endpoint-absent"),
    ],
)
def test_a_runtime_file_the_driver_cannot_read_is_refused(
    change: Callable[[dict[str, Any]], object], code: str
) -> None:
    with pytest.raises(Refused, match=f"^{code}: "):
        read(edited(change))


def test_what_a_driver_cannot_run_is_refused() -> None:
    """A mover kind with no protocol; an input absent, of another shape, or of other
    frames than another input's; an output mover not idle; an element the driver cannot
    pack."""

    def renamed(found: dict[str, Any]) -> None:
        found["cores"][0]["inputs"][0]["mover"] = "pushed_adapter"

    with pytest.raises(Refused, match="^mover-unknown: in0: its mover kind 'pushed_adapter'"):
        Driver(FakeDevice(), read(edited(renamed)))
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    driver = Driver(FakeDevice(), read(found))
    with pytest.raises(Refused, match="^input-absent: "):
        driver.run_core(core["name"], {})
    with pytest.raises(Refused, match="^input-shape: "):
        driver.run_core(core["name"], {given["tensor"]: np.zeros((1, 3))})
    with pytest.raises(Refused, match="^core-unknown: DataflowCore_7"):
        driver.run_core("DataflowCore_7", {})
    busy = Driver(FakeDevice(), read(found))
    with pytest.raises(Refused, match="^mover-busy: out0"):
        busy.run_core(core["name"], {given["tensor"]: np.zeros(given["shape"])})
    with pytest.raises(Refused, match="^element-unsupported: SCALEDINT<8>"):
        Element.named("SCALEDINT<8>")
    with pytest.raises(Refused, match=r"^value-outside: UINT2 holds no \[4\]"):
        pack(np.array([[1, 4]]), Element.named("UINT2"))


#: Every kind of element the movers move, at widths within a byte, across a byte, across a
#: 32-bit word and of a whole word, against FINN's packing.
ELEMENTS = [
    "BINARY",
    "BIPOLAR",
    "TERNARY",
    "UINT1",
    "INT2",
    "UINT3",
    "INT4",
    "INT7",
    "UINT8",
    "INT8",
    "UINT13",
    "INT16",
    "INT17",
    "UINT24",
    "INT31",
    "UINT32",
    "INT33",
    "UINT45",
    "FIXED<8,4>",
    "FIXED<12,3>",
    "FIXED<34,10>",
    "FLOAT32",
]


@pytest.mark.parametrize("name", ELEMENTS)
@pytest.mark.parametrize("lanes", [1, 3, 8, 16])
def test_packing_is_finn_s_in_the_movers_lane_order_both_ways(name: str, lanes: int) -> None:
    """The driver's packing of values of each element, ``lanes`` a beat, equals FINN's with
    its inner and its byte order reversed (the movers' order), and its unpacking FINN's
    unpacking of the same bytes."""
    element = DataType[name]
    rng = np.random.default_rng(lanes)
    values = gen_finn_dt_tensor(element, (2, 5, lanes))
    if name == "FLOAT32":
        values = rng.standard_normal((2, 5, lanes)).astype(np.float32)
    packed = pack(values, Element.named(name))
    finn = finn_packed(values, element)
    assert np.array_equal(packed, finn), (name, lanes)
    back = unpack(np.ascontiguousarray(finn), Element.named(name), lanes)
    assert np.array_equal(back, np.asarray(values, dtype=np.float32)), (name, lanes)
    # FINN's unpacking of a fixed point of 8 or 16 bits in this order reads its bytes as
    # floats and fails; there the values themselves are the reference.
    if not (name.startswith("FIXED") and element.bitwidth() in (8, 16)):
        theirs = finn_unpacked(finn, element, (2, 5, lanes))
        assert np.array_equal(back, theirs), (name, lanes)


def test_the_recorded_texts_hold_no_machine_s_path() -> None:
    """The recorded texts are FINN's test data: no path of a machine."""
    for name in ("tfc-integration.json", "tfc-ops.txt"):
        text = (RECORDED / name).read_text()
        for marker in ("/home", "/tmp", "/proj", "/scratch", "\\", str(Path.home())):
            assert marker not in text, (name, marker)
