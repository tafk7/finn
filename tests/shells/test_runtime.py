# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shared host driver (``finn.shells.runtime``) against a device double
(``support.devices.FakeDevice``): TFC's runtime file, recorded (``recorded/``), read as a
runtime reads it, its core run, every call to the device against its recorded sequence; the
parent graph run, its host nodes by ONNX execution; a node's run-time weights; what the
driver refuses; and its packing against FINN's own, both ways, in the order the movers
move lanes. Then the ``pynq`` runtime that ships it (``finn.shells.pynq.runtime``): the
driver it writes, run as on a board with numpy and a stubbed PYNQ alone
(``support.pynq_stub``), its PYNQ calls against their recorded sequence."""

from __future__ import annotations

import ast
import gc
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import EXACT_INTEGERS
from qonnx.util.basic import gen_finn_dt_tensor
from support.devices import BASE, STEP, FakeDevice
from support.packaging import bitfile_default
from support.pynq_stub import stub
from support.refusals import refused
from support.tfc import partition

from finn.dataflow.tensor import ScalarEncoding
from finn.shells.build import ShellRefused
from finn.shells.integration import MAIN, Linked, describe, runtime_file
from finn.shells.pynq.linker import linked_row, unit_bases
from finn.shells.pynq.runtime import PynqRuntime, shipped
from finn.shells.runtime import Buffer, Description, Driver, Refused, cli, read, validate
from finn.shells.runtime.driver import RETAINED
from finn.shells.runtime.movers import DONE, IDLE, START
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


@pytest.mark.parametrize("exact", ["1", "0"], ids=["finn", "shipped"])
def test_the_parent_graph_runs_its_host_nodes_around_the_core(
    tfc_parent: Path, monkeypatch: pytest.MonkeyPatch, exact: str
) -> None:
    """The parent graph: the Reshape before the core and the TopK after it by ONNX
    execution, the core on the device by its name, as ``run_core`` runs it, its output
    given to the TopK in the graph's float32, not its int8 container. So also where qonnx's
    exact integers are off, as in the shipped driver, which imports no FINN (ONNX Runtime
    refuses an int8 input the graph declares float)."""
    monkeypatch.setenv(EXACT_INTEGERS, exact)
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
    assert run[taken["tensor"]].dtype == np.int8
    assert np.array_equal(outputs[output.name].reshape(-1), top.reshape(-1))


def test_a_core_s_output_the_graph_s_type_would_change_is_refused(
    tfc_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A core's output past what the graph's float32 holds exactly (an integer past 2**24)
    is refused before the host nodes run (``value-inexact``); one it holds goes on."""
    found = recorded()
    (core,) = found["cores"]
    (taken,) = core["outputs"]
    model = ModelWrapper(str(tfc_parent))
    shape = model.get_tensor_shape(model.graph.input[0].name)
    assert shape is not None
    image = np.zeros(shape, np.float32)
    driver = Driver(device_for(found), read(found))
    width = taken["shape"][-1]
    for value, exact in ((2**24, True), (2**24 + 1, False)):
        given = np.full((1, width), value, np.uint32)
        given[0, 1] = 3

        def core_gives(
            name: str, inputs: object, out: np.ndarray[Any, Any] = given
        ) -> dict[str, np.ndarray[Any, Any]]:
            return {taken["tensor"]: out}

        monkeypatch.setattr(driver, "run_core", core_gives)
        if exact:
            outputs = driver.run_graph(tfc_parent, {model.graph.input[0].name: image})
            assert next(iter(outputs.values())).reshape(-1).tolist() == [0]
            continue
        with pytest.raises(
            Refused,
            match=r"^value-inexact: MatMul_3_out0: the parent graph holds it as float32, which "
            r"changes \[16777217, ",
        ):
            driver.run_graph(tfc_parent, {model.graph.input[0].name: image})


def with_weights(found: dict[str, Any]) -> dict[str, Any]:
    """``found`` with an endpoint of run-time weights: Thresholding_0's, four words from the
    offset 64 of its core's unit."""
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
    return found


def test_a_node_s_weights_are_written_at_its_endpoint_and_read_back() -> None:
    """A node's run-time weights: words from its endpoint's offset in its unit, each read
    back; more words than its endpoint holds refused."""
    found = with_weights(recorded())
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
    for unsupported in ("SCALEDINT<8>", "FLOAT64", "FIXED<8,8>"):
        with pytest.raises(Refused, match=f"^element-unsupported: {re.escape(unsupported)}"):
            Element.named(unsupported)
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
    assert back.dtype == element.to_numpy_dt(), (name, lanes)
    assert np.array_equal(back, values), (name, lanes)
    # FINN's unpacking of a fixed point of 8 or 16 bits in this order reads its bytes as
    # floats and fails; there the values themselves are the reference.
    if not (name.startswith("FIXED") and element.bitwidth() in (8, 16)):
        theirs = finn_unpacked(finn, element, (2, 5, lanes))
        assert np.array_equal(back, theirs), (name, lanes)


@pytest.mark.parametrize(
    "name, values, codes",
    [
        ("UINT64", np.array([3 * 2**62], np.float32), [0xC000000000000000]),
        ("UINT64", np.array([2**64 - 1], np.uint64), [0xFFFFFFFFFFFFFFFF]),
        ("UINT64", np.array([2**24 + 1], np.uint64), [2**24 + 1]),
        ("INT32", np.array([2**24 + 1, -(2**31)], np.int64), [2**24 + 1, 1 << 31]),
        ("INT64", np.array([2**63 - 1, -(2**63)], np.int64), [0x7FFFFFFFFFFFFFFF, 1 << 63]),
        ("UINT63", np.array([2**63 - 1], np.uint64), [0x7FFFFFFFFFFFFFFF]),
        ("INT63", np.array([2**62 - 1, -(2**62)], np.int64), [0x3FFFFFFFFFFFFFFF, 1 << 62]),
        ("FIXED<64,1>", np.array([-1.0, 0.5]), [1 << 63, 1 << 62]),
        ("FIXED<64,1>", np.array([-1, 0], np.int64), [1 << 63, 0]),
    ],
)
def test_wide_integers_pack_exactly_and_unpack_by_their_sign(
    name: str, values: np.ndarray[Any, Any], codes: list[int]
) -> None:
    """A code of up to 64 bits is packed from integers or floats exactly, and an unsigned
    code unpacks as unsigned: the beat as the movers move it, lane 0 in its least
    significant bits, and back, exactly, in qonnx's container for the type (an integer
    past float32's 2**24 kept)."""
    element = Element.named(name)
    packed = pack(values.reshape(1, -1), element)
    beat = sum(code << (element.bits * lane) for lane, code in enumerate(codes))
    nbytes = -(-element.bits * len(codes) // 8)
    assert packed.reshape(-1).tobytes() == beat.to_bytes(nbytes, "little")
    back = unpack(packed, element, values.size).reshape(-1)
    assert back.dtype == DataType[name].to_numpy_dt()
    assert np.array_equal(back, values)


@pytest.mark.parametrize(
    "name, value",
    [
        ("UINT64", np.float64(2.0**64)),
        ("INT64", np.float64(2.0**63)),
        ("UINT8", np.int8(-1)),
        ("INT8", np.uint8(200)),
        ("INT8", np.float32(np.inf)),
        ("INT8", np.float32(np.nan)),
        ("FIXED<8,4>", np.int64(8)),
        ("FIXED<8,4>", np.float32(0.03125)),
    ],
)
def test_a_value_outside_an_element_is_refused_at_every_width(name: str, value: Any) -> None:
    """The bounds are compared exactly: the value past the greatest code, which a float
    rounds the greatest to, is outside; so are a negative unsigned, an infinity, a NaN and a
    fraction finer than a fixed point's."""
    with pytest.raises(Refused, match=f"^value-outside: {re.escape(name)} holds no "):
        pack(np.array([[value]]), Element.named(name))


def test_a_mover_holds_one_buffer_allocated_when_staged() -> None:
    """Staging allocates both directions' buffers, so a launch allocates nothing; a call of
    another batch releases each mover's buffer and allocates one of its size, and a call of
    the same size reuses it."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    device = device_for(found)
    driver = Driver(device, read(found))
    for frames in (1, 2, 2):
        driver.run_core(core["name"], {given["tensor"]: np.zeros((frames, 784))})
        if frames == 1:
            staged = len(device.calls)
            driver.stage(core["name"], {given["tensor"]: np.zeros((1, 784))})
            launching = len(device.calls)
            driver.launch(core["name"], 1)
            assert not any(line.startswith("buffer") for line in device.calls[launching:])
            assert any(line.startswith("buffer") for line in device.calls[:staged])
    assert [line for line in device.calls if line.startswith(("buffer", "release"))] == [
        f"buffer in0 m_axi_gmem 784 HP0 -> {BASE:#x}",
        f"buffer out0 m_axi_gmem 9 HP0 -> {BASE + STEP:#x}",
        f"release in0 {BASE:#x}",
        f"buffer in0 m_axi_gmem 1568 HP0 -> {BASE + 2 * STEP:#x}",
        f"release out0 {BASE + STEP:#x}",
        f"buffer out0 m_axi_gmem 18 HP0 -> {BASE + 3 * STEP:#x}",
    ]
    assert sorted(device.buffers.values()) == ["in0", "out0"]
    with pytest.raises(Refused, match="^batch-invalid: a batch of 0 frames"):
        cli.throughput(driver, 0)


class Stalling(FakeDevice):
    """The device double whose movers' control, once started, fails to read ``failures``
    times, then reads as written while ``stalled`` (started: neither idle nor done), else
    idle and done."""

    failures = 0
    stalled = False

    def read32(self, unit: str, offset: int) -> int:
        if self.registers.get((unit, offset)) == START:
            if self.failures:
                self.failures -= 1
                raise OSError(f"{unit}: its status did not read")
            if self.stalled:
                return START
        return super().read32(unit, offset)


def test_a_launch_that_did_not_complete_keeps_its_buffers_until_its_movers_settle() -> None:
    """A status read that fails after the movers started leaves them armed: a call of
    another batch or of the same, and a collection, are refused before any buffer is
    written, released or read; settling finds each mover idle, then the driver goes on. A
    mover not idle is refused at settling, and stays armed."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    device = Stalling(answers=device_for(found).answers)
    driver = Driver(device, read(found))
    device.failures = 1
    with pytest.raises(OSError, match="out0: its status did not read"):
        driver.run_core(core["name"], {given["tensor"]: np.zeros((1, 784))})
    held, made = dict(device.buffers), len(device.calls)
    for frames in (2, 1):
        with pytest.raises(
            Refused, match=r"^mover-armed: DataflowCore_0: a launch on \['in0', 'out0'\]"
        ):
            driver.run_core(core["name"], {given["tensor"]: np.zeros((frames, 784))})
    with pytest.raises(Refused, match="^mover-armed: "):
        driver.collect(core["name"], 1)
    assert (device.buffers, device.calls[made:]) == (held, [])
    device.stalled = True
    with pytest.raises(Refused, match="^mover-busy: out0: its mover is not idle"):
        driver.settle(core["name"])
    with pytest.raises(Refused, match="^mover-armed: "):
        driver.run_core(core["name"], {given["tensor"]: np.zeros((1, 784))})
    device.stalled = False
    driver.settle(core["name"])
    driver.run_core(core["name"], {given["tensor"]: np.zeros((2, 784))})
    assert [line for line in device.calls[made:] if line.startswith("release")] == [
        f"release in0 {BASE:#x}",
        f"release out0 {BASE + STEP:#x}",
    ]


def test_closing_releases_every_buffer_once_its_movers_are_idle() -> None:
    """A driver closes on its context's exit: every buffer it holds released, once; a
    closed driver is refused. An armed mover not idle refuses closing, every buffer kept;
    leaving the context on an error keeps them so, and raises the error."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    inputs = {given["tensor"]: np.zeros((1, 784))}
    device = device_for(found)
    with Driver(device, read(found)) as driver:
        driver.run_core(core["name"], inputs)
    assert device.buffers == {}
    assert device.calls[-2:] == [f"release in0 {BASE:#x}", f"release out0 {BASE + STEP:#x}"]
    driver.close()
    assert device.calls[-1] == f"release out0 {BASE + STEP:#x}"
    with pytest.raises(Refused, match="^driver-closed: DataflowCore_0"):
        driver.run_core(core["name"], inputs)
    stalling = Stalling(answers=device.answers)
    stalling.failures, stalling.stalled = 1, True
    kept = Driver(stalling, read(found))
    with pytest.raises(OSError):
        kept.run_core(core["name"], inputs)
    with pytest.raises(Refused, match="^mover-busy: in0: its mover is not idle"):
        kept.close()
    assert sorted(stalling.buffers.values()) == ["in0", "out0"]
    stalling = Stalling(answers=device.answers)
    stalling.failures, stalling.stalled = 1, True
    with pytest.raises(OSError), Driver(stalling, read(found)) as driver:
        driver.run_core(core["name"], inputs)
    assert sorted(stalling.buffers.values()) == ["in0", "out0"]
    assert not any(line.startswith("release") for line in stalling.calls)


class Unstartable(FakeDevice):
    """The device double whose mover ``failing`` does not start: writing any of its
    registers raises."""

    failing = ""

    def write32(self, unit: str, offset: int, value: int) -> None:
        if unit == self.failing:
            raise OSError(f"{unit}: its register did not write")
        super().write32(unit, offset, value)


def test_a_partial_launch_faults_the_driver_until_the_image_is_loaded_again() -> None:
    """The output mover started and the input mover failing to start: the driver refuses
    every use, settling and closing included, and keeps every buffer, retained past the
    driver's life, also when its context is left on the error. Loading the image again
    resets the movers: the driver goes on, and closes."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    inputs = {given["tensor"]: np.zeros((1, 784))}
    device = Unstartable(answers=device_for(found).answers)
    driver = Driver(device, read(found))
    driver.stage(core["name"], inputs)
    held, before = dict(device.buffers), set(RETAINED)
    device.failing = "in0"
    with pytest.raises(OSError, match="in0: its register did not write"):
        driver.launch(core["name"], 1)
    faulted = (
        r"^driver-faulted: \S+: a launch of DataflowCore_0: in0 failed to start after "
        r"\['out0'\] started"
    )
    for use in (
        lambda: driver.run_core(core["name"], inputs),
        lambda: driver.settle(core["name"]),
        driver.close,
    ):
        with pytest.raises(Refused, match=faulted):
            use()
    assert device.buffers == held
    retained = set(RETAINED) - before
    assert sorted(RETAINED[each][2].address for each in retained) == sorted(held)
    device.failing = ""
    driver.load(Path("finn-accel.bit"))
    assert not retained & set(RETAINED)
    driver.run_core(core["name"], inputs)
    driver.close()
    assert device.buffers == {}
    # Left on the error, the driver dropped: its buffers retained, none released.
    device = Unstartable(answers=device.answers)
    before = set(RETAINED)
    with pytest.raises(OSError), Driver(device, read(found)) as dropped:
        dropped.stage(core["name"], inputs)
        device.failing = "in0"
        dropped.launch(core["name"], 1)
    del dropped
    gc.collect()
    assert len(set(RETAINED) - before) == 2
    assert not any(line.startswith("release") for line in device.calls)


def abandoned(device: Unstartable, found: dict[str, Any]) -> None:
    """A partial launch on ``device``, its driver left on the error and dropped."""
    (core,) = found["cores"]
    (given,) = core["inputs"]
    with pytest.raises(OSError), Driver(device, read(found)) as driver:
        driver.stage(core["name"], {given["tensor"]: np.zeros((1, 784))})
        device.failing = "in0"
        driver.launch(core["name"], 1)
    device.failing = ""


def test_loading_a_device_s_image_releases_what_its_dropped_drivers_retained() -> None:
    """A driver dropped after a partial launch leaves its buffers retained; a new driver of
    the same device loading the image again releases them, each once, whatever the
    drivers dropped before; another device's stay retained, and so do those of a driver
    still alive."""
    found = recorded()
    before = set(RETAINED)
    other = Unstartable(answers=device_for(found).answers)
    abandoned(other, found)
    device = Unstartable(answers=other.answers)
    alive = Driver(device, read(found))
    (core,) = found["cores"]
    (given,) = core["inputs"]
    alive.stage(core["name"], {given["tensor"]: np.zeros((1, 784))})
    device.failing = "in0"
    with pytest.raises(OSError):
        alive.launch(core["name"], 1)
    device.failing = ""
    kept = set(RETAINED) - before
    for _ in range(3):
        abandoned(device, found)
        gc.collect()
        assert len(set(RETAINED) - before - kept) == 2
        released = len(device.calls)
        Driver(device, read(found)).load(Path("finn-accel.bit"))
        assert sorted(line.split()[1] for line in device.calls[released:] if "release" in line) == [
            "in0",
            "out0",
        ]
        assert set(RETAINED) - before == kept
    assert len(kept) == 4
    assert not any(line.startswith("release") for line in other.calls)
    alive.load(Path("finn-accel.bit"))
    assert len(set(RETAINED) - before) == 2


def test_a_mover_failing_to_start_faults_the_driver_though_none_started_before() -> None:
    """The first mover failing to start may have started: the driver is faulted as after a
    partial launch, its buffer retained, until the image is loaded again."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    inputs = {given["tensor"]: np.zeros((1, 784))}
    device = Unstartable(answers=device_for(found).answers)
    driver = Driver(device, read(found))
    driver.stage(core["name"], inputs)
    before = set(RETAINED)
    device.failing = "out0"
    with pytest.raises(OSError, match="out0: its register did not write"):
        driver.launch(core["name"], 1)
    with pytest.raises(
        Refused,
        match=r"^driver-faulted: DataflowCore_0: a launch of DataflowCore_0: out0 failed to "
        r"start: a started mover",
    ):
        driver.run_core(core["name"], inputs)
    assert [RETAINED[each][2].address for each in set(RETAINED) - before] == [BASE + STEP]
    device.failing = ""
    driver.load(Path("finn-accel.bit"))
    assert set(RETAINED) == before
    driver.run_core(core["name"], inputs)


def test_weights_are_written_between_launches_on_a_driver_in_use(tmp_path: Path) -> None:
    """Run-time weights are refused while a launch has not completed, on a faulted driver
    and on a closed one, before any register is written."""
    found = with_weights(recorded())
    (tmp_path / "Thresholding_0.dat").write_text("1")
    (core,) = found["cores"]
    (given,) = core["inputs"]
    inputs = {given["tensor"]: np.zeros((1, 784))}
    stalling = Stalling(answers=device_for(found).answers)
    stalling.failures = 1
    armed = Driver(stalling, read(found))
    with pytest.raises(OSError):
        armed.run_core(core["name"], inputs)
    device = Unstartable(answers=stalling.answers)
    faulted = Driver(device, read(found))
    faulted.stage(core["name"], inputs)
    device.failing = "in0"
    with pytest.raises(OSError):
        faulted.launch(core["name"], 1)
    closed = Driver(FakeDevice(), read(found))
    closed.close()
    for driver, code in (
        (armed, "mover-armed"),
        (faulted, "driver-faulted"),
        (closed, "driver-closed"),
    ):
        written = len(driver.device.calls)  # type: ignore[attr-defined]
        with pytest.raises(Refused, match=f"^{code}: Thresholding_0: "):
            driver.write_weights("Thresholding_0", [1])
        with pytest.raises(Refused, match=f"^{code}: "):
            driver.write_weight_files(tmp_path)
        assert driver.device.calls[written:] == []  # type: ignore[attr-defined]
    stalling.registers.clear()
    armed.settle(core["name"])
    armed.write_weights("Thresholding_0", [1])


def test_a_core_is_staged_whole_or_not_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restaging a core that its packing refuses, or its inputs' shape or presence, touches
    none of its buffers, and one whose sync fails stops there: each leaves the core
    unstaged, a launch and a collection refused, until it is staged again."""
    found = recorded()
    (core,) = found["cores"]
    (given,) = core["inputs"]
    device = device_for(found)
    driver = Driver(device, read(found))
    driver.stage(core["name"], {given["tensor"]: np.ones((1, 784))})
    made = len(device.calls)
    with pytest.raises(Refused, match="^value-outside: "):
        driver.stage(core["name"], {given["tensor"]: np.full((1, 784), 10**6)})
    assert device.calls[made:] == []
    unstaged = "^core-unstaged: DataflowCore_0: not staged for 1 frames"
    with pytest.raises(Refused, match=unstaged):
        driver.launch(core["name"], 1)
    with pytest.raises(Refused, match=unstaged):
        driver.collect(core["name"], 1)
    driver.stage(core["name"], {given["tensor"]: np.ones((1, 784))})
    # An input of another shape, or none: refused before any buffer, the core unstaged.
    for refusal, given_inputs in (
        ("input-shape", {given["tensor"]: np.ones((1, 783))}),
        ("input-absent", {}),
    ):
        made = len(device.calls)
        with pytest.raises(Refused, match=f"^{refusal}: "):
            driver.stage(core["name"], given_inputs)
        assert device.calls[made:] == []
        with pytest.raises(Refused, match=unstaged):
            driver.launch(core["name"], 1)
        driver.stage(core["name"], {given["tensor"]: np.ones((1, 784))})

    def unsynced(buffer: Buffer, to_device: bool) -> None:
        raise OSError("the buffer did not sync")

    monkeypatch.setattr(device, "sync", unsynced)
    with pytest.raises(OSError, match="did not sync"):
        driver.stage(core["name"], {given["tensor"]: np.zeros((1, 784))})
    with pytest.raises(Refused, match=unstaged):
        driver.launch(core["name"], 1)
    monkeypatch.undo()
    driver.stage(core["name"], {given["tensor"]: np.zeros((1, 784))})
    driver.launch(core["name"], 1)


def test_the_pynq_stand_in_keeps_each_device_s_state(tmp_path: Path) -> None:
    """The PYNQ stand-in acts on the device each overlay, MMIO and buffer was made on, and
    raises on a register of a device with no overlay loaded, one outside the MMIO's window,
    a freed buffer synced and a buffer freed twice."""
    breaches = {
        "pynq.MMIO(0xA0000000, 4096, device=device0).read(0)": "device0, which has no overlay",
        "mmio.write(4096, 1)": "outside the window",
        "freed.sync_from_device()": "sync_from_device of the buffer 0x120000000 on device1",
        "freed.freebuffer()": "freebuffer of the buffer 0x120000000 on device1",
    }
    setup = (
        "import numpy as np\nimport pynq\n"
        "device0, device1 = pynq.Device.devices\n"
        "pynq.Overlay('finn-accel.bit', device=device1)\n"
        "mmio = pynq.MMIO(0xA0000000, 4096, device=device1)\n"
        "mmio.write(0, 1)\n"
        "freed = pynq.allocate((8,), np.uint8, target=device1)\n"
        "freed.freebuffer()\n"
        "assert (device0.registers, device1.registers) == ({}, {0xA0000000: 1})\n"
        "assert (device0.memory, device1.memory) == (set(), set())\n"
    )
    env = {**os.environ, "PYTHONPATH": str(stub(tmp_path)), "PYTHONDONTWRITEBYTECODE": "1"}
    for breach, said in breaches.items():
        ran = subprocess.run(
            [sys.executable, "-c", setup + breach],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert ran.returncode == 1 and "RuntimeError: " in ran.stderr, (breach, ran.stderr)
        assert said in ran.stderr, (breach, ran.stderr)


def test_the_recorded_texts_hold_no_machine_s_path() -> None:
    """The recorded texts are FINN's test data: no path of a machine."""
    for name in ("tfc-integration.json", "tfc-ops.txt", "tfc-pynq-calls.txt"):
        text = (RECORDED / name).read_text()
        for marker in ("/home", "/tmp", "/proj", "/scratch", "\\", str(Path.home())):
            assert marker not in text, (name, marker)


def emitted(parent: ModelWrapper, directory: Path) -> Path:
    """TFC's driver as the ``pynq`` runtime writes it into ``directory``/driver, linked by
    the ``pynq`` linker's placement, its bitfile at bitfile/finn-accel.bit beside it."""
    described = describe(parent)
    linked = Linked(
        image=directory / "bitfile" / "finn-accel.bit",
        placement=unit_bases(described, linked_row(described)),
        clocks_mhz=DELIVERED,
        reports={},
    )
    PynqRuntime().emit(described, linked, directory / "driver")
    return directory / "driver"


def test_the_pynq_runtime_writes_the_shared_driver_its_device_and_the_runtime_file(
    tfc_parent: Path, tmp_path: Path
) -> None:
    """``emit`` writes driver/ afresh: the shared driver's modules and the PYNQ device as
    FINN holds them, the runtime file (the recorded text, as the build's report writes it)
    and the two scripts, each running the bitfile beside driver/ unless told another."""
    (tmp_path / "driver").mkdir()
    (tmp_path / "driver" / "stale.py").write_text("an earlier build's")
    driver = emitted(ModelWrapper(str(tfc_parent)), tmp_path)
    written = sorted(str(path.relative_to(driver)) for path in driver.rglob("*") if path.is_file())
    assert written == sorted(
        [
            *(str(path) for path in shipped()),
            "finn_runtime/devices/__init__.py",
            "integration.json",
            "driver.py",
            "validate.py",
        ]
    )
    for path, source in shipped().items():
        assert (driver / path).read_bytes() == source.read_bytes(), path
    recorded_text = (RECORDED / "tfc-integration.json").read_text()
    assert (driver / "integration.json").read_text() + "\n" == recorded_text
    for script in ("driver.py", "validate.py"):
        assert bitfile_default(driver / script, tmp_path) == str(
            tmp_path / "bitfile" / "finn-accel.bit"
        )


def board_run(driver: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """``driver``'s driver.py run from ``cwd`` with ``args``, PYNQ stubbed and onnx, qonnx and
    finn unimportable (``support.pynq_stub``), its calls to PYNQ logged in ``cwd``/pynq.log."""
    env = {
        **os.environ,
        "PYTHONPATH": str(stub(cwd / "pynq_stub")),
        "PYNQ_STUB_LOG": str(cwd / "pynq.log"),
        "PYNQ_STUB_BLOCK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, str(driver / "driver.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_shipped_driver_runs_tfc_on_a_board_with_numpy_alone(
    tfc_parent: Path, tmp_path: Path
) -> None:
    """The driver as shipped, run on a board with numpy and PYNQ alone (PYNQ stubbed): the
    device of the index asked for every call, not PYNQ's active one, the bitfile beside it
    loaded and PL0 set to the delivered clock, each mover reached at its base, the input
    staged and moved and the output collected, every PYNQ call against the recorded
    sequence; the output saved. A throughput test writes what it measured."""
    driver = emitted(ModelWrapper(str(tfc_parent)), tmp_path)
    board = tmp_path / "board"
    board.mkdir()
    np.save(board / "input.npy", (np.arange(784) % 256).reshape(1, 784).astype(np.float32))
    ran = board_run(driver, board, "--device", "1")
    assert ran.returncode == 0, ran.stderr
    assert (board / "pynq.log").read_text() == (RECORDED / "tfc-pynq-calls.txt").read_text()
    output = np.load(board / "output.npy")
    assert output.shape == (1, 10) and not output.any()
    (board / "pynq.log").unlink()
    ran = board_run(driver, board, "--exec_mode", "throughput_test", "--batchsize", "2")
    assert ran.returncode == 0, ran.stderr
    metrics = ast.literal_eval((board / "nw_metrics.txt").read_text())
    assert sorted(metrics) == sorted(
        [
            "runtime[ms]",
            "throughput[images/s]",
            "DRAM_in_bandwidth[MB/s]",
            "DRAM_out_bandwidth[MB/s]",
            "fclk[mhz]",
            "batch_size",
            "pack_input[ms]",
            "unpack_output[ms]",
        ]
    )
    assert (metrics["fclk[mhz]"], metrics["batch_size"]) == (187.512, 2)
    assert "allocate (1568,) uint8" in (board / "pynq.log").read_text()


def test_the_pynq_runtime_refuses_an_edge_the_shared_driver_cannot_drive(
    tfc_parent: Path, tmp_path: Path
) -> None:
    """An element the shared driver does not pack, or a mover kind it has no protocol for,
    is refused before anything is written."""
    described = describe(ModelWrapper(str(tfc_parent)))
    (core,) = described.cores
    (edge,) = core.inputs
    linked = Linked(image=tmp_path / "x.bit", placement={}, clocks_mhz={}, reports={})
    floated = replace(edge, element=ScalarEncoding(DataType["FLOAT<5,10,15>"]))
    pushed = replace(edge, contract=replace(edge.contract, kind="pushed_adapter"))
    for changed, code in ((floated, "element-unsupported"), (pushed, "mover-unknown")):
        changed_core = replace(core, inputs=(changed,))
        with refused(code, error=ShellRefused, owner="PynqRuntime", message="^in0: "):
            PynqRuntime().emit(replace(described, cores=(changed_core,)), linked, tmp_path / "d")
        assert not (tmp_path / "d").exists()


def test_an_absent_directory_of_weights_writes_nothing() -> None:
    """A directory of run-time weights that is not there writes none (the command line's
    default, where a build has none)."""
    device = FakeDevice()
    assert Driver(device, read(recorded())).write_weight_files(RECORDED / "absent") == []
    assert device.calls == []


def test_the_command_line_runs_each_core_on_its_files_and_refuses_another_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command line on the device it opens: each input file an input edge, each output
    saved; the run-time weights in their directory written first; another count of files
    refused, its code printed, status 1."""
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
    (tmp_path / "integration.json").write_text(json.dumps(found))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "Thresholding_0.dat").write_text("1 a\nff\n")
    np.save(tmp_path / "in.npy", np.zeros((2, 784), dtype=np.float32))
    devices: list[FakeDevice] = []

    def opened(description: Description, index: int) -> FakeDevice:
        devices.append(device_for(found, {"out0": bytes(18)}))
        return devices[-1]

    args = ["--inputfile", str(tmp_path / "in.npy"), "--outputfile", str(tmp_path / "out.npy")]
    weights = ["--runtime_weight_dir", str(tmp_path / "weights")]
    assert cli.main([*args, *weights], opened, tmp_path, tmp_path / "x.bit") == 0
    assert devices[0].calls[:4] == [
        "load x.bit",
        "write32 DataflowCore_0 0x40 0x1",
        "write32 DataflowCore_0 0x44 0xa",
        "write32 DataflowCore_0 0x48 0xff",
    ]
    assert np.load(tmp_path / "out.npy").shape == (2, 10)
    assert cli.main(["--inputfile"], opened, tmp_path, tmp_path / "x.bit") == 1
    assert capsys.readouterr().err.startswith("refused: inputs-count: 0 input and 1 output")


def test_validation_counts_each_frame_s_label_against_the_test_set() -> None:
    """Validation runs the one core a batch at a time, the last batch what remains, and
    counts over the whole test set; a frame's label is the place of its largest output:
    here every output is zero, so every label 0. A batch of no frame and an empty test set
    are refused."""
    found = recorded()
    driver = Driver(device_for(found, {"out0": bytes(9)}), read(found))
    images = np.zeros((4, 1, 28, 28), dtype=np.float32)
    assert validate.validate(driver, images, np.array([0, 3, 0, 0]), 1) == 75.0
    five, labels = np.zeros((5, 1, 28, 28), dtype=np.float32), np.array([0, 0, 0, 0, 1])
    assert validate.validate(driver, five, labels, 2) == 80.0
    assert validate.validate(driver, five, labels, 7) == 80.0
    assert validate.labels_of(np.array([[1.0], [4.0]]), 2).tolist() == [1.0, 4.0]
    with pytest.raises(Refused, match="^batch-invalid: "):
        validate.validate(driver, five, labels, 0)
    with pytest.raises(Refused, match="^dataset-empty: "):
        validate.validate(driver, five[:0], labels[:0], 2)


def test_validation_writes_the_run_time_weights_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validation writes the run-time weights in their directory once the image is loaded,
    as the command line does, then runs the test set."""
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
    (tmp_path / "integration.json").write_text(json.dumps(found))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "Thresholding_0.dat").write_text("1 a\n")
    test_set = (np.zeros((3, 1, 28, 28), dtype=np.float32), np.array([0, 0, 1]))
    monkeypatch.setattr(validate, "dataset_of", lambda dataset, root: test_set)
    devices: list[FakeDevice] = []

    def opened(description: Description, index: int) -> FakeDevice:
        devices.append(device_for(found, {"out0": bytes(9)}))
        return devices[-1]

    argv = ["--dataset", "mnist", "--runtime_weight_dir", str(tmp_path / "weights")]
    assert validate.main(argv, opened, tmp_path, tmp_path / "x.bit") == 0
    assert devices[0].calls[:3] == [
        "load x.bit",
        "write32 DataflowCore_0 0x40 0x1",
        "write32 DataflowCore_0 0x44 0xa",
    ]
    assert capsys.readouterr().out.endswith("Final accuracy: 66.666667\n")
