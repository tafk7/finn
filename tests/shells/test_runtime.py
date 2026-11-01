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
import json
import os
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
from qonnx.util.basic import gen_finn_dt_tensor
from support.devices import BASE, FakeDevice
from support.packaging import bitfile_default
from support.pynq_stub import stub
from support.refusals import refused
from support.tfc import partition

from finn.dataflow.tensor import ScalarEncoding
from finn.shells.integration import MAIN, Linked, describe, runtime_file
from finn.shells.pynq.linker import linked_row, unit_bases
from finn.shells.pynq.runner import PynqRefused
from finn.shells.pynq.runtime import PynqRuntime, shipped
from finn.shells.runtime import Description, Driver, Refused, cli, read, validate
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
    device of the index asked, the bitfile beside it loaded and PL0 set to the delivered
    clock, each mover reached at its base, the input staged and moved and the output
    collected, every PYNQ call against the recorded sequence; the output saved. A throughput
    test writes what it measured."""
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
        with refused(code, error=PynqRefused, owner="PynqRuntime", message="^in0: "):
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
    """Validation runs the one core a batch at a time; a frame's label is the place of its
    largest output: here every output is zero, so every label 0."""
    found = recorded()
    driver = Driver(device_for(found, {"out0": bytes(9)}), read(found))
    images = np.zeros((4, 1, 28, 28), dtype=np.float32)
    assert validate.validate(driver, images, np.array([0, 3, 0, 0]), 1) == 75.0
    assert validate.labels_of(np.array([[1.0], [4.0]]), 2).tolist() == [1.0, 4.0]
