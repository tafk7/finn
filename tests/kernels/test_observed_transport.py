# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The observed scoreboard rejects a correct prefix followed by an extra beat; the XSI
drivers pace their streams as ``finn.core.rtlsim_pacing`` states; a design's runs share
one compilation, each simulating its own copy."""

import pytest

from finn.core.rtlsim_pacing import FREE, Pace
from finn.xsi._artifacts import write_record
from kernels.sweeps import rtl_transport

#: The toolchain the transport is given: nothing here runs a tool.
STUB = object()


class Port:
    def __init__(self, sim, name):
        self.sim, self.name, self.value = sim, name, 0

    def read(self):
        return self

    def as_bool(self):
        if self.name == "in0_V_tvalid":
            return self.sim.ticks == 1
        if self.name == "in0_V_tready":
            return True
        if self.name == "out0_V_tvalid":
            return self.sim.ticks in self.sim.output_ticks
        return bool(self.value)

    def as_hexstr(self):
        return "13"


class Watchdog:
    def reset(self):
        pass


class Sim:
    def __init__(self, output_ticks):
        self.output_ticks = output_ticks
        self.ticks = 0
        self.tasks = []
        self.watchdogs = []
        self.ports = {}
        self.top = self

    def getPort(self, name):
        return self.ports.setdefault(name, Port(self, name))

    def stream_input(self, *args, **kwargs):
        pass

    def create_watchdog(self, *args):
        value = Watchdog()
        self.watchdogs.append(value)
        return value

    def remove_watchdog(self, value):
        self.watchdogs.remove(value)

    def enlist(self, task):
        self.tasks.append(task)

    def run(self, *, cycles):
        for _ in range(cycles):
            self.ticks += 1
            remaining = []
            for task in self.tasks:
                updates = task(self)
                if updates is not None:
                    remaining.append(task)
                    for port, value in updates.items():
                        port.value = int(value, 16)
            self.tasks = remaining
            if not remaining:
                return []
        return ["run timeout"]


@pytest.mark.parametrize(
    "ticks, error", [((3,), None), ((3, 3500), "extra output"), ((), "missing output")]
)
def test_observation_drain_detects_extra_and_missing_outputs(monkeypatch, ticks, error):
    sim = Sim(ticks)
    monkeypatch.setattr(rtl_transport, "load_sim_obj", lambda *_, toolchain: sim)
    monkeypatch.setattr(rtl_transport, "reset_rtlsim", lambda *_: None)
    monkeypatch.setattr(rtl_transport, "close_rtlsim", lambda *_: None)
    request = {
        "stimulus": {"in0_V": [0x13]},
        "expected_outputs": {"out0_V": 1},
        "observations": {},
        "drain_cycles": 4000,
        "pacing": FREE.as_json(),
    }
    if error:
        with pytest.raises(AssertionError, match=error):
            rtl_transport._simulate_observed("unused", "unused", request, toolchain=STUB)
    else:
        result = rtl_transport._simulate_observed("unused", "unused", request, toolchain=STUB)
        assert result["outputs"] == {"out0_V": [0x13]}
        assert result["input_counts"] == {"in0_V": 1}
        assert result["drain_cycles"] == 4000
        assert sim.ports["out0_V_tready"].value == 1


class Pin:
    """A pin as the XSI drivers read it: what the driver last drove, or a fixed level."""

    def __init__(self, level=None):
        self.level, self.value = level, "0"

    def read(self):
        return self

    def as_bool(self):
        return bool(self.level) if self.level is not None else self.value == "1"

    def as_hexstr(self):
        return "7"


class Top:
    def __init__(self, stream, fixed):
        self.pins = {
            f"{stream}_{member}": Pin(fixed.get(member)) for member in ("tvalid", "tready", "tdata")
        }

    def getPort(self, name):
        return self.pins[name]


def _levels(driver, pin, cycles):
    """The level ``driver`` drives ``pin`` at, cycle by cycle."""
    found = []
    for _ in range(cycles):
        updates = driver(None)
        if updates is None:
            break
        for port, value in updates.items():
            port.value = value
        found.append(int(pin.as_bool()))
    return found


def test_the_xsi_input_driver_paces_by_handshakes_as_the_testbench_does():
    """Pace(2, 1), the consumer always ready: two beats, one idle cycle, and so on, as the
    stream testbench's counters drive valid (``finn.core.rtlsim_bench``)."""
    top = Top("in0_V", {"tready": 1})
    driver = rtl_transport._PacedInput(top, "in0_V", [1, 2, 3, 4, 5], Pace(2, 1))
    assert _levels(driver, top.pins["in0_V_tvalid"], 12) == [1, 1, 0, 1, 1, 0, 1, 0]


def test_the_xsi_output_driver_pauses_after_each_burst():
    top = Top("out0_V", {"tvalid": 1})

    class Watchdog:
        resets = 0

        def reset(self):
            Watchdog.resets += 1

    driver = rtl_transport._PacedOutput(top, "out0_V", 3, Pace(1, 5), Watchdog())
    levels = _levels(driver, top.pins["out0_V_tready"], 20)
    assert levels == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    assert driver.words == [7, 7, 7] and Watchdog.resets == 3


def test_a_design_is_compiled_once_and_each_run_simulates_its_own_copy(monkeypatch, tmp_path):
    """The first run compiles into the design's ``compiled``; every run copies it (XSI
    writes into the snapshot), and a compilation is reused only as its record states it."""
    glbl = tmp_path / "glbl.v"
    glbl.write_text("module glbl; endmodule\n")
    source = tmp_path / "top.sv"
    source.write_text("module top; endmodule\n")
    compiles = []

    def compile_sim_obj(top, sources, directory, *, behav, toolchain):
        compiles.append(top)
        so = directory / "xsim.dir" / top / "xsimk.so"
        so.parent.mkdir(parents=True)
        so.write_bytes(b"elaborated")
        write_record(so, kind="design", tool="xelab test", sources=[*sources, glbl])

    monkeypatch.setattr(rtl_transport, "compile_sim_obj", compile_sim_obj)
    monkeypatch.setattr(rtl_transport, "identity", lambda _toolchain, tool: f"{tool} test")
    monkeypatch.setattr(rtl_transport.vivado, "glbl", lambda _toolchain: glbl)
    request = {
        "top_module": "top",
        "sources": [str(source)],
        "compiled": str(tmp_path / "compiled"),
        "data_files": {"weights.dat": "0f\n"},
    }
    runs = [tmp_path / mode for mode in ("free", "stalled")]
    for run in runs:
        directory, so = rtl_transport._private_copy(request, run, toolchain=STUB)
        assert (directory, so) == (str(run), "xsim.dir/top/xsimk.so")
        assert (run / so).read_bytes() == b"elaborated"
        assert (run / "weights.dat").read_text() == "0f\n"
    assert compiles == ["top"]
    (runs[0] / "xsim.dir/top/xsimkernel.log").write_text("written by a run")
    assert not (tmp_path / "compiled/xsim.dir/top/xsimkernel.log").exists()
    assert not (runs[1] / "xsim.dir/top/xsimkernel.log").exists()

    other = tmp_path / "other.sv"
    other.write_text("module other; endmodule\n")
    with pytest.raises(ValueError, match="another design"):
        rtl_transport._private_copy(
            {**request, "sources": [str(other)]}, tmp_path / "another", toolchain=STUB
        )
    source.write_text("module top; wire changed; endmodule\n")
    with pytest.raises(ValueError, match="input changed"):
        rtl_transport._private_copy(request, tmp_path / "changed", toolchain=STUB)
    assert compiles == ["top"]
