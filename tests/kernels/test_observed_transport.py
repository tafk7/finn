# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The observed scoreboard rejects a correct prefix followed by an extra beat; the XSI
drivers pace their streams as ``finn.core.executors.xsim.pacing`` states."""

import pytest

from finn.core.executors.xsim.pacing import FREE, Pace
from kernels.sweeps import rtl_transport


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
    monkeypatch.setattr(rtl_transport, "load_sim_obj", lambda *_: sim)
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
            rtl_transport._simulate_observed("unused", "unused", request)
    else:
        result = rtl_transport._simulate_observed("unused", "unused", request)
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
    stream testbench's counters drive valid (``finn.core.executors.xsim.rtl``)."""
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
