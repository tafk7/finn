# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The observed scoreboard rejects a correct prefix followed by an extra beat."""

import pytest

from dataflow.rtlsim import rtl_transport


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
        "stalls": False,
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
