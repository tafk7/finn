# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import sys
import types

# finn_xsi is not pip-installed: finn.xsi adds this dir to sys.path at runtime
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XSI_ROOT = os.path.join(os.environ.get("FINN_ROOT", _REPO_ROOT), "finn_xsi")


class FakePort:
    def __init__(self, design, name):
        self.design = design
        self.name = name

    def isInput(self):
        return True

    def clear(self):
        return self

    def set(self, value):
        self.design.writes.append((self.name, str(value)))
        return self

    def set_hexstr(self, value):
        self.design.writes.append((self.name, value))
        return self

    def write_back(self):
        return self


class FakeDesign:
    """Stands in for xsi.Design. fail_in names the method that raises, and
    fail_with the exception it raises."""

    fail_in = None
    fail_with = RuntimeError
    opened = []

    def __init__(self, kernel, design, log=None, wdb=None):
        if self.fail_in == "open":
            raise self.fail_with("cannot open design")
        self.close_calls = 0
        self.writes = []
        self._ports = {name: FakePort(self, name) for name in ("ap_clk", "ap_rst_n")}
        FakeDesign.opened.append(self)

    def getPort(self, name):
        return self._ports.get(name)

    def ports(self):
        if self.fail_in == "ports":
            raise self.fail_with("cannot list ports")
        return list(self._ports.values())

    def trace_all(self):
        if self.fail_in == "trace_all":
            raise self.fail_with("cannot trace")

    def run(self, ticks):
        pass

    def close(self):
        self.close_calls += 1


@pytest.fixture
def adapter(monkeypatch):
    """finn_xsi.adapter, simulating on FakeDesign."""
    monkeypatch.syspath_prepend(_XSI_ROOT)
    try:
        import xsi  # noqa: F401, PLC0415
    except ImportError:
        # finn_xsi is not built: sim_engine only needs an xsi module to import
        monkeypatch.setitem(sys.modules, "xsi", types.ModuleType("xsi"))
    from finn_xsi import adapter, sim_engine  # noqa: PLC0415

    fake_xsi = types.SimpleNamespace(Kernel=lambda path: path, Design=FakeDesign)
    monkeypatch.setattr(sim_engine, "xsi", fake_xsi)
    monkeypatch.setattr(FakeDesign, "fail_in", None)
    monkeypatch.setattr(FakeDesign, "fail_with", RuntimeError)
    monkeypatch.setattr(FakeDesign, "opened", [])
    return adapter


def load(adapter, sim_dir, tracefile=None):
    return adapter.load_sim_obj(
        str(sim_dir), "xsim.dir/top/xsimk.so", tracefile, simkernel_so="libsimkernel.so"
    )


@pytest.mark.util
@pytest.mark.parametrize(
    "fail_in, fail_with",
    [
        ("open", RuntimeError),
        ("ports", RuntimeError),
        ("trace_all", KeyboardInterrupt),
    ],
)
def test_load_sim_obj_cleans_up_on_failure(adapter, tmp_path, monkeypatch, fail_in, fail_with):
    monkeypatch.chdir(tmp_path)
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    FakeDesign.fail_in = fail_in
    FakeDesign.fail_with = fail_with
    with pytest.raises(fail_with):
        load(adapter, sim_dir, tracefile="trace.wdb")
    assert os.getcwd() == str(tmp_path)
    # a design that was opened is closed exactly once
    assert [design.close_calls for design in FakeDesign.opened] == (
        [] if fail_in == "open" else [1]
    )


def reset_cycles(writes):
    """Count the rising clock edges while ap_rst_n is low."""
    rst_low = False
    cycles = 0
    for name, value in writes:
        if name == "ap_rst_n":
            rst_low = value == "0"
        elif name == "ap_clk" and value == "1" and rst_low:
            cycles += 1
    return cycles


@pytest.mark.util
@pytest.mark.parametrize("n_cycles", [None, 32])
def test_reset_rtlsim_holds_reset_for_n_cycles(adapter, tmp_path, n_cycles):
    sim = load(adapter, tmp_path)
    if n_cycles is None:
        adapter.reset_rtlsim(sim)
        n_cycles = 16
    else:
        adapter.reset_rtlsim(sim, n_cycles=n_cycles)
    writes = sim.top.writes
    assert ("ap_rst_n", "0") in writes
    assert writes.index(("ap_rst_n", "0")) < writes.index(("ap_rst_n", "1"))
    assert reset_cycles(writes) == n_cycles


@pytest.mark.util
@pytest.mark.parametrize("n_cycles, error", [(0, ValueError), (1.5, TypeError)])
def test_reset_rtlsim_refuses_other_than_positive_integers(adapter, tmp_path, n_cycles, error):
    sim = load(adapter, tmp_path)
    with pytest.raises(error):
        adapter.reset_rtlsim(sim, n_cycles=n_cycles)
    assert sim.tasks == []
