############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Driving one RTL top under XSI, and the process discipline that requires.

Every sweep drives its design through this one transport -- the marshalling,
the paced drivers and the watchdog handling -- and compares the results
itself. Its streams are paced by the harness's one spec (``finn.harness.pacing``),
as the stream testbench paces its own: a ``Pacing`` gives each input and output
its ``Pace`` by position.  A second copy of the transport would be a second place for the
one-simulation-per-process rule to be got wrong.

**Each simulation runs in its own process.**  XSI keeps state that outlives
``close_rtlsim``: the third ``load_sim_obj`` in a process hangs -- not the
third load of the same object, the third load at all -- with no diagnostic and
without the watchdog firing.  Isolating per configuration is not enough,
because one configuration is already several simulations.  So :func:`drive`
marshals a request to a fresh interpreter running *this* module, whose whole
job is one simulation.

The worker is this module and not the caller's, deliberately.  A sweep that
spawned another sweep's file would re-enter that sweep's argument parser, and
the two would have to keep agreeing about flags they do not share.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, cast

import numpy as np

from finn.harness.pacing import Pace, Pacing
from finn.kernels.artifacts.module import RegisterMap
from finn.xsi import close_rtlsim, compile_sim_obj, load_sim_obj, reset_rtlsim

#: Watchdog budget, in cycles without an accepted output beat.
LIVENESS = 4000


def random_word(generator: np.random.RandomState, bits: int) -> int:
    """A random integer of exactly ``bits`` bits.

    Built from bytes rather than ``randint`` because a packed weight beat is
    ``PE * SIMD * WEIGHT_WIDTH`` wide and reaches 64 bits at modest folding,
    which ``randint`` cannot represent.

    ``.astype(np.uint8).tobytes()`` and not ``bytes(...)``.  ``randint``
    returns ``int64``, and ``bytes()`` on a numpy array reads its *buffer* --
    eight bytes per value, seven of them zero.  A 32-bit word would then be one
    random low byte and 24 zero bits, a four-lane weight beat would have three
    lanes stuck at zero, and a comparison would run on a quarter of the stimulus
    it appeared to use, without failing.
    """

    raw = generator.randint(0, 256, size=(bits + 7) // 8).astype(np.uint8).tobytes()
    return int.from_bytes(raw, "little") & ((1 << bits) - 1)


class _PacedInput:
    """Present ``values`` on an input stream in order, paced by ``pace`` as the stream
    testbench paces it (``finn.harness.pacing``): after every ``burst`` handshakes, valid
    low for ``pause`` cycles.

    Called before each rising edge with the pins as they are during the cycle (a
    handshake completes at that edge); what it returns is driven after the edge."""

    def __init__(self, top: Any, stream: str, values: list[int], pace: Pace) -> None:
        self.valid = top.getPort(f"{stream}_tvalid")
        self.ready = top.getPort(f"{stream}_tready")
        self.data = top.getPort(f"{stream}_tdata")
        if None in (self.valid, self.ready, self.data):
            raise ValueError(f"missing stream pins of {stream}")
        self.values, self.pace = values, pace
        self.sent = self.burst = self.idle = 0
        self.presenting = False

    def __call__(self, _sim: object) -> dict[Any, str] | None:
        if self.presenting and self.ready.read().as_bool():
            self.sent += 1
            self.burst += 1
            if self.burst == self.pace.burst:
                self.burst, self.idle = 0, self.pace.pause
        elif self.idle:
            self.idle -= 1
        if self.sent == len(self.values):
            if not self.presenting:
                return None
            self.presenting = False
            return {self.valid: "0"}
        self.presenting = not self.idle
        if not self.presenting:
            return {self.valid: "0"}
        return {self.valid: "1", self.data: f"{self.values[self.sent]:x}"}


class _PacedOutput:
    """Collect ``size`` words of an output stream, paced by ``pace``: after every ``burst``
    handshakes, ready low for ``pause`` cycles. Each accepted word resets ``watchdog``, so a
    deliberate stall does not read as a hang."""

    def __init__(self, top: Any, stream: str, size: int, pace: Pace, watchdog: Any) -> None:
        self.valid = top.getPort(f"{stream}_tvalid")
        self.ready = top.getPort(f"{stream}_tready")
        self.data = top.getPort(f"{stream}_tdata")
        if None in (self.valid, self.ready, self.data):
            raise ValueError(f"missing stream pins of {stream}")
        self.size, self.pace, self.watchdog = size, pace, watchdog
        self.words: list[int] = []
        self.burst = self.idle = 0
        self.taking = False

    def __call__(self, _sim: object) -> dict[Any, str] | None:
        if self.taking and self.valid.read().as_bool():
            self.watchdog.reset()
            self.words.append(int(self.data.read().as_hexstr(), 16))
            self.burst += 1
            if self.burst == self.pace.burst:
                self.burst, self.idle = 0, self.pace.pause
        elif self.idle:
            self.idle -= 1
        if len(self.words) == self.size:
            if not self.taking:
                return None
            self.taking = False
            return {self.ready: "0"}
        self.taking = not self.idle
        return {self.ready: "1" if self.taking else "0"}


def _die_with_parent() -> None:
    # Leading its own process group, the worker would outlive a parent that is
    # killed outright (a stopped background task); PR_SET_PDEATHSIG ends it then.
    # An xvlog/xelab it had started at that moment is orphaned and runs to its end.
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _run_worker(top_module: str, request: Path, response: Path, log: IO[str] | None = None) -> int:
    """Run one simulation process to completion or to its deadline; its exit status.

    XSI's hang gives no diagnostic and fires no watchdog, so without a deadline
    a hung load waits forever and takes the whole run with it. The worker leads
    its own process group, as ``finn.util.toolchain.run_process`` runs tools,
    so a deadline also ends the xvlog/xelab it started; unlike that, its output
    streams as it runs. ``FINN_XSI_TIMEOUT`` (seconds, default 1200) covers
    compile and simulation.
    """
    timeout = float(os.environ.get("FINN_XSI_TIMEOUT", "1200"))
    worker = subprocess.Popen(
        [sys.executable, __file__, "--simulate", str(request), "--out", str(response)],
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
        start_new_session=True,
        preexec_fn=_die_with_parent if sys.platform == "linux" else None,
    )
    try:
        return worker.wait(timeout=timeout)
    except BaseException as stopped:
        # A deadline or an interruption: the whole group, even if its leader has
        # exited while a descendant still runs.
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait()
        if isinstance(stopped, subprocess.TimeoutExpired):
            raise AssertionError(
                f"{top_module}: simulation timed out after {timeout:g} s (probable XSI hang; "
                "FINN_XSI_TIMEOUT sets the deadline)"
            ) from None
        raise


def drive(
    top_module: str,
    sources: list[str],
    stimulus: dict[str, list[int]],
    expected: int,
    *,
    pacing: Pacing,
    data_files: Mapping[str, str] | None = None,
) -> list[int]:
    """Compile and run one DUT in a fresh process, its streams paced by ``pacing``.

    Only marshals arguments across the process boundary; the simulation itself
    happens in :func:`simulate_once`, which is the whole body of that process.
    """

    with tempfile.TemporaryDirectory() as scratch:
        request = Path(scratch) / "request.json"
        response = Path(scratch) / "response.json"
        request.write_text(
            json.dumps(
                {
                    "top_module": top_module,
                    "sources": sources,
                    "stimulus": stimulus,
                    "expected": expected,
                    "pacing": pacing.as_json(),
                    "data_files": dict(data_files or {}),
                }
            )
        )
        returncode = _run_worker(top_module, request, response)
        if returncode != 0 or not response.is_file():
            raise AssertionError(f"{top_module}: simulation subprocess failed (exit {returncode})")
        payload = json.loads(response.read_text())
        if payload.get("error"):
            raise AssertionError(f"{top_module}: {payload['error']}")
        return cast("list[int]", payload["output"])


def drive_observed(
    top_module: str,
    sources: list[str],
    stimulus: dict[str, list[int]],
    expected_outputs: dict[str, int],
    observations: dict[str, dict[str, str]],
    *,
    pacing: Pacing,
    directory: Path,
    drain_cycles: int = LIVENESS,
    data_files: dict[str, str] | None = None,
    registers: Mapping[str, RegisterMap] | None = None,
) -> dict[str, Any]:
    """Run an observed production artifact, retaining request, response and compile files.

    Stream names are complete ABI bus names, paced by ``pacing`` in the order
    ``stimulus`` and ``expected_outputs`` name them. Each observation names actual
    read-only data/valid/ready pins and optionally last; absent pins refuse.
    data_files (name -> text) are placed where the simulation resolves relative
    file names, such as an INIT_FILE. ``registers`` (bus -> its writes, as a
    module declares them: ``finn.harness.rtl.declared_registers``) are carried
    out, in order, after reset and before any stream starts.
    """
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    response = directory / "response.json"
    request.write_text(
        json.dumps(
            {
                "top_module": top_module,
                "sources": sources,
                "stimulus": stimulus,
                "expected_outputs": expected_outputs,
                "observations": observations,
                "pacing": pacing.as_json(),
                "work_directory": str(directory / "compile"),
                "drain_cycles": drain_cycles,
                "data_files": data_files or {},
                "axilite_writes": {
                    bus: list(found.writes) for bus, found in (registers or {}).items()
                },
            },
            indent=2,
        )
    )
    with (directory / "simulation.log").open("w") as log:
        returncode = _run_worker(top_module, request, response, log=log)
    if returncode != 0 or not response.is_file():
        detail = response.read_text() if response.is_file() else "no response"
        raise AssertionError(f"{top_module}: simulation subprocess exit {returncode}: {detail}")
    payload = json.loads(response.read_text())
    if payload.get("error"):
        raise AssertionError(f"{top_module}: {payload['error']}")
    return cast("dict[str, Any]", payload)


class _AxiLiteWriter:
    """Carry out AXI-Lite writes one at a time: address and data, then the response.

    AWPROT and ARPROT are driven when the bus has them (a FinnLib wrapper ignores
    them and does not present them)."""

    def __init__(self, top: Any, bus: str, writes: list[tuple[int, int]]) -> None:
        self.pins = {
            name: top.getPort(f"{bus}_{name}")
            for name in (
                "AWVALID",
                "AWREADY",
                "AWADDR",
                "AWPROT",
                "WVALID",
                "WREADY",
                "WDATA",
                "WSTRB",
                "BVALID",
                "BREADY",
                "ARVALID",
                "ARPROT",
                "ARADDR",
                "RREADY",
            )
        }
        missing = [
            name
            for name, port in self.pins.items()
            if port is None and name not in ("AWPROT", "ARPROT")
        ]
        if missing:
            raise ValueError(f"{bus}: missing AXI-Lite pins {missing}")
        self.pins = {name: port for name, port in self.pins.items() if port is not None}
        self.writes = list(writes)
        self.address = self.data = self.response = False
        self.started = False

    def __call__(self, _sim: Any) -> dict[Any, str] | None:
        pins = self.pins
        if not self.started:
            self.started = True
            idle = ("ARVALID", "ARPROT", "ARADDR", "RREADY")
            return {pins[name]: "0" for name in idle if name in pins}
        updates: dict[Any, str] = {}
        if self.address and pins["AWREADY"].read().as_bool():
            self.address = False
            updates[pins["AWVALID"]] = "0"
        if self.data and pins["WREADY"].read().as_bool():
            self.data = False
            updates[pins["WVALID"]] = "0"
        if self.response and pins["BVALID"].read().as_bool():
            self.response = False
            updates[pins["BREADY"]] = "0"
        if self.address or self.data or self.response:
            return updates
        if not self.writes:
            return None
        address, data = self.writes.pop(0)
        self.address = self.data = self.response = True
        updates.update(
            {
                pins["AWVALID"]: "1",
                pins["AWADDR"]: f"{address:x}",
                **({pins["AWPROT"]: "0"} if "AWPROT" in pins else {}),
                pins["WVALID"]: "1",
                pins["WDATA"]: f"{data:x}",
                pins["WSTRB"]: "f",
                pins["BREADY"]: "1",
            }
        )
        return updates


def _simulate_observed(sim_dir: str, so_rel: str, request: dict[str, Any]) -> dict[str, Any]:
    sim = load_sim_obj(sim_dir, so_rel)
    reset_rtlsim(sim)
    for bus, writes in request.get("axilite_writes", {}).items():
        sim.enlist(_AxiLiteWriter(sim.top, bus, [(int(a), int(d)) for a, d in writes]))
        if sim.run(cycles=LIVENESS):
            raise AssertionError(f"{bus}: AXI-Lite writes did not complete")
    stimulus = request["stimulus"]
    expected = request["expected_outputs"]
    drain_cycles = request["drain_cycles"]
    if not expected or drain_cycles < LIVENESS:
        raise ValueError("observed runs require outputs and at least 4000 drain cycles")

    def pin(name: str) -> Any:
        value = sim.top.getPort(name)
        if value is None:
            raise ValueError(f"missing observation/stream pin {name}")
        return value

    def bus(name: str) -> dict[str, Any]:
        return {key: pin(f"{name}_{key}") for key in ("tdata", "tvalid", "tready")}

    inputs = {name: bus(name) for name in stimulus}
    outputs = {name: bus(name) for name in expected}
    monitors = {
        name: {key: pin(value) for key, value in names.items()}
        for name, names in request["observations"].items()
    }
    for ports in monitors.values():
        if not {"data", "valid", "ready"} <= set(ports) or set(ports) - {
            "data",
            "valid",
            "ready",
            "last",
        }:
            raise ValueError("an observation names exactly data/valid/ready and optional last")
    pacing = Pacing.from_json(request["pacing"])
    for index, (name, values) in enumerate(stimulus.items()):
        sim.enlist(_PacedInput(sim.top, name, list(values), pacing.input(index)))
    paces = {name: pacing.output(index) for index, name in enumerate(outputs)}
    watchdogs = {name: sim.create_watchdog(f"{name} timeout", LIVENESS) for name in outputs}

    class ObservedCollector:
        def __init__(self) -> None:
            self.outputs: dict[str, list[int]] = {name: [] for name in outputs}
            self.traces: dict[str, dict[str, list[int]]] = {
                name: {"words": [], "last": []} for name in monitors
            }
            self.inputs = dict.fromkeys(inputs, 0)
            self.burst = dict.fromkeys(outputs, 0)
            self.stall = dict.fromkeys(outputs, 0)
            self.drain = 0
            self.held: dict[str, tuple[int, int | None]] = {}

        def sample(self, name: str, ports: dict[str, Any]) -> tuple[int, int | None] | None:
            valid = ports["valid"].read().as_bool()
            ready = ports["ready"].read().as_bool()
            value = (
                (
                    int(ports["data"].read().as_hexstr(), 16),
                    int(ports["last"].read().as_bool()) if "last" in ports else None,
                )
                if valid
                else None
            )
            if name in self.held and (not valid or self.held[name] != value):
                raise AssertionError(f"{name}: payload/framing changed while stalled")
            if valid and not ready:
                assert value is not None
                self.held[name] = value
            else:
                self.held.pop(name, None)
            return value if valid and ready else None

        def __call__(self, _sim: Any) -> dict[Any, str] | None:
            updates = {}
            for name, ports in inputs.items():
                if ports["tvalid"].read().as_bool() and ports["tready"].read().as_bool():
                    self.inputs[name] += 1
                    if self.inputs[name] > len(stimulus[name]):
                        raise AssertionError(f"{name}: extra accepted input")
            for name, ports in monitors.items():
                value = self.sample(name, ports)
                if value is not None:
                    self.traces[name]["words"].append(value[0])
                    if value[1] is not None:
                        self.traces[name]["last"].append(value[1])
            for name, ports in outputs.items():
                value = self.sample(
                    name,
                    {"data": ports["tdata"], "valid": ports["tvalid"], "ready": ports["tready"]},
                )
                if value is not None:
                    self.outputs[name].append(value[0])
                    watchdogs[name].reset()
                    if len(self.outputs[name]) > expected[name]:
                        raise AssertionError(
                            f"{name}: extra output after expected {expected[name]} transfers"
                        )
                    if len(self.outputs[name]) == expected[name]:
                        sim.remove_watchdog(watchdogs[name])
                    else:  # after every burst, a pause (finn.harness.pacing)
                        self.burst[name] += 1
                        if self.burst[name] == paces[name].burst:
                            self.burst[name], self.stall[name] = 0, paces[name].pause
                # Keep ready high after the expected prefix to observe extras.
                if len(self.outputs[name]) >= expected[name]:
                    self.stall[name] = 0
                updates[ports["tready"]] = "0" if self.stall[name] else "1"
                self.stall[name] = max(0, self.stall[name] - 1)
            if all(self.inputs[name] == len(values) for name, values in stimulus.items()):
                self.drain += 1
                if self.drain >= drain_cycles:
                    if any(len(self.outputs[name]) != count for name, count in expected.items()):
                        raise AssertionError("missing output transfers after drain window")
                    return None
            return updates

    collector = ObservedCollector()
    sim.enlist(collector)
    try:
        timeouts = sim.run(cycles=20 * LIVENESS)
        if timeouts:
            raise AssertionError(f"deadlock or incomplete drain: {timeouts}")
        return {
            "outputs": collector.outputs,
            "observations": collector.traces,
            "input_counts": collector.inputs,
            "drain_cycles": collector.drain,
            "cycles": sim.ticks,
        }
    finally:
        close_rtlsim(sim)


def simulate_once(request_path: str, response_path: str) -> int:
    """Run exactly one simulation described by a JSON request, then exit.

    The whole body of the process: compile, load, run, write the outputs.
    Nothing else may load a simulation object here, which is the point.
    """

    request = json.loads(Path(request_path).read_text())
    payload: dict[str, object]
    try:
        if "observations" in request:
            scratch = Path(request["work_directory"])
            scratch.mkdir(parents=True, exist_ok=False)
            for name, contents in cast("dict[str, str]", request.get("data_files", {})).items():
                (scratch / name).write_text(contents)
            sim_dir, so_rel = compile_sim_obj(
                request["top_module"],
                request["sources"],
                str(scratch),
                behav=True,
            )
            payload = _simulate_observed(sim_dir, so_rel, request)
            Path(response_path).write_text(json.dumps(payload, indent=2))
            return 0
        with tempfile.TemporaryDirectory() as scratch:
            for name, contents in cast("dict[str, str]", request.get("data_files", {})).items():
                (Path(scratch) / name).write_text(contents)
            sim_dir, so_rel = compile_sim_obj(
                request["top_module"], request["sources"], scratch, behav=True
            )
            payload = {
                "output": _simulate(
                    sim_dir,
                    so_rel,
                    request["stimulus"],
                    request["expected"],
                    pacing=Pacing.from_json(request["pacing"]),
                    label=request["top_module"],
                )
            }
    except Exception as failure:  # noqa: BLE001 - reported across the boundary
        payload = {"error": f"{type(failure).__name__}: {failure}"}
    Path(response_path).write_text(json.dumps(payload))
    return 1 if payload.get("error") else 0


def _simulate(
    sim_dir: str,
    so_rel: str,
    stimulus: dict[str, list[int]],
    expected: int,
    *,
    pacing: Pacing,
    label: str,
) -> list[int]:
    sim = load_sim_obj(sim_dir, so_rel)
    reset_rtlsim(sim)
    # Paced by position, so two inputs also arrive out of step rather than in lockstep.
    for index, (name, values) in enumerate(stimulus.items()):
        sim.enlist(_PacedInput(sim.top, f"{name}_V", list(values), pacing.input(index)))
    watchdog = sim.create_watchdog("out0_V timeout", LIVENESS)
    collected = _PacedOutput(sim.top, "out0_V", expected, pacing.output(0), watchdog)
    sim.enlist(collected)
    timeouts = sim.run()
    if timeouts:
        raise AssertionError(f"{label}: deadlock, watchdogs fired: {timeouts}")
    result = collected.words
    if watchdog in sim.watchdogs:
        sim.remove_watchdog(watchdog)
    close_rtlsim(sim)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", metavar="REQUEST", required=True)
    parser.add_argument("--out", metavar="RESPONSE", required=True)
    arguments = parser.parse_args(argv)
    return simulate_once(arguments.simulate, arguments.out)


if __name__ == "__main__":
    sys.exit(main())
