############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Driving one RTL top under XSI, and the process discipline that requires.

Extracted from fixture 5, unchanged.  Fixture 8 compares the same composed RTL
against arithmetic rather than against the fused core, and the Phase 6 plan is
explicit that it should reuse fixture 5's *transport* and not its comparison --
the marshalling, the backpressure collector and the watchdog handling are
correct and hard-won, and a second copy of them would be a second place for the
one-simulation-per-process rule to be got wrong.

**Each simulation runs in its own process.**  XSI keeps state that outlives
``close_rtlsim``: the third ``load_sim_obj`` in a process hangs -- not the
third load of the same object, the third load at all -- with no diagnostic and
without the watchdog firing.  Isolating per configuration is not enough,
because one configuration is already several simulations.  So :func:`drive`
marshals a request to a fresh interpreter running *this* module, whose whole
job is one simulation.

The worker is this module and not the caller's, deliberately.  A fixture that
spawned another fixture's file would re-enter that fixture's argument parser,
and the two would have to keep agreeing about flags they do not share.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]

from finn.xsi import close_rtlsim, compile_sim_obj, load_sim_obj, reset_rtlsim

#: Watchdog budget, in cycles without an accepted output beat.
LIVENESS = 4000

#: Output beats accepted between stalls, and how long each stall lasts.
BACKPRESSURE_PERIOD = 1
BACKPRESSURE_TICKS = 5


def random_word(generator: np.random.RandomState, bits: int) -> int:
    """A random integer of exactly ``bits`` bits.

    Built from bytes rather than ``randint`` because a packed weight beat is
    ``PE * SIMD * WEIGHT_WIDTH`` wide and reaches 64 bits at modest folding,
    which ``randint`` cannot represent.

    ``.astype(np.uint8).tobytes()`` and not ``bytes(...)``.  ``randint``
    returns ``int64``, and ``bytes()`` on a numpy array reads its *buffer* --
    eight bytes per value, seven of them zero.  A 32-bit word therefore came
    out as one random low byte and 24 zero bits, so a four-lane weight beat had
    three lanes stuck at zero and fixture 5 compared two DUTs on a quarter of
    the stimulus it appeared to be using.  Nothing failed, because both DUTs
    got the same impoverished stream.
    """

    raw = generator.randint(0, 256, size=(bits + 7) // 8).astype(np.uint8).tobytes()
    return int.from_bytes(raw, "little") & ((1 << bits) - 1)


def _collect_with_backpressure(sim: object, stream: str, size: int, watchdog: object) -> object:
    """Collect outputs while de-asserting ready every few accepted beats.

    ``rtlsim_multi_io`` holds ready high forever, which never exercises the
    stall path: the obligation is that the composition does not drop, duplicate
    or reorder anything when the consumer is not listening, and a consumer that
    always listens cannot show that.

    The watchdog is reset on every accepted beat, exactly as the stock
    collector does; a deliberate stall must not read as a hang.
    """

    class ThrottledCollector:
        def __init__(self) -> None:
            # The bus-port accessor lives on the engine, which is what
            # SimEngine.collect_output passes its own collector as ``top``.
            self.vld = sim.get_bus_port(stream, "tvalid")  # type: ignore[attr-defined]
            self.rdy = sim.get_bus_port(stream, "tready")  # type: ignore[attr-defined]
            self.dat = sim.get_bus_port(stream, "tdata")  # type: ignore[attr-defined]
            self.buf: list[str] = []
            self.stall = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.buf)

        def __call__(self, _sim: object) -> object:
            if self.stall > 0:
                self.stall -= 1
                return {self.rdy: "0"} if self.rdy.as_bool() else {}
            if self.rdy.as_bool():
                if self.vld.read().as_bool():
                    watchdog.reset()  # type: ignore[attr-defined]
                    self.buf.append(self.dat.read().as_hexstr())
                    if len(self.buf) == size:
                        return {self.rdy: "0"}
                    if len(self.buf) % BACKPRESSURE_PERIOD == 0:
                        self.stall = BACKPRESSURE_TICKS
                        return {self.rdy: "0"}
                return {}
            if len(self.buf) < size:
                return {self.rdy: "1"}
            return None

    collector = ThrottledCollector()
    sim.enlist(collector)  # type: ignore[attr-defined]
    return collector


def drive(
    top_module: str,
    sources: list[str],
    stimulus: dict[str, list[int]],
    expected: int,
    *,
    stalls: bool,
    data_files: Mapping[str, str] | None = None,
) -> list[int]:
    """Compile and run one DUT in a fresh process, optionally stalling it.

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
                    "stalls": stalls,
                    "data_files": dict(data_files or {}),
                }
            )
        )
        completed = subprocess.run(
            [sys.executable, __file__, "--simulate", str(request), "--out", str(response)],
            check=False,
        )
        if completed.returncode != 0 or not response.is_file():
            raise AssertionError(
                f"{top_module}: simulation subprocess failed (exit {completed.returncode})"
            )
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
    stalls: bool,
    directory: Path,
    drain_cycles: int = LIVENESS,
) -> dict[str, Any]:
    """Run an observed production artifact, retaining request, response and compile files.

    Stream names are complete ABI bus names. Each observation names actual
    read-only data/valid/ready pins and optionally last; absent pins refuse.
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
                "stalls": stalls,
                "work_directory": str(directory / "compile"),
                "drain_cycles": drain_cycles,
            },
            indent=2,
        )
    )
    with (directory / "simulation.log").open("w") as log:
        completed = subprocess.run(
            [sys.executable, __file__, "--simulate", str(request), "--out", str(response)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not response.is_file():
        detail = response.read_text() if response.is_file() else "no response"
        raise AssertionError(
            f"{top_module}: simulation subprocess exit {completed.returncode}: {detail}"
        )
    payload = json.loads(response.read_text())
    if payload.get("error"):
        raise AssertionError(f"{top_module}: {payload['error']}")
    return cast("dict[str, Any]", payload)


def _simulate_observed(sim_dir: str, so_rel: str, request: dict[str, Any]) -> dict[str, Any]:
    sim = load_sim_obj(sim_dir, so_rel)
    reset_rtlsim(sim)
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
    for index, (name, values) in enumerate(stimulus.items()):
        throttle = (2 + index % 2, 3 + index % 3) if request["stalls"] else (float("inf"), 0)
        sim.stream_input(name, iter(f"{value:x}" for value in values), throttle=throttle)
    watchdogs = {name: sim.create_watchdog(f"{name} timeout", LIVENESS) for name in outputs}

    class ObservedCollector:
        def __init__(self) -> None:
            self.outputs: dict[str, list[int]] = {name: [] for name in outputs}
            self.traces: dict[str, dict[str, list[int]]] = {
                name: {"words": [], "last": []} for name in monitors
            }
            self.inputs = dict.fromkeys(inputs, 0)
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
                    elif request["stalls"]:
                        self.stall[name] = BACKPRESSURE_TICKS
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
                    stalls=request["stalls"],
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
    stalls: bool,
    label: str,
) -> list[int]:
    sim = load_sim_obj(sim_dir, so_rel)
    reset_rtlsim(sim)
    # Different throttles per stream, so the two inputs also arrive out of step
    # with each other rather than in lockstep.
    throttles = {"in0": (2, 3), "in1": (3, 2)} if stalls else {}
    for name, values in stimulus.items():
        sim.stream_input(
            f"{name}_V",
            map(lambda value: f"{value:0x}", list(values)),
            throttle=throttles.get(name, (float("inf"), 0)),
        )
    watchdog = sim.create_watchdog("out0_V timeout", LIVENESS)
    if stalls:
        collected = _collect_with_backpressure(sim, "out0_V", expected, watchdog)
    else:
        collected = sim.collect_output("out0_V", expected, watchdog=watchdog)
    timeouts = sim.run()
    if timeouts:
        raise AssertionError(f"{label}: deadlock, watchdogs fired: {timeouts}")
    # Both collectors are iterables of hex strings; neither is typed as one.
    result = [int(value, base=16) for value in cast("Iterable[str]", collected)]
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
