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
from collections.abc import Iterable
from pathlib import Path
from typing import cast

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
    """

    raw = int.from_bytes(bytes(generator.randint(0, 256, size=(bits + 7) // 8)), "little")
    return raw & ((1 << bits) - 1)


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


def simulate_once(request_path: str, response_path: str) -> int:
    """Run exactly one simulation described by a JSON request, then exit.

    The whole body of the process: compile, load, run, write the outputs.
    Nothing else may load a simulation object here, which is the point.
    """

    request = json.loads(Path(request_path).read_text())
    payload: dict[str, object]
    try:
        with tempfile.TemporaryDirectory() as scratch:
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
