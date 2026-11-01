# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pynq shell's driver for the build's parent graph (``MakePynqDriver``): the ``pynq``
runtime's (``runtime.PynqRuntime.emit``), from the partition's integration description and
what the bitfile's build linked of it: the bitfile the parent graph states
(``shipped_bitfile``), each unit's base as the build's runtime file states it
(report/integration.json, ``linked_placement``), and the clock the bitfile delivers
(report/delivered_clock.json, ``delivered_mhz``), which the driver sets PL0 to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.core.graph.annotations.outputs import OUTPUT_BITFILE, OUTPUT_BITFILE_IP_KEY
from finn.core.graph.partition import design_digest, partition_body
from finn.shells.integration import MAIN, RUNTIME_SCHEMA, Integration, Linked, describe
from finn.shells.pynq.runner import (
    DELIVERED_CLOCK,
    REPORTS,
    PynqRefused,
    pynq_row,
    refuse_stale_bitfile,
)
from finn.shells.pynq.runtime import PynqRuntime

#: Where ``MakePynqDriver`` writes the driver and its description, under the output
#: directory it is given.
DRIVER_DIR = Path("driver")
DRIVER_REPORT = Path("report") / "driver.json"


def delivered_mhz(directory: Path) -> float:
    """The clock the bitfile built into ``directory`` delivers, in MHz, from its
    delivered-clock report there (``runner.DELIVERED_CLOCK``); no report (a
    ``PynqRefused``), or one that found no PL clock, is refused."""
    report = Path(directory) / DELIVERED_CLOCK
    if not report.is_file():
        raise PynqRefused(
            "pynq_driver",
            "clock-absent",
            f"no delivered clock ({report}): the driver sets the clock its bitfile "
            "delivers; list step_build_bitfile before step_make_driver",
        )
    clock = json.loads(report.read_text())
    if "delivered_mhz" not in clock:
        known = "; ".join(finding["message"] for finding in clock.get("findings", ()))
        raise PynqRefused(
            "pynq_driver", "clock-unknown", f"the bitfile's clock is not known: {known}"
        )
    return float(clock["delivered_mhz"])


def linked_placement(directory: Path) -> dict[str, int]:
    """Each unit's base as the bitfile built into ``directory`` placed it, from its runtime
    file there (``runner.REPORTS``); none, or one of another schema than this FINN's
    (``RUNTIME_SCHEMA``), is refused (``PynqRefused``)."""
    report = Path(directory) / REPORTS["integration"]
    if not report.is_file():
        raise PynqRefused(
            "pynq_driver",
            "integration-absent",
            f"no runtime file ({report}): the driver reaches each unit where the bitfile's "
            "build placed it; list step_build_bitfile before step_make_driver",
        )
    found = json.loads(report.read_text())
    if found.get("schema") != RUNTIME_SCHEMA:
        raise PynqRefused(
            "pynq_driver",
            "integration-schema",
            f"the runtime file {report} is of the schema {found.get('schema')!r}, and this "
            f"FINN's is {RUNTIME_SCHEMA}: build the bitfile again (step_build_bitfile)",
        )
    return {unit: int(base) for unit, base in found["placement"].items()}


def shipped_bitfile(model: ModelWrapper, directory: Path) -> Path:
    """The bitfile the parent graph ``model`` states (finn.outputs), relative to the output
    ``directory``, which the deployment package mirrors (bitfile/ beside driver/); a
    model that states none is refused (``PynqRefused``)."""
    bitfile = model.get(OUTPUT_BITFILE)
    if bitfile is None:
        raise PynqRefused(
            "pynq_driver",
            "bitfile-absent",
            "the model states no bitfile (finn.outputs): the driver runs the bitfile the "
            "build ships; list step_build_bitfile before step_make_driver",
        )
    return Path(bitfile).relative_to(Path(directory))


class MakePynqDriver(Transformation):
    """The ``pynq`` runtime's driver of the build's parent graph, into driver/ under the
    output ``directory``, written afresh (``PynqRuntime.emit``): its shell the target the
    partition's body states (``pynq_row``), its description the partition's
    (``describe``, ``integration``), linked as the bitfile's build linked it (the module
    docstring). A bitfile built from another design than the partition's now, or another
    IP than its body states, is refused, named (``refuse_stale_bitfile``), before anything
    is written. report/driver.json states the bitfile, by its path in the output directory
    and in the deployment package, both of which hold bitfile/ beside driver/; the runtime;
    the digest of the design the driver was made from (``made_from``: the partition's,
    which its bitfile was built from) and the key of the IP its bitfile holds
    (``ip_key``). ``description`` holds it."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.description: dict[str, Any] = {}
        self.integration: Integration | None = None

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        pynq_row(model, "pynq_driver")
        fclk_mhz = delivered_mhz(self.directory)
        bitfile = shipped_bitfile(model, self.directory)
        refuse_stale_bitfile(model, "pynq_driver")
        placement = linked_placement(self.directory)
        described = describe(model)
        linked = Linked(
            image=self.directory / bitfile,
            placement=placement,
            clocks_mhz={MAIN: fclk_mhz},
            reports={},
        )
        PynqRuntime().emit(described, linked, self.directory / DRIVER_DIR)
        _, body, _ = partition_body(model)
        self.integration = described
        self.description = {
            "bitfile": str(bitfile),
            "runtime": described.runtime,
            "made_from": design_digest(body),
            "ip_key": model.get(OUTPUT_BITFILE_IP_KEY),
        }
        report = self.directory / DRIVER_REPORT
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(self.description, indent=2))
        return model, False


__all__ = [
    "DRIVER_DIR",
    "DRIVER_REPORT",
    "MakePynqDriver",
    "delivered_mhz",
    "linked_placement",
    "shipped_bitfile",
]
