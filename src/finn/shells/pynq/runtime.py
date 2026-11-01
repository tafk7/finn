# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` runtime (``PynqRuntime``): what runs a build on a PYNQ board, the driver
written beside its runtime file.

``PynqRuntime.emit`` writes into the driver's directory, afresh:

- ``finn_runtime/``: the shared driver (``finn.shells.runtime``) as it is, and this
  runtime's device (``finn.shells.pynq.device``) as ``finn_runtime/devices/pynq.py``
  (``shipped``);
- ``integration.json``: the runtime file (``finn.shells.integration.runtime_text``), the
  same text as the build's report of it;
- ``driver.py``: the command line (``finn_runtime.cli``) over the PYNQ device;
- ``validate.py``: the top-1 validation (``finn_runtime.validate``) over it.

Both scripts run the device image the linker made unless told another, by its path relative
to the driver's directory, from wherever they are run. An edge whose element the shared
driver cannot pack, or whose mover kind it has no protocol for, is refused before anything
is written (``ShellRefused``: ``element-unsupported``, ``mover-unknown``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath

from finn.shells import runtime as shared
from finn.shells.build import ShellRefused
from finn.shells.integration import Integration, Linked, runtime_text
from finn.shells.pynq import device
from finn.shells.runtime.errors import Refused
from finn.shells.runtime.movers import PROTOCOLS
from finn.shells.runtime.packing import Element

#: The shared driver's package name in the driver's directory.
PACKAGE = "finn_runtime"

#: The scripts ``emit`` writes, each by the shared driver's module whose ``main`` it runs.
SCRIPTS = {"driver.py": "cli", "validate.py": "validate"}

LAUNCHER = '''"""FINN's {name} for this build: ``{package}.{module}`` over the PYNQ device, by the
runtime file beside it."""

import os
import sys
from pathlib import Path

from {package}.devices.pynq import PynqDevice
from {package}.{module} import main

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], PynqDevice, HERE, Path(os.path.normpath(HERE / {image!r}))))
'''
"""A script ``emit`` writes: its ``name``, the shared driver's ``module`` whose ``main`` it
runs, and the device ``image``, relative to its directory."""


def shipped() -> dict[PurePosixPath, Path]:
    """The files of ``finn_runtime/`` by their paths in it, each the FINN source it is
    copied from: the shared driver's modules and the PYNQ device."""
    sources = Path(shared.__file__).parent
    files = {PurePosixPath(PACKAGE, path.name): path for path in sorted(sources.glob("*.py"))}
    files[PurePosixPath(PACKAGE, "devices", "pynq.py")] = Path(device.__file__)
    return files


def refuse_undriven(integration: Integration) -> None:
    """Refuse an edge of ``integration`` the shared driver cannot drive: its element not one
    it packs, or its mover kind not one it has a protocol for (``ShellRefused``)."""
    for edge in integration.edges:
        try:
            Element.named(edge.element.dtype.name)
        except Refused as refused:
            raise ShellRefused(
                "driver",
                refused.code,
                f"{edge.unit}: {refused.message}",
                owner="PynqRuntime",
            ) from refused
        if edge.contract.kind not in PROTOCOLS:
            raise ShellRefused(
                "driver",
                "mover-unknown",
                f"{edge.unit}: its mover kind {edge.contract.kind!r} has no protocol in the "
                f"shared driver (the kinds: {sorted(PROTOCOLS)})",
                owner="PynqRuntime",
            )


class PynqRuntime:
    """The ``pynq`` runtime's code (``finn.shells.integration.Runtime``; the module
    docstring)."""

    def emit(self, integration: Integration, linked: Linked, directory: Path) -> None:
        """Write the driver of ``integration``, linked as ``linked``, into ``directory``,
        afresh."""
        refuse_undriven(integration)
        directory = Path(directory)
        shutil.rmtree(directory, ignore_errors=True)
        for path, source in shipped().items():
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (directory / PACKAGE / "devices" / "__init__.py").write_text(
            '"""The devices of the runtimes that ship this driver."""\n'
        )
        (directory / "integration.json").write_text(runtime_text(integration, linked))
        image = os.path.relpath(linked.image, directory)
        for name, module in SCRIPTS.items():
            (directory / name).write_text(
                LAUNCHER.format(name=name, package=PACKAGE, module=module, image=image)
            )


__all__ = ["LAUNCHER", "PACKAGE", "SCRIPTS", "PynqRuntime", "refuse_undriven", "shipped"]
