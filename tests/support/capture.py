# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The functions on the tests' side that the sweep's keying tool replaces.

``scripts/emitted_text.py`` captures what a job simulates without Vivado: it runs the
job's test or sweep with these functions replaced, one by one, by wrappers that record
what the simulation was asked to read. ``SEAMS`` declares them, each under a stable name
the tool uses; renaming or removing one is a change here (``tests/xsim_sweep`` resolves
every seam), not a silent break of the sweep's keys.

FINN's own seams, which the tool also replaces (``finn.core.rtlsim_bench.simulate``,
``built_hls``, ``finn.toolchain.ip.ip_store``, ``packaging.emit_module``), are not
declared here: they are ``src``'s.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

#: A seam's stable name: the module (importable with ``tests`` on the path) and the
#: attribute the tool replaces.
SEAMS: Mapping[str, tuple[str, str]] = {
    # the conformance check of one kernel: counted, so the tool can tell a capture that
    # ran to its end from one that stopped
    "conformance.check": ("support.conformance", "conformance"),
    # the conformance check's stream through the IP-XACT interface: the tool writes the
    # interface text it sees
    "conformance.stream": ("support.conformance", "stream_through"),
    # whether the XSim tests are selected: always, since nothing simulates
    "xsim.selected": ("support.xsim", "vivado_simulator"),
    # the numeric sweeps' transport to the simulator's worker process
    "sweep.transport": ("kernels.sweeps.rtl_transport", "_run_worker"),
}


def replace(seam: str, make: Callable[[Any], Any]) -> Any:
    """Replace the seam's attribute by ``make(current)``, and return the replacement."""
    module_name, attribute = SEAMS[seam]
    module = importlib.import_module(module_name)
    replacement = make(getattr(module, attribute))
    setattr(module, attribute, replacement)
    return replacement
