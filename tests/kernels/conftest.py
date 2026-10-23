# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each netlist a test reads with the shell's bus counts checked against its top's pins
(``support.top_pins``); ``purity_unchecked`` for a test that counts the calls the Space
engine's memo saves (``support.purity``)."""

from support.purity import purity_unchecked  # noqa: F401
from support.top_pins import _top_pins  # noqa: F401
