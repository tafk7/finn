# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Module build values and their emission, below every Space.

``module`` holds what is built: a FinnLib module (``Leaf``) or a flat netlist
of them (``Composed``), with its pins (``abi``) and files (``contributions``).
``build`` writes a module's sources (ordered by ``sources``) and, for a
composed module, its netlist; ``hls`` stages an HLS request's sources; ``rtl``
checks declared pins against the RTL; ``ipxact`` emits the Tcl that packages an
emitted module as a Vivado IP, and ``interface`` describes its pins
(``interface.json``); ``projection`` takes the canonical digest that keys a module.
A one-way import rule, a row of the layer table in ``tests/layering.py``:
``artifacts`` imports the standard library and its approved dependencies;
kernels import ``artifacts``, never the reverse. What crosses into it is a
detached ``Module``.
"""
