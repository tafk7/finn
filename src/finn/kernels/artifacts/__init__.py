# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Module build values and their emission, below every Space.

``module`` holds what is built: a FinnLib module (``Leaf``) or a flat netlist
of them (``Composed``), with its pins (``abi``) and files (``contributions``);
``requirements`` the older requirement values, until the composite path that
uses them is retired. ``build`` writes a module's sources (ordered by
``sources``, templates rendered by ``render``), ``rtl`` checks declared pins
against the RTL. A one-way import rule, tested in
``tests/kernels/artifacts/test_isolation.py``: ``artifacts`` imports the
standard library and its approved dependencies; kernels import ``artifacts``,
never the reverse. What crosses into it is a detached ``ModuleBuildRequirements``.
"""
