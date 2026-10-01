# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Module build values and their emission, below every Space.

``requirements`` holds what a module needs to be built (``abi``: its pins;
``contributions``: its files), ``build`` writes its sources (ordered by
``sources``, templates rendered by ``render``), ``rtl`` checks declared pins
against the RTL. A one-way import rule, tested in
``tests/kernels/artifacts/test_isolation.py``: ``artifacts`` imports the
standard library and its approved dependencies; kernels import ``artifacts``,
never the reverse. What crosses into it is a detached ``ModuleBuildRequirements``.
"""
