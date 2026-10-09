# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The XSim executor and the testbench it runs.

``executor``: ``XSim``, which runs a partition node whose body is a partition of
KernelOps as its hardware, in XSim: its inputs streamed at the boundary in the order
each port presents, its outputs read back from the words that arrive. ``rtl``: the
stream testbench a module is simulated in, and ``measure``, its cycles. ``pacing``: how
a simulation stalls its streams, for the testbench and the XSI numeric transport alike.

The package re-exports nothing: each name is imported from the module that owns it,
so a job that imports only ``pacing`` loads neither the testbench nor the executor
(``scripts/emitted_text.py`` keys a job by the files it imports).
"""
