# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernels' numeric XSI sweeps: command-line programs, not tests.

Each module is run as ``python -m kernels.sweeps.<name>`` by
``scripts/xsim-sweep.sh``, one simulation per process (XSI keeps state across
loads). Nothing here is collected by pytest; ``test_observed_transport`` checks
the transport they share.
"""
