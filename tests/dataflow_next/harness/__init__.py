# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The comparison harness: ``finn.dataflow`` against its revision, ``finn.dataflow_next``.

FINN is run as it derives a design (``sources``), every question it asks ``finn.dataflow``
recorded (``record``); each is asked again of the revision through one stated translation
of values (``translate``), and the answers compared and classified (``replay``).

    python -m dataflow_next.harness SOURCE [SOURCE ...] [--out DIRECTORY]

run from ``tests`` with ``src`` and ``tests`` importable, writes one table a source.
"""
