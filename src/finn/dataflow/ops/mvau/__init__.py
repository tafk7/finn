# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU: canonical semantic values today, the operation itself in U4.

What survives the legacy reset is the part that was never an experiment --
the normalized Region declarations and the decomposed Network they form.  They
are the reference semantics every later MVAU implementation is measured
against, and they depend on nothing but ``region`` and ``network``.

The DataflowOp, its source projection, persistence and association arrive here
in U4 and are authored against the occurrence lifecycle rather than restored
from the retired stack.  This namespace performs no eager operation import.
"""

__all__: list[str] = []
