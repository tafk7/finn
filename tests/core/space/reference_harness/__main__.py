# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Print the acceptance matrix as a markdown table: ``python -m core.space.reference_harness``
from ``tests``."""

from .matrix import markdown

print(markdown())
