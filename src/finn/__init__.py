# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN.

Importing any FINN module turns qonnx's exact integers on for the process: wherever
qonnx's execute_node runs (FINN's execution, constant folding in qonnx's and FINN's
transformations, the builder's steps), integers are computed exactly, as the hardware
computes them, where the graph's float32 containers would round them, and constant
folding leaves a node unfolded where float32 cannot hold its value, rather than
rounding it. The setting is the environment variable qonnx reads on every call, so it
holds in every thread and in child processes, and setting it here imports nothing.
"""

import os

os.environ["QONNX_EXACT_INTEGERS"] = "1"
