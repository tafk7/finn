# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""How FINN runs external tools (Vivado, Vitis HLS, v++, compilers, ...).

- ``launcher``: runs a tool, keeps its transcript and raises ``ToolFailed`` when it fails.
- ``vivado``: reads the messages that the Vivado tools print.
"""
