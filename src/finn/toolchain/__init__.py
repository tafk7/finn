# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What FINN knows of the external tools it runs.

``launcher``: the route a tool is launched on (``Selection``, ``Toolchain``,
``run_process``). ``host``: the host file, where this host's Xilinx tools and
licence server are (``~/.config/finn/xilinx.env``). ``vivado`` and ``hls``: one
module per tool.

The package re-exports nothing: each name is imported from the module that owns it.
"""
