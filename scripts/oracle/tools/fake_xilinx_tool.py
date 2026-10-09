#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A stand-in for ``vivado``, ``vitis_hls`` and ``vitis-run``, by the name it is called
as, for the oracle's legacy builds: it answers a version or help probe, and otherwise
makes only the files the HWCustomOp flow looks for after a tool run, running nothing.
The scripts the flow wrote are the evidence.

- ``vitis_hls [-f] SCRIPT``, ``vitis-run --mode hls --tcl SCRIPT`` (``hls_syn_<node>.tcl``):
  ``project_<node>/sol1/impl/ip``.
- ``vivado -mode batch -source make_project.tcl`` (``CreateStitchedIP``): the block
  design's wrapper, where the script's ``make_wrapper`` names it.
- ``vivado -mode batch -source ip_config.tcl`` (``MakeZYNQProject``): an empty bitfile
  and hardware handoff where ``MakeZYNQProject`` reads them.

Each call is appended to ``$FAKE_TOOL_LOG``, if set.
"""

import os
import re
import sys
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
if os.environ.get("FAKE_TOOL_LOG"):
    with open(os.environ["FAKE_TOOL_LOG"], "a") as log:
        log.write(" ".join([tool, *args]) + "\n")
if args[:1] in (["-version"], ["--version"]):
    print(f"{tool} v2025.2 (64-bit): the oracle's stand-in")
elif args == ["--help"]:
    print("--mode hls (the oracle's stand-in)")
elif tool in ("vitis_hls", "vitis-run"):
    script = next((args[i + 1] for i, a in enumerate(args) if a in ("-f", "--tcl")), args[-1])
    node = Path(script).name.removeprefix("hls_syn_").removesuffix(".tcl")
    Path(f"project_{node}/sol1/impl/ip").mkdir(parents=True, exist_ok=True)
elif tool == "vivado":
    script = Path(args[args.index("-source") + 1])
    if script.name == "make_project.tcl":
        design = re.search(r"make_wrapper -files \[get_files (\S+)\.bd\]", script.read_text())
        bd = Path(design.group(1))
        wrapper = bd.parent / "hdl" / f"{bd.name}_wrapper.v"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(f"// the oracle's stand-in wrapper for {bd.name}\n")
    elif script.name == "ip_config.tcl":
        for made in (
            "finn_zynq_link.runs/impl_1/top_wrapper.bit",
            "finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh",
        ):
            Path(made).parent.mkdir(parents=True, exist_ok=True)
            Path(made).write_text("")
    else:
        sys.exit(f"{tool} (stand-in): no outputs known for {script}")
else:
    sys.exit(f"{tool} (stand-in): not a tool it stands in for")
