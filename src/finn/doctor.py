# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""finn doctor: what this environment gives FINN, and what FINN can do in it.

A read-only report: it builds, fetches and writes nothing. It reads the AMD tools'
environment as FINN does (``finn.util.launcher``), and prints the entries of the
release table (``finn.util.releases``) for the release in use. It exits with 1 if
something is wrong (tools of different releases, HLS headers missing), not if
something is merely absent: FINN without the AMD tools is a valid setup.
"""

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import sysconfig
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from finn.util import launcher, releases
from finn.util.launcher import ToolchainMismatch, ToolUnavailable

OK, NOTE, WARN, FAIL = "ok", "note", "warn", "fail"


@dataclass
class Finding:
    section: str
    label: str
    state: str
    detail: str


class Report:
    def __init__(self):
        self.findings: List[Finding] = []
        self.release: Optional[Tuple[int, int]] = None

    def add(self, section, label, state, detail=""):
        self.findings.append(Finding(section, label, state, detail))
        return state == OK

    def failed(self) -> bool:
        return any(f.state == FAIL for f in self.findings)


def _release_name(release) -> str:
    return "%d.%d" % release if release else "unknown"


def _tool(report, name) -> bool:
    try:
        return report.add("AMD tools", name, OK, launcher.xilinx_tool(name))
    except ToolchainMismatch as error:
        return report.add("AMD tools", name, FAIL, str(error))
    except ToolUnavailable as error:
        return report.add("AMD tools", name, WARN, error.reason)


def _first_line(command) -> str:
    try:
        out = launcher.run_tool(command).stdout.decode()
    except (OSError, launcher.ToolFailed, ToolUnavailable):
        return ""
    return out.splitlines()[0] if out else ""


def check_tools(report):
    section = "AMD tools"
    set_up = {name: launcher.amd_path(name) for name in launcher.AMD_VARIABLES}
    if not any(set_up.values()):
        report.add(section, "release", NOTE, "none set up: " + launcher.setup_hint())
        return {}
    try:
        report.release = launcher.expected_release()
    except ToolchainMismatch as error:
        report.add(section, "release", FAIL, str(error))
        return {}
    report.add(section, "release", OK, _release_name(report.release))
    for name, path in set_up.items():
        if path:
            report.add(section, name, OK, path)
    override = os.environ.get(launcher.TOOL_DIR_ENV)
    if override:
        report.add(section, launcher.TOOL_DIR_ENV, NOTE, f"tools run through {override}")
    found = {}
    found["vivado"] = _tool(report, "vivado") if set_up["XILINX_VIVADO"] else False
    found["xelab"] = _tool(report, "xelab") if set_up["XILINX_VIVADO"] else False
    hls_tool = "vitis-run" if releases.affects("hls-via-vitis-run", report.release) else "vitis_hls"
    found["hls"] = (
        _tool(report, hls_tool) if set_up["XILINX_HLS"] or set_up["XILINX_VITIS"] else False
    )
    found["v++"] = _tool(report, "v++") if set_up["XILINX_VITIS"] else False
    if set_up["XILINX_HLS"] or set_up["XILINX_VITIS"]:
        try:
            found["headers"] = report.add(section, "HLS headers", OK, launcher.hls_root())
        except ToolUnavailable as error:
            found["headers"] = report.add(section, "HLS headers", FAIL, error.reason)
    current = report.release
    others = [r for r, _ in launcher.installed_releases() if r != current]
    if others:
        report.add(section, "also installed", NOTE, ", ".join(map(_release_name, others)))
    return found


def check_host(report):
    section = "Host"
    found = {}
    compiler = shutil.which("g++")
    found["g++"] = report.add(
        section,
        "g++",
        OK if compiler else WARN,
        f"{compiler}: {_first_line([compiler, '--version'])}"
        if compiler
        else "not found: cppsim and the XSim binding need it",
    )
    header = os.path.join(sysconfig.get_paths()["include"], "Python.h")
    found["headers"] = report.add(
        section,
        "Python headers",
        OK if os.path.isfile(header) else WARN,
        header if os.path.isfile(header) else f"no {header}: the XSim binding needs them",
    )
    licences = [n for n in ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE") if os.environ.get(n)]
    report.add(
        section,
        "licence",
        OK if licences else NOTE,
        "; ".join(f"{n}={os.environ[n]}" for n in licences)
        or "no licence server set (a node-locked licence or none)",
    )
    site_file = getattr(sys.modules.get("sitecustomize"), "__file__", None)
    vitis = launcher.amd_path("XILINX_VITIS")
    if site_file and vitis and _inside(site_file, vitis):
        report.add(
            section,
            "sitecustomize",
            WARN,
            f"this Python ran {site_file}, which preloads Vitis's libstdc++ (see below)",
        )
    return found


def _inside(path, directory) -> bool:
    path, directory = os.path.realpath(path), os.path.realpath(directory)
    return os.path.commonpath([path, directory]) == directory


def check_xsi(report) -> Optional[str]:
    """The XSim binding's state: 'built', 'first-use', 'failed', or None without Vivado."""
    if not launcher.amd_path("XILINX_VIVADO"):
        return None
    try:
        from finn.xsi.paths import build_dir  # noqa: PLC0415

        directory = build_dir()
    except Exception as error:  # noqa: BLE001 (a report, not a build)
        report.add("FINN", "XSim binding", WARN, f"cannot locate: {error}")
        return "failed"
    if (directory / "xsi.so").exists():
        report.add("FINN", "XSim binding", OK, f"built in {directory}")
        return "built"
    log = directory / "compile_xsi.stderr.log"
    if log.exists():
        report.add("FINN", "XSim binding", WARN, f"its last build failed; see {log}")
        return "failed"
    report.add("FINN", "XSim binding", NOTE, f"builds on first use, into {directory}")
    return "first-use"


def check_resources(report):
    from finn import resources  # noqa: PLC0415

    try:
        declared = resources.declarations()
    except Exception as error:  # noqa: BLE001
        report.add("Resources", "declarations", FAIL, str(error))
        return
    for name in declared:
        try:
            status = resources.status(name)
        except Exception as error:  # noqa: BLE001
            report.add("Resources", name, FAIL, str(error))
            continue
        if status.state == "missing":
            report.add("Resources", name, NOTE, "fetched on first use")
        else:
            report.add("Resources", name, OK, f"{status.state}: {status.path}")


def check_capabilities(report, tools, host, xsi):
    section = "FINN can run"
    report.add(section, "transformations, estimates, code generation", OK)
    hls = tools.get("headers", False)
    _can(report, section, "cppsim", hls and host["g++"], "needs Vitis HLS headers and g++")
    _can(report, section, "HLS synthesis", tools.get("hls", False), "needs Vitis HLS")
    _can(
        report,
        section,
        "rtlsim",
        tools.get("xelab", False) and xsi,
        "needs Vivado, and g++ with Python headers to build the XSim binding",
    )
    _can(
        report,
        section,
        "stitched IP, Zynq and Versal bitfiles",
        tools.get("vivado"),
        "needs Vivado",
    )
    alveo = [] if tools.get("v++") else ["Vitis"]
    alveo += [n for n in ("PLATFORM_REPO_PATHS", "XILINX_XRT") if not os.environ.get(n)]
    _can(report, section, "Alveo builds", not alveo, "needs " + ", ".join(alveo))


def _can(report, section, label, available, needs):
    report.add(section, label, OK if available else NOTE, "" if available else needs)


def check_finn(report):
    try:
        version = importlib.metadata.version("finn")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    where = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report.add("FINN", "version", OK, f"{version}, from {where}")
    report.add("FINN", "Python", OK, f"{sys.version.split()[0]} ({sys.executable})")


def diagnose() -> Report:
    report = Report()
    check_finn(report)
    tools = check_tools(report)
    host = check_host(report)
    state = check_xsi(report)
    xsi = state == "built" or (state == "first-use" and host["g++"] and host["headers"])
    check_resources(report)
    check_capabilities(report, tools, host, xsi)
    if report.release:
        for note in releases.notes_for(report.release):
            report.add(
                f"Known differences of release {_release_name(report.release)}",
                f"{note.tools} ({note.releases()})",
                NOTE,
                f"{note.differs} FINN {note.finn_does}",
            )
    return report


_MARK = {OK: "ok  ", NOTE: "--  ", WARN: "warn", FAIL: "FAIL"}


def render(report) -> str:
    sections: dict[str, list[str]] = {}
    for f in report.findings:
        line = f"  {_MARK[f.state]}  {f.label}" + (f": {f.detail}" if f.detail else "")
        sections.setdefault(f.section, []).append(line)
    return "\n\n".join("\n".join([name, *lines]) for name, lines in sections.items())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="finn doctor",
        description="Report what this environment gives FINN. Changes nothing.",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = diagnose()
    if args.json:
        print(
            json.dumps(
                {
                    "release": _release_name(report.release) if report.release else None,
                    "findings": [asdict(f) for f in report.findings],
                },
                indent=2,
            )
        )
    else:
        print(render(report))
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
