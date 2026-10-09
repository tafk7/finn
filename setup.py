# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# -*- coding: utf-8 -*-
"""Build hooks for FINN: package data and source provenance.

Project metadata and dependencies are declared in pyproject.toml.
"""
from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

import json
import subprocess
from pathlib import Path


def resource_files(directory, prefix="", extra=()):
    """Ship source resources and notices, excluding tests and native artifacts.

    Paths are relative to the package that owns ``directory``: ``prefix`` is the
    directory's path inside that package. ``extra`` adds suffixes to ship.
    """
    allowed = {
        ".v",
        ".sv",
        ".vh",
        ".svh",
        ".tcl",
        ".xml",
        ".cpp",
        ".hpp",
        ".h",
        ".template",
        ".dat",
        ".mem",
        ".abc",
        ".mdd",
        ".mld",
        ".toml",
    }
    result = []
    for path in Path(directory).rglob("*"):
        rel = path.relative_to(directory)
        notice = any(x in path.name.lower() for x in ("license", "licence", "notice", "copying"))
        if not path.is_file() or any(
            x in rel.parts for x in ("test", "tests", "testcase", "tb", "__pycache__", "build")
        ):
            continue
        # rtllib/sim/hdl holds the stitched-IP simulation controller, not a testbench.
        if "_tb." in path.name or ("sim" in rel.parts and rel.parts[:2] != ("sim", "hdl")):
            continue
        if notice or path.suffix in allowed or path.suffix in extra:
            if path.suffix not in {".pyc", ".so"}:
                result.append(prefix + rel.as_posix())
    return sorted(result)


def provenance():
    saved = Path("FINN_BUILD_INFO.json")
    if saved.exists():
        return json.loads(saved.read_text())
    info = {"version": Path("VERSION").read_text().strip(), "revision": None, "dirty": None}
    if Path(".git").exists():
        try:
            info["revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            info["dirty"] = bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True)
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return info


class BuildPy(build_py):
    def run(self):
        super().run()
        path = Path(self.build_lib) / "finn" / "_build_info.json"
        path.write_text(json.dumps(provenance(), sort_keys=True) + "\n")


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        (Path(base_dir) / "FINN_BUILD_INFO.json").write_text(
            json.dumps(provenance(), sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    packages = find_namespace_packages(
        "src",
        exclude=[
            "finn.qnn-data",
            "finn.qnn-data.*",
            # Data directories inside packages, not packages of their own.
            "finn.rtllib.*",
            "finn.custom_hls.*",
            "finn.xsi.src",
            "finn.xsi.src.*",
            "finn.deploy.data",
            "finn.deploy.data.*",
            "finn.shells.pynq.data",
            "finn.shells.pynq.data.*",
            "finn.platform.data",
        ],
    )
    setup(
        cmdclass={"build_py": BuildPy, "sdist": Sdist},
        version=Path("VERSION").read_text().strip(),
        packages=packages,
        package_dir={"": "src"},
        package_data={
            "finn": ["resources.toml"],
            "finn.rtllib": resource_files("src/finn/rtllib"),
            "finn.custom_hls": resource_files("src/finn/custom_hls"),
            "finn.xsi": resource_files("src/finn/xsi/src", prefix="src/"),
            "finn.deploy": resource_files("src/finn/deploy/data", prefix="data/", extra=(".py",)),
            "finn.shells.pynq": resource_files(
                "src/finn/shells/pynq/data", prefix="data/", extra=(".py",)
            ),
            "finn.core.space": ["py.typed"],
            "finn.dataflow": ["py.typed"],
            "finn.kernels": ["py.typed"],
            "finn.platform": ["catalog.tcl", "data/*.jsonl", "data/manifest.json"],
        },
    )
