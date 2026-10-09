# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The finn-dev oracle: legacy FINN at its pin, and the captures the kernel path's tests read.

The kernel path's tests compare with values only the HWCustomOp flow computes (its FIFO
model, its cycle estimates, SetFolding, the IODMA it inserts). That flow is not part of
the kernel path, so the values come from finn-dev itself, at a pinned commit, run here
once and committed as data under ``tests/oracle/captures``. The fast gate reads the
captures; nothing in it runs the oracle.

    python3 scripts/oracle/generate.py --work DIR [--source REPO] [--lock] [PROBE ...]

1. **The tree.** ``DIR/finn-<pin>``: a clone of ``--source`` (by default this checkout's
   repository, which holds the pin) checked out detached at ``PIN``. A tree that exists
   is used if it is clean and at the pin, and refused otherwise.
2. **The venv.** ``DIR/venv-<pin>``, Python 3.12 (the pin's ``docker/Dockerfile.finn``:
   Ubuntu noble), with the pin's own dependencies: its ``requirements.txt`` less the
   documentation and development tools (``EXCLUDED``), qonnx and brevitas at the commits
   its ``fetch-repos.sh`` names, and torch, torchvision and pytest at its Dockerfile's
   versions, from PyPI. Every version is held by ``constraints-<pin>.txt`` beside this script;
   ``--lock`` writes that file from the venv it builds instead of reading it.
3. **The probes** (``probes/<name>.py``, all of them in ``ORDER`` unless named), each in a fresh
   interpreter of the venv, with an environment of its own: the tree's ``src`` on
   ``PYTHONPATH``, ``FINN_ROOT`` the tree, ``FINN_BUILD_DIR`` under ``DIR``, and no
   Xilinx tool but the stand-in ``tools/fake_xilinx_tool.py`` (``FINN_TOOL_DIR_OVERRIDE``).
4. **The captures.** ``tests/oracle/captures/<name>.json`` (and any file a probe names
   beside it): the probe's values, stamped with the oracle's commit, the probe's name
   and the venv's versions. A probe that writes a model leaves it beside its capture,
   its sha256 in the capture.

Re-running reproduces every capture byte for byte: no stamp carries a time or a path,
and the versions only move with the constraints file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PIN = "b507fec5865f07013d590fc798ec9f73127354de"
"""finn-dev's ``dev`` at the line's merge-base with upstream (2026-09-25, "Merge pull
request #1707 from Xilinx/feature/requantf"): the project's ``finn-oracle`` pin."""
SHORT = PIN[:9]
PYTHON = "3.12"
"""The pin's Docker image is Ubuntu noble, whose Python is 3.12."""
EXCLUDED = {"gspread", "pre-commit", "pyscaffold", "setupext-janitor", "sphinx", "sphinx-rtd-theme"}
"""The pin's requirements that only build its documentation or its releases."""
ORDER = ("fifo_cost", "exp_cycles", "tfc_streamlined", "tfc_set_folding", "tfc_zynq_build")
"""The probes, in the order they run: the TFC probes after ``tfc_streamlined``, whose
committed model they start from."""
PACKAGES = ("numpy", "onnx", "onnxruntime", "protobuf", "qonnx", "brevitas", "torch")
"""The versions each capture is stamped with."""

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
CAPTURES = CHECKOUT / "tests" / "oracle" / "captures"
CONSTRAINTS = HERE / f"constraints-{SHORT}.txt"
PROBES = HERE / "probes"
TOOLS = HERE / "tools"


def run(*command: str | Path, **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def git(tree: Path, *args: str) -> str:
    return run("git", "-C", tree, *args, capture_output=True, text=True).stdout.strip()


def materialise(work: Path, source: str) -> Path:
    """The oracle's tree: a clone of ``source`` at ``PIN``, clean."""
    tree = work / f"finn-{SHORT}"
    if not tree.exists():
        run("git", "clone", "--quiet", "--no-checkout", source, tree)
        run("git", "-C", tree, "checkout", "--quiet", "--detach", PIN)
    head = git(tree, "rev-parse", "HEAD")
    status = git(tree, "status", "--porcelain")
    if head != PIN or status:
        sys.exit(f"{tree}: at {head}, {'dirty' if status else 'clean'}; want {PIN}, clean")
    return tree


def pinned(tree: Path) -> list[str]:
    """The pin's own dependencies, as requirement specifiers."""
    requirements = []
    for line in (tree / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        name = re.split(r"[=<>!~ \[]", line)[0].lower().replace("_", "-")
        if line and name not in EXCLUDED:
            requirements.append(line)
    fetch = (tree / "fetch-repos.sh").read_text()
    for name, variable, url in (
        ("qonnx", "QONNX_COMMIT", "https://github.com/fastmachinelearning/qonnx"),
        ("brevitas", "BREVITAS_COMMIT", "https://github.com/Xilinx/brevitas"),
    ):
        commit = re.search(rf'^{variable}="([0-9a-f]{{40}})"', fetch, re.M).group(1)
        # From git, as fetch-repos.sh takes them: their versions come from their tags.
        requirements.append(f"{name} @ git+{url}@{commit}")
    docker = (tree / "docker" / "Dockerfile.finn").read_text()
    # finn.util.test, which the TFC probe reads its network from, imports pytest.
    for name in ("torch", "torchvision", "pytest"):
        version = re.search(rf"pip install .*\b{name}==(\S+)", docker).group(1)
        requirements.append(f"{name}=={version}")
    return requirements


def venv(work: Path, tree: Path, lock: bool) -> Path:
    """The oracle's venv, built once for its requirements and constraints."""
    env = work / f"venv-{SHORT}"
    python = env / "bin" / "python"
    requirements = pinned(tree)
    constraints = None if lock else CONSTRAINTS.read_text()
    spec = json.dumps({"requirements": requirements, "constraints": constraints})
    stamp = env / "oracle-spec.json"
    if stamp.is_file() and stamp.read_text() == spec:
        return python
    if env.exists():
        shutil.rmtree(env)
    run("uv", "venv", "--quiet", "--python", PYTHON, env)
    listed = work / "requirements.txt"
    listed.write_text("\n".join(requirements) + "\n")
    install = ["uv", "pip", "install", "--python", python, "-r", listed]
    run(*install, *([] if lock else ["-c", CONSTRAINTS]))
    if lock:
        frozen = run(
            "uv",
            "--color",
            "never",
            "pip",
            "freeze",
            "--python",
            python,
            capture_output=True,
            text=True,
        )
        CONSTRAINTS.write_text(frozen.stdout)
    stamp.write_text(spec)
    return python


def versions(python: Path) -> dict[str, str]:
    """The stamp's versions: Python and ``PACKAGES``, as the venv reports them, qonnx and
    brevitas with the commit they were installed from."""
    script = (
        "import json, platform, importlib.metadata as m\n"
        f"names = {list(PACKAGES)!r}\n"
        "found = {'python': platform.python_version()}\n"
        "for name in names:\n"
        "    found[name] = m.version(name)\n"
        "    url = json.loads(m.distribution(name).read_text('direct_url.json') or '{}')\n"
        "    if 'vcs_info' in url:\n"
        "        found[name] += ' @ ' + url['vcs_info']['commit_id'][:12]\n"
        "print(json.dumps(found))\n"
    )
    return json.loads(run(python, "-c", script, capture_output=True, text=True).stdout)


def environment(work: Path, tree: Path, python: Path, probe: str) -> dict[str, str]:
    """A probe's environment: the oracle and nothing of the host's FINN or Xilinx setup."""
    tools = work / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("vivado", "vitis_hls", "vitis-run"):
        link = tools / name
        if not link.exists():
            link.symlink_to(TOOLS / "fake_xilinx_tool.py")
    build = work / "build" / probe
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    return {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": f"{tree / 'src'}:{PROBES}",
        "FINN_ROOT": str(tree),
        "FINN_BUILD_DIR": str(build),
        "FINN_TOOL_DIR_OVERRIDE": str(tools),
        # Read only for its release (CallHLS picks vitis-run after 2024.2); no tool is
        # taken from it.
        "XILINX_VIVADO": "/oracle-stand-in/2025.2/Vivado",
        "PWD": str(build),
        "FAKE_TOOL_LOG": str(work / "build" / f"{probe}.tools.log"),
        "NUM_DEFAULT_WORKERS": "1",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(work: Path, tree: Path, python: Path, probe: str, stamp: dict) -> Path:
    """Run ``probe`` and write its capture."""
    raw = work / "raw" / probe
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    env = environment(work, tree, python, probe)
    run(python, PROBES / f"{probe}.py", raw, CAPTURES, env=env, cwd=env["FINN_BUILD_DIR"])
    values = json.loads((raw / "values.json").read_text())
    files = {}
    for path in sorted(raw.iterdir()):
        if path.name != "values.json":
            shutil.copyfile(path, CAPTURES / path.name)
            files[path.name] = sha256(path)
    document = {**stamp, "probe": probe, "files": files, "values": values}
    out = CAPTURES / f"{probe}.json"
    out.write_text(json.dumps(document, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probes", nargs="*", help="probes/<name>.py; all (ORDER) when none")
    parser.add_argument("--work", type=Path, required=True, help="the tree, venv and builds")
    parser.add_argument("--source", default=str(CHECKOUT), help="a FINN repository holding PIN")
    parser.add_argument("--lock", action="store_true", help=f"write {CONSTRAINTS.name}")
    args = parser.parse_args()
    unknown = sorted(set(args.probes) - set(ORDER))
    if unknown:
        sys.exit(f"unknown probes {unknown}; known: {list(ORDER)}")
    probes = [probe for probe in ORDER if probe in args.probes or not args.probes]
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    tree = materialise(work, args.source)
    python = venv(work, tree, args.lock)
    stamp = {
        "oracle": {"commit": PIN, "subject": git(tree, "log", "-1", "--format=%s")},
        "generator": "scripts/oracle/generate.py",
        "versions": versions(python),
        "xilinx": "none: Vivado and Vitis HLS stubbed (scripts/oracle/tools)",
    }
    CAPTURES.mkdir(parents=True, exist_ok=True)
    for probe in probes:
        print(f"== {probe}", flush=True)
        print("wrote", capture(work, tree, python, probe, stamp).relative_to(CHECKOUT), flush=True)


if __name__ == "__main__":
    main()
