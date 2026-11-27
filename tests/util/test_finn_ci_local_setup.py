# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""CI's native Local Setup stage (ci/Jenkinsfile): the job's release, without the container."""
import pytest

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.util


def local_setup_step():
    """The Local Setup stage's bash step from ci/Jenkinsfile, as Jenkins runs it."""
    text = (REPO / "ci/Jenkinsfile").read_text()
    stage = text[text.index("void runLocalSetupTest()") :]
    match = re.search(r"sh '''(#!/bin/bash\n.*?)'''", stage, re.DOTALL)
    assert match, "no bash step in runLocalSetupTest"
    return match.group(1)


@pytest.mark.parametrize("missing", [False, True], ids=["release", "missing-release"])
def test_the_stage_gets_the_jobs_release_ahead_of_the_venv(tmp_path, missing):
    # The FINN job's names (jenkins-deploy's finn.dsl) for xbuilds' separate trees:
    # <path>/Vivado/2024.2 and <path>/Vitis/2024.2.
    root = tmp_path / "Xilinx"
    trees = {
        tool: root / tool / ("2023.2" if missing else "2024.2") for tool in ("Vivado", "Vitis")
    }
    for tree in trees.values():
        tree.mkdir(parents=True)
        (tree / "settings64.sh").write_text(f'export PATH="{tree}/bin:$PATH"\n')
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / ".venv/bin").mkdir(parents=True)
    shutil.copy(REPO / "scripts/xilinx-release.sh", workspace / "scripts")
    (workspace / ".venv/bin/activate").write_text(f'export PATH="{workspace}/.venv/bin:$PATH"\n')
    quicktest = workspace / "scripts/quicktest-local.sh"
    quicktest.write_text('#!/bin/bash\necho "$1|$XILINX_VIVADO|$XILINX_VITIS|$PATH"\n')
    quicktest.chmod(0o755)
    step = tmp_path / "step"
    step.write_text(local_setup_step())
    proc = subprocess.run(
        ["/bin/bash", str(step)],
        cwd=workspace,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "FINN_XILINX_PATH": str(root),
            "FINN_XILINX_VERSION": "2024.2",
        },
    )
    if missing:
        # The stage fails before its tests, naming the release it did not find.
        assert proc.returncode == 1 and proc.stdout == ""
        assert "no settings64.sh for FINN_XILINX_VERSION=2024.2" in proc.stderr
        return
    assert proc.returncode == 0, proc.stderr
    mode, vivado, vitis, path = proc.stdout.strip().split("|")
    assert mode == "vivado"
    assert vivado == str(trees["Vivado"]) and vitis == str(trees["Vitis"])
    # The venv's Python ahead of the release's tools.
    assert path.split(":")[:3] == [
        f"{workspace}/.venv/bin",
        f"{trees['Vitis']}/bin",
        f"{trees['Vivado']}/bin",
    ]
