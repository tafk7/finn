# Copyright (c) 2021 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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


"""What FINN knows of HLS: its frontends and installation, the command that runs a
script, the pynq shell's ``CallHLS``, and the kernel system's ``synthesize``.

A selection names its frontend (``FRONTENDS``; the machine's is ``host_frontend``'s),
and ``command`` refuses one its probed release does not match. ``installation`` is
where the HLS C++ headers and C simulation libraries are.

``CallHLS`` runs an IODMA's generated Tcl script where the pynq shell's IP generation
wrote it (``finn.shells.pynq.iodma``), replayed by ``ipgen.sh``, its transcript
``Toolchain.run``'s (in a build, the step's log directory; else beside ``ipgen.sh``).

Both read what HLS prints as Vivado's tools print it (``SEVERITY: [HLS N-M] text``):
``finn.toolchain.vivado.messages`` is their parser too.

``synthesize`` builds a staged HLS request (``finn.kernels.artifacts.hls``: a
directory holding ``script.tcl``, the top and its headers, by relative paths) for
one part, through the toolchain's HLS frontend, and caches the product under
``$FINN_HOME/hls`` (``hls_cache``) with the resource store's discipline
(``finn.resources``): built in a temporary directory beside the entry, marked with
its full key, renamed into place under the entry's lock, so a failed or concurrent
build never publishes a partial product and a second caller waits for the first.
The key is the request's content, the part and the HLS installation
(``finn.toolchain.launcher.identity`` of the frontend): a hit never runs the tool. A
product (``HlsProduct``) holds the exported Verilog with its memory images
(``verilog/``), the synthesis reports (``report/``), the request it was built from
(``request/``) and the run's replay script with its transcript (``synthesize.sh``,
``.stdout.log``, ``.stderr.log``, written as the tool runs; in a build too, since the
product is every build's). A failed build is kept beside the entry
(``<entry>.failed``) and named by the error, a ``ToolFailed`` whose transcript is the
run's output.

It takes a directory, never a kernel value: ``finn.toolchain`` imports no kernel layer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from finn.resources import home, tree_digest
from finn.resources._store import complete, locked, publish

# Modules, not names: the launcher reads this module in turn (``FRONTENDS``).
from finn.toolchain import host, launcher

if TYPE_CHECKING:
    from finn.toolchain.launcher import Toolchain

#: The HLS frontends a selection may name.
FRONTENDS = ("vivado_hls", "vitis_hls", "vitis-run")
_LOG = logging.getLogger(__name__)
# The vitis-run routes that advertised HLS, by probe key: asked once a process.
_CAPABLE: set[tuple[Hashable, ...]] = set()


def installation(toolchain: Toolchain) -> Path:
    """The local installation whose HLS C++ headers (``include/``) and C simulation
    libraries (``lnx64/``) code compiled by ``g++`` uses: ``toolchain``'s ``XILINX_HLS``,
    else ``XILINX_VITIS`` (whose installation carries HLS from 2025.1).
    ``ToolUnavailable`` when it names neither."""
    for variable in ("XILINX_HLS", "XILINX_VITIS"):
        if toolchain.environment.get(variable):
            return Path(toolchain.environment[variable])
    raise launcher.ToolUnavailable(
        "HLS",
        "this toolchain names no HLS installation (XILINX_HLS or XILINX_VITIS) "
        "for HLS C++ headers and C simulation libraries",
    )


def selected_frontend(toolchain: Toolchain) -> str:
    """The HLS frontend ``toolchain``'s selection names; ``ValueError`` when it names
    none."""
    if toolchain.selection.hls_frontend is None:
        raise ValueError(
            "This selection names no HLS frontend; state one, or lay it over the "
            "machine's (host_selection)"
        )
    return toolchain.selection.hls_frontend


def command(
    toolchain: Toolchain, script: str, timeout: float | None = None
) -> tuple[str, list[str]]:
    """The selected HLS frontend and its arguments to run ``script``, its probes within
    ``timeout`` (``finn.toolchain.launcher.PROBE_TIMEOUT`` by default); refused
    (``ValueError``) when the probed release does not match the frontend."""
    timeout = launcher.PROBE_TIMEOUT if timeout is None else timeout
    frontend = selected_frontend(toolchain)
    version = toolchain.probe(frontend, timeout)
    # These are FINN code-generation constraints, separate from whether a
    # command exists. Preserve the standalone/unified transition deliberately.
    supported = {
        "vivado_hls": version <= (2020, 1),
        "vitis_hls": (2020, 1) <= version <= (2024, 2),
        "vitis-run": version >= (2025, 1),
    }[frontend]
    if not supported:
        raise ValueError(
            f"FINN HLS frontend {frontend} is incompatible with {version}; "
            "select the matching frontend explicitly"
        )
    capability = toolchain.probe_key("hls-capability", frontend)
    if frontend == "vitis-run" and capability not in _CAPABLE:
        # A capability probe keeps no transcript, as a version probe keeps none.
        help_result = toolchain.transcripts_in(None).run(frontend, ["--help"], timeout=timeout)
        help_text = (help_result.stdout + help_result.stderr).decode("utf-8", errors="replace")
        if "hls" not in help_text.lower():
            raise RuntimeError("Selected vitis-run does not advertise HLS capability")
        _LOG.info("capability tool=vitis-run hls=advertised")
        _CAPABLE.add(capability)
    args = ["--mode", "hls", "--tcl", script] if frontend == "vitis-run" else ["-f", script]
    return frontend, args


def host_frontend(environ: Mapping[str, str] | None = None) -> str:
    """The HLS frontend of the release the host file selects
    (``FINN_XILINX_VERSION``, which the environment may override): ``vitis-run``
    from 2025.1, else ``vitis_hls``, which is also the frontend when it names no
    release. ``ValueError`` when the release is not ``YEAR.MINOR``."""
    version = host.settings(environ).get("FINN_XILINX_VERSION")
    if version is None:
        return "vitis_hls"
    match = re.fullmatch(r"(20\d{2})\.(\d+)", version)
    if match is None:
        raise ValueError(f"FINN_XILINX_VERSION={version} is not a release (YEAR.MINOR)")
    return "vitis-run" if tuple(map(int, match.groups())) >= (2025, 1) else "vitis_hls"


#: The staged request's script, which reads ``$part`` (``finn.kernels.artifacts.hls``).
REQUEST_SCRIPT = "script.tcl"
#: How long one synthesis may run, in seconds.
SYNTHESIS_TIMEOUT = 3 * 3600


def hls_cache() -> Path:
    """Where HLS products are cached: ``$FINN_HOME/hls``."""
    return home() / "hls"


def synthesis_key(request: Path, part: str, identity: str) -> str:
    """The key of a request's product: its staged content, the part and the HLS tool."""
    preimage = "\0".join(("hls-synthesis-v1", tree_digest(request), part, identity))
    return hashlib.sha256(preimage.encode()).hexdigest()


@dataclass(frozen=True)
class HlsProduct:
    """A cached synthesis: ``directory``, its entry, and where its parts are."""

    directory: Path

    @property
    def verilog(self) -> Path:
        """The exported Verilog and the images its memories read."""
        return self.directory / "verilog"

    @property
    def report(self) -> Path:
        """The synthesis reports (``csynth.xml``, ``<top>_csynth.xml``, ...)."""
        return self.directory / "report"


def _exported(work: Path) -> Path | None:
    """The one solution's ``syn`` directory below the run's directory; None unless
    there is exactly one."""
    found = sorted(work.glob("*/*/syn/verilog"))
    return found[0].parent if len(found) == 1 else None


def synthesize(
    request: Path,
    part: str,
    *,
    name: str,
    toolchain: Toolchain | None = None,
    cache: Path | None = None,
) -> HlsProduct:
    """The product of the staged ``request`` for ``part``, synthesized unless cached.

    ``name`` names the entry, ``<name>-<key16>`` under ``cache`` (``hls_cache()`` by
    default); ``toolchain`` runs the frontend (the machine's by default). Raises
    ``ToolFailed`` when the tool fails or leaves no exported RTL, ``ToolUnavailable``
    when the toolchain has no HLS installation or frontend to run."""
    toolchain = toolchain or launcher.host_toolchain()
    cache = hls_cache() if cache is None else cache
    if not (Path(request) / REQUEST_SCRIPT).is_file():
        raise ValueError(f"{request} holds no {REQUEST_SCRIPT}: it is not a staged request")
    key = synthesis_key(
        Path(request), part, launcher.identity(toolchain, selected_frontend(toolchain))
    )
    entry_name = f"{name}-{key[:16]}"
    with locked(cache, entry_name) as entry:
        if complete(entry, key):
            return HlsProduct(entry)
        work = Path(tempfile.mkdtemp(dir=cache, prefix=f".{entry_name}."))
        failed = cache / f"{entry_name}.failed"
        try:
            _build(Path(request), part, work, toolchain, kept=failed)
        except BaseException:
            shutil.rmtree(failed, ignore_errors=True)
            work.rename(failed)
            raise
        publish(work, entry, key)
    return HlsProduct(entry)


def _build(request: Path, part: str, work: Path, toolchain: Toolchain, *, kept: Path) -> None:
    """Run the request in ``work/request`` and collect its product into ``work``; a
    failure names the build where it is ``kept`` (``work``, once moved there)."""
    run = work / "request"
    shutil.copytree(request, run)
    driver = run / "synthesize.tcl"
    driver.write_text(f"set part {{{part}}}\nsource {REQUEST_SCRIPT}\n")
    frontend, args = command(toolchain, driver.name)
    replay = work / "synthesize.sh"
    # The transcript is the product's, beside its replay, in a build too: the product
    # is shared by every build, and a cached one runs nothing. A failure names it where
    # the kept build holds it.
    transcript = launcher.transcript_files(kept / replay.name)[0]
    try:
        result = toolchain.run(
            frontend,
            args,
            cwd=run,
            timeout=SYNTHESIS_TIMEOUT,
            check=False,
            replay=replay,
            transcript=replay,
        )
    except (subprocess.SubprocessError, InterruptedError, OSError) as error:
        raise launcher.ToolFailed.of(
            frontend,
            getattr(error, "output", None) or b"",
            transcript=transcript,
            what=f"it did not finish ({type(error).__name__})",
        ) from error
    if result.returncode != 0:
        raise launcher.ToolFailed.of(
            frontend, result.stdout + result.stderr, status=result.returncode, transcript=transcript
        )
    syn = _exported(run)
    if syn is None:
        raise launcher.ToolFailed.of(
            frontend,
            result.stdout + result.stderr,
            transcript=transcript,
            what="the run left no one exported RTL directory",
        )
    # The replay runs where the product is published, not in the temporary directory.
    replay.write_text(
        re.sub(r"^cd -- .*$", 'cd -- "$(dirname -- "$0")/request"', replay.read_text(), flags=re.M)
    )
    shutil.copytree(syn / "verilog", work / "verilog")
    shutil.copytree(syn / "report", work / "report")
    # The request alone stays: the project's intermediate files are the tool's.
    for item in run.iterdir():
        if item.is_dir() and any(item.glob("*/syn")):
            shutil.rmtree(item)


class CallHLS:
    """Execute a deliberately selected HLS frontend with child-scoped settings:
    ``toolchain``'s, by default the machine's (``host_toolchain``)."""

    def __init__(self, toolchain: Toolchain | None = None):
        self.toolchain = toolchain
        self.tcl_script = ""
        self.ipgen_path = ""
        self.code_gen_dir = ""
        self.ipgen_script = ""

    def append_tcl(self, tcl_script):
        self.tcl_script = tcl_script

    def set_ipgen_path(self, path):
        self.ipgen_path = path

    def build(self, code_gen_dir):
        toolchain = self.toolchain or launcher.host_toolchain()
        self.code_gen_dir = os.path.abspath(code_gen_dir)
        frontend, args = command(toolchain, self.tcl_script)
        self.ipgen_script = str(Path(self.code_gen_dir) / "ipgen.sh")
        # The replay requires the selected tool's environment, but holds neither
        # secrets nor an environment snapshot. What the tool prints is its transcript
        # (Toolchain.run), not this process's output.
        result = toolchain.run(
            frontend, args, cwd=self.code_gen_dir, check=False, replay=self.ipgen_script
        )
        if result.returncode != 0:
            raise launcher.ToolFailed.of_run(frontend, result, what=f"{self.tcl_script}")
