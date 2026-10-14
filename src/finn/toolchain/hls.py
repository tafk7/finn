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
script, and ``synthesize``, which every HLS synthesis goes through: the kernel system's
HLS leaves and the pynq shell's IODMAs.

A selection names its frontend (``FRONTENDS``; the machine's comes from
``host_frontend``), and ``command`` refuses one that does not match its probed release.
``installation`` is where the HLS C++ headers and C simulation libraries are.

HLS prints messages in the form Vivado's tools use (``SEVERITY: [HLS N-M] text``), so
both read them with ``finn.toolchain.vivado.messages``.

``synthesize`` builds a staged HLS request (a directory holding ``script.tcl`` and the
files it reads, by relative paths: ``finn.kernels.artifacts.hls`` stages a leaf's, the
pynq shell's IODMA writes its own) for one part with the toolchain's HLS frontend, and
caches the product under ``$FINN_HOME/hls`` (``hls_cache``) the way the resource store
does (``finn.resources``). The script reads the part from the Tcl variable ``part``, and
each library it reads in place (finn-hlslib, for an IODMA) from a variable of the
library's name. The product is built in a temporary directory beside the entry, marked
with its full key and renamed into place under the entry's lock, so a failed or
concurrent build never publishes a partial product, and a second caller waits for the
first. The key is the request's content, the part, each library's content (never its
location) and the HLS installation (``finn.toolchain.launcher.identity`` of the
frontend); a cache hit runs nothing. A product (``HlsProduct``) holds the exported
Verilog with its memory images (``verilog/``), the synthesis reports (``report/``), the
IP a script exports into the IP catalog (``ip/``, ``export_design -format ip_catalog``),
the request it was built from (``request/``), and the run's replay script with its
transcript (``synthesize.sh``, ``.stdout.log``, ``.stderr.log``). The transcript stays
with the product even in a build, since every build shares the product. A failed build
is kept beside the entry (``<entry>.failed``), and the ``ToolFailed`` it raises names
its transcript.

It takes a directory, never a kernel value: ``finn.toolchain`` imports no kernel layer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.reporting.manifest import stored
from finn.resources import home, tree_digest
from finn.resources._store import complete, locked, publish

# Modules, not names: the launcher imports this module in turn (``FRONTENDS``).
from finn.toolchain import launcher

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
    """The selected HLS frontend and its arguments to run ``script``, probing it within
    ``timeout`` (``finn.toolchain.launcher.PROBE_TIMEOUT`` by default). Refuses a
    frontend the probed release does not match (``hls-frontend-mismatch``, owned by the
    selection's ``hls_frontend``)."""
    timeout = launcher.PROBE_TIMEOUT if timeout is None else timeout
    frontend = selected_frontend(toolchain)
    version = toolchain.probe(frontend, timeout)
    # The releases FINN's generated scripts support with each frontend, whatever
    # commands exist: the standalone HLS tools up to 2024.2, the unified vitis-run after.
    supported = {
        "vivado_hls": version <= (2020, 1),
        "vitis_hls": (2020, 1) <= version <= (2024, 2),
        "vitis-run": version >= (2025, 1),
    }[frontend]
    if not supported:
        message = (
            f"FINN HLS frontend {frontend} is incompatible with {version}; "
            "select the matching frontend explicitly"
        )
        mismatch = Finding(
            FindingKind.REJECTION, "hls-frontend-mismatch", "toolchain.hls_frontend", message
        )
        raise Refusal((mismatch,))
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
    release. A release that is not ``YEAR.MINOR`` is refused (``host-setting-invalid``,
    owned by the setting)."""
    version = launcher.host_settings(environ).get("FINN_XILINX_VERSION")
    if version is None:
        return "vitis_hls"
    match = re.fullmatch(r"(20\d{2})\.(\d+)", version)
    if match is None:
        raise launcher.setting_refused(
            "FINN_XILINX_VERSION", f"{version} is not a release (YEAR.MINOR)"
        )
    return "vitis-run" if tuple(map(int, match.groups())) >= (2025, 1) else "vitis_hls"


#: The staged request's script, which reads ``$part`` (``finn.kernels.artifacts.hls``).
REQUEST_SCRIPT = "script.tcl"
#: How long one synthesis may run, in seconds.
SYNTHESIS_TIMEOUT = 3 * 3600


def hls_cache() -> Path:
    """Where HLS products are cached: ``$FINN_HOME/hls``."""
    return home() / "hls"


def synthesis_key(
    request: Path, part: str, identity: str, libraries: Mapping[str, Path] | None = None
) -> str:
    """The key of a request's product: its staged content, the part, the HLS tool and
    each library's content, by its name."""
    read = [f"{name}={tree_digest(path)}" for name, path in sorted((libraries or {}).items())]
    preimage = "\0".join(("hls-synthesis-v1", tree_digest(request), part, identity, *read))
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

    @property
    def ip(self) -> Path:
        """The IP the script exported into the IP catalog; none unless it exported one."""
        return self.directory / "ip"


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
    toolchain: Toolchain,
    cache: Path | None = None,
    libraries: Mapping[str, Path] | None = None,
) -> HlsProduct:
    """The product of the staged ``request`` for ``part``, synthesized unless cached.

    ``name`` names the entry, ``<name>-<key16>`` under ``cache`` (``hls_cache()`` by
    default); ``toolchain`` runs the frontend; ``libraries`` are the directories the
    script reads in place, each by a Tcl variable of its name. The running step's
    manifest lists the entry, built or reused (``finn.core.reporting.manifest.stored``).
    Raises ``ToolFailed`` when the tool fails or leaves no exported RTL,
    ``ToolUnavailable`` when the toolchain has no HLS installation or frontend to run."""
    cache = hls_cache() if cache is None else cache
    libraries = dict(libraries or {})
    if not (Path(request) / REQUEST_SCRIPT).is_file():
        raise ValueError(f"{request} holds no {REQUEST_SCRIPT}: it is not a staged request")
    key = synthesis_key(
        Path(request), part, launcher.identity(toolchain, selected_frontend(toolchain)), libraries
    )
    entry_name = f"{name}-{key[:16]}"
    with locked(cache, entry_name) as entry:
        if complete(entry, key):
            stored("hls", entry, key, built=False)
            return HlsProduct(entry)
        work = Path(tempfile.mkdtemp(dir=cache, prefix=f".{entry_name}."))
        failed = cache / f"{entry_name}.failed"
        try:
            _build(Path(request), part, libraries, work, toolchain, kept=failed)
        except BaseException:
            shutil.rmtree(failed, ignore_errors=True)
            work.rename(failed)
            raise
        publish(work, entry, key)
    stored("hls", entry, key, built=True)
    return HlsProduct(entry)


def _build(
    request: Path,
    part: str,
    libraries: Mapping[str, Path],
    work: Path,
    toolchain: Toolchain,
    *,
    kept: Path,
) -> None:
    """Run the request in ``work/request`` and collect its product into ``work``; a
    failure names the build where it is ``kept`` (``work``, once moved there)."""
    run = work / "request"
    shutil.copytree(request, run)
    driver = run / "synthesize.tcl"
    variables = {"part": part, **{name: str(path) for name, path in libraries.items()}}
    driver.write_text(
        "".join(f"set {name} {{{value}}}\n" for name, value in variables.items())
        + f"source {REQUEST_SCRIPT}\n"
    )
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
    if (syn.parent / "impl" / "ip").is_dir():
        shutil.copytree(syn.parent / "impl" / "ip", work / "ip")
    # The request alone stays: the project's intermediate files are the tool's.
    for item in run.iterdir():
        if item.is_dir() and any(item.glob("*/syn")):
            shutil.rmtree(item)
