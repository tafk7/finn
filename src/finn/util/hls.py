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


"""Running HLS: the pynq shell's ``CallHLS``, and the kernel system's ``synthesize``.

``CallHLS`` runs an IODMA's generated Tcl script where the pynq shell's IP generation
wrote it (``finn.shells.pynq.iodma``).

``synthesize`` builds a staged HLS request (``finn.kernels.artifacts.hls``: a
directory holding ``script.tcl``, the top and its headers, by relative paths) for
one part, through the toolchain's HLS frontend, and caches the product under
``$FINN_HOME/hls`` (``hls_cache``) with the resource store's discipline
(``finn.resources``): built in a temporary directory beside the entry, marked with
its full key, renamed into place under the entry's lock, so a failed or concurrent
build never publishes a partial product and a second caller waits for the first.
The key is the request's content, the part and the HLS installation
(``hls_identity``): a hit never runs the tool. A product (``HlsProduct``) holds the
exported Verilog with its memory images (``verilog/``), the synthesis reports
(``report/``), the request it was built from (``request/``) and the run's replay
script with its output (``synthesize.sh``, ``.stdout.log``, ``.stderr.log``). A
failed build is kept beside the entry (``<entry>.failed``) and named by the error.

It takes a directory, never a kernel value: ``finn.util`` imports no kernel layer.
"""

import hashlib
import os
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from finn.resources import home, tree_digest
from finn.resources._store import complete, locked, publish
from finn.util.toolchain import Toolchain, machine_toolchain

#: The staged request's script, which reads ``$part`` (``finn.kernels.artifacts.hls``).
REQUEST_SCRIPT = "script.tcl"
#: How long one synthesis may run, in seconds.
SYNTHESIS_TIMEOUT = 3 * 3600


class HlsFailed(RuntimeError):
    """A synthesis exited with an error, or left no exported RTL; the message names the
    kept build and ends with what the tool printed last."""


def hls_cache() -> Path:
    """Where HLS products are cached: ``$FINN_HOME/hls``."""
    return home() / "hls"


def hls_identity(toolchain: Toolchain) -> str:
    """The toolchain's HLS installation by what it is, never where it is installed: its
    release (the ``XILINX_VERSION_DEFAULT`` its ``data/version.sh`` states, ``unstated``
    without one) and its version file's digest, beside the frontend that runs it."""
    data = toolchain.hls_installation() / "data"
    release = "unstated"
    if (data / "version.sh").is_file():
        stated = re.search(
            r"^XILINX_VERSION_DEFAULT=(\S+)$", (data / "version.sh").read_text(), re.MULTILINE
        )
        release = stated.group(1) if stated else release
    version = data / "version.dat"
    stamp = hashlib.sha256(version.read_bytes()).hexdigest()[:16] if version.is_file() else "none"
    return f"{toolchain.selection.hls_frontend} {release} {stamp}"


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


def _exported(work: Path) -> Path:
    """The one solution's ``syn`` directory below the run's directory."""
    found = sorted(work.glob("*/*/syn/verilog"))
    if len(found) != 1:
        raise HlsFailed(f"{work}: the run left {len(found)} exported RTL directories, not one")
    return found[0].parent


def _tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


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
    ``HlsFailed`` when the tool fails or leaves no exported RTL."""
    toolchain = toolchain or machine_toolchain()
    cache = hls_cache() if cache is None else cache
    if not (Path(request) / REQUEST_SCRIPT).is_file():
        raise HlsFailed(f"{request} holds no {REQUEST_SCRIPT}: it is not a staged request")
    key = synthesis_key(Path(request), part, hls_identity(toolchain))
    entry_name = f"{name}-{key[:16]}"
    with locked(cache, entry_name) as entry:
        if complete(entry, key):
            return HlsProduct(entry)
        work = Path(tempfile.mkdtemp(dir=cache, prefix=f".{entry_name}."))
        try:
            _build(Path(request), part, work, toolchain)
        except BaseException:
            failed = cache / f"{entry_name}.failed"
            shutil.rmtree(failed, ignore_errors=True)
            work.rename(failed)
            raise
        publish(work, entry, key)
    return HlsProduct(entry)


def _build(request: Path, part: str, work: Path, toolchain: Toolchain) -> None:
    """Run the request in ``work/request`` and collect its product into ``work``."""
    run = work / "request"
    shutil.copytree(request, run)
    driver = run / "synthesize.tcl"
    driver.write_text(f"set part {{{part}}}\nsource {REQUEST_SCRIPT}\n")
    frontend, args = toolchain.hls_command(driver.name)
    replay = work / "synthesize.sh"
    try:
        result = toolchain.run(
            frontend, args, cwd=run, timeout=SYNTHESIS_TIMEOUT, check=False, replay=replay
        )
    except Exception as error:
        raise HlsFailed(f"HLS did not finish; the build is kept at {work}") from error
    if result.returncode != 0:
        raise HlsFailed(
            f"HLS exited with {result.returncode}; the build is kept at {work}:\n"
            + _tail(Path(f"{replay}.stdout.log"))
        )
    try:
        syn = _exported(run)
    except HlsFailed as error:
        raise HlsFailed(
            f"{error}; the build is kept at {work}:\n" + _tail(Path(f"{replay}.stdout.log"))
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
    ``toolchain``'s, by default the machine's (``machine_toolchain``)."""

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
        toolchain = self.toolchain or machine_toolchain()
        self.code_gen_dir = os.path.abspath(code_gen_dir)
        frontend, args = toolchain.hls_command(self.tcl_script)
        self.ipgen_script = str(Path(self.code_gen_dir) / "ipgen.sh")
        # Retain the useful replay artifact. It requires the selected tool's
        # environment, but contains neither secrets nor an environment snapshot.
        Path(self.ipgen_script).write_text(
            "#!/bin/bash\nset -e\ncd "
            + shlex.quote(self.code_gen_dir)
            + "\nexec "
            + shlex.join(toolchain.command(frontend, *args))
            + "\n"
        )
        result = toolchain.run(frontend, args, cwd=self.code_gen_dir, check=False)
        sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        result.check_returncode()
