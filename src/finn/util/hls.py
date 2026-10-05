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


import os
import shlex
import sys
from pathlib import Path

from finn.util._legacy_build_env import toolchain as legacy_toolchain
from finn.util.toolchain import Toolchain


class CallHLS:
    """Execute a deliberately selected HLS frontend with child-scoped settings:
    ``toolchain``'s, by default the legacy environment's."""

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
        toolchain = self.toolchain or legacy_toolchain()
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
