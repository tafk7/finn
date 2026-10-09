# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Support code the kernel path, graph preparation and the tests share: build
directories (``make_build_dir``, ``robust_rmtree``), the Vivado release the environment
selects, and the Resize input a streamlining transform reads."""

import errno
import os
import re
import shutil
import tempfile
import time
from typing import Optional, Tuple

from finn import resources


def get_vivado_version() -> Optional[Tuple[int, int]]:
    """Extract Vivado version as (year, minor) tuple from XILINX_VIVADO."""
    path = os.environ.get("XILINX_VIVADO", "")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", path)
    return (int(match.group(1)), int(match.group(2))) if match else None


def make_build_dir(prefix=""):
    """Creates a folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    build_dir = resources.scratch()
    os.makedirs(build_dir, exist_ok=True)
    new_dir = tempfile.mkdtemp(prefix=prefix, dir=build_dir)
    os.chmod(new_dir, 0o755)
    return new_dir


def robust_rmtree(path, retries=6, initial_delay=0.1, backoff=2.0):
    """Remove a directory tree with retries for transient NFS cleanup races.
    Retries ``ENOTEMPTY``/``EBUSY``. Other errors propagate immediately.
    """
    if not path or not os.path.exists(path):
        return
    delay = initial_delay
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            transient = exc.errno in (errno.ENOTEMPTY, errno.EBUSY)
            if not transient or attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff


def resolve_resize_param_input(model, node):
    """Identify which input of an ONNX Resize node carries the resampling
    parameter, across the different opset input signatures, and whether that
    parameter is a target output size (``sizes``) rather than ``scales``.

    Returns a ``(param_index, is_sizes)`` tuple where ``param_index`` is the
    index into ``node.input`` and ``is_sizes`` is True if the parameter is a
    ``sizes`` input. Handles:

    * ``(X, scales)`` (Resize-10)                  -> ``(1, False)``
    * ``(X, roi, scales)`` (Resize-11+, no sizes)  -> ``(2, False)``
    * ``(X, roi, scales, sizes)`` (Resize-11+)     -> ``(2, False)`` or ``(3, True)``
    """
    num_inputs = len(node.input)
    if num_inputs == 2:
        # Resize-10: (X, scales)
        return 1, False
    elif num_inputs == 3:
        # Resize-11+: (X, roi, scales), no sizes input
        return 2, False
    elif num_inputs == 4:
        # Resize-11+: (X, roi, scales, sizes); exactly one of scales/sizes is set
        scales_init = model.get_initializer(node.input[2])
        sizes_init = model.get_initializer(node.input[3])
        scales_exists = scales_init is not None and len(scales_init) != 0
        sizes_exists = sizes_init is not None and len(sizes_init) != 0
        assert scales_exists ^ sizes_exists, (
            "%s: Either scales or the target output size must be specified. "
            "Specifying both is prohibited." % node.name
        )
        return (2, False) if scales_exists else (3, True)
    else:
        raise ValueError("%s: Unsupported number of Resize inputs (%d)." % (node.name, num_inputs))
