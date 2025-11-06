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

import logging
import os
import subprocess
import sys
import tempfile
import threading
from qonnx.util.basic import roundup_to_integer_multiple
from typing import Optional, Tuple, Union

# test boards
test_board_map = ["Pynq-Z1", "KV260_SOM", "ZCU104", "U250"]

# mapping from PYNQ board names to FPGA part names
pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Ultra96-V2"] = "xczu3eg-sbva484-1-i"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU102"] = "xczu9eg-ffvb1156-2-e"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"
pynq_part_map["ZCU111"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC2x2"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC4x2"] = "xczu48dr-ffvg1517-2-e"
pynq_part_map["KV260_SOM"] = "xck26-sfvc784-2LV-c"


# native AXI HP port width (in bits) for PYNQ boards
pynq_native_port_width = dict()
pynq_native_port_width["Pynq-Z1"] = 64
pynq_native_port_width["Pynq-Z2"] = 64
pynq_native_port_width["Ultra96"] = 128
pynq_native_port_width["Ultra96-V2"] = 128
pynq_native_port_width["ZCU102"] = 128
pynq_native_port_width["ZCU104"] = 128
pynq_native_port_width["ZCU111"] = 128
pynq_native_port_width["RFSoC2x2"] = 128
pynq_native_port_width["RFSoC4x2"] = 128
pynq_native_port_width["KV260_SOM"] = 128

# Alveo device and platform mappings
alveo_part_map = dict()
alveo_part_map["U50"] = "xcu50-fsvh2104-2L-e"
alveo_part_map["U200"] = "xcu200-fsgd2104-2-e"
alveo_part_map["U250"] = "xcu250-figd2104-2L-e"
alveo_part_map["U280"] = "xcu280-fsvh2892-2L-e"
alveo_part_map["U55C"] = "xcu55c-fsvh2892-2L-e"

alveo_default_platform = dict()
alveo_default_platform["U50"] = "xilinx_u50_gen3x16_xdma_5_202210_1"
alveo_default_platform["U200"] = "xilinx_u200_gen3x16_xdma_2_202110_1"
alveo_default_platform["U250"] = "xilinx_u250_gen3x16_xdma_2_1_202010_1"
alveo_default_platform["U280"] = "xilinx_u280_gen3x16_xdma_1_202211_1"
alveo_default_platform["U55C"] = "xilinx_u55c_gen3x16_xdma_3_202210_1"

# Create a joint part map, encompassing other boards too
part_map = {**pynq_part_map, **alveo_part_map}
part_map["VEK280"] = "xcve2802-vsvh1760-2MP-e-S"
part_map["VCK190"] = "xcvc1902-vsva2197-2MP-e-S"
part_map["V80"] = "xcv80-lsva4737-2MHP-e-s"


def get_rtlsim_trace_depth():
    """Return the trace depth for rtlsim. Controllable
    via the RTLSIM_TRACE_DEPTH environment variable. If the env.var. is
    undefined, the default value of 1 is returned. A trace depth of 1
    will only show top-level signals and yield smaller .vcd files.

    The following depth values are of interest for whole-network stitched IP
    rtlsim:
    - level 1 shows top-level input/output streams
    - level 2 shows per-layer input/output streams
    - level 3 shows per full-layer I/O including FIFO count signals
    """

    try:
        return int(os.environ["RTLSIM_TRACE_DEPTH"])
    except KeyError:
        return 1


def get_finn_root():
    "Return the root directory that FINN is cloned into."

    try:
        return os.environ["FINN_ROOT"]
    except KeyError:
        raise Exception(
            """Environment variable FINN_ROOT must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_vivado_root():
    "Return the root directory that Vivado is installed into."

    try:
        return os.environ["XILINX_VIVADO"]
    except KeyError:
        raise Exception(
            """Environment variable XILINX_VIVADO must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_liveness_threshold_cycles():
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""

    return int(os.getenv("LIVENESS_THRESHOLD", 1000000))


def make_build_dir(prefix=""):
    """Creates a folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    try:
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        newdir = tmpdir.replace("/tmp", os.environ["FINN_BUILD_DIR"])
        os.makedirs(newdir)
        return newdir
    except KeyError:
        raise Exception(
            """Environment variable FINN_BUILD_DIR must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


class CppBuilder:
    """Builds the g++ compiler command to produces the executable of the c++ code
    in code_gen_dir which is passed to the function build() of this class."""

    def __init__(self):
        self.include_paths = []
        self.cpp_files = []
        self.executable_path = ""
        self.code_gen_dir = ""
        self.compile_components = []
        self.compile_script = ""

    def append_includes(self, library_path):
        """Adds given library path to include_paths list."""
        self.include_paths.append(library_path)

    def append_sources(self, cpp_file):
        """Adds given c++ file to cpp_files list."""
        self.cpp_files.append(cpp_file)

    def set_executable_path(self, path):
        """Sets member variable "executable_path" to given path."""
        self.executable_path = path

    def build(self, code_gen_dir):
        """Builds the g++ compiler command according to entries in include_paths
        and cpp_files lists. Saves it in bash script in given folder and
        executes it."""
        # raise error if includes are empty
        self.code_gen_dir = code_gen_dir
        self.compile_components.append("g++ -o " + str(self.executable_path))
        for cpp_file in self.cpp_files:
            self.compile_components.append(cpp_file)
        for lib in self.include_paths:
            self.compile_components.append(lib)
        bash_compile = ""
        for component in self.compile_components:
            bash_compile += str(component) + " "
        self.compile_script = str(self.code_gen_dir) + "/compile.sh"
        with open(self.compile_script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write(bash_compile + "\n")
        bash_command = ["bash", self.compile_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()


def _detect_log_level(line: str, default: int) -> int:
    """
    Parse tool output to assign appropriate log level.

    Detects patterns from Xilinx tools (xelab, Vivado) and standard output.

    Args:
        line: Output line to analyze
        default: Default level if no pattern matches

    Returns:
        Appropriate logging level constant
    """
    line_upper = line.upper()

    # Error patterns (highest priority)
    if any(x in line_upper for x in ["ERROR:", "FATAL:", "FAILED", "EXCEPTION"]):
        return logging.ERROR

    # Warning patterns
    if any(x in line_upper for x in ["WARNING:", "WARN:"]):
        return logging.WARNING

    # Verbose/debug patterns (tool spam)
    if any(
        x in line_upper
        for x in [
            "COMPILING MODULE",
            "COMPILING ARCHITECTURE",
            "ANALYZING ENTITY",
            "ELABORATING ENTITY",
        ]
    ):
        return logging.DEBUG

    # Info patterns
    if any(x in line_upper for x in ["INFO:", "NOTE:"]):
        return logging.INFO

    return default


def launch_process_helper(
    args,
    proc_env=None,
    cwd=None,
    logger: Optional[logging.Logger] = None,
    use_logging: bool = False,
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.WARNING,
    detect_levels: bool = True,
    raise_on_error: bool = False,
) -> Union[Tuple[str, str], int]:
    """
    Launch subprocess with configurable output handling.

    This function provides two modes of operation:

    **Legacy Mode (default, use_logging=False)**:
    - Buffers all output via communicate()
    - Writes to sys.stdout/sys.stderr directly
    - Returns (stdout_str, stderr_str) tuple
    - No exit code checking
    - Backward compatible with all existing code

    **Logging Mode (use_logging=True)**:
    - Streams output line-by-line through Python logging
    - Enables application-level log filtering and formatting
    - Returns exit code integer
    - Optional automatic error raising on non-zero exit
    - Thread-safe streaming of stdout and stderr

    Args:
        args: Command and arguments as list (e.g., ["xelab", "-prj", "sim.prj"])
        proc_env: Environment variables dict (default: os.environ.copy())
        cwd: Working directory for subprocess (default: current directory)
        logger: Logger instance for logging mode (default: 'finn.subprocess')
        use_logging: Enable logging mode (default: False for compatibility)
        stdout_level: Base log level for stdout (default: INFO)
        stderr_level: Base log level for stderr (default: WARNING)
        detect_levels: Auto-detect log levels from output patterns (default: True)
        raise_on_error: Raise CalledProcessError on non-zero exit (default: False)

    Returns:
        - If use_logging=False: (stdout: str, stderr: str) tuple
        - If use_logging=True: exit_code: int

    Raises:
        subprocess.CalledProcessError: If use_logging=True, raise_on_error=True,
                                       and subprocess exits with non-zero code

    Examples:
        Legacy mode (backward compatible)::

            out, err = launch_process_helper(["echo", "hello"])
            # Writes "hello" to sys.stdout, returns tuple

        Logging mode (new)::

            import logging
            logging.basicConfig(level=logging.INFO)

            exitcode = launch_process_helper(
                ["xelab", "work.top", "-prj", "sim.prj"],
                use_logging=True,
                logger=logging.getLogger('finn.xsi'),
                stdout_level=logging.DEBUG,
                stderr_level=logging.WARNING,
                raise_on_error=True
            )

    Notes:
        - Return value change when use_logging=True is safe: all existing FINN
          code discards return values (verified by codebase analysis)
        - Logging mode uses threads for non-blocking stdout/stderr reading
        - Pattern detection (detect_levels=True) automatically adjusts levels
          based on tool output (ERROR:, WARNING:, etc.)
        - For long-running processes, logging mode is more memory-efficient
          than legacy mode (streams vs. buffers)
    """
    if proc_env is None:
        proc_env = os.environ.copy()

    # ============================================================
    # LEGACY MODE: Preserve existing behavior exactly
    # ============================================================
    if not use_logging:
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env, cwd=cwd
        ) as proc:
            (cmd_out, cmd_err) = proc.communicate()

        if cmd_out is not None:
            cmd_out = cmd_out.decode("utf-8")
            sys.stdout.write(cmd_out)
        if cmd_err is not None:
            cmd_err = cmd_err.decode("utf-8")
            sys.stderr.write(cmd_err)

        return (cmd_out, cmd_err)

    # ============================================================
    # LOGGING MODE: Stream through Python logging
    # ============================================================

    # Default logger if not specified
    if logger is None:
        logger = logging.getLogger("finn.subprocess")

    # Log command being executed (at DEBUG level)
    logger.debug(f"Launching: {' '.join(args)}")
    if cwd:
        logger.debug(f"Working directory: {cwd}")

    # Start subprocess with pipes
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=proc_env,
        cwd=cwd,
        text=True,  # Text mode (not bytes)
        bufsize=1,  # Line buffered
    )

    def stream_output(pipe, base_level, stream_name):
        """Read from pipe and log line-by-line."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break

                line = line.rstrip()
                if not line:  # Skip empty lines
                    continue

                # Determine log level (with optional detection)
                if detect_levels:
                    level = _detect_log_level(line, base_level)
                else:
                    level = base_level

                logger.log(level, line)
        except Exception as e:
            logger.exception(f"Error streaming {stream_name}: {e}")
        finally:
            pipe.close()

    # Create threads for parallel stdout/stderr reading
    # (prevents deadlock if one buffer fills up)
    t_out = threading.Thread(
        target=stream_output,
        args=(proc.stdout, stdout_level, "stdout"),
        daemon=True,
        name=f"finn-stdout-{proc.pid}",
    )
    t_err = threading.Thread(
        target=stream_output,
        args=(proc.stderr, stderr_level, "stderr"),
        daemon=True,
        name=f"finn-stderr-{proc.pid}",
    )

    t_out.start()
    t_err.start()

    # Wait for process to complete
    returncode = proc.wait()

    # Wait for threads to finish reading
    t_out.join(timeout=5.0)
    t_err.join(timeout=5.0)

    # Handle errors
    if returncode != 0:
        cmd_str = " ".join(args)
        logger.error(f"Command failed with exit code {returncode}: {cmd_str}")

        if raise_on_error:
            raise subprocess.CalledProcessError(returncode, args)

    return returncode


def which(program):
    "Python equivalent of the shell cmd 'which'."

    # source:
    # https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


mem_primitives_versal = {
    "URAM_72x4096": (72, 4096),
    "URAM_36x8192": (36, 8192),
    "URAM_18x16384": (18, 16384),
    "URAM_9x32768": (9, 32768),
    "BRAM18_36x512": (36, 512),
    "BRAM18_18x1024": (18, 1024),
    "BRAM18_9x2048": (9, 2048),
    "LUTRAM": (1, 64),
}


def get_memutil_alternatives(
    req_mem_spec, mem_primitives=mem_primitives_versal, sort_min_waste=True
):
    """Computes how many instances of a memory primitive are necessary to
    implement a desired memory size, where req_mem_spec is the desired
    size and the primitive_spec is the primitve size. The sizes are expressed
    as tuples of (mem_width, mem_depth). Returns a list of tuples of the form
    (primitive_name, (primitive_count, efficiency, waste)) where efficiency in
    range [0,1] indicates how much of the total capacity is utilized, and waste
    indicates how many bits of storage are wasted. If sort_min_waste is True,
    the list is sorted by increasing waste.
    """
    ret = [
        (primitive_name, memutil(req_mem_spec, primitive_spec))
        for (primitive_name, primitive_spec) in mem_primitives.items()
    ]
    if sort_min_waste:
        ret = sorted(ret, key=lambda x: x[1][2])
    return ret


def memutil(req_mem_spec, primitive_spec):
    """Computes how many instances of a memory primitive are necessary to
    implemented a desired memory size, where req_mem_spec is the desired
    size and the primitive_spec is the primitve size. The sizes are expressed
    as tuples of (mem_width, mem_depth). Returns (primitive_count, efficiency, waste)
    where efficiency in range [0,1] indicates how much of the total capacity is
    utilized, and waste indicates how many bits of storage are wasted."""

    req_width, req_depth = req_mem_spec
    prim_width, prim_depth = primitive_spec

    match_width = roundup_to_integer_multiple(req_width, prim_width)
    match_depth = roundup_to_integer_multiple(req_depth, prim_depth)
    count_width = match_width // prim_width
    count_depth = match_depth // prim_depth
    count = count_depth * count_width
    eff = (req_width * req_depth) / (count * prim_width * prim_depth)
    waste = (count * prim_width * prim_depth) - (req_width * req_depth)
    return (count, eff, waste)


def is_versal(fpgapart):
    """Returns whether board is part of the Versal family"""
    return fpgapart[0:4] in ["xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm"] or fpgapart[0:5] in [
        "xqrvc",
        "xcv80",
    ]


def get_dsp_block(fpgapart):
    if is_versal(fpgapart):
        return "DSP58"
    elif fpgapart[2] == "7":
        return "DSP48E1"
    else:
        return "DSP48E2"
