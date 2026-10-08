# Copyright (c) 2020 Xilinx, Inc.
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

import clize
import json
import logging
import os
import pdb  # NOQA
import subprocess
import sys
import time
import traceback
from qonnx.core.modelwrapper import ModelWrapper

from finn import resources
from finn.builder.build_dataflow_checks import (
    format_report,
    run_all_config_checks,
    save_report,
)
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    default_build_dataflow_steps,
)
from finn.builder.build_dataflow_phases import (
    build_dataflow_phase_lookup,
    recorded_step_times,
)
from finn.builder.build_dataflow_steps import (
    _maybe_enable_verify_behavioral,
    build_dataflow_step_lookup,
)
from finn.builder.kernel_build_config import (
    KernelBuildConfig,
    default_kernel_build_steps,
)
from finn.builder.kernel_build_steps import kernel_build_step_lookup

#: A build's configuration: the HWCustomOp flow's, or the kernel path's.
BuildConfig = DataflowBuildConfig | KernelBuildConfig

#: The file a build directory states its configuration in, by the configuration's type
#: (build_dataflow_directory): one of them, beside model.onnx.
CONFIG_FILES = {
    "dataflow_build_config.json": DataflowBuildConfig,
    "kernel_build_config.json": KernelBuildConfig,
}


# adapted from https://stackoverflow.com/a/39215961
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def _step_lookup(cfg: BuildConfig) -> tuple[list, dict]:
    """The default steps and the steps and phases by name of the build ``cfg``
    configures: the kernel path's for a KernelBuildConfig, the HWCustomOp flow's for
    a DataflowBuildConfig."""
    if isinstance(cfg, KernelBuildConfig):
        return default_kernel_build_steps, kernel_build_step_lookup
    return default_build_dataflow_steps, {
        **build_dataflow_step_lookup,
        **build_dataflow_phase_lookup,
    }


def resolve_build_steps(cfg: BuildConfig, partial: bool = True):
    """Resolve build steps from config, supporting both phases and fine-grained steps:
    those of the configuration's flow (a KernelBuildConfig's or a DataflowBuildConfig's).

    Note: When using phase-based builds with start_step/stop_step, specify phase names
    (e.g., start_step="phase_build_hardware") rather than fine-grained step names.
    Phases save intermediate models for each internal step, so checkpoints like
    step_hw_ipgen.onnx will exist, but the build loop operates at the phase level.
    """
    default_steps, all_steps = _step_lookup(cfg)
    steps = cfg.steps
    if steps is None:
        steps = default_steps

    steps_as_fxns = []
    for transform_step in steps:
        step_name = None

        # Get step function and name
        if type(transform_step) is str:
            step_name = transform_step
            if transform_step in all_steps:
                step_fn = all_steps[transform_step]
            else:
                raise ValueError(f"Unknown step or phase: {transform_step}")
        elif callable(transform_step):
            step_fn = transform_step
            step_name = getattr(transform_step, "__name__", None)
        else:
            raise ValueError(f"Invalid step type: {type(transform_step)}")

        # Inject steps BEFORE this step
        if step_name and step_name in cfg.inject_steps_before:
            for injected_step in cfg.inject_steps_before[step_name]:
                steps_as_fxns.append(injected_step)

        # Add the main step
        steps_as_fxns.append(step_fn)

        # Inject steps AFTER this step
        if step_name and step_name in cfg.inject_steps_after:
            for injected_step in cfg.inject_steps_after[step_name]:
                steps_as_fxns.append(injected_step)

    if partial:
        step_names = list(map(lambda x: x.__name__, steps_as_fxns))
        if cfg.start_step is None:
            start_ind = 0
        else:
            start_ind = step_names.index(cfg.start_step)
        if cfg.stop_step is None:
            stop_ind = len(step_names) - 1
        else:
            stop_ind = step_names.index(cfg.stop_step)
        steps_as_fxns = steps_as_fxns[start_ind : (stop_ind + 1)]

    return steps_as_fxns


def resolve_step_filename(step_name: str, cfg: BuildConfig, step_delta: int = 0):
    step_names = list(map(lambda x: x.__name__, resolve_build_steps(cfg, partial=False)))
    assert step_name in step_names, "start_step %s not found" + step_name
    step_no = step_names.index(step_name) + step_delta
    assert step_no >= 0, "Invalid step+delta combination"
    assert step_no < len(step_names), "Invalid step+delta combination"
    filename = cfg.output_dir + "/intermediate_models/"
    filename += "%s.onnx" % (step_names[step_no])
    return filename


def _run_build_steps(model, cfg, build_dataflow_steps, log):
    """Run the resolved build steps in order, logging to `log`.

    Writes time_per_step.json: the seconds each step took, by name, and for a phase
    also each step it ran, as ``<phase>/<step>``. Returns 0 on success and -1 on the
    first failing step.
    """
    step_num = 1
    time_per_step = dict()
    stdout_logger = StreamToLogger(log, logging.INFO)
    stderr_logger = StreamToLogger(log, logging.ERROR)
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    for transform_step in build_dataflow_steps:
        try:
            step_name = transform_step.__name__
            print("Running step: %s [%d/%d]" % (step_name, step_num, len(build_dataflow_steps)))
            # redirect output to logfile
            if not cfg.verbose:
                sys.stdout = stdout_logger
                sys.stderr = stderr_logger
                # also log current step name to logfile
                print("Running step: %s [%d/%d]" % (step_name, step_num, len(build_dataflow_steps)))
            # run the step
            step_start = time.time()
            with recorded_step_times() as inner_times:
                model = transform_step(model, cfg)
            step_end = time.time()
            # restore stdout/stderr
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            time_per_step[step_name] = step_end - step_start
            for inner_name, seconds in inner_times.items():
                time_per_step[f"{step_name}/{inner_name}"] = seconds
            chkpt_name = "%s.onnx" % (step_name)
            if cfg.save_intermediate_models:
                intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                if not os.path.exists(intermediate_model_dir):
                    os.makedirs(intermediate_model_dir)
                model.save("%s/%s" % (intermediate_model_dir, chkpt_name))
            step_num += 1
        except:  # noqa
            # restore stdout/stderr
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            # print exception info and traceback
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            # start postmortem debug if configured
            if cfg.enable_build_pdb_debug:
                pdb.post_mortem(tb)
            else:
                print("enable_build_pdb_debug not set in build config, exiting...")
            print("Build failed")
            return -1

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    print("Completed successfully")
    return 0


def build_dataflow_cfg(model_filename, cfg: BuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration, by its
    type: a DataflowBuildConfig through the HWCustomOp flow's steps, a
    KernelBuildConfig through the kernel path's (finn.builder.kernel_build_steps).

    :param model_filename: ONNX model filename to build
    :param cfg: Build configuration
    """
    # if start_step is specified, override the input model
    if cfg.start_step is None:
        print("Building dataflow accelerator from " + model_filename)
        model = ModelWrapper(model_filename)
    else:
        intermediate_model_filename = resolve_step_filename(cfg.start_step, cfg, -1)
        print(
            "Building dataflow accelerator from intermediate checkpoint"
            + intermediate_model_filename
        )
        model = ModelWrapper(intermediate_model_filename)
    assert type(model) is ModelWrapper
    finn_build_dir = resources.scratch()

    print("Intermediate outputs will be generated in " + str(finn_build_dir))
    print("Final outputs will be generated in " + cfg.output_dir)
    print("Build log is at " + cfg.output_dir + "/build_dataflow.log")
    # create the output dir if it doesn't exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    if isinstance(cfg, DataflowBuildConfig):
        _maybe_enable_verify_behavioral(cfg)

    # Run configuration checks
    config_report = run_all_config_checks(cfg, model)
    print(format_report(config_report))
    report_path = save_report(config_report, cfg.output_dir)
    print(f"Configuration check report saved to: {report_path}")

    if config_report.has_errors():
        # The kernel path's configuration mutes nothing: its errors stop the build.
        if isinstance(cfg, DataflowBuildConfig) and cfg.mute_config_assertions is True:
            print("WARNING: Configuration errors detected but muted by mute_config_assertions=True")
            print("Build may fail or produce unexpected results.")
        else:
            muting = (
                " or set mute_config_assertions=True"
                if isinstance(cfg, DataflowBuildConfig)
                else ""
            )
            raise AssertionError(
                f"Configuration check failed with errors. Fix the issues above{muting} to proceed."
            )

    build_dataflow_steps = resolve_build_steps(cfg)
    # set up logger
    # Note: basicConfig() is ignored if logging already configured (e.g., in Jupyter)
    # So we explicitly add a file handler to ensure build_dataflow.log is created
    log = logging.getLogger("build_dataflow")
    log.setLevel(logging.DEBUG)
    log.propagate = (
        False  # Prevent propagation to root logger (avoids duplicate console output in Jupyter)
    )

    # Create file handler for build_dataflow.log
    log_file_handler = logging.FileHandler(cfg.output_dir + "/build_dataflow.log", mode="a")
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))

    # Remove any existing handlers from this logger to avoid duplicates
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    # Add the file handler (only file output, no console output)
    log.addHandler(log_file_handler)

    try:
        return _run_build_steps(model, cfg, build_dataflow_steps, log)
    finally:
        log.removeHandler(log_file_handler)
        log_file_handler.close()


def read_build_config(path_to_cfg_dir: str) -> BuildConfig:
    """The build configuration a build directory states: its one configuration file
    (CONFIG_FILES), read as the type the file's name gives. A directory with none, or
    with more than one, is refused, naming the files."""
    stated = [name for name in CONFIG_FILES if os.path.isfile(os.path.join(path_to_cfg_dir, name))]
    if len(stated) != 1:
        raise FileNotFoundError(
            f"{path_to_cfg_dir}: a build directory states one build configuration, one of "
            f"{sorted(CONFIG_FILES)}; it has {stated or 'none'}"
        )
    (name,) = stated
    with open(os.path.join(path_to_cfg_dir, name)) as f:
        return CONFIG_FILES[name].from_json(f.read())


def build_dataflow_directory(path_to_cfg_dir: str):
    """Best-effort build a dataflow accelerator from the specified directory.

    :param path_to_cfg_dir: Directory containing the model and build config

    The specified directory path_to_cfg_dir must contain the following files:

    * model.onnx : ONNX model to be converted to dataflow accelerator
    * one build configuration (read_build_config), whose file name gives its type:
      dataflow_build_config.json, a DataflowBuildConfig (the HWCustomOp flow), or
      kernel_build_config.json, a KernelBuildConfig (the kernel path)

    """
    # get absolute path
    path_to_cfg_dir = os.path.abspath(path_to_cfg_dir)
    assert os.path.isdir(path_to_cfg_dir), "Directory not found: " + path_to_cfg_dir
    onnx_filename = path_to_cfg_dir + "/model.onnx"
    assert os.path.isfile(onnx_filename), "ONNX not found: " + onnx_filename
    cfg = read_build_config(path_to_cfg_dir)
    # Isolate cwd and the pre-start native loader environment for this worker
    # tree, from the toolchain the configuration selects, prepared over this
    # environment so that the child keeps FINN's own settings beside the tools'.
    # The build directory is resolved here, where a relative FINN_BUILD_DIR
    # means what it says. Relative config paths retain their historical
    # directory semantics.
    scratch = resources.scratch()
    scratch.mkdir(parents=True, exist_ok=True)
    toolchain = cfg._resolve_selection().prepare({**os.environ, "FINN_BUILD_DIR": str(scratch)})
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from finn.builder.build_dataflow import "
            "build_dataflow_cfg, read_build_config; "
            "sys.exit(build_dataflow_cfg('model.onnx', read_build_config('.')))",
        ],
        cwd=path_to_cfg_dir,
        env=toolchain.simulation_environment(),
    )
    return child.returncode


def main():
    """Entry point for dataflow builds. Invokes `build_dataflow_directory` using
    command line arguments"""

    def cli(path_to_cfg_dir: str):
        """Build the model and configuration in path_to_cfg_dir."""
        status = build_dataflow_directory(path_to_cfg_dir)
        if status:
            raise SystemExit(status)

    clize.run(cli)


if __name__ == "__main__":
    main()
