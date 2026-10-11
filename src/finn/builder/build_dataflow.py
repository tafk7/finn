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
import sys
import time
import traceback
from dataclasses import replace
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
from finn.builder.build_dataflow_phases import build_dataflow_phase_lookup
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup

#: The file a build directory states its configuration in (build_dataflow_directory),
#: beside model.onnx.
CONFIG_FILE = "dataflow_build_config.json"


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


def _expanded_steps(cfg: DataflowBuildConfig):
    """The build's steps in the order they run (``cfg.steps``, else
    default_build_dataflow_steps), each phase expanded into its steps
    (finn.builder.build_dataflow_phases), and the steps ``inject_steps_before`` and
    ``inject_steps_after`` name around each step or phase placed; with each phase's
    first and last index in that list. A name neither a step nor a phase is refused,
    naming both."""
    steps = cfg.steps
    if steps is None:
        steps = default_build_dataflow_steps

    steps_as_fxns = []
    phase_spans = {}

    def place(step_name, step_fn):
        steps_as_fxns.extend(cfg.inject_steps_before.get(step_name, []))
        steps_as_fxns.append(step_fn)
        steps_as_fxns.extend(cfg.inject_steps_after.get(step_name, []))

    for transform_step in steps:
        if type(transform_step) is str and transform_step in build_dataflow_phase_lookup:
            first = len(steps_as_fxns)
            steps_as_fxns.extend(cfg.inject_steps_before.get(transform_step, []))
            for step_name in build_dataflow_phase_lookup[transform_step]:
                place(step_name, build_dataflow_step_lookup[step_name])
            steps_as_fxns.extend(cfg.inject_steps_after.get(transform_step, []))
            phase_spans.setdefault(transform_step, (first, len(steps_as_fxns) - 1))
        elif type(transform_step) is str:
            if transform_step not in build_dataflow_step_lookup:
                raise ValueError(
                    f"Unknown step or phase: {transform_step}; the steps: "
                    f"{', '.join(build_dataflow_step_lookup)}; the phases: "
                    f"{', '.join(build_dataflow_phase_lookup)}"
                )
            place(transform_step, build_dataflow_step_lookup[transform_step])
        elif callable(transform_step):
            place(getattr(transform_step, "__name__", None), transform_step)
        else:
            raise ValueError(f"Invalid step type: {type(transform_step)}")
    return steps_as_fxns, phase_spans


def _step_index(name: str, steps_as_fxns, phase_spans, last: bool) -> int:
    """The index of the step ``name`` in the build's expanded steps; for a phase, of its
    first step, or its last if ``last``. A name the build does not run is refused,
    naming the build's steps and phases."""
    if name in phase_spans:
        return phase_spans[name][1 if last else 0]
    step_names = [step_fn.__name__ for step_fn in steps_as_fxns]
    if name in step_names:
        return step_names.index(name)
    raise ValueError(
        f"{name!r} is not one of this build's steps or phases; its steps: "
        f"{', '.join(step_names)}; its phases: {', '.join(phase_spans) or 'none'}"
    )


def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True):
    """The build's steps, as functions in the order they run: every phase expanded into
    its steps (finn.builder.build_dataflow_phases), the injected steps placed, and, if
    ``partial``, only those from ``cfg.start_step`` to ``cfg.stop_step``. Either may name
    a step, or a phase: its first step to start from, its last to stop after."""
    steps_as_fxns, phase_spans = _expanded_steps(cfg)
    if partial:
        if cfg.start_step is None:
            start_ind = 0
        else:
            start_ind = _step_index(cfg.start_step, steps_as_fxns, phase_spans, last=False)
        if cfg.stop_step is None:
            stop_ind = len(steps_as_fxns) - 1
        else:
            stop_ind = _step_index(cfg.stop_step, steps_as_fxns, phase_spans, last=True)
        steps_as_fxns = steps_as_fxns[start_ind : (stop_ind + 1)]
    return steps_as_fxns


def resolve_step_filename(step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0):
    """The intermediate model of the step ``step_delta`` after ``step_name`` (a step, or
    a phase: its first step) in the build's steps."""
    steps_as_fxns, phase_spans = _expanded_steps(cfg)
    step_no = _step_index(step_name, steps_as_fxns, phase_spans, last=False) + step_delta
    assert step_no >= 0, "Invalid step+delta combination"
    assert step_no < len(steps_as_fxns), "Invalid step+delta combination"
    filename = cfg.output_dir + "/intermediate_models/"
    filename += "%s.onnx" % (steps_as_fxns[step_no].__name__)
    return filename


def _run_build_steps(model, cfg, build_dataflow_steps, log):
    """Run the resolved build steps in order, logging to `log`.

    Writes time_per_step.json: the seconds each step took, by name. Returns 0 on
    success and -1 on the first failing step.
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
            model = transform_step(model, cfg)
            step_end = time.time()
            # restore stdout/stderr
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            time_per_step[step_name] = step_end - step_start
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
            # start postmortem debug if configured, and only where someone can answer it:
            # with stdin not a terminal (pytest, a batch job) pdb's read would fail and
            # report itself instead of the error above
            if not cfg.enable_build_pdb_debug:
                print("enable_build_pdb_debug not set in build config, exiting...")
            elif sys.stdin is None or not sys.stdin.isatty():
                print("enable_build_pdb_debug set, but stdin is not a terminal, exiting...")
            else:
                pdb.post_mortem(tb)
            print("Build failed")
            return -1

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    print("Completed successfully")
    return 0


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration, through
    the kernel path's steps (finn.builder.build_dataflow_steps).

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

    # Run configuration checks
    config_report = run_all_config_checks(cfg)
    print(format_report(config_report))
    report_path = save_report(config_report, cfg.output_dir)
    print(f"Configuration check report saved to: {report_path}")

    if config_report.has_errors():
        raise AssertionError(
            "Configuration check failed with errors. Fix the issues above to proceed."
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


def _resolved_against(cfg: DataflowBuildConfig, directory: str) -> DataflowBuildConfig:
    """``cfg`` with each relative path it states joined to ``directory``: ``output_dir``,
    ``verify_input_npy``, the preparation's ``preprocessing`` model, a strategy's
    ``path`` (``pinned``'s choices file), the toolchain's settings scripts and command
    directory. An absolute path, and an empty command directory, stay as stated."""

    def resolve(path: str) -> str:
        return os.path.join(directory, path)

    preparation = cfg.preparation
    if preparation.preprocessing is not None:
        preparation = replace(preparation, preprocessing=resolve(preparation.preprocessing))
    exploration = [
        {**spec, "path": resolve(spec["path"])} if isinstance(spec.get("path"), str) else spec
        for spec in cfg.kernel_exploration
    ]
    toolchain = cfg.toolchain
    if toolchain is not None:
        toolchain = replace(
            toolchain,
            settings=tuple(map(resolve, toolchain.settings)),
            command_dir=resolve(toolchain.command_dir)
            if toolchain.command_dir
            else toolchain.command_dir,
        )
    return replace(
        cfg,
        output_dir=resolve(cfg.output_dir),
        verify_input_npy=resolve(cfg.verify_input_npy),
        preparation=preparation,
        kernel_exploration=exploration,
        toolchain=toolchain,
    )


def read_build_config(path_to_cfg_dir: str) -> DataflowBuildConfig:
    """The build configuration a build directory states: its CONFIG_FILE, each relative
    path in it resolved against the directory (``_resolved_against``), so that it names
    the same file from whatever directory the build runs. A directory without one is
    refused, naming the file."""
    path = os.path.join(path_to_cfg_dir, CONFIG_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path_to_cfg_dir}: a build directory states {CONFIG_FILE}")
    with open(path) as f:
        cfg = DataflowBuildConfig.from_json(f.read())
    return _resolved_against(cfg, os.path.abspath(path_to_cfg_dir))


def build_dataflow_directory(path_to_cfg_dir: str):
    """Best-effort build a dataflow accelerator from the specified directory, in this
    process (build_dataflow_cfg), its working directory unchanged: the configuration's
    relative paths name files in the directory (read_build_config), and a relative
    FINN_BUILD_DIR the caller's. Returns build_dataflow_cfg's result, 0 or -1 for a
    failed step; a configuration it refuses raises.

    :param path_to_cfg_dir: Directory containing the model and build config

    The specified directory path_to_cfg_dir must contain the following files:

    * model.onnx : ONNX model to be converted to dataflow accelerator
    * dataflow_build_config.json : its DataflowBuildConfig (read_build_config)

    """
    # get absolute path
    path_to_cfg_dir = os.path.abspath(path_to_cfg_dir)
    assert os.path.isdir(path_to_cfg_dir), "Directory not found: " + path_to_cfg_dir
    onnx_filename = path_to_cfg_dir + "/model.onnx"
    assert os.path.isfile(onnx_filename), "ONNX not found: " + onnx_filename
    return build_dataflow_cfg(onnx_filename, read_build_config(path_to_cfg_dir))


def main():
    """Entry point for dataflow builds. Invokes `build_dataflow_directory` using
    command line arguments, and exits with its result: 0, or 255 for a failed step
    (its -1); a configuration it refuses raises, exiting 1 with the traceback."""

    def cli(path_to_cfg_dir: str):
        """Build the model and configuration in path_to_cfg_dir."""
        status = build_dataflow_directory(path_to_cfg_dir)
        if status:
            raise SystemExit(status)

    clize.run(cli)


if __name__ == "__main__":
    main()
